"""Hybrid legal + ML compliance engine for FinAudit-AI.

Deterministic Indian tax rules decide statutory breaches. The Isolation Forest
from ``ml_engine.detect_anomaly`` only contributes a statistical penalty. A
checksum or a section of the Income-tax Act is never left to a probability.

Per-invoice Audit Readiness Score (0–100)::

    100
    − sum(penalty of unresolved rule flags)
    − ml_penalty

``ml_penalty`` is ``anomaly_score × 15`` when the forest flags the invoice, and
``anomaly_score × 4`` otherwise, so a quiet score still nudges the books and a
true outlier costs more. The firm-level score is the unweighted mean of the
invoice scores.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy.orm import Session

from ml_engine import (
    BLOCKED_ITC_CATEGORIES,
    blocked_itc_tags,
    build_ledger_text,
    detect_anomaly,
    predict_ledger,
)
from models import AuditFlag, FlagSource, Invoice, ReviewStatus, Severity
from ocr_pipeline import validate_gstin

# Points removed from one invoice's readiness score.
PENALTY_CRITICAL = 22.0
PENALTY_WARNING = 8.0
ML_ANOMALY_WEIGHT = 15.0
ML_ELEVATED_WEIGHT = 4.0

CASH_DISALLOWANCE_LIMIT = 10_000.0  # Income-tax Act s.40A(3)
TDS_SINGLE_LIMIT = 30_000.0  # s.194C / s.194J single-payment threshold used here
GST_MATH_TOLERANCE = 2.0  # rupees

# The brief scopes the TDS alert to these two ledgers, not every 194C tag
# the classifier also emits (marketing, repairs).
TDS_BY_CATEGORY = {
    "Contractor Fees": "194C",
    "Professional Fees": "194J",
}

RULE_GSTIN = "GSTIN_CHECKSUM"
RULE_CASH = "CASH_40A3"
RULE_ITC = "BLOCKED_ITC_17_5"
RULE_TDS = "TDS_194C_194J"
RULE_DUPLICATE = "DUPLICATE_INVOICE"
RULE_MISSING = "MISSING_INVOICE"
RULE_GST_MATH = "GST_MATH"
RULE_ML = "ML_ANOMALY"


def _round2(value: float) -> float:
    return round(float(value) + 0.0, 2)


def _party_gstin(invoice: Invoice) -> Optional[str]:
    if (invoice.entry_type or "EXPENSE") == "SALE":
        return invoice.buyer_gstin or None
    return invoice.vendor_gstin or None


def party_key(invoice: Invoice) -> str:
    """Identity used for duplicate-invoice detection: GSTIN, else the party name."""
    gstin = (_party_gstin(invoice) or "").strip().upper()
    if gstin:
        return f"gstin:{gstin}"
    name = (invoice.vendor_name or "").strip().lower()
    return f"name:{name}"


def _invoice_no(invoice: Invoice) -> str:
    return (invoice.invoice_no or "").strip().upper()


def _amount(invoice: Invoice) -> float:
    if invoice.total_amount is not None:
        return float(invoice.total_amount)
    if invoice.base_amount is not None:
        return float(invoice.base_amount) + float(invoice.gst_amount or 0)
    return 0.0


def _fee_amount(invoice: Invoice) -> float:
    """TDS is tested on the professional/contractor fee, which is the taxable value."""
    if invoice.base_amount is not None:
        return float(invoice.base_amount)
    return _amount(invoice)


def normalise_totals(invoice: Invoice) -> None:
    """Fill GST total from components, and the invoice total from base + GST.

    Does not invent a GST figure from the rate. The math-mismatch rule needs
    the number that was actually recorded.
    """
    if invoice.gst_amount is None:
        parts = [invoice.cgst_amount, invoice.sgst_amount, invoice.igst_amount]
        if any(p is not None for p in parts):
            invoice.gst_amount = _round2(sum(float(p or 0) for p in parts))
    if invoice.total_amount is None and invoice.base_amount is not None:
        invoice.total_amount = _round2(float(invoice.base_amount) + float(invoice.gst_amount or 0))
    if invoice.entry_type not in ("SALE", "EXPENSE"):
        invoice.entry_type = "EXPENSE"


def apply_gstin_metadata(invoice: Invoice) -> None:
    """Persist the checksum result for the GSTIN that matters on this voucher."""
    raw = _party_gstin(invoice)
    if not raw:
        invoice.gstin_valid = False if (invoice.entry_type == "EXPENSE" and (invoice.gst_amount or 0) > 0) else None
        return
    result = validate_gstin(raw)
    invoice.gstin_valid = bool(result.is_valid)
    if result.state_code:
        invoice.gstin_state_code = result.state_code


def classify_invoice(invoice: Invoice) -> Dict[str, Any]:
    """Run the ledger classifier and copy its outputs onto the invoice row.

    A category already marked ``ledger_overridden`` is kept. Sales never carry
    a blocked-ITC flag: section 17(5) restricts input credit, not output tax.
    """
    descriptions = [item.description for item in (invoice.line_items or []) if item.description]
    text = build_ledger_text(invoice.vendor_name, descriptions, invoice.raw_text)
    prediction = predict_ledger(text or (invoice.vendor_name or ""))
    if not invoice.ledger_overridden:
        invoice.ledger_category = prediction["category"]
        invoice.ledger_confidence = float(prediction["confidence"])
    invoice.ledger_alternatives = prediction.get("top_k") or []
    invoice.ledger_model_version = prediction.get("model_version") or None
    tags = list(prediction.get("itc_tags") or [])
    extra = blocked_itc_tags(" ".join(filter(None, [text, invoice.ledger_category])))
    merged: List[str] = []
    for tag in tags + extra:
        if tag not in merged:
            merged.append(tag)
    invoice.itc_tags = merged
    blocked_category = (invoice.ledger_category or "") in BLOCKED_ITC_CATEGORIES
    invoice.is_blocked_itc = invoice.entry_type != "SALE" and (blocked_category or bool(merged))
    return prediction


def _feature_dict(invoice: Invoice) -> Dict[str, Any]:
    when = invoice.invoice_date
    return {
        "total_amount": invoice.total_amount,
        "base_amount": invoice.base_amount,
        "gst_rate": invoice.gst_rate,
        "gst_amount": invoice.gst_amount,
        "invoice_date": when.isoformat() if isinstance(when, date) else when,
        "ledger_category": invoice.ledger_category,
    }


def score_anomaly(invoice: Invoice, vendor_history: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Persist Isolation Forest outputs. ``vendor_history`` excludes this invoice."""
    result = detect_anomaly(_feature_dict(invoice), vendor_history)
    invoice.anomaly_score = float(result["score"])
    invoice.is_anomaly = bool(result["is_anomaly"])
    invoice.anomaly_reasons = list(result.get("reasons") or [])
    invoice.anomaly_features = dict(result.get("features") or {})
    invoice.anomaly_model_version = result.get("model_version") or None
    return result


def _hit(
    code: str,
    severity: str,
    title: str,
    message: str,
    legal_reference: str,
    evidence: Dict[str, Any],
    source: str = FlagSource.RULE.value,
) -> Dict[str, Any]:
    penalty = PENALTY_CRITICAL if severity == Severity.CRITICAL.value else PENALTY_WARNING
    if severity not in (Severity.CRITICAL.value, Severity.WARNING.value):
        penalty = 0.0
    return {
        "rule_code": code,
        "source": source,
        "severity": severity,
        "title": title,
        "message": message,
        "legal_reference": legal_reference,
        "penalty": penalty,
        "evidence": evidence,
    }


def _is_b2b(invoice: Invoice) -> bool:
    """A sale is B2B only when a buyer GSTIN was recorded.

    An expense that charges GST, or that quotes a supplier GSTIN, is treated as
    a B2B purchase. A cash retail slip with no GST and no GSTIN is not.
    """
    if invoice.entry_type == "SALE":
        return bool((invoice.buyer_gstin or "").strip())
    if (invoice.vendor_gstin or "").strip():
        return True
    return float(invoice.gst_amount or 0) > 0


def evaluate_rules(invoice: Invoice, peers: Sequence[Invoice]) -> List[Dict[str, Any]]:
    """Return the statutory hits for one invoice. Does not touch the session."""
    hits: List[Dict[str, Any]] = []
    entry = invoice.entry_type or "EXPENSE"

    if _is_b2b(invoice):
        raw = _party_gstin(invoice)
        if not raw:
            hits.append(_hit(
                RULE_GSTIN, Severity.CRITICAL.value,
                "B2B invoice has no GSTIN",
                "This B2B invoice charges GST (or is a sale to a registered buyer) but the "
                "15-digit counterparty GSTIN is missing, so the credit and the invoice both fail section 16 / rule 46.",
                "CGST Act s.16 & s.31; CGST Rules, rule 46 (tax invoice must carry the GSTIN)",
                {"entry_type": entry, "gst_amount": invoice.gst_amount},
            ))
        else:
            checked = validate_gstin(raw)
            if not checked.is_valid:
                hits.append(_hit(
                    RULE_GSTIN, Severity.CRITICAL.value,
                    "GSTIN failed the modulo-36 checksum",
                    checked.reason or f"GSTIN {raw} is not a valid 15-character GSTIN.",
                    "GSTIN structure (CGST Rules, rule 46); modulo-36 check character",
                    {
                        "gstin": checked.gstin,
                        "expected_check_char": checked.expected_check_char,
                        "state_code": checked.state_code,
                        "reason": checked.reason,
                    },
                ))

    if entry == "EXPENSE" and (invoice.payment_mode or "").upper() == "CASH" and _amount(invoice) > CASH_DISALLOWANCE_LIMIT:
        paid = _amount(invoice)
        hits.append(_hit(
            RULE_CASH, Severity.CRITICAL.value,
            "Cash payment above ₹10,000",
            f"Cash expense of ₹{paid:,.2f} exceeds ₹{CASH_DISALLOWANCE_LIMIT:,.0f}. "
            "The expenditure is liable to be disallowed.",
            "Income-tax Act, 1961 — section 40A(3)",
            {"payment_mode": invoice.payment_mode, "amount": paid, "limit": CASH_DISALLOWANCE_LIMIT},
        ))

    if entry == "EXPENSE":
        blob = " ".join(filter(None, [
            invoice.vendor_name,
            invoice.raw_text,
            invoice.ledger_category,
            " ".join(invoice.itc_tags or []),
            " ".join(item.description for item in (invoice.line_items or []) if item.description),
        ]))
        tags = list(invoice.itc_tags or [])
        for tag in blocked_itc_tags(blob):
            if tag not in tags:
                tags.append(tag)
        category_blocked = (invoice.ledger_category or "") in BLOCKED_ITC_CATEGORIES
        if category_blocked or tags or invoice.is_blocked_itc:
            why = invoice.ledger_category or "unclassified"
            if tags:
                why = f"{why}; keyword tags: {', '.join(tags)}"
            hits.append(_hit(
                RULE_ITC, Severity.WARNING.value,
                "Input tax credit looks blocked",
                f"Food, restaurant, club or personal-use indicators ({why}). "
                "ITC on this expense should not be claimed without a documented statutory exception.",
                "CGST Act, 2017 — section 17(5)",
                {"ledger_category": invoice.ledger_category, "itc_tags": tags, "ml_confidence": invoice.ledger_confidence},
            ))
            invoice.is_blocked_itc = True
            invoice.itc_tags = tags

        section = TDS_BY_CATEGORY.get(invoice.ledger_category or "")
        fee = _fee_amount(invoice)
        if section and fee > TDS_SINGLE_LIMIT:
            hits.append(_hit(
                RULE_TDS, Severity.WARNING.value,
                f"TDS threshold crossed under section {section}",
                f"{invoice.ledger_category} of ₹{fee:,.2f} is above ₹{TDS_SINGLE_LIMIT:,.0f}. "
                f"Confirm that tax was deducted under section {section} before the books are signed off.",
                f"Income-tax Act, 1961 — section {section}",
                {
                    "ledger_category": invoice.ledger_category,
                    "section": section,
                    "fee_amount": fee,
                    "limit": TDS_SINGLE_LIMIT,
                    "ml_confidence": invoice.ledger_confidence,
                },
            ))

    number = _invoice_no(invoice)
    if not number:
        hits.append(_hit(
            RULE_MISSING, Severity.CRITICAL.value,
            "Invoice number is missing",
            "No invoice number was captured. A tax invoice without a consecutive serial number cannot support a credit or a sale.",
            "CGST Rules, rule 46(b) — consecutive serial number",
            {"vendor_name": invoice.vendor_name, "entry_type": entry},
        ))
    else:
        key = (party_key(invoice), number)
        twins = [
            peer.id for peer in peers
            if _invoice_no(peer) and (party_key(peer), _invoice_no(peer)) == key
        ]
        if len(twins) > 1:
            hits.append(_hit(
                RULE_DUPLICATE, Severity.CRITICAL.value,
                "Duplicate vendor invoice number",
                f"Invoice {invoice.invoice_no} for this party appears on {len(twins)} rows. "
                "Claiming it twice would double-count turnover or input credit.",
                "CGST Rules, rule 46; duplicate-invoice control in the books of account",
                {"invoice_no": invoice.invoice_no, "party_key": key[0], "invoice_ids": twins},
            ))

    if invoice.base_amount is not None and invoice.gst_rate is not None and invoice.gst_amount is not None:
        expected = _round2(float(invoice.base_amount) * float(invoice.gst_rate) / 100.0)
        actual = _round2(float(invoice.gst_amount))
        diff = _round2(abs(expected - actual))
        if diff > GST_MATH_TOLERANCE:
            hits.append(_hit(
                RULE_GST_MATH, Severity.WARNING.value,
                "GST amount does not match the rate",
                f"Taxable value ₹{float(invoice.base_amount):,.2f} × {float(invoice.gst_rate):g}% "
                f"= ₹{expected:,.2f}, but GST recorded is ₹{actual:,.2f} (difference ₹{diff:,.2f}, tolerance ₹{GST_MATH_TOLERANCE:,.0f}).",
                "CGST Act, section 15 (value of supply) read with the rate notification",
                {
                    "base_amount": invoice.base_amount,
                    "gst_rate": invoice.gst_rate,
                    "expected_gst": expected,
                    "gst_amount": actual,
                    "difference": diff,
                },
            ))
    return hits


def _ml_hit(invoice: Invoice, anomaly: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not anomaly.get("is_anomaly"):
        return None
    reasons = anomaly.get("reasons") or ["Multivariate outlier on amount, tax and timing features."]
    score = float(anomaly.get("score") or 0)
    severity = Severity.HIGH.value if score >= 0.75 else Severity.MEDIUM.value
    return {
        "rule_code": RULE_ML,
        "source": FlagSource.ML.value,
        "severity": severity,
        "title": "Isolation Forest anomaly",
        "message": " ".join(reasons),
        "legal_reference": "Statistical control — not a statutory disallowance",
        "penalty": 0.0,  # charged once via ml_penalty so the score is not double-counted
        "evidence": {
            "score": anomaly.get("score"),
            "raw_score": anomaly.get("raw_score"),
            "threshold": anomaly.get("threshold"),
            "reasons": reasons,
            "model_version": anomaly.get("model_version"),
        },
    }


def _ml_penalty(anomaly: Dict[str, Any]) -> float:
    score = max(0.0, min(1.0, float(anomaly.get("score") or 0)))
    weight = ML_ANOMALY_WEIGHT if anomaly.get("is_anomaly") else ML_ELEVATED_WEIGHT
    return _round2(score * weight)


def readiness_score(rule_hits: Sequence[Dict[str, Any]], anomaly: Dict[str, Any]) -> Dict[str, float]:
    rule_penalty = _round2(sum(float(hit["penalty"]) for hit in rule_hits))
    ml_penalty = _ml_penalty(anomaly)
    score = _round2(max(0.0, min(100.0, 100.0 - rule_penalty - ml_penalty)))
    return {"audit_readiness_score": score, "rule_penalty": rule_penalty, "ml_penalty": ml_penalty}


def _status_for(invoice: Invoice, open_hits: Sequence[Dict[str, Any]]) -> str:
    """Do not overwrite a status an advocate has already chosen."""
    locked = {ReviewStatus.IN_REVIEW.value, ReviewStatus.DISALLOWED.value}
    if invoice.status in locked:
        return invoice.status
    anomaly_open = bool(invoice.is_anomaly) and not _is_kept_resolved(invoice, RULE_ML)
    if any(hit["severity"] == Severity.CRITICAL.value for hit in open_hits):
        return ReviewStatus.ESCALATED.value
    if open_hits or anomaly_open:
        return ReviewStatus.PENDING_REVIEW.value
    return ReviewStatus.CLEARED.value


def attach_prediction(invoice: Invoice, override: Optional[str] = None) -> Dict[str, Any]:
    """Run the ledger classifier. ``override`` is a human label and replaces the model category."""
    invoice.ledger_overridden = False
    invoice.ledger_category = None
    invoice.ledger_model_version = None
    prediction = classify_invoice(invoice)
    if override and override != prediction["category"]:
        invoice.ledger_category = override
        invoice.ledger_overridden = True
        invoice.ledger_confidence = 1.0
        invoice.is_blocked_itc = invoice.entry_type != "SALE" and (
            override in BLOCKED_ITC_CATEGORIES or bool(invoice.itc_tags)
        )
    return prediction


def _upsert_flags(db: Session, invoice: Invoice, hits: Sequence[Dict[str, Any]]) -> None:
    """Update flags in place so advocate remarks keep their ``flag_id``.

    A resolved flag is left alone and is not re-opened. An unresolved flag whose
    rule no longer fires is deleted.
    """
    existing = {flag.rule_code: flag for flag in list(invoice.flags or [])}
    seen = set()
    for hit in hits:
        seen.add(hit["rule_code"])
        current = existing.get(hit["rule_code"])
        if current is not None and current.is_resolved:
            continue
        if current is None:
            current = AuditFlag(invoice_id=invoice.id, rule_code=hit["rule_code"])
            invoice.flags.append(current)
            db.add(current)
        current.source = hit["source"]
        current.severity = hit["severity"]
        current.title = hit["title"]
        current.message = hit["message"]
        current.legal_reference = hit["legal_reference"]
        current.penalty = float(hit["penalty"])
        current.evidence = hit["evidence"]
        current.is_resolved = False
    for code, flag in existing.items():
        if code not in seen and not flag.is_resolved:
            invoice.flags.remove(flag)
            db.delete(flag)


def vendor_histories(invoices: Sequence[Invoice]) -> Dict[int, List[float]]:
    grouped: Dict[str, List[Invoice]] = defaultdict(list)
    for invoice in invoices:
        grouped[party_key(invoice)].append(invoice)
    histories: Dict[int, List[float]] = {}
    for invoice in invoices:
        histories[invoice.id] = [
            float(peer.total_amount)
            for peer in grouped[party_key(invoice)]
            if peer.id != invoice.id and peer.total_amount
        ]
    return histories


def audit_invoice(
    db: Session,
    invoice: Invoice,
    peers: Optional[Sequence[Invoice]] = None,
    vendor_history: Optional[Sequence[float]] = None,
) -> Invoice:
    """Classify (if needed), score the forest, apply rules, and store the readiness score."""
    normalise_totals(invoice)
    if not invoice.ledger_category or not invoice.ledger_model_version:
        classify_invoice(invoice)
    apply_gstin_metadata(invoice)
    if peers is None:
        peers = db.query(Invoice).all()
    if vendor_history is None:
        vendor_history = vendor_histories(peers).get(invoice.id, [])
    anomaly = score_anomaly(invoice, vendor_history)
    rule_hits = evaluate_rules(invoice, peers)
    ml = _ml_hit(invoice, anomaly)
    all_hits = list(rule_hits)
    if ml is not None:
        all_hits.append(ml)
    _upsert_flags(db, invoice, all_hits)
    # Resolved hits stay on the row but do not reduce the score.
    open_hits = [hit for hit in all_hits if not _is_kept_resolved(invoice, hit["rule_code"])]
    scored = readiness_score([hit for hit in open_hits if hit["source"] == FlagSource.RULE.value], anomaly if not _is_kept_resolved(invoice, RULE_ML) else {"score": 0.0, "is_anomaly": False})
    invoice.audit_score = scored["audit_readiness_score"]
    invoice.rule_penalty = scored["rule_penalty"]
    invoice.ml_penalty = scored["ml_penalty"]
    invoice.status = _status_for(invoice, open_hits)
    return invoice


def _is_kept_resolved(invoice: Invoice, rule_code: str) -> bool:
    for flag in invoice.flags or []:
        if flag.rule_code == rule_code and flag.is_resolved:
            return True
    return False


def audit_books(db: Session) -> List[Invoice]:
    """Re-score every invoice. Call this after inserts so duplicate detection sees the new row."""
    invoices = db.query(Invoice).all()
    histories = vendor_histories(invoices)
    for invoice in invoices:
        audit_invoice(db, invoice, peers=invoices, vendor_history=histories.get(invoice.id, []))
    return invoices


def itc_eligible(invoice: Invoice) -> bool:
    """ITC a CA can put on the credit sheet: inward, checksum-valid, not blocked, no open critical flag."""
    if (invoice.entry_type or "EXPENSE") != "EXPENSE":
        return False
    if invoice.is_blocked_itc:
        return False
    if not invoice.gstin_valid:
        return False
    if float(invoice.gst_amount or 0) <= 0:
        return False
    for flag in invoice.flags or []:
        if flag.severity == Severity.CRITICAL.value and not flag.is_resolved:
            return False
    return True


def _latest_remark(invoice: Invoice, flag_id: Optional[int]) -> Optional[str]:
    remarks = list(invoice.remarks or [])
    if not remarks:
        return None
    tied = [remark for remark in remarks if flag_id is not None and remark.flag_id == flag_id]
    pool = tied or remarks
    pool = sorted(pool, key=lambda remark: remark.created_at or datetime.min)
    return pool[-1].remark


def _issue_dict(invoice: Invoice, flag: AuditFlag) -> Dict[str, Any]:
    created = flag.created_at or datetime.utcnow()
    return {
        "id": flag.id,
        "invoice_id": invoice.id,
        "rule_code": flag.rule_code,
        "source": flag.source,
        "severity": flag.severity,
        "title": flag.title,
        "message": flag.message,
        "legal_reference": flag.legal_reference,
        "penalty": float(flag.penalty or 0),
        "is_resolved": bool(flag.is_resolved),
        "created_at": created,
        "vendor_name": invoice.vendor_name,
        "invoice_no": invoice.invoice_no,
        "invoice_date": invoice.invoice_date,
        "entry_type": invoice.entry_type or "EXPENSE",
        "total_amount": invoice.total_amount,
        "ledger_category": invoice.ledger_category,
        "ml_confidence": invoice.ledger_confidence,
        "anomaly_score": invoice.anomaly_score,
        "is_anomaly": bool(invoice.is_anomaly),
        "advocate_remark": _latest_remark(invoice, flag.id),
        "audit_status": invoice.status,
    }


def build_audit_summary(invoices: Iterable[Invoice]) -> Dict[str, Any]:
    """Firm-level readiness, GST position, cashflow and every flag."""
    rows = list(invoices)
    scores = [float(row.audit_score) for row in rows if row.audit_score is not None]
    readiness = _round2(sum(scores) / len(scores)) if scores else 100.0
    readiness = max(0.0, min(100.0, readiness))

    sales = [row for row in rows if row.entry_type == "SALE"]
    expenses = [row for row in rows if row.entry_type != "SALE"]
    output_gst = _round2(sum(float(row.gst_amount or 0) for row in sales))
    eligible = [row for row in expenses if itc_eligible(row)]
    eligible_itc = _round2(sum(float(row.gst_amount or 0) for row in eligible))

    issues: List[Dict[str, Any]] = []
    severity_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        for flag in row.flags or []:
            issues.append(_issue_dict(row, flag))
            if not flag.is_resolved:
                severity_counts[flag.severity] += 1
    rank = {Severity.CRITICAL.value: 0, Severity.HIGH.value: 1, Severity.MEDIUM.value: 2, Severity.WARNING.value: 3, Severity.LOW.value: 4, Severity.INFO.value: 5}
    issues.sort(key=lambda item: (item["is_resolved"], rank.get(item["severity"], 9), item["invoice_id"], item["id"] or 0))

    cash: Dict[str, Dict[str, float]] = defaultdict(lambda: {"sales": 0.0, "expenses": 0.0})
    for row in rows:
        if row.invoice_date is None:
            continue
        bucket = cash[row.invoice_date.strftime("%Y-%m")]
        if row.entry_type == "SALE":
            bucket["sales"] += float(row.total_amount or 0)
        else:
            bucket["expenses"] += float(row.total_amount or 0)
    monthly = [
        {
            "month": month,
            "sales": _round2(vals["sales"]),
            "expenses": _round2(vals["expenses"]),
            "net": _round2(vals["sales"] - vals["expenses"]),
        }
        for month, vals in sorted(cash.items())
    ]

    by_category: Dict[str, Dict[str, Any]] = {}
    for row in expenses:
        name = row.ledger_category or "Uncategorised"
        slot = by_category.setdefault(name, {"category": name, "count": 0, "total_amount": 0.0, "confidence_sum": 0.0, "confidence_n": 0})
        slot["count"] += 1
        slot["total_amount"] += float(row.total_amount or 0)
        if row.ledger_confidence is not None:
            slot["confidence_sum"] += float(row.ledger_confidence)
            slot["confidence_n"] += 1
    categories = []
    for slot in sorted(by_category.values(), key=lambda item: item["total_amount"], reverse=True):
        n = slot.pop("confidence_n")
        total_conf = slot.pop("confidence_sum")
        slot["total_amount"] = _round2(slot["total_amount"])
        slot["avg_confidence"] = round(total_conf / n, 4) if n else None
        categories.append(slot)

    pending = sum(1 for row in rows if row.status in {
        ReviewStatus.PENDING_REVIEW.value, ReviewStatus.IN_REVIEW.value, ReviewStatus.ESCALATED.value,
    })
    return {
        "audit_readiness_score": readiness,
        "total_invoices": len(rows),
        "total_sales": _round2(sum(float(row.total_amount or 0) for row in sales)),
        "total_expenses": _round2(sum(float(row.total_amount or 0) for row in expenses)),
        "output_gst": output_gst,
        "eligible_itc": eligible_itc,
        "net_gst_payable": _round2(output_gst - eligible_itc),
        "anomaly_count": sum(1 for row in rows if row.is_anomaly),
        "critical_count": severity_counts.get(Severity.CRITICAL.value, 0),
        "warning_count": severity_counts.get(Severity.WARNING.value, 0),
        "blocked_itc_count": sum(1 for row in rows if row.is_blocked_itc),
        "pending_review": pending,
        "flags_by_severity": dict(severity_counts),
        "categories": categories,
        "monthly_cashflow": monthly,
        "issues": issues,
    }
