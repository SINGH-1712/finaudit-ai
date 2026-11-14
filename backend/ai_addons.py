"""AI add-on endpoints for FinAudit-AI.

Three capstone enhancements isolated from the v1.0-stable routes:

1. GET  /api/ai/forecast            — LinearRegression cash-flow + GST forecast
2. POST /api/ai/retrain-classifier  — active-learning retrain using CA-verified labels
3. POST /api/ai/ask-ledger          — natural-language copilot over the ledger

Mounted from ``main.py`` via ``app.include_router(router)``.
Uses the exact column names defined in ``models.py``.
"""
from __future__ import annotations

import importlib
import logging
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlalchemy.orm import Session

from database import get_db
from models import Invoice, MLModelRun, ReviewStatus

log = logging.getLogger("finaudit.ai_addons")

router = APIRouter(prefix="/api/ai", tags=["AI Add-ons"])
_MODELS_DIR = Path(__file__).parent / "models"


# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------
def _inr(n) -> str:
    """Indian lakh/crore formatted rupee string, e.g. ₹1,23,456.78."""
    if n is None:
        n = 0
    try:
        n = float(n)
    except (TypeError, ValueError):
        n = 0
    neg = n < 0
    n = abs(n)
    whole, _, frac = f"{n:.2f}".partition(".")
    if len(whole) > 3:
        last3, rest, chunks = whole[-3:], whole[:-3], []
        while len(rest) > 2:
            chunks.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            chunks.insert(0, rest)
        whole = ",".join(chunks) + "," + last3
    return f"₹{'-' if neg else ''}{whole}.{frac}"


def _month_key(d: Optional[date]) -> Optional[str]:
    if d is None:
        return None
    return f"{d.year:04d}-{d.month:02d}"


def _next_month(key: str) -> str:
    y, m = (int(x) for x in key.split("-"))
    return f"{y + 1:04d}-01" if m == 12 else f"{y:04d}-{m + 1:02d}"


def _iter_invoices(db: Session) -> List[Invoice]:
    return db.query(Invoice).order_by(Invoice.invoice_date, Invoice.id).all()


# =========================================================================
# Feature 1 — Predictive cash-flow + GST forecast
# =========================================================================
@router.get("/forecast")
def forecast(db: Session = Depends(get_db)) -> dict:
    """Aggregate monthly Sales / Expenses / GST and forecast the next month."""
    invoices = _iter_invoices(db)
    monthly: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"sales": 0.0, "expenses": 0.0, "output_gst": 0.0, "input_gst": 0.0}
    )
    for inv in invoices:
        mk = _month_key(inv.invoice_date)
        if mk is None:
            continue
        total = float(inv.total_amount or 0.0)
        gst = float(inv.gst_amount or 0.0)
        if inv.entry_type == "SALE":
            monthly[mk]["sales"] += total
            monthly[mk]["output_gst"] += gst
        else:
            monthly[mk]["expenses"] += total
            if not inv.is_blocked_itc:
                monthly[mk]["input_gst"] += gst

    months = sorted(monthly)
    history = [
        {
            "month": m,
            "sales": round(monthly[m]["sales"], 2),
            "expenses": round(monthly[m]["expenses"], 2),
            "output_gst": round(monthly[m]["output_gst"], 2),
            "input_gst": round(monthly[m]["input_gst"], 2),
            "net_gst": round(monthly[m]["output_gst"] - monthly[m]["input_gst"], 2),
        }
        for m in months
    ]

    if len(months) < 2:
        return {
            "history": history,
            "forecast": None,
            "note": (
                f"Need at least 2 distinct months of dated vouchers to forecast. "
                f"Currently have {len(months)} month(s)."
            ),
        }

    try:
        from sklearn.linear_model import LinearRegression
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"scikit-learn not available: {exc}") from exc

    X = np.array(range(len(months)), dtype=float).reshape(-1, 1)
    next_x = np.array([[len(months)]], dtype=float)

    def _predict(series: List[float]) -> float:
        model = LinearRegression().fit(X, np.array(series, dtype=float))
        return float(model.predict(next_x)[0])

    pred_sales = max(0.0, _predict([monthly[m]["sales"] for m in months]))
    pred_exp = max(0.0, _predict([monthly[m]["expenses"] for m in months]))
    pred_out = max(0.0, _predict([monthly[m]["output_gst"] for m in months]))
    pred_in = max(0.0, _predict([monthly[m]["input_gst"] for m in months]))
    pred_net = pred_out - pred_in

    confidence = "low" if len(months) < 3 else ("medium" if len(months) < 6 else "high")
    next_key = _next_month(months[-1])

    return {
        "history": history,
        "forecast": {
            "month": next_key,
            "sales": round(pred_sales, 2),
            "expenses": round(pred_exp, 2),
            "output_gst": round(pred_out, 2),
            "input_gst": round(pred_in, 2),
            "net_gst": round(pred_net, 2),
            "method": "linear_regression",
            "n_months": len(months),
            "confidence": confidence,
        },
        "note": (
            f"Forecast for {next_key} via LinearRegression over {len(months)} months. "
            f"Confidence: {confidence}. Directional estimate only."
        ),
    }


# =========================================================================
# Feature 2 — Active-learning retrain of the ledger classifier
# =========================================================================
class RetrainRequest(BaseModel):
    include_pseudo: bool = Field(
        default=True,
        description="Augment verified labels with high-confidence predictions when verified set is small.",
    )


@router.post("/retrain-classifier")
def retrain_classifier(
    body: Optional[RetrainRequest] = None,
    db: Session = Depends(get_db),
) -> dict:
    """Retrain the ledger classifier using CA-verified transactions."""
    include_pseudo = True if body is None else body.include_pseudo

    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"ML dependencies unavailable: {exc}") from exc

    from ml_engine import build_ledger_text  # already imported by main.py

    verified_statuses = [ReviewStatus.CLEARED.value, ReviewStatus.DISALLOWED.value]
    verified = (
        db.query(Invoice)
        .filter(
            or_(
                Invoice.ledger_overridden.is_(True),
                Invoice.status.in_(verified_statuses),
            ),
            Invoice.ledger_category.isnot(None),
        )
        .all()
    )

    samples = list(verified)
    data_mode = "verified_only"
    if include_pseudo and len(samples) < 8:
        excluded = [s.id for s in samples]
        q = db.query(Invoice).filter(
            Invoice.ledger_category.isnot(None),
            Invoice.ledger_confidence.isnot(None),
            Invoice.ledger_confidence >= 0.65,
        )
        if excluded:
            q = q.filter(~Invoice.id.in_(excluded))
        samples.extend(q.all())
        data_mode = "verified_plus_pseudo"

    if len(samples) < 4:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Not enough training samples ({len(samples)}). "
                "Mark vouchers CLEARED/DISALLOWED or correct the ledger label "
                "on at least 4 transactions, then retry."
            ),
        )

    X_text: List[str] = []
    y_labels: List[str] = []
    for inv in samples:
        descriptions = [li.description for li in inv.line_items if li.description]
        try:
            text = build_ledger_text(inv.vendor_name, descriptions, inv.raw_text)
        except Exception:
            text = " ".join(filter(None, [inv.vendor_name or "", inv.raw_text or ""] + descriptions))
        if not text or not text.strip():
            continue
        X_text.append(text)
        y_labels.append(inv.ledger_category)

    if len(X_text) < 4:
        raise HTTPException(status_code=400, detail="Not enough labelled text to train.")

    classes = sorted(set(y_labels))
    if len(classes) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need ≥ 2 distinct ledger classes. Currently: {classes}",
        )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    n = len(X_text)
    class_counts = {c: y_labels.count(c) for c in classes}
    use_holdout = n >= 10 and min(class_counts.values()) >= 2
    if use_holdout:
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y_labels, test_size=0.3, random_state=42, stratify=y_labels
        )
        pipeline.fit(X_train, y_train)
        val_acc = float(accuracy_score(y_test, pipeline.predict(X_test))) * 100.0
        eval_mode, train_size, eval_size = "holdout", len(X_train), len(X_test)
    else:
        pipeline.fit(X_text, y_labels)
        val_acc = float(accuracy_score(y_labels, pipeline.predict(X_text))) * 100.0
        eval_mode, train_size, eval_size = "train_only", n, n

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    version = f"al_{ts}"
    artifact = _MODELS_DIR / f"ledger_classifier_{version}.pkl"
    joblib.dump(pipeline, artifact)

    db.add(MLModelRun(
        model_name="ledger_classifier",
        version=version,
        n_samples=len(X_text),
        params={
            "vectorizer": "TfidfVectorizer(1,2)",
            "classifier": "LogisticRegression",
            "data_mode": data_mode,
        },
        metrics={
            "validation_accuracy_pct": round(val_acc, 2),
            "eval_mode": eval_mode,
            "train_size": train_size,
            "eval_size": eval_size,
            "n_verified": len(verified),
            "n_classes": len(classes),
        },
        artifact_path=str(artifact),
    ))
    db.commit()

    activation = "requires_restart"
    try:
        import ml_engine
        importlib.reload(ml_engine)
        activation = "loaded_in_memory"
    except Exception as exc:  # pragma: no cover
        log.warning("ml_engine hot-reload failed: %s", exc)

    return {
        "status": "retrained",
        "model_name": "ledger_classifier",
        "version": version,
        "n_samples": len(X_text),
        "n_verified": len(verified),
        "data_mode": data_mode,
        "classes": classes,
        "validation_accuracy_pct": round(val_acc, 2),
        "eval_mode": eval_mode,
        "artifact_path": str(artifact),
        "activation": activation,
        "note": (
            f"Retrained on {len(X_text)} samples "
            f"({len(verified)} CA-verified, {len(X_text) - len(verified)} pseudo). "
            f"Validation accuracy {val_acc:.2f}% ({eval_mode})."
        ),
    }


# =========================================================================
# Feature 3 — Natural-language "Ask Your Ledger" copilot
# =========================================================================
class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "blocked_itc": ["blocked itc", "blocked", "ineligible", "17(5)", "section 17"],
    "duplicates":  ["duplicate", "repeated", "twice", "double booked"],
    "tds":         ["tds", "tax deducted", "194"],
    "cash":        ["cash", "40a", "40a3"],
    "gst_position":["gst", "tax payable", "net gst", "how much gst", "gst paid", "gst due"],
    "anomalies":   ["anomaly", "anomalies", "flagged", "suspicious", "unusual", "outlier"],
    "high_value":  ["high value", "high-value", "top", "largest", "biggest"],
    "pending":     ["pending", "to review", "awaiting review", "needs review"],
    "sales":       ["sales", "revenue", "turnover", "income", "sold"],
    "expenses":    ["expenses", "spending", "spent", "costs", "purchases"],
    "summary":     ["summary", "overview", "status", "how am i doing", "position", "total"],
}
_INTENT_PRIORITY = [
    "blocked_itc", "duplicates", "tds", "cash", "gst_position",
    "anomalies", "high_value", "pending", "sales", "expenses", "summary",
]


def _parse_intent(question: str) -> tuple[str, List[str]]:
    q = question.lower()
    scores: Dict[str, int] = {k: 0 for k in _INTENT_KEYWORDS}
    hits: Dict[str, List[str]] = {k: [] for k in _INTENT_KEYWORDS}
    for intent, kws in _INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                scores[intent] += 1
                hits[intent].append(kw)
    max_score = max(scores.values()) if scores else 0
    if max_score == 0:
        return "summary", []
    for intent in _INTENT_PRIORITY:
        if scores[intent] == max_score:
            return intent, hits[intent]
    return "summary", []


def _tx_brief(inv: Invoice) -> dict:
    return {
        "id": inv.id,
        "vendor_name": inv.vendor_name,
        "invoice_no": inv.invoice_no,
        "invoice_date": inv.invoice_date.isoformat() if inv.invoice_date else None,
        "entry_type": inv.entry_type,
        "total_amount": inv.total_amount,
        "gst_amount": inv.gst_amount,
        "ledger_category": inv.ledger_category,
        "is_anomaly": inv.is_anomaly,
        "is_blocked_itc": inv.is_blocked_itc,
        "status": inv.status,
    }


@router.post("/ask-ledger")
def ask_ledger(body: AskRequest, db: Session = Depends(get_db)) -> dict:
    question = body.question.strip()
    intent, hits = _parse_intent(question)
    invoices = _iter_invoices(db)

    answer, matching, aggregates = "", [], {}

    if intent == "blocked_itc":
        blocked = [i for i in invoices if i.is_blocked_itc]
        total_gst = sum(float(i.gst_amount or 0) for i in blocked)
        answer = (
            f"You have {len(blocked)} blocked-ITC transaction(s) under CGST s.17(5), "
            f"totalling {_inr(total_gst)} of ineligible GST."
        ) if blocked else "No blocked ITC transactions found — all input credits are eligible."
        matching, aggregates = blocked, {"blocked_count": len(blocked), "blocked_gst_value": round(total_gst, 2)}

    elif intent == "duplicates":
        dupes = [i for i in invoices if any("DUP" in (f.rule_code or "") for f in i.flags)]
        answer = (
            f"Found {len(dupes)} potential duplicate booking(s) flagged by the rule engine."
        ) if dupes else "No duplicate bookings detected."
        matching, aggregates = dupes, {"duplicate_count": len(dupes)}

    elif intent == "tds":
        tds_items = [i for i in invoices if any("TDS" in (f.rule_code or "") for f in i.flags)]
        answer = (
            f"{len(tds_items)} transaction(s) are TDS-applicable (professional / contract / rent style payments)."
        ) if tds_items else "No TDS-applicable transactions identified."
        matching, aggregates = tds_items, {"tds_applicable_count": len(tds_items)}

    elif intent == "cash":
        cash_items = [
            i for i in invoices
            if (i.payment_mode or "").upper() == "CASH" and float(i.total_amount or 0) > 10000
        ]
        total = sum(float(i.total_amount or 0) for i in cash_items)
        answer = (
            f"{len(cash_items)} cash payment(s) exceed ₹10,000 (total {_inr(total)}), "
            f"disallowed under s.40A(3) of the Income Tax Act."
        ) if cash_items else "No cash payments above ₹10,000 were found."
        matching = cash_items
        aggregates = {"cash_over_10k_count": len(cash_items), "cash_total": round(total, 2)}

    elif intent == "gst_position":
        out_gst = sum(float(i.gst_amount or 0) for i in invoices if i.entry_type == "SALE")
        itc = sum(float(i.gst_amount or 0) for i in invoices
                  if i.entry_type == "EXPENSE" and not i.is_blocked_itc)
        blocked = sum(float(i.gst_amount or 0) for i in invoices
                      if i.entry_type == "EXPENSE" and i.is_blocked_itc)
        net = out_gst - itc
        answer = (
            f"Output GST on sales: {_inr(out_gst)}. Eligible ITC: {_inr(itc)}. "
            f"Blocked ITC: {_inr(blocked)}. Net GST payable: {_inr(net)}."
        )
        matching = [i for i in invoices if i.entry_type == "SALE"][:10]
        aggregates = {
            "output_gst": round(out_gst, 2),
            "eligible_itc": round(itc, 2),
            "blocked_itc": round(blocked, 2),
            "net_gst_payable": round(net, 2),
        }

    elif intent == "anomalies":
        flagged = [i for i in invoices if i.is_anomaly or any(not f.is_resolved for f in i.flags)]
        total = sum(float(i.total_amount or 0) for i in flagged)
        answer = (
            f"{len(flagged)} transaction(s) are flagged for review (ML anomaly or rule violation), "
            f"combined value {_inr(total)}."
        ) if flagged else "No flagged transactions — your books are clean."
        matching = flagged
        aggregates = {"flagged_count": len(flagged), "flagged_value": round(total, 2)}

    elif intent == "high_value":
        top = sorted(
            [i for i in invoices if i.total_amount is not None],
            key=lambda i: abs(float(i.total_amount or 0)),
            reverse=True,
        )[:5]
        answer = f"Top {len(top)} highest-value transactions by amount." if top else "No amounts recorded."
        matching = top
        aggregates = {
            "count": len(top),
            "total_top_value": round(sum(float(i.total_amount or 0) for i in top), 2),
        }

    elif intent == "pending":
        pending = [i for i in invoices if i.status == ReviewStatus.PENDING_REVIEW.value]
        answer = (
            f"{len(pending)} transaction(s) are pending review."
        ) if pending else "No pending transactions — everything has been reviewed."
        matching = pending
        aggregates = {"pending_count": len(pending)}

    elif intent == "sales":
        sales = [i for i in invoices if i.entry_type == "SALE"]
        total = sum(float(i.total_amount or 0) for i in sales)
        gst = sum(float(i.gst_amount or 0) for i in sales)
        answer = (
            f"{len(sales)} sale(s), total {_inr(total)}, with {_inr(gst)} output GST."
        ) if sales else "No sales recorded yet."
        matching = sales
        aggregates = {"sale_count": len(sales), "total_sales": round(total, 2), "output_gst": round(gst, 2)}

    elif intent == "expenses":
        expenses = [i for i in invoices if i.entry_type == "EXPENSE"]
        total = sum(float(i.total_amount or 0) for i in expenses)
        gst = sum(float(i.gst_amount or 0) for i in expenses)
        answer = (
            f"{len(expenses)} expense(s), total {_inr(total)}, with {_inr(gst)} input GST."
        ) if expenses else "No expenses recorded yet."
        matching = expenses
        aggregates = {"expense_count": len(expenses), "total_expenses": round(total, 2), "input_gst": round(gst, 2)}

    else:  # summary
        sales = [i for i in invoices if i.entry_type == "SALE"]
        expenses = [i for i in invoices if i.entry_type == "EXPENSE"]
        anomalies = sum(1 for i in invoices if i.is_anomaly)
        pending = sum(1 for i in invoices if i.status == ReviewStatus.PENDING_REVIEW.value)
        answer = (
            f"Books contain {len(invoices)} transactions — {len(sales)} sale(s), {len(expenses)} expense(s). "
            f"{anomalies} anomaly/anomalies, {pending} pending review."
        )
        matching = invoices[:10]
        aggregates = {
            "total_transactions": len(invoices),
            "sale_count": len(sales),
            "expense_count": len(expenses),
            "anomaly_count": anomalies,
            "pending_count": pending,
        }

    return {
        "question": question,
        "intent": intent,
        "matched_keywords": hits,
        "answer": answer,
        "aggregates": aggregates,
        "matching_transactions": [_tx_brief(i) for i in matching[:20]],
        "match_count": len(matching),
        "suggestions": [
            "What is my net GST payable?",
            "Show blocked ITC transactions",
            "List flagged or high-value expenses",
            "How many transactions are pending review?",
            "Show duplicate bookings",
        ],
    }