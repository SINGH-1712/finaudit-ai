"""FinAudit-AI API: receipt scan, ledger, hybrid audit, and CA exports.

Run from this directory:

    uvicorn main:app --reload --port 8000

The first launch trains the Phase 1 scikit-learn artifacts if ``backend/models``
is empty, then seeds fifteen Indian vouchers when the database has no invoices.
"""
from __future__ import annotations

import hashlib
import logging
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy.orm import Session, selectinload

from audit_engine import (
    RULE_CASH,
    RULE_DUPLICATE,
    RULE_GST_MATH,
    RULE_GSTIN,
    RULE_ITC,
    RULE_MISSING,
    RULE_ML,
    RULE_TDS,
    attach_prediction,
    audit_books,
    build_audit_summary,
)
from database import DEBUG_DIR, UPLOAD_DIR, get_db, init_db, session_scope
from export_service import generate_gst_excel, generate_tally_xml
from ml_engine import build_ledger_text, detect_anomaly, predict_ledger
from models import AdvocateRemark, AuditFlag, Invoice, LineItem, ReviewStatus
from ocr_pipeline import GSTIN_CHARSET, generate_valid_gstin, process_receipt, validate_gstin
from schemas import (
    AuditSummary,
    InvoiceCreate,
    InvoiceListItem,
    InvoiceRead,
    RemarkCreate,
    RemarkResolution,
)

log = logging.getLogger("finaudit.api")

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
_MAX_UPLOAD_BYTES = 12 * 1024 * 1024
_LOAD = (
    selectinload(Invoice.line_items),
    selectinload(Invoice.flags),
    selectinload(Invoice.remarks),
)


def _load_invoices(db: Session) -> List[Invoice]:
    return db.query(Invoice).options(*_LOAD).order_by(Invoice.invoice_date, Invoice.id).all()


def _load_one(db: Session, invoice_id: int) -> Invoice:
    invoice = db.query(Invoice).options(*_LOAD).filter(Invoice.id == invoice_id).one_or_none()
    if invoice is None:
        raise HTTPException(status_code=404, detail=f"Transaction {invoice_id} was not found")
    return invoice


def _summary(db: Session) -> AuditSummary:
    return AuditSummary(**build_audit_summary(_load_invoices(db)))


def _invalid_gstin(state_code: str, pan: str) -> str:
    """15-character GSTIN that passes the shape check and fails the checksum."""
    valid = generate_valid_gstin(state_code, pan)
    for char in GSTIN_CHARSET:
        if char == valid[-1]:
            continue
        candidate = valid[:-1] + char
        if not validate_gstin(candidate).is_valid:
            return candidate
    raise RuntimeError("could not derive an invalid GSTIN")


def _intra(base: float, rate: float) -> dict:
    gst = round(base * rate / 100.0, 2)
    cgst = round(gst / 2.0, 2)
    sgst = round(gst - cgst, 2)
    return {
        "base_amount": base,
        "gst_rate": rate,
        "cgst_amount": cgst,
        "sgst_amount": sgst,
        "igst_amount": 0.0,
        "gst_amount": round(cgst + sgst, 2),
        "total_amount": round(base + cgst + sgst, 2),
    }


def _inter(base: float, rate: float) -> dict:
    gst = round(base * rate / 100.0, 2)
    return {
        "base_amount": base,
        "gst_rate": rate,
        "cgst_amount": 0.0,
        "sgst_amount": 0.0,
        "igst_amount": gst,
        "gst_amount": gst,
        "total_amount": round(base + gst, 2),
    }


def _wipe(db: Session) -> None:
    db.query(AdvocateRemark).delete(synchronize_session=False)
    db.query(AuditFlag).delete(synchronize_session=False)
    db.query(LineItem).delete(synchronize_session=False)
    db.query(Invoice).delete(synchronize_session=False)
    db.commit()


def seed_demo_data() -> dict:
    """Replace the books with 15 vouchers that exercise every Phase 2 rule and the forest."""
    nova = generate_valid_gstin("29", "AABCN1234A")
    horizon = generate_valid_gstin("27", "AABCH5678B")
    mehta = generate_valid_gstin("24", "AABCM9012C")
    redington = generate_valid_gstin("29", "AABCR1111D")
    airtel = generate_valid_gstin("29", "AABCA2222E")
    aggarwal = generate_valid_gstin("29", "AABCS3333F")
    indigo = generate_valid_gstin("27", "AABCI4444G")
    hardware = generate_valid_gstin("29", "AABCL5555H")
    zomato = generate_valid_gstin("29", "AABCZ6666J")
    gupta = generate_valid_gstin("29", "AABCG7777K")
    bescom = generate_valid_gstin("29", "AABCB8888L")
    apex_bad = _invalid_gstin("29", "AABCP9999M")

    rows = [
        dict(file_name="demo-b2b-nova.txt", entry_type="SALE", vendor_name="Nova Retail Pvt Ltd",
             buyer_gstin=nova, invoice_no="SAL/26-27/001", invoice_date=date(2026, 4, 8),
             payment_mode="BANK_TRANSFER", raw_text="B2B sale of analytics platform annual subscription",
             **_intra(85000, 18)),
        dict(file_name="demo-b2b-horizon.txt", entry_type="SALE", vendor_name="Horizon Logistics Pvt Ltd",
             buyer_gstin=horizon, invoice_no="SAL/26-27/014", invoice_date=date(2026, 5, 14),
             payment_mode="BANK_TRANSFER", raw_text="B2B sale of implementation support hours",
             **_inter(126000, 18)),
        dict(file_name="demo-b2b-mehta.txt", entry_type="SALE", vendor_name="Mehta Distributors",
             buyer_gstin=mehta, invoice_no="SAL/26-27/027", invoice_date=date(2026, 6, 19),
             payment_mode="BANK_TRANSFER", raw_text="B2B sale of packaged software licence",
             **_inter(54000, 12)),
        dict(file_name="demo-b2c-counter.txt", entry_type="SALE", vendor_name="Walk-in Retail Counter",
             buyer_gstin=None, invoice_no="SAL/26-27/041", invoice_date=date(2026, 7, 2),
             payment_mode="UPI", raw_text="counter sale printed user manual",
             **_intra(4200, 18)),
        dict(file_name="demo-exp-laptop.txt", entry_type="EXPENSE", vendor_name="Redington India Ltd",
             vendor_gstin=redington, invoice_no="RDN/88421", invoice_date=date(2026, 4, 18),
             payment_mode="BANK_TRANSFER",
             raw_text="Dell Latitude laptop Redington India Ltd",
             line="Dell Latitude 5440 laptop", **_intra(72000, 18)),
        dict(file_name="demo-exp-airtel.txt", entry_type="EXPENSE", vendor_name="Bharti Airtel",
             vendor_gstin=airtel, invoice_no="ATL/26/5521", invoice_date=date(2026, 5, 2),
             payment_mode="BANK_TRANSFER", raw_text="airtel broadband bill",
             line="airtel broadband bill", **_intra(2499, 18)),
        dict(file_name="demo-exp-stationery.txt", entry_type="EXPENSE", vendor_name="Aggarwal Office Solutions",
             vendor_gstin=aggarwal, invoice_no="AGG/1042", invoice_date=date(2026, 6, 11),
             payment_mode="UPI", raw_text="a4 paper 75 gsm ream stationery items",
             line="A4 paper 75 gsm ream", **_intra(3200, 18)),
        dict(file_name="demo-exp-indigo.txt", entry_type="EXPENSE", vendor_name="IndiGo",
             vendor_gstin=indigo, invoice_no="6E-442190", invoice_date=date(2026, 7, 16),
             payment_mode="CARD", raw_text="indigo flight del-bom",
             line="IndiGo flight DEL-BOM", **_inter(8450, 5)),
        dict(file_name="demo-exp-cash-40a3.txt", entry_type="EXPENSE", vendor_name="Sharma Hardware Mart",
             vendor_gstin=hardware, invoice_no="SHM/773", invoice_date=date(2026, 8, 4),
             payment_mode="CASH", raw_text="office chairs ergonomic cash purchase",
             line="Ergonomic office chairs", **_intra(12711.86, 18)),
        dict(file_name="demo-exp-zomato.txt", entry_type="EXPENSE", vendor_name="Zomato",
             vendor_gstin=zomato, invoice_no="ZOM/992811", invoice_date=date(2026, 8, 21),
             payment_mode="UPI", raw_text="Zomato food order team dinner restaurant bill",
             line="Zomato team dinner", **_intra(1860, 5)),
        dict(file_name="demo-exp-outlier.txt", entry_type="EXPENSE", vendor_name="Aggarwal Office Solutions",
             vendor_gstin=aggarwal, invoice_no="AGG/8891", invoice_date=date(2026, 9, 13),
             payment_mode="BANK_TRANSFER",
             raw_text="a4 paper 75 gsm ream stationery items printer cartridge toner",
             line="Stationery stock — statistical outlier", **_intra(580000, 18)),
        dict(file_name="demo-exp-bad-gstin.txt", entry_type="EXPENSE", vendor_name="Apex Traders",
             vendor_gstin=apex_bad, invoice_no="APX/310", invoice_date=date(2026, 9, 18),
             payment_mode="BANK_TRANSFER", raw_text="laptop repair service",
             line="Laptop repair", **_intra(22000, 18)),
        dict(file_name="demo-exp-tds.txt", entry_type="EXPENSE", vendor_name="Gupta & Co Chartered Accountants",
             vendor_gstin=gupta, invoice_no="GA/221", invoice_date=date(2026, 10, 1),
             payment_mode="BANK_TRANSFER", raw_text="chartered accountant audit fee Gupta & Co Chartered Accountants",
             line="Statutory audit fee", **_intra(48000, 18)),
        dict(file_name="demo-exp-duplicate.txt", entry_type="EXPENSE", vendor_name="Gupta & Co Chartered Accountants",
             vendor_gstin=gupta, invoice_no="GA/221", invoice_date=date(2026, 10, 6),
             payment_mode="BANK_TRANSFER", raw_text="chartered accountant audit fee duplicate booking",
             line="Audit fee booked again", **_intra(12000, 18)),
        dict(file_name="demo-exp-missing-math.txt", entry_type="EXPENSE", vendor_name="BESCOM",
             vendor_gstin=bescom, invoice_no=None, invoice_date=date(2026, 10, 20),
             payment_mode="BANK_TRANSFER", raw_text="electricity bill",
             line="Electricity charges",
             base_amount=8000.0, gst_rate=18.0, cgst_amount=500.0, sgst_amount=500.0,
             igst_amount=0.0, gst_amount=1000.0, total_amount=9000.0),
    ]

    with session_scope() as db:
        _wipe(db)
        created: List[Invoice] = []
        for spec in rows:
            line = spec.pop("line", None)
            invoice = Invoice(file_name=spec.pop("file_name"), **spec)
            if line:
                invoice.line_items.append(LineItem(description=line, amount=invoice.base_amount))
            db.add(invoice)
            created.append(invoice)
        db.flush()
        for invoice in created:
            attach_prediction(invoice)
        audit_books(db)
        outlier = next(invoice for invoice in created if invoice.file_name == "demo-exp-outlier.txt")
        if not outlier.is_anomaly:
            boosted = _intra(1_500_000, 18)
            for key, value in boosted.items():
                setattr(outlier, key, value)
            audit_books(db)
        db.flush()
        summary = build_audit_summary(created)
        fired = {issue["rule_code"] for issue in summary["issues"] if not issue["is_resolved"]}
        required = {RULE_CASH, RULE_ITC, RULE_GSTIN, RULE_TDS, RULE_DUPLICATE, RULE_MISSING, RULE_GST_MATH, RULE_ML}
        missing_rules = sorted(required - fired)
        if missing_rules:
            preview = [
                {
                    "file": invoice.file_name,
                    "category": invoice.ledger_category,
                    "confidence": invoice.ledger_confidence,
                    "anomaly": invoice.is_anomaly,
                    "score": invoice.anomaly_score,
                    "flags": [flag.rule_code for flag in invoice.flags],
                }
                for invoice in created
            ]
            raise RuntimeError(f"Demo seed did not raise {missing_rules}. Books: {preview}")
        return {
            "seeded": len(created),
            "audit_readiness_score": summary["audit_readiness_score"],
            "anomaly_count": summary["anomaly_count"],
            "critical_count": summary["critical_count"],
            "warning_count": summary["warning_count"],
            "rules_fired": sorted(fired),
            "transactions": [
                {
                    "id": invoice.id,
                    "party": invoice.vendor_name,
                    "entry_type": invoice.entry_type,
                    "invoice_no": invoice.invoice_no,
                    "total_amount": invoice.total_amount,
                    "ledger_category": invoice.ledger_category,
                    "ml_confidence": invoice.ledger_confidence,
                    "is_anomaly": invoice.is_anomaly,
                    "audit_score": invoice.audit_score,
                    "audit_status": invoice.status,
                }
                for invoice in created
            ],
        }


def _known_vendors(db: Session) -> List[str]:
    names = [name for (name,) in db.query(Invoice.vendor_name).distinct() if name]
    return names


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    with session_scope() as db:
        existing = db.query(Invoice).count()
    if existing == 0:
        log.info("Empty ledger — seeding the 15-voucher demo and loading ML artifacts.")
        seed_demo_data()
    yield


app = FastAPI(
    title="FinAudit-AI",
    version="2.0.0",
    summary="Hybrid legal rules and scikit-learn audit workbench for Indian GST invoices.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "FinAudit-AI"}


@app.post("/api/scan-receipt")
async def scan_receipt(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict:
    """OCR the upload, then classify the ledger and score the Isolation Forest."""
    original = file.filename or "receipt.jpg"
    suffix = Path(original).suffix.lower()
    if suffix not in _IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Upload a PNG, JPG, WEBP, TIFF or BMP receipt image.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Receipt image must be under 12 MB.")
    digest = hashlib.sha256(raw).hexdigest()
    dest = UPLOAD_DIR / f"{digest[:20]}{suffix}"
    dest.write_bytes(raw)
    try:
        parsed = process_receipt(dest, known_vendors=_known_vendors(db), debug_dir=DEBUG_DIR)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    extracted = parsed.get("extracted") or {}
    descriptions = [item.get("description", "") for item in parsed.get("line_items") or []]
    text = build_ledger_text(extracted.get("vendor_name"), descriptions, parsed.get("raw_text"))
    ledger = predict_ledger(text)
    anomaly = detect_anomaly(
        {
            "total_amount": extracted.get("total_amount"),
            "base_amount": extracted.get("base_amount"),
            "gst_rate": extracted.get("gst_rate"),
            "gst_amount": extracted.get("gst_amount"),
            "invoice_date": extracted.get("invoice_date"),
            "ledger_category": ledger["category"],
        }
    )
    return jsonable_encoder({
        **parsed,
        "ledger": ledger,
        "anomaly": anomaly,
        "file_name": original,
        "file_path": str(dest),
        "file_hash": digest,
    })


@app.post("/api/transactions", response_model=InvoiceRead, status_code=201)
def create_transaction(body: InvoiceCreate, db: Session = Depends(get_db)) -> Invoice:
    payload = body.model_dump(exclude={"line_items", "ledger_category"})
    invoice = Invoice(**payload)
    for item in body.line_items:
        invoice.line_items.append(LineItem(**item.model_dump()))
    db.add(invoice)
    db.flush()
    attach_prediction(invoice, body.ledger_category)
    audit_books(db)
    db.commit()
    return _load_one(db, invoice.id)


@app.get("/api/transactions", response_model=List[InvoiceListItem])
def list_transactions(
    q: Optional[str] = Query(default=None, description="Match party, invoice number or ledger"),
    entry_type: Optional[str] = Query(default=None, pattern="^(SALE|EXPENSE)$"),
    anomalies_only: bool = False,
    db: Session = Depends(get_db),
) -> List[Invoice]:
    query = db.query(Invoice)
    if entry_type:
        query = query.filter(Invoice.entry_type == entry_type)
    if anomalies_only:
        query = query.filter(Invoice.is_anomaly.is_(True))
    if q:
        like = f"%{q.strip()}%"
        query = query.filter(
            Invoice.vendor_name.ilike(like)
            | Invoice.invoice_no.ilike(like)
            | Invoice.ledger_category.ilike(like)
        )
    return query.order_by(Invoice.invoice_date.desc(), Invoice.id.desc()).all()


@app.patch("/api/transactions/{invoice_id}/remark", response_model=InvoiceRead)
def save_remark(invoice_id: int, body: RemarkCreate, db: Session = Depends(get_db)) -> Invoice:
    invoice = _load_one(db, invoice_id)
    flag = None
    if body.flag_id is not None:
        flag = next((item for item in invoice.flags if item.id == body.flag_id), None)
        if flag is None:
            raise HTTPException(status_code=404, detail="That flag does not belong to this transaction.")
    if body.resolves_flag and flag is None:
        raise HTTPException(status_code=400, detail="flag_id is required when resolves_flag is true.")
    remark = AdvocateRemark(
        invoice_id=invoice.id,
        flag_id=flag.id if flag is not None else None,
        author=body.author,
        remark=body.remark,
        resolution=body.resolution.value if isinstance(body.resolution, RemarkResolution) else body.resolution,
    )
    db.add(remark)
    if body.resolves_flag and flag is not None:
        flag.is_resolved = True
    audit_books(db)
    if body.resolution == RemarkResolution.DISALLOW:
        invoice.status = ReviewStatus.DISALLOWED.value
    elif body.resolution == RemarkResolution.ESCALATE_TO_CA:
        invoice.status = ReviewStatus.ESCALATED.value
    db.commit()
    return _load_one(db, invoice.id)


@app.get("/api/audit-summary", response_model=AuditSummary)
def audit_summary(db: Session = Depends(get_db)) -> AuditSummary:
    return _summary(db)


@app.get("/api/export/tally-xml")
def export_tally_xml(db: Session = Depends(get_db)) -> Response:
    xml = generate_tally_xml(_load_invoices(db))
    return Response(
        content=xml,
        media_type="application/xml",
        headers={"Content-Disposition": "attachment; filename=finaudit-tally-prime.xml"},
    )


@app.get("/api/export/gst-excel")
def export_gst_excel(db: Session = Depends(get_db)) -> Response:
    invoices = _load_invoices(db)
    workbook = generate_gst_excel(invoices, build_audit_summary(invoices))
    return Response(
        content=workbook,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=finaudit-gst-working-papers.xlsx"},
    )


@app.post("/api/seed-demo")
def seed_demo() -> dict:
    """Wipe the ledger and insert the 15-voucher demonstration set."""
    try:
        return seed_demo_data()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
