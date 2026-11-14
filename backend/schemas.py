"""Pydantic v2 schemas (API contracts) for FinAudit-AI.

The `Processing*` family mirrors the dict returned by `ocr_pipeline.process_receipt`
plus the ML inference outputs from `ml_engine`, so the FastAPI layer can validate
and serialise them without any translation code.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models import FlagSource, RemarkResolution, ReviewStatus, Severity

_GSTIN_SHAPE = re.compile(r"^[0-9A-Z]{15}$")


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


# --------------------------------------------------------------------------
# OCR / extraction
# --------------------------------------------------------------------------
class OCRSummary(BaseModel):
    engine: str
    mean_confidence: float = Field(ge=0, le=1)
    median_confidence: float = Field(ge=0, le=1)
    low_conf_ratio: float = Field(ge=0, le=1, description="share of tokens below the low-confidence threshold")
    token_count: int = Field(ge=0)
    quality: float = Field(ge=0, le=1, description="pass-selection score = mean_conf x coverage")
    preprocess: Dict[str, Any] = Field(default_factory=dict)
    passes: List[Dict[str, Any]] = Field(default_factory=list)


class ExtractedFields(BaseModel):
    vendor_name: Optional[str] = None
    vendor_gstin: Optional[str] = None
    gstin_valid: Optional[bool] = None
    gstin_state_code: Optional[str] = None
    gstin_state_name: Optional[str] = None
    buyer_gstin: Optional[str] = None
    invoice_no: Optional[str] = None
    invoice_date: Optional[date] = None
    base_amount: Optional[float] = None
    gst_rate: Optional[float] = None
    cgst_amount: Optional[float] = None
    sgst_amount: Optional[float] = None
    igst_amount: Optional[float] = None
    gst_amount: Optional[float] = None
    total_amount: Optional[float] = None
    payment_mode: Optional[str] = None
    hsn_code: Optional[str] = None


class LineItemBase(BaseModel):
    description: str = Field(min_length=1, max_length=512)
    hsn_sac: Optional[str] = None
    quantity: Optional[float] = Field(default=None, ge=0)
    rate: Optional[float] = Field(default=None, ge=0)
    amount: Optional[float] = Field(default=None, ge=0)


class LineItemRead(LineItemBase, ORMModel):
    id: int
    ledger_category: Optional[str] = None
    ledger_confidence: Optional[float] = None


# --------------------------------------------------------------------------
# ML outputs
# --------------------------------------------------------------------------
class LedgerAlternative(BaseModel):
    category: str
    confidence: float = Field(ge=0, le=1)


class LedgerPrediction(BaseModel):
    category: str
    confidence: float = Field(ge=0, le=1)
    confidence_pct: float = Field(ge=0, le=100)
    label: str = Field(description='Display string, e.g. "IT Infrastructure – 94.2% confidence"')
    top_k: List[LedgerAlternative] = Field(default_factory=list)
    is_blocked_itc: bool = False
    itc_tags: List[str] = Field(default_factory=list)
    tds_section: Optional[str] = None
    needs_review: bool = False
    model_version: str = ""


class AnomalyResult(BaseModel):
    score: float = Field(ge=0, le=1)
    raw_score: float
    threshold: float
    is_anomaly: bool
    reasons: List[str] = Field(default_factory=list)
    features: Dict[str, float] = Field(default_factory=dict)
    model_version: str = ""


class ProcessingResult(BaseModel):
    """Everything produced for one uploaded receipt before human review."""

    ocr: OCRSummary
    raw_text: str
    lines: List[Dict[str, Any]] = Field(default_factory=list)
    extracted: ExtractedFields
    field_confidences: Dict[str, float] = Field(default_factory=dict)
    field_sources: Dict[str, str] = Field(default_factory=dict)
    line_items: List[LineItemBase] = Field(default_factory=list)
    all_gstins: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_ms: int = 0
    ledger: Optional[LedgerPrediction] = None
    anomaly: Optional[AnomalyResult] = None


# --------------------------------------------------------------------------
# Invoice CRUD
# --------------------------------------------------------------------------
class InvoiceUpdate(BaseModel):
    """Fields a reviewer may correct after OCR. Absent fields are left unchanged."""

    vendor_name: Optional[str] = Field(default=None, max_length=255)
    vendor_gstin: Optional[str] = None
    buyer_gstin: Optional[str] = None
    invoice_no: Optional[str] = Field(default=None, max_length=64)
    invoice_date: Optional[date] = None
    base_amount: Optional[float] = Field(default=None, ge=0)
    gst_rate: Optional[float] = Field(default=None, ge=0, le=40)
    cgst_amount: Optional[float] = Field(default=None, ge=0)
    sgst_amount: Optional[float] = Field(default=None, ge=0)
    igst_amount: Optional[float] = Field(default=None, ge=0)
    gst_amount: Optional[float] = Field(default=None, ge=0)
    total_amount: Optional[float] = Field(default=None, ge=0)
    payment_mode: Optional[str] = None
    hsn_code: Optional[str] = Field(default=None, max_length=8)
    ledger_category: Optional[str] = None
    entry_type: Optional[Literal["SALE", "EXPENSE"]] = None

    @field_validator("vendor_gstin", "buyer_gstin")
    @classmethod
    def _gstin_shape(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        v = re.sub(r"\s+", "", v).upper()
        if not _GSTIN_SHAPE.match(v):
            raise ValueError("GSTIN must be 15 alphanumeric characters")
        return v

    @field_validator("payment_mode")
    @classmethod
    def _payment_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().upper().replace(" ", "_")
        allowed = {"CASH", "UPI", "CARD", "BANK_TRANSFER", "CHEQUE", "CREDIT"}
        if v not in allowed:
            raise ValueError(f"payment_mode must be one of {sorted(allowed)}")
        return v


class AuditFlagRead(ORMModel):
    id: int
    invoice_id: int
    rule_code: str
    source: FlagSource
    severity: Severity
    title: str
    message: str
    legal_reference: Optional[str] = None
    penalty: float
    evidence: Optional[Dict[str, Any]] = None
    is_resolved: bool
    created_at: datetime


class RemarkCreate(BaseModel):
    author: str = Field(default="Advocate", min_length=1, max_length=80)
    remark: str = Field(min_length=3, max_length=4000)
    resolution: RemarkResolution = RemarkResolution.CLARIFIED
    flag_id: Optional[int] = Field(default=None, description="attach to a specific flag; omit for invoice-level remark")
    resolves_flag: bool = Field(default=False, description="mark the linked flag resolved and re-score the invoice")


class RemarkRead(ORMModel):
    id: int
    invoice_id: int
    flag_id: Optional[int] = None
    author: str
    remark: str
    resolution: RemarkResolution
    created_at: datetime


class InvoiceCreate(BaseModel):
    """Manual entry or the confirmed payload from the receipt scanner."""

    file_name: str = Field(default="manual-entry", max_length=255)
    file_path: Optional[str] = Field(default=None, max_length=512)
    file_hash: Optional[str] = Field(default=None, max_length=64)
    entry_type: Literal["SALE", "EXPENSE"] = "EXPENSE"
    vendor_name: Optional[str] = Field(default=None, max_length=255)
    vendor_gstin: Optional[str] = None
    buyer_gstin: Optional[str] = None
    invoice_no: Optional[str] = Field(default=None, max_length=64)
    invoice_date: Optional[date] = None
    base_amount: Optional[float] = Field(default=None, ge=0)
    gst_rate: Optional[float] = Field(default=None, ge=0, le=40)
    cgst_amount: Optional[float] = Field(default=None, ge=0)
    sgst_amount: Optional[float] = Field(default=None, ge=0)
    igst_amount: Optional[float] = Field(default=None, ge=0)
    gst_amount: Optional[float] = Field(default=None, ge=0)
    total_amount: Optional[float] = Field(default=None, ge=0)
    payment_mode: Optional[str] = None
    hsn_code: Optional[str] = Field(default=None, max_length=8)
    ledger_category: Optional[str] = Field(
        default=None,
        description="Set when a human overrides the classifier. Omit to keep the model label.",
    )
    raw_text: Optional[str] = None
    line_items: List[LineItemBase] = Field(default_factory=list)
    ocr_engine: Optional[str] = None
    ocr_mean_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    ocr_token_count: Optional[int] = Field(default=None, ge=0)
    field_confidences: Optional[Dict[str, float]] = None
    field_sources: Optional[Dict[str, str]] = None
    preprocess_meta: Optional[Dict[str, Any]] = None

    @field_validator("vendor_gstin", "buyer_gstin")
    @classmethod
    def _gstin_shape(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        v = re.sub(r"\s+", "", v).upper()
        if not _GSTIN_SHAPE.match(v):
            raise ValueError("GSTIN must be 15 alphanumeric characters")
        return v

    @field_validator("payment_mode")
    @classmethod
    def _payment_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        v = v.strip().upper().replace(" ", "_")
        allowed = {"CASH", "UPI", "CARD", "BANK_TRANSFER", "CHEQUE", "CREDIT"}
        if v not in allowed:
            raise ValueError(f"payment_mode must be one of {sorted(allowed)}")
        return v


class InvoiceListItem(ORMModel):
    id: int
    file_name: str
    entry_type: str = "EXPENSE"
    vendor_name: Optional[str] = None
    invoice_no: Optional[str] = None
    invoice_date: Optional[date] = None
    total_amount: Optional[float] = None
    gst_amount: Optional[float] = None
    payment_mode: Optional[str] = None
    hsn_code: Optional[str] = None
    ledger_category: Optional[str] = None
    ledger_confidence: Optional[float] = None
    ocr_mean_confidence: Optional[float] = None
    is_anomaly: bool = False
    is_blocked_itc: bool = False
    audit_score: Optional[float] = None
    status: ReviewStatus
    uploaded_at: datetime


class InvoiceRead(ORMModel):
    id: int
    file_name: str
    file_hash: Optional[str] = None
    uploaded_at: datetime

    ocr_engine: Optional[str] = None
    ocr_mean_confidence: Optional[float] = None
    ocr_token_count: Optional[int] = None
    preprocess_meta: Optional[Dict[str, Any]] = None
    raw_text: Optional[str] = None
    field_confidences: Optional[Dict[str, float]] = None
    field_sources: Optional[Dict[str, str]] = None

    entry_type: str = "EXPENSE"
    vendor_name: Optional[str] = None
    vendor_gstin: Optional[str] = None
    gstin_valid: Optional[bool] = None
    gstin_state_code: Optional[str] = None
    buyer_gstin: Optional[str] = None
    invoice_no: Optional[str] = None
    invoice_date: Optional[date] = None
    base_amount: Optional[float] = None
    gst_rate: Optional[float] = None
    cgst_amount: Optional[float] = None
    sgst_amount: Optional[float] = None
    igst_amount: Optional[float] = None
    gst_amount: Optional[float] = None
    total_amount: Optional[float] = None
    payment_mode: Optional[str] = None
    hsn_code: Optional[str] = None
    vendor_gstin_status: Optional[str] = None
    ledger_category: Optional[str] = None
    ledger_confidence: Optional[float] = None
    ledger_alternatives: Optional[List[Dict[str, Any]]] = None
    ledger_model_version: Optional[str] = None
    ledger_overridden: bool = False
    is_blocked_itc: bool = False
    itc_tags: Optional[List[str]] = None

    anomaly_score: Optional[float] = None
    is_anomaly: bool = False
    anomaly_reasons: Optional[List[str]] = None
    anomaly_features: Optional[Dict[str, float]] = None
    anomaly_model_version: Optional[str] = None

    audit_score: Optional[float] = None
    rule_penalty: float = 0.0
    ml_penalty: float = 0.0
    status: ReviewStatus
    reviewed_at: Optional[datetime] = None

    line_items: List[LineItemRead] = Field(default_factory=list)
    flags: List[AuditFlagRead] = Field(default_factory=list)
    remarks: List[RemarkRead] = Field(default_factory=list)


class StatusUpdate(BaseModel):
    status: ReviewStatus

# --------------------------------------------------------------------------
# GSTIN verification (Feature 2)
# --------------------------------------------------------------------------
class GSTINVerification(BaseModel):
    gstin: str
    is_format_valid: bool
    is_checksum_valid: bool
    state_code: Optional[str] = None
    state_name: Optional[str] = None
    entity_type: Optional[str] = None
    pan: Optional[str] = None
    status: str = "Unknown"  # Active | Cancelled | Composition | Suspended | Invalid | Unknown
    legal_name: Optional[str] = None
    trade_name: Optional[str] = None
    registration_date: Optional[str] = None
    cancellation_date: Optional[str] = None
    source: str = "offline"  # offline | live_api
    message: Optional[str] = None

# --------------------------------------------------------------------------
# Dashboard / MLOps
# --------------------------------------------------------------------------
class CategoryCount(BaseModel):
    category: str
    count: int
    total_amount: float = 0.0
    avg_confidence: Optional[float] = None


class HistogramBin(BaseModel):
    label: str
    count: int


class DashboardSummary(BaseModel):
    total_invoices: int
    total_value: float
    avg_audit_score: Optional[float] = None
    avg_ocr_confidence: Optional[float] = None
    anomaly_count: int
    blocked_itc_count: int
    blocked_itc_value: float = 0.0
    pending_review: int
    flags_by_severity: Dict[str, int] = Field(default_factory=dict)
    flags_by_rule: Dict[str, int] = Field(default_factory=dict)
    categories: List[CategoryCount] = Field(default_factory=list)
    audit_score_histogram: List[HistogramBin] = Field(default_factory=list)
    anomaly_score_histogram: List[HistogramBin] = Field(default_factory=list)
    status_counts: Dict[str, int] = Field(default_factory=dict)


class CashflowMonth(BaseModel):
    month: str
    sales: float = 0.0
    expenses: float = 0.0
    net: float = 0.0


class AuditIssue(BaseModel):
    """One open or resolved flag, flattened with the invoice context the portal renders.

    ``ml_confidence`` is ``Invoice.ledger_confidence``. ``audit_status`` is ``Invoice.status``.
    ``advocate_remark`` is the latest ``AdvocateRemark.remark`` on this flag (else the invoice).
    Those three names are API-facing; the columns themselves stay as Phase 1 defined them.
    """

    id: int
    invoice_id: int
    rule_code: str
    source: str
    severity: str
    title: str
    message: str
    legal_reference: Optional[str] = None
    penalty: float = 0.0
    is_resolved: bool = False
    created_at: datetime
    vendor_name: Optional[str] = None
    invoice_no: Optional[str] = None
    invoice_date: Optional[date] = None
    entry_type: str = "EXPENSE"
    total_amount: Optional[float] = None
    ledger_category: Optional[str] = None
    ml_confidence: Optional[float] = None
    anomaly_score: Optional[float] = None
    is_anomaly: bool = False
    advocate_remark: Optional[str] = None
    audit_status: Optional[str] = None


class AuditSummary(BaseModel):
    audit_readiness_score: float = Field(ge=0, le=100)
    total_invoices: int
    total_sales: float
    total_expenses: float
    output_gst: float
    eligible_itc: float
    net_gst_payable: float
    anomaly_count: int
    critical_count: int
    warning_count: int
    blocked_itc_count: int
    pending_review: int
    flags_by_severity: Dict[str, int] = Field(default_factory=dict)
    categories: List[CategoryCount] = Field(default_factory=list)
    monthly_cashflow: List[CashflowMonth] = Field(default_factory=list)
    issues: List[AuditIssue] = Field(default_factory=list)


class MLModelRunRead(ORMModel):
    id: int
    model_name: str
    version: str
    n_samples: Optional[int] = None
    params: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    trained_at: datetime
