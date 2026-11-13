"""SQLAlchemy 2.0 ORM models for FinAudit-AI.

Design notes
------------
* Every ML output is persisted next to the data it describes (confidence scores,
  anomaly score + human-readable reasons, model version) so an Advocate can audit
  *why* the system decided something, and so predictions can be re-scored after a
  model upgrade.
* Rule-engine violations and ML anomaly flags share one table (`audit_flags`)
  distinguished by `source`; the Audit Readiness Score is derived from them.
* Advocate remarks are append-only (an audit trail), optionally tied to one flag.
"""
from __future__ import annotations

import enum
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Severity(str, enum.Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FlagSource(str, enum.Enum):
    RULE = "RULE"  # deterministic legal / arithmetic rule
    ML = "ML"  # statistical model (Isolation Forest / low classifier confidence)


class ReviewStatus(str, enum.Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    IN_REVIEW = "IN_REVIEW"
    CLEARED = "CLEARED"
    DISALLOWED = "DISALLOWED"
    ESCALATED = "ESCALATED"


class RemarkResolution(str, enum.Enum):
    CLARIFIED = "CLARIFIED"  # advocate accepted the explanation
    NEEDS_DOCUMENT = "NEEDS_DOCUMENT"  # supporting document requested from client
    ACCEPT_RISK = "ACCEPT_RISK"
    DISALLOW = "DISALLOW"  # expense/ITC to be disallowed
    ESCALATE_TO_CA = "ESCALATE_TO_CA"


class Vendor(Base):
    __tablename__ = "vendors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    normalized_name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    gstin: Mapped[Optional[str]] = mapped_column(String(15), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    invoices: Mapped[List["Invoice"]] = relationship(back_populates="vendor")


class Invoice(Base):
    __tablename__ = "invoices"
    __table_args__ = (
        Index("ix_invoice_vendor_gstin_no", "vendor_gstin", "invoice_no"),
        Index("ix_invoice_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # ---- file provenance -------------------------------------------------
    file_name: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[Optional[str]] = mapped_column(String(512))
    file_hash: Mapped[Optional[str]] = mapped_column(String(64), index=True)  # sha256 -> exact-duplicate upload
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # ---- OCR / CV telemetry ---------------------------------------------
    ocr_engine: Mapped[Optional[str]] = mapped_column(String(32))
    ocr_mean_confidence: Mapped[Optional[float]] = mapped_column(Float)  # 0..1
    ocr_token_count: Mapped[Optional[int]] = mapped_column(Integer)
    preprocess_meta: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    raw_text: Mapped[Optional[str]] = mapped_column(Text)
    field_confidences: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)  # per extracted field
    field_sources: Mapped[Optional[Dict[str, str]]] = mapped_column(JSON)  # regex | fuzzy | derived | llm | manual

    # ---- extracted / user-corrected fields ------------------------------
    vendor_id: Mapped[Optional[int]] = mapped_column(ForeignKey("vendors.id", ondelete="SET NULL"))
    vendor_name: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    vendor_gstin: Mapped[Optional[str]] = mapped_column(String(15))
    gstin_valid: Mapped[Optional[bool]] = mapped_column(Boolean)  # modulo-36 checksum result
    gstin_state_code: Mapped[Optional[str]] = mapped_column(String(2))
    buyer_gstin: Mapped[Optional[str]] = mapped_column(String(15))
    invoice_no: Mapped[Optional[str]] = mapped_column(String(64))
    invoice_date: Mapped[Optional[date]] = mapped_column(Date)
    base_amount: Mapped[Optional[float]] = mapped_column(Float)
    gst_rate: Mapped[Optional[float]] = mapped_column(Float)
    cgst_amount: Mapped[Optional[float]] = mapped_column(Float)
    sgst_amount: Mapped[Optional[float]] = mapped_column(Float)
    igst_amount: Mapped[Optional[float]] = mapped_column(Float)
    gst_amount: Mapped[Optional[float]] = mapped_column(Float)
    total_amount: Mapped[Optional[float]] = mapped_column(Float)
    payment_mode: Mapped[Optional[str]] = mapped_column(String(24))  # CASH | UPI | CARD | BANK_TRANSFER | CHEQUE
    # SALE = outward supply (we are the supplier). EXPENSE = inward supply / cost.
    # Party name stays in vendor_name; the counterparty GSTIN is buyer_gstin (sales) or vendor_gstin (expenses).
    entry_type: Mapped[str] = mapped_column(String(16), default="EXPENSE", index=True)

    # ---- ML: ledger classifier ------------------------------------------
    ledger_category: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    ledger_confidence: Mapped[Optional[float]] = mapped_column(Float)  # 0..1
    ledger_alternatives: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)  # top-k
    ledger_model_version: Mapped[Optional[str]] = mapped_column(String(32))
    ledger_overridden: Mapped[bool] = mapped_column(Boolean, default=False)  # human corrected the label
    is_blocked_itc: Mapped[bool] = mapped_column(Boolean, default=False)
    itc_tags: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # ---- ML: anomaly detector -------------------------------------------
    anomaly_score: Mapped[Optional[float]] = mapped_column(Float)  # 0..1 percentile of training distribution
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    anomaly_reasons: Mapped[Optional[List[str]]] = mapped_column(JSON)
    anomaly_features: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)
    anomaly_model_version: Mapped[Optional[str]] = mapped_column(String(32))

    # ---- audit outcome ---------------------------------------------------
    audit_score: Mapped[Optional[float]] = mapped_column(Float)  # Audit Readiness Score, 0..100
    rule_penalty: Mapped[float] = mapped_column(Float, default=0.0)
    ml_penalty: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(20), default=ReviewStatus.PENDING_REVIEW.value)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    vendor: Mapped[Optional[Vendor]] = relationship(back_populates="invoices")
    line_items: Mapped[List["LineItem"]] = relationship(
        back_populates="invoice", cascade="all, delete-orphan", order_by="LineItem.id"
    )
    flags: Mapped[List["AuditFlag"]] = relationship(
        back_populates="invoice", cascade="all, delete-orphan", order_by="AuditFlag.id"
    )
    remarks: Mapped[List["AdvocateRemark"]] = relationship(
        back_populates="invoice", cascade="all, delete-orphan", order_by="AdvocateRemark.id"
    )


class LineItem(Base):
    __tablename__ = "line_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id", ondelete="CASCADE"), index=True)
    description: Mapped[str] = mapped_column(String(512))
    hsn_sac: Mapped[Optional[str]] = mapped_column(String(8))
    quantity: Mapped[Optional[float]] = mapped_column(Float)
    rate: Mapped[Optional[float]] = mapped_column(Float)
    amount: Mapped[Optional[float]] = mapped_column(Float)
    ledger_category: Mapped[Optional[str]] = mapped_column(String(64))
    ledger_confidence: Mapped[Optional[float]] = mapped_column(Float)

    invoice: Mapped[Invoice] = relationship(back_populates="line_items")


class AuditFlag(Base):
    __tablename__ = "audit_flags"
    __table_args__ = (Index("ix_flag_invoice_code", "invoice_id", "rule_code"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id", ondelete="CASCADE"), index=True)
    rule_code: Mapped[str] = mapped_column(String(40))  # e.g. GSTIN_CHECKSUM, CASH_40A3, ML_ANOMALY
    source: Mapped[str] = mapped_column(String(8), default=FlagSource.RULE.value)
    severity: Mapped[str] = mapped_column(String(10), default=Severity.MEDIUM.value)
    title: Mapped[str] = mapped_column(String(160))
    message: Mapped[str] = mapped_column(Text)
    legal_reference: Mapped[Optional[str]] = mapped_column(String(160))  # "IT Act s.40A(3)", "CGST s.17(5)"
    penalty: Mapped[float] = mapped_column(Float, default=0.0)  # points deducted from the readiness score
    evidence: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    is_resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    invoice: Mapped[Invoice] = relationship(back_populates="flags")
    remarks: Mapped[List["AdvocateRemark"]] = relationship(back_populates="flag")


class AdvocateRemark(Base):
    """Append-only Advocate/CA clarification trail."""

    __tablename__ = "advocate_remarks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id", ondelete="CASCADE"), index=True)
    flag_id: Mapped[Optional[int]] = mapped_column(ForeignKey("audit_flags.id", ondelete="SET NULL"))
    author: Mapped[str] = mapped_column(String(80), default="Advocate")
    remark: Mapped[str] = mapped_column(Text)
    resolution: Mapped[str] = mapped_column(String(20), default=RemarkResolution.CLARIFIED.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    invoice: Mapped[Invoice] = relationship(back_populates="remarks")
    flag: Mapped[Optional[AuditFlag]] = relationship(back_populates="remarks")


class MLModelRun(Base):
    """Registry of trained model artifacts + evaluation metrics (MLOps lineage)."""

    __tablename__ = "ml_model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(48), index=True)  # ledger_classifier | anomaly_detector
    version: Mapped[str] = mapped_column(String(32))
    n_samples: Mapped[Optional[int]] = mapped_column(Integer)
    params: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    artifact_path: Mapped[Optional[str]] = mapped_column(String(512))
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
