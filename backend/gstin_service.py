"""GSTIN decoder + verification service for FinAudit-AI.

Provides:
  * Mathematical Mod-36 checksum validation (reuses ocr_pipeline.validate_gstin)
  * 2-digit State code → State name decode (reuses ocr_pipeline.STATE_CODES)
  * PAN 4th-char → Entity type decode
  * Optional LIVE lookup when GST_API_KEY env var is set (via a third-party
    provider); falls back to a built-in realistic Indian GSTIN registry so the
    demo works 100% offline during the viva.

Public API:
    verify_gstin(gstin: str) -> GSTINVerification
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

from ocr_pipeline import generate_valid_gstin, validate_gstin
from schemas import GSTINVerification

log = logging.getLogger("finaudit.gstin")

# ── PAN 4th-character → entity type ─────────────────────────────────────
# Position 6 (0-indexed 5) of a GSTIN = PAN[3] = entity-type indicator.
ENTITY_TYPES: dict[str, str] = {
    "P": "Proprietorship",
    "C": "Company",
    "F": "Partnership Firm",
    "H": "Hindu Undivided Family (HUF)",
    "A": "Association of Persons (AOP)",
    "T": "Trust",
    "B": "Body of Individuals (BOI)",
    "L": "Local Authority",
    "J": "Artificial Juridical Person",
    "G": "Government",
}


# ── Offline registry: sample Active / Cancelled / Composition GSTINs ────
# Two entries intentionally match seeded demo vendors so the audit engine
# raises CRITICAL ITC-ineligibility flags during the offline viva.
_OFFLINE_REGISTRY: dict[str, dict] = {
    # Seeded vendor: Aggarwal Office Solutions → CANCELLED
    generate_valid_gstin("29", "AABCS3333F"): {
        "status": "Cancelled",
        "legal_name": "Aggarwal Office Solutions Pvt Ltd",
        "trade_name": "Aggarwal Office Solutions",
        "registration_date": "2019-04-01",
        "cancellation_date": "2025-08-15",
    },
    # Seeded vendor: Sharma Hardware Mart → COMPOSITION (cannot charge GST)
    generate_valid_gstin("29", "AABCL5555H"): {
        "status": "Composition",
        "legal_name": "Sharma Hardware Mart",
        "trade_name": "Sharma Hardware Mart",
        "registration_date": "2021-07-12",
    },
    # A generic Cancelled sample that lives outside the seed
    generate_valid_gstin("27", "AAACR5055K"): {
        "status": "Cancelled",
        "legal_name": "Sample Cancelled Trading Co",
        "trade_name": "Cancelled Trading",
        "registration_date": "2018-01-15",
        "cancellation_date": "2023-06-30",
    },
}


# ── Optional live lookup (third-party GST API) ──────────────────────────
def _live_lookup(gstin: str) -> Optional[dict]:
    """Return {status, legal_name, ...} from a live GST API, or None.

    Set GST_API_KEY (and optionally GST_API_URL) to enable. Never raises.
    """
    api_key = os.getenv("GST_API_KEY")
    if not api_key:
        return None
    url = os.getenv(
        "GST_API_URL",
        f"https://api.mastersindia.co/gst-information?gstin={gstin}",
    )
    try:
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read())
        st = str(data.get("status") or data.get("lgnm_status") or "").lower()
        if not st:
            return None
        status = (
            "Active" if "active" in st else
            "Cancelled" if "cancel" in st else
            "Composition" if "composition" in st else
            "Suspended" if "suspend" in st else
            "Active"
        )
        return {
            "status": status,
            "legal_name": data.get("lgnm") or data.get("legal_name"),
            "trade_name": data.get("tradeNam") or data.get("trade_name"),
            "registration_date": data.get("rgdt") or data.get("registration_date"),
        }
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
        log.warning("live GST lookup failed for %s: %s", gstin, exc)
        return None


# ── Main entry point ────────────────────────────────────────────────────
def verify_gstin(gstin: str) -> GSTINVerification:
    checked = validate_gstin(gstin)

    # 1. Fail fast on format / checksum — no lookup needed
    if not checked.is_valid:
        is_fmt = len(checked.gstin) == 15
        return GSTINVerification(
            gstin=checked.gstin, is_format_valid=is_fmt, is_checksum_valid=False,
            state_code=checked.state_code, state_name=checked.state_name,
            status="Invalid", source="offline",
            message=checked.reason or "Invalid GSTIN format or checksum",
        )

    # 2. Decode
    entity_char = checked.gstin[5]
    entity_type = ENTITY_TYPES.get(entity_char, "Unknown")

    # 3. Lookup (live first, offline fallback)
    record = _live_lookup(checked.gstin)
    source = "live_api" if record else "offline"
    if record is None:
        record = _OFFLINE_REGISTRY.get(checked.gstin)
    if record is None:
        # Not in registry — treat as Active (most valid GSTINs are active)
        return GSTINVerification(
            gstin=checked.gstin,
            is_format_valid=True, is_checksum_valid=True,
            state_code=checked.state_code, state_name=checked.state_name,
            entity_type=entity_type, pan=checked.pan,
            status="Active", source=source,
            message="Not in offline registry — assuming Active (checksum valid).",
        )

    return GSTINVerification(
        gstin=checked.gstin,
        is_format_valid=True, is_checksum_valid=True,
        state_code=checked.state_code, state_name=checked.state_name,
        entity_type=entity_type, pan=checked.pan,
        status=record.get("status", "Unknown"),
        legal_name=record.get("legal_name"),
        trade_name=record.get("trade_name"),
        registration_date=record.get("registration_date"),
        cancellation_date=record.get("cancellation_date"),
        source=source,
    )