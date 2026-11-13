"""CA working-paper and Tally Prime export for FinAudit-AI.

``generate_tally_xml`` builds one import envelope. Each sale becomes a Sales
voucher and each expense a Purchase voucher. Ledger lines are signed so the
voucher sums to zero (Tally rejects an unbalanced import).

``generate_gst_excel`` writes four sheets a chartered accountant can filter:

* B2B_Sales — outward supplies with a checksum-valid buyer GSTIN
* B2C_Sales — outward supplies without one
* ITC_Eligible_Expenses — inward supplies that ``audit_engine.itc_eligible`` accepts
* Advocate_Audit_RedFlags — every rule and ML flag, with the advocate's remark
  and the classifier confidence
"""
from __future__ import annotations

from datetime import date, datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence
from xml.sax.saxutils import escape

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from audit_engine import itc_eligible
from ocr_pipeline import validate_gstin

COMPANY_NAME = "FinAudit Analytics Pvt Ltd"

_HEADER_FILL = PatternFill("solid", fgColor="1F4D3A")
_HEADER_FONT = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
_TITLE_FONT = Font(name="Calibri", bold=True, size=14, color="1C1915")
_CRITICAL_FILL = PatternFill("solid", fgColor="F8D6D0")
_WARNING_FILL = PatternFill("solid", fgColor="FDE8C8")
_ML_FILL = PatternFill("solid", fgColor="E4EAF3")
_RESOLVED_FILL = PatternFill("solid", fgColor="E7F0EA")
_ZEBRA = PatternFill("solid", fgColor="F7F4EE")
_THIN = Border(
    left=Side(style="thin", color="E4DDD2"),
    right=Side(style="thin", color="E4DDD2"),
    top=Side(style="thin", color="E4DDD2"),
    bottom=Side(style="thin", color="E4DDD2"),
)
_INR = '₹#,##0.00'
_MONEY_NAMES = {
    "Taxable Value", "CGST", "SGST", "IGST", "GST Amount", "Invoice Value", "Penalty",
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _num(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return round(float(value), 2)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%d-%m-%Y")
    if isinstance(value, date):
        return value.strftime("%d-%m-%Y")
    return str(value)


def _tally_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    if isinstance(value, str) and len(value) >= 10 and value[4] == "-":
        return value[:10].replace("-", "")
    return datetime.utcnow().strftime("%Y%m%d")


def _party_name(txn: Any) -> str:
    name = (_get(txn, "vendor_name") or "").strip()
    return name or "Suspense Party"


def _split_tax(txn: Any) -> Dict[str, float]:
    """Use recorded CGST/SGST/IGST. If only a combined GST amount exists, split by state.

    Same state code as a recorded counterparty GSTIN means intra-state (half CGST,
    half SGST). A different state code means IGST. With no GSTIN, intra-state is
    the default because the demo company is Karnataka and most seeded bills are too.
    """
    cgst = _get(txn, "cgst_amount")
    sgst = _get(txn, "sgst_amount")
    igst = _get(txn, "igst_amount")
    if any(part is not None for part in (cgst, sgst, igst)):
        return {"cgst": _num(cgst), "sgst": _num(sgst), "igst": _num(igst)}
    gst = _num(_get(txn, "gst_amount"))
    party_gstin = _get(txn, "buyer_gstin") if _get(txn, "entry_type") == "SALE" else _get(txn, "vendor_gstin")
    checked = validate_gstin(party_gstin) if party_gstin else None
    our_state = "29"
    if checked and checked.state_code and checked.state_code != our_state:
        return {"cgst": 0.0, "sgst": 0.0, "igst": gst}
    half = round(gst / 2.0, 2)
    other = round(gst - half, 2)
    return {"cgst": half, "sgst": other, "igst": 0.0}


def _base_amount(txn: Any, tax: Dict[str, float]) -> float:
    if _get(txn, "base_amount") is not None:
        return _num(_get(txn, "base_amount"))
    total = _num(_get(txn, "total_amount"))
    gst = tax["cgst"] + tax["sgst"] + tax["igst"]
    if total:
        return round(total - gst, 2)
    return 0.0


def _xml(value: Any) -> str:
    return escape("" if value is None else str(value))


def _ledger_xml(name: str, amount: float, deemed_positive: bool, party: bool = False) -> str:
    party_tag = "\n      <ISPARTYLEDGER>Yes</ISPARTYLEDGER>" if party else ""
    return (
        "    <ALLLEDGERENTRIES.LIST>\n"
        f"      <LEDGERNAME>{_xml(name)}</LEDGERNAME>\n"
        f"      <ISDEEMEDPOSITIVE>{'Yes' if deemed_positive else 'No'}</ISDEEMEDPOSITIVE>"
        f"{party_tag}\n"
        f"      <AMOUNT>{amount:.2f}</AMOUNT>\n"
        "    </ALLLEDGERENTRIES.LIST>"
    )


def _voucher_xml(txn: Any, index: int) -> str:
    entry = _get(txn, "entry_type") or "EXPENSE"
    is_sale = entry == "SALE"
    vch_type = "Sales" if is_sale else "Purchase"
    tax = _split_tax(txn)
    base = _base_amount(txn, tax)
    party = _party_name(txn)
    income_or_expense = "Sales" if is_sale else (_get(txn, "ledger_category") or "Purchase")
    # Credits are positive. Debits are negative. ISDEEMEDPOSITIVE Yes marks a debit.
    lines: List[str] = []
    signed_body = 0.0
    if is_sale:
        lines.append(_ledger_xml(income_or_expense, base, deemed_positive=False))
        signed_body += base
        if tax["cgst"]:
            lines.append(_ledger_xml("Output CGST", tax["cgst"], False))
            signed_body += tax["cgst"]
        if tax["sgst"]:
            lines.append(_ledger_xml("Output SGST", tax["sgst"], False))
            signed_body += tax["sgst"]
        if tax["igst"]:
            lines.append(_ledger_xml("Output IGST", tax["igst"], False))
            signed_body += tax["igst"]
        party_amount = round(-signed_body, 2)
        lines.insert(0, _ledger_xml(party, party_amount, deemed_positive=True, party=True))
    else:
        lines.append(_ledger_xml(income_or_expense, base, deemed_positive=True))
        signed_body += base
        if tax["cgst"]:
            lines.append(_ledger_xml("Input CGST", tax["cgst"], True))
            signed_body += tax["cgst"]
        if tax["sgst"]:
            lines.append(_ledger_xml("Input SGST", tax["sgst"], True))
            signed_body += tax["sgst"]
        if tax["igst"]:
            lines.append(_ledger_xml("Input IGST", tax["igst"], True))
            signed_body += tax["igst"]
        party_amount = round(-signed_body, 2)
        lines.insert(0, _ledger_xml(party, party_amount, deemed_positive=False, party=True))

    gstin = _get(txn, "buyer_gstin") if is_sale else _get(txn, "vendor_gstin")
    confidence = _get(txn, "ledger_confidence")
    conf_txt = f"{float(confidence) * 100:.1f}% ML confidence" if confidence is not None else "ML confidence n/a"
    score = _get(txn, "audit_score")
    score_txt = f"{float(score):.1f}" if score is not None else "n/a"
    narration = (
        f"FinAudit-AI | {vch_type} | {income_or_expense} | {conf_txt} | "
        f"readiness {score_txt} | payment {_get(txn, 'payment_mode') or 'unspecified'}"
    )
    number = _get(txn, "invoice_no") or f"FA-{index:04d}"
    remote = f"finaudit-{_get(txn, 'id') or index}"
    gstin_xml = f"\n    <PARTYGSTIN>{_xml(gstin)}</PARTYGSTIN>" if gstin else ""
    return (
        f"   <TALLYMESSAGE xmlns:UDF=\"TallyUDF\">\n"
        f"    <VOUCHER VCHTYPE=\"{_xml(vch_type)}\" ACTION=\"Create\" OBJVIEW=\"Invoice Voucher View\" REMOTEID=\"{_xml(remote)}\">\n"
        f"     <DATE>{_tally_date(_get(txn, 'invoice_date'))}</DATE>\n"
        f"     <VOUCHERTYPENAME>{vch_type}</VOUCHERTYPENAME>\n"
        f"     <VOUCHERNUMBER>{_xml(number)}</VOUCHERNUMBER>\n"
        f"     <PARTYLEDGERNAME>{_xml(party)}</PARTYLEDGERNAME>"
        f"{gstin_xml}\n"
        f"     <PERSISTEDVIEW>Invoice Voucher View</PERSISTEDVIEW>\n"
        f"     <ISINVOICE>Yes</ISINVOICE>\n"
        f"     <NARRATION>{_xml(narration)}</NARRATION>\n"
        + "\n".join(lines) + "\n"
        "    </VOUCHER>\n"
        "   </TALLYMESSAGE>"
    )


def generate_tally_xml(transactions: Sequence[Any]) -> str:
    """Valid Tally Prime import XML: ENVELOPE / BODY / IMPORTDATA, Sales and Purchase vouchers."""
    vouchers = "\n".join(_voucher_xml(txn, i) for i, txn in enumerate(transactions, start=1))
    if not vouchers:
        vouchers = "   <!-- No vouchers in the current books -->"
    return (
        "<ENVELOPE>\n"
        " <HEADER>\n"
        "  <TALLYREQUEST>Import Data</TALLYREQUEST>\n"
        " </HEADER>\n"
        " <BODY>\n"
        "  <IMPORTDATA>\n"
        "   <REQUESTDESC>\n"
        "    <REPORTNAME>Vouchers</REPORTNAME>\n"
        "    <STATICVARIABLES>\n"
        f"     <SVCURRENTCOMPANY>{_xml(COMPANY_NAME)}</SVCURRENTCOMPANY>\n"
        "    </STATICVARIABLES>\n"
        "   </REQUESTDESC>\n"
        "   <REQUESTDATA>\n"
        f"{vouchers}\n"
        "   </REQUESTDATA>\n"
        "  </IMPORTDATA>\n"
        " </BODY>\n"
        "</ENVELOPE>\n"
    )


def _place_of_supply(txn: Any) -> str:
    gstin = _get(txn, "buyer_gstin") if _get(txn, "entry_type") == "SALE" else _get(txn, "vendor_gstin")
    if not gstin:
        code = _get(txn, "gstin_state_code")
        return str(code or "")
    checked = validate_gstin(gstin)
    if checked.state_name and checked.state_code:
        return f"{checked.state_code} {checked.state_name}"
    return checked.state_code or ""


def _confidence_pct(txn: Any) -> Optional[float]:
    raw = _get(txn, "ledger_confidence")
    if raw is None:
        raw = _get(txn, "ml_confidence")
    if raw is None:
        return None
    value = float(raw)
    return round(value * 100, 1) if value <= 1 else round(value, 1)


def _sales_row(txn: Any) -> Dict[str, Any]:
    tax = _split_tax(txn)
    return {
        "Invoice Date": _text(_get(txn, "invoice_date")),
        "Invoice Number": _text(_get(txn, "invoice_no")),
        "Customer Name": _party_name(txn),
        "Customer GSTIN": _text(_get(txn, "buyer_gstin")),
        "Place of Supply": _place_of_supply(txn),
        "Taxable Value": _base_amount(txn, tax),
        "GST Rate %": _get(txn, "gst_rate"),
        "CGST": tax["cgst"],
        "SGST": tax["sgst"],
        "IGST": tax["igst"],
        "Invoice Value": _num(_get(txn, "total_amount")) or round(_base_amount(txn, tax) + tax["cgst"] + tax["sgst"] + tax["igst"], 2),
        "ML Category": _text(_get(txn, "ledger_category")),
        "ML Confidence %": _confidence_pct(txn),
    }


def _itc_row(txn: Any) -> Dict[str, Any]:
    tax = _split_tax(txn)
    gst = round(tax["cgst"] + tax["sgst"] + tax["igst"], 2)
    return {
        "Invoice Date": _text(_get(txn, "invoice_date")),
        "Invoice Number": _text(_get(txn, "invoice_no")),
        "Vendor Name": _party_name(txn),
        "Vendor GSTIN": _text(_get(txn, "vendor_gstin")),
        "Ledger Category": _text(_get(txn, "ledger_category")),
        "ML Confidence %": _confidence_pct(txn),
        "Payment Mode": _text(_get(txn, "payment_mode")),
        "Taxable Value": _base_amount(txn, tax),
        "GST Rate %": _get(txn, "gst_rate"),
        "GST Amount": gst or _num(_get(txn, "gst_amount")),
        "Invoice Value": _num(_get(txn, "total_amount")),
        "ITC Eligible": "Yes",
    }


def _coerce_issues(transactions: Sequence[Any], audit_results: Any) -> List[Dict[str, Any]]:
    if isinstance(audit_results, dict) and isinstance(audit_results.get("issues"), list):
        return list(audit_results["issues"])
    if isinstance(audit_results, list) and audit_results and isinstance(audit_results[0], dict) and "rule_code" in audit_results[0]:
        return list(audit_results)
    issues: List[Dict[str, Any]] = []
    for txn in transactions:
        flags = _get(txn, "flags") or []
        remarks = list(_get(txn, "remarks") or [])
        for flag in flags:
            flag_id = _get(flag, "id")
            tied = [r for r in remarks if _get(r, "flag_id") == flag_id]
            pool = tied or remarks
            pool = sorted(pool, key=lambda r: _get(r, "created_at") or datetime.min)
            remark = _get(pool[-1], "remark") if pool else None
            issues.append({
                "severity": _get(flag, "severity"),
                "source": _get(flag, "source"),
                "rule_code": _get(flag, "rule_code"),
                "legal_reference": _get(flag, "legal_reference"),
                "invoice_no": _get(txn, "invoice_no"),
                "vendor_name": _get(txn, "vendor_name"),
                "entry_type": _get(txn, "entry_type"),
                "total_amount": _get(txn, "total_amount"),
                "ledger_category": _get(txn, "ledger_category"),
                "ml_confidence": _get(txn, "ledger_confidence", _get(txn, "ml_confidence")),
                "anomaly_score": _get(txn, "anomaly_score"),
                "title": _get(flag, "title"),
                "message": _get(flag, "message"),
                "advocate_remark": remark,
                "audit_status": _get(txn, "status", _get(txn, "audit_status")),
                "is_resolved": _get(flag, "is_resolved"),
                "penalty": _get(flag, "penalty"),
            })
    return issues


def _flag_row(issue: Dict[str, Any]) -> Dict[str, Any]:
    confidence = issue.get("ml_confidence")
    if confidence is not None:
        value = float(confidence)
        confidence = round(value * 100, 1) if value <= 1 else round(value, 1)
    return {
        "Severity": _text(issue.get("severity")),
        "Source": _text(issue.get("source")),
        "Rule Code": _text(issue.get("rule_code")),
        "Legal Reference": _text(issue.get("legal_reference")),
        "Invoice Number": _text(issue.get("invoice_no")),
        "Party Name": _text(issue.get("vendor_name")),
        "Entry Type": _text(issue.get("entry_type")),
        "Invoice Value": _num(issue.get("total_amount")) if issue.get("total_amount") is not None else None,
        "ML Category": _text(issue.get("ledger_category")),
        "ML Confidence %": confidence,
        "Anomaly Score": issue.get("anomaly_score"),
        "Issue": _text(issue.get("message") or issue.get("title")),
        "Advocate Clarification Remark": _text(issue.get("advocate_remark")),
        "Audit Status": _text(issue.get("audit_status")),
        "Resolved": "Yes" if issue.get("is_resolved") else "No",
        "Penalty": issue.get("penalty"),
    }


def _style_sheet(ws: Worksheet, money_headers: Sequence[str], severity_col: Optional[int] = None) -> None:
    ws.auto_filter.ref = ws.dimensions
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.oddHeader.left.text = "FinAudit-AI"
    ws.oddFooter.right.text = "Page &P of &N"
    headers = [cell.value for cell in ws[1]]
    for cell in ws[1]:
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 30
    money_idx = {headers.index(name) + 1 for name in money_headers if name in headers}
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=ws.max_column):
        severity = row[severity_col - 1].value if severity_col else None
        source = None
        if "Source" in headers:
            source = row[headers.index("Source")].value
        resolved = None
        if "Resolved" in headers:
            resolved = row[headers.index("Resolved")].value
        fill = None
        if resolved == "Yes":
            fill = _RESOLVED_FILL
        elif severity == "CRITICAL":
            fill = _CRITICAL_FILL
        elif severity == "WARNING":
            fill = _WARNING_FILL
        elif source == "ML":
            fill = _ML_FILL
        elif row[0].row % 2 == 0:
            fill = _ZEBRA
        for cell in row:
            cell.border = _THIN
            cell.alignment = Alignment(vertical="center", wrap_text=(cell.column == headers.index("Issue") + 1 if "Issue" in headers else False))
            if fill is not None:
                cell.fill = fill
            if cell.column in money_idx and isinstance(cell.value, (int, float)):
                cell.number_format = _INR
    for idx, header in enumerate(headers, start=1):
        width = min(46, max(14, len(str(header)) + 4))
        if header in {"Issue", "Legal Reference", "Advocate Clarification Remark"}:
            width = 46
        if header in {"Customer Name", "Vendor Name", "Party Name", "ML Category"}:
            width = 32
        ws.column_dimensions[get_column_letter(idx)].width = width


def _write_frame(writer: pd.ExcelWriter, frame: pd.DataFrame, sheet: str, columns: Sequence[str]) -> Worksheet:
    if frame.empty:
        frame = pd.DataFrame(columns=list(columns))
    else:
        frame = frame.reindex(columns=list(columns))
    frame.to_excel(writer, sheet_name=sheet, index=False)
    return writer.sheets[sheet]


def generate_gst_excel(transactions: Sequence[Any], audit_results: Any) -> bytes:
    """Formatted workbook with B2B_Sales, B2C_Sales, ITC_Eligible_Expenses, Advocate_Audit_RedFlags."""
    sales = [txn for txn in transactions if (_get(txn, "entry_type") or "EXPENSE") == "SALE"]
    b2b = []
    b2c = []
    for txn in sales:
        checked = validate_gstin(_get(txn, "buyer_gstin"))
        (b2b if checked.is_valid else b2c).append(_sales_row(txn))
    itc = [_itc_row(txn) for txn in transactions if itc_eligible(txn)]
    flags = [_flag_row(issue) for issue in _coerce_issues(transactions, audit_results)]

    sales_columns = [
        "Invoice Date", "Invoice Number", "Customer Name", "Customer GSTIN", "Place of Supply",
        "Taxable Value", "GST Rate %", "CGST", "SGST", "IGST", "Invoice Value", "ML Category", "ML Confidence %",
    ]
    itc_columns = [
        "Invoice Date", "Invoice Number", "Vendor Name", "Vendor GSTIN", "Ledger Category",
        "ML Confidence %", "Payment Mode", "Taxable Value", "GST Rate %", "GST Amount", "Invoice Value", "ITC Eligible",
    ]
    flag_columns = [
        "Severity", "Source", "Rule Code", "Legal Reference", "Invoice Number", "Party Name", "Entry Type",
        "Invoice Value", "ML Category", "ML Confidence %", "Anomaly Score", "Issue",
        "Advocate Clarification Remark", "Audit Status", "Resolved", "Penalty",
    ]
    frames = {
        "B2B_Sales": (pd.DataFrame(b2b), sales_columns),
        "B2C_Sales": (pd.DataFrame(b2c), sales_columns),
        "ITC_Eligible_Expenses": (pd.DataFrame(itc), itc_columns),
        "Advocate_Audit_RedFlags": (pd.DataFrame(flags), flag_columns),
    }

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        money = ["Taxable Value", "CGST", "SGST", "IGST", "Invoice Value", "GST Amount", "Penalty"]
        for name, (frame, columns) in frames.items():
            ws = _write_frame(writer, frame, name, columns)
            severity_col = (columns.index("Severity") + 1) if "Severity" in columns else None
            if ws.max_row >= 1:
                _style_sheet(ws, money, severity_col if name == "Advocate_Audit_RedFlags" else None)
            ws.sheet_view.showGridLines = False
    return buffer.getvalue()
