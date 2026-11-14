# FinAudit-AI: Automated GST Bookkeeping, OCR & CA Compliance Engine

An AI-powered accounting, receipt scanning, anomaly auditing, and Chartered Accountant (CA) export platform built as a B.Tech AI/ML Final-Year Capstone Project.

## Key Features
- **Smart Receipt OCR & Entity Extraction:** Extracts vendor names, 15-digit Indian GSTINs, invoice numbers, dates, and tax splits (`CGST`/`SGST`/`IGST`) using Tesseract OCR and Regex pipelines.
- **Automated Ledger Classification:** Uses a trained `scikit-learn` text classifier to categorize expenses and sales into standard Indian accounting ledgers with confidence scoring.
- **Hybrid AI & GST Audit Engine:** Combines deterministic Indian GST compliance rules (Section 17(5) blocked ITC, duplicate voucher checks, GSTIN validation) with unsupervised Isolation Forest anomaly detection.
- **CA-Ready Exports:** Generates one-click **Tally Prime XML** (`<ENVELOPE>` vouchers) and multi-sheet **GSTR Working Paper Excel** reports.
- **Human-in-the-Loop CA Review:** Tracks audit readiness scores and allows Chartered Accountants to review flagged items and log remarks before filing.

## Tech Stack
- **Frontend:** React, Vite, Tailwind CSS
- **Backend:** Python, FastAPI, SQLAlchemy, SQLite
- **AI / ML & OCR:** Tesseract OCR (`pytesseract`), Scikit-Learn, Pandas, OpenPyXL

## How to Run Locally (macOS / Linux)

### 1. Start the FastAPI Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Start the React Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.
