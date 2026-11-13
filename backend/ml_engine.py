"""ml_engine.py - Training + inference for FinAudit-AI's two scikit-learn models.

1. Ledger classifier  : word+char TF-IDF  ->  Logistic Regression (multinomial, class-balanced)
2. Anomaly detector   : engineered tax/amount features -> Isolation Forest

Self-contained: running `python ml_engine.py train` generates a realistic synthetic
Indian business dataset (optionally merged with your own CSV at data/real_expenses.csv
having columns `description,category`), trains + evaluates both models, and serialises
them to ./models/*.joblib together with a metrics JSON. Application code only needs:

    from ml_engine import predict_ledger, detect_anomaly
    predict_ledger("Swiggy team lunch")            -> dict with confidence
    detect_anomaly({...invoice fields...}, vendor_history=[...])

CLI:
    python ml_engine.py train [--samples 5000] [--seed 42]
    python ml_engine.py predict "Dell Latitude laptop Redington"
    python ml_engine.py anomaly --total 49500 --base 41949 --gst-gst 7551 --rate 18 --category "Office Supplies"
    python ml_engine.py demo
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

log = logging.getLogger("finaudit.ml")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("FINAUDIT_MODEL_DIR", BASE_DIR / "models"))
DATA_DIR = Path(os.getenv("FINAUDIT_DATA_DIR", BASE_DIR / "data"))
LEDGER_PATH = MODEL_DIR / "ledger_classifier.joblib"
ANOMALY_PATH = MODEL_DIR / "anomaly_detector.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
REAL_DATA_PATH = DATA_DIR / "real_expenses.csv"

SEED = 42
LOW_CONFIDENCE_THRESHOLD = 0.55  # below this the UI asks a human to confirm the ledger
STATUTORY_THRESHOLDS = (10_000, 30_000, 50_000, 200_000)  # 40A(3) cash, 194C single, PAN/e-way, 269ST
GST_SLABS = (0.0, 0.25, 3.0, 5.0, 12.0, 18.0, 28.0)

# ---------------------------------------------------------------------------
# Domain configuration: ledgers, vendors, vocab, amount profiles
# ---------------------------------------------------------------------------
# weight  = share of synthetic samples (deliberately imbalanced, like real books)
# median/sigma = lognormal amount profile (INR, base value before GST)
CATEGORY_SPECS: Dict[str, Dict[str, Any]] = {
    "IT Infrastructure": {
        "weight": 14, "blocked": False, "tds": None, "gst_rates": [18], "median": 42000, "sigma": 0.9,
        "vendors": ["Redington India Ltd", "Ingram Micro India", "Croma Electronics", "Vijay Sales",
                    "Amazon Web Services India", "Supertron Electronics", "Rashi Peripherals",
                    "Compuage Infocom", "Zoho Corporation", "Sify Technologies"],
        "items": ["dell latitude laptop", "macbook air m2", "hp laserjet printer", "cisco 24 port switch",
                  "tp-link wifi router", "aws cloud hosting charges", "azure vm subscription",
                  "digitalocean droplet", "godaddy domain renewal", "ssl certificate renewal",
                  "ssd 1tb nvme", "ups battery backup", "server rack installation", "lan cabling",
                  "monitor 27 inch led", "zoho workplace licence", "microsoft 365 business subscription",
                  "google workspace seats", "github enterprise seats", "antivirus licence renewal"],
    },
    "Travel & Conveyance": {
        "weight": 14, "blocked": False, "tds": None, "gst_rates": [5, 12, 18], "median": 4500, "sigma": 0.8,
        "vendors": ["MakeMyTrip India", "IndiGo", "Ola Cabs", "Uber India", "IRCTC", "Taj Hotels",
                    "Lemon Tree Hotels", "Zoomcar", "Yatra Online", "Redbus"],
        "items": ["ola cab airport transfer", "uber ride client visit", "indigo flight del-bom",
                  "air india ticket", "irctc rajdhani ticket", "redbus volvo ticket",
                  "hotel accommodation outstation 2 nights", "oyo room booking", "fuel petrol reimbursement",
                  "toll fastag recharge", "taxi hire bangalore", "makemytrip booking", "local conveyance",
                  "airport parking charges", "visa processing travel"],
    },
    "Office Supplies": {
        "weight": 14, "blocked": False, "tds": None, "gst_rates": [12, 18], "median": 1800, "sigma": 0.8,
        "vendors": ["Staples India", "Navneet Education", "Camlin Stationers", "Sharma Stationery Mart",
                    "Aggarwal Office Solutions", "Godrej Interio", "Pilot Pen India", "JK Paper Ltd",
                    "Bharat Stationers", "Classmate Traders"],
        "items": ["a4 paper 75 gsm ream", "stationery items", "pen and notebooks", "printer cartridge toner",
                  "whiteboard markers", "stapler pins files", "office chairs ergonomic", "steel almirah",
                  "cleaning supplies housekeeping", "water purifier cans", "visiting cards printing",
                  "courier packing material", "sticky notes", "office furniture workstation"],
    },
    "Contractor Fees": {
        "weight": 11, "blocked": False, "tds": "194C", "gst_rates": [18, 12], "median": 38000, "sigma": 0.8,
        "vendors": ["Sai Construction Co", "Shree Ganesh Contractors", "Bhagwati Manpower Services",
                    "Sentinel Security Services", "Om Sai Enterprises", "Raj Electrical Works",
                    "Balaji Labour Contractors", "Krishna Interiors", "Apex Facility Services",
                    "Mahalaxmi Fabricators"],
        "items": ["civil works labour contract", "housekeeping manpower supply", "security guard services contract",
                  "transportation contractor charges", "electrical wiring contract work", "painting contractor labour",
                  "interior fit-out contract", "packing and forwarding contract", "cutting and stitching job work",
                  "manpower outsourcing agency", "loading unloading labour contractor", "printing job work",
                  "sub-contract fabrication work"],
    },
    "Professional Fees": {
        "weight": 10, "blocked": False, "tds": "194J", "gst_rates": [18], "median": 45000, "sigma": 0.8,
        "vendors": ["Kapoor & Associates", "Singh Legal Associates", "Gupta & Co Chartered Accountants",
                    "Deloitte Haskins", "Khaitan & Co", "Rao Consulting LLP", "Verma CS Services",
                    "Iyer Advisory LLP", "Sharma Law Chambers", "Nair Architects"],
        "items": ["chartered accountant audit fee", "legal consultation advocate fee", "tax advisory retainer",
                  "company secretary compliance filing fee", "software consulting professional fee",
                  "architect design consultancy", "gst return filing professional charges",
                  "trademark registration attorney fee", "management consulting fees",
                  "technical consultancy services", "valuation report fee", "legal drafting agreement charges",
                  "statutory audit fees"],
    },
    "Rent & Lease": {
        "weight": 6, "blocked": False, "tds": "194I", "gst_rates": [18], "median": 60000, "sigma": 0.6,
        "vendors": ["WeWork India", "Awfis Space Solutions", "Prestige Estates", "DLF Commercial",
                    "Anil Mehta Landlord", "Smartworks Coworking", "Embassy Property", "Gupta Properties",
                    "Brigade Enterprises", "Sunrise Realty"],
        "items": ["office rent monthly", "warehouse rental charges", "co-working desk rent", "godown lease rent",
                  "shop rent for month", "equipment lease rental", "coworking space membership",
                  "parking space rent", "rent for premises sector 62"],
    },
    "Utilities & Telecom": {
        "weight": 9, "blocked": False, "tds": None, "gst_rates": [18], "median": 6500, "sigma": 0.7,
        "vendors": ["Bharti Airtel", "Reliance Jio", "BESCOM", "Tata Power", "Vodafone Idea", "ACT Fibernet",
                    "BSNL", "UPPCL", "MSEDCL", "Hathway"],
        "items": ["electricity bill", "airtel broadband bill", "jio fiber connection", "vodafone idea postpaid",
                  "bsnl landline", "water charges municipal", "internet leased line",
                  "mobile recharge corporate plan", "power bill commercial", "fibernet broadband"],
    },
    "Marketing & Advertising": {
        "weight": 8, "blocked": False, "tds": "194C", "gst_rates": [18], "median": 26000, "sigma": 0.9,
        "vendors": ["Google India", "Meta Platforms", "Times Group", "Ogilvy India", "WebChutney", "Dentsu",
                    "Hoarding Media Co", "Ads2Go", "Vistaprint India", "Pixel Perfect Studio"],
        "items": ["google ads campaign spend", "facebook meta ads", "instagram influencer promotion",
                  "newspaper advertisement", "hoarding banner printing", "seo agency monthly retainer",
                  "exhibition stall booking", "brochure and flyer printing", "linkedin sponsored posts",
                  "youtube video promotion", "bulk sms marketing", "trade fair participation fee",
                  "branding merchandise tshirts"],
    },
    "Repairs & Maintenance": {
        "weight": 7, "blocked": False, "tds": "194C", "gst_rates": [18], "median": 7500, "sigma": 0.8,
        "vendors": ["Voltas Service", "Blue Star Care", "Urban Company", "Otis Elevators", "Kirloskar Services",
                    "Rentokil India", "Hikvision Service", "Sodexo Facilities", "Cool Tech Services",
                    "Sharma Electricals"],
        "items": ["ac servicing annual maintenance", "laptop repair service", "printer repair", "lift amc maintenance",
                  "generator servicing", "plumbing repair", "forklift servicing", "office renovation repair",
                  "pest control services", "cctv camera maintenance", "carpentry repair work",
                  "building painting maintenance", "fire extinguisher refill"],
    },
    "Blocked: Food & Beverages": {
        "weight": 12, "blocked": True, "tds": None, "gst_rates": [5, 18], "median": 1100, "sigma": 0.9,
        "vendors": ["Swiggy", "Zomato", "Dominos Pizza", "Cafe Coffee Day", "Starbucks", "Barbeque Nation",
                    "Haldirams", "McDonalds", "Paradise Biryani", "Saravana Bhavan"],
        "items": ["team lunch restaurant bill", "dinner at hotel restaurant", "swiggy order", "zomato food order",
                  "pizza party", "coffee and snacks", "staff birthday cake celebration",
                  "catering for diwali party", "outdoor catering banquet", "tea and snacks",
                  "biryani order", "client dinner restaurant", "food and beverage bill", "sweets boxes mithai"],
    },
    "Blocked: Personal & Club": {
        "weight": 4, "blocked": True, "tds": None, "gst_rates": [18, 12, 3], "median": 3500, "sigma": 1.0,
        "vendors": ["Tanishq", "Cult Fit", "Lakme Salon", "Shoppers Stop", "PVR Cinemas", "Taj Club House",
                    "Max Lifestyle", "Westside", "Kalyan Jewellers", "Naturals Salon"],
        "items": ["health club gym membership", "spa and salon", "jewellery purchase", "personal shopping clothing",
                  "club membership fee golf", "gift vouchers for employees", "family vacation resort",
                  "cosmetics beauty parlour", "life insurance personal premium", "school fees",
                  "personal car purchase for director", "movie tickets", "liquor bar bill", "designer clothes"],
    },
}
CATEGORIES: List[str] = list(CATEGORY_SPECS)
BLOCKED_ITC_CATEGORIES = {c for c, s in CATEGORY_SPECS.items() if s["blocked"]}
TDS_SECTION_BY_CATEGORY = {c: s["tds"] for c, s in CATEGORY_SPECS.items() if s["tds"]}

# NLP "tags" for CGST s.17(5)-type blocked credits: independent of the classifier so the
# rule engine can still flag a restaurant bill the model happened to put under another ledger.
ITC_TAG_PATTERNS: Dict[str, re.Pattern] = {
    "food_beverage": re.compile(r"\b(restaurant|swiggy|zomato|lunch|dinner|breakfast|biryani|pizza|cafe|coffee|"
                                r"catering|banquet|snacks|mithai|sweets|bar bill|bakery|canteen)\b", re.I),
    "health_club": re.compile(r"\b(gym|health club|fitness|cult fit|spa|salon|beauty|parlou?r|yoga)\b", re.I),
    "club_membership": re.compile(r"\b(club membership|golf|club house|clubhouse)\b", re.I),
    "gift_personal": re.compile(r"\b(gift voucher|gifts?|jewell?ery|tanishq|kalyan jewellers|liquor|designer clothes|"
                                r"cosmetics)\b", re.I),
    "personal_use": re.compile(r"\b(family vacation|school fees|personal|movie tickets|pvr|life insurance)\b", re.I),
    "motor_vehicle": re.compile(r"\b(personal car|motor car purchase|car purchase for director)\b", re.I),
}

EXTRAS = ["for march", "inv no 4521", "as per po 8841", "bill dated 12/04/2024", "gst invoice", "qty 2",
          "monthly", "advance payment", "head office", "branch bangalore", "reimbursement", "pune office",
          "q3 fy 2023-24", "tax invoice", "ref 77120"]
OCR_CONFUSIONS = {"o": "0", "l": "1", "s": "5", "b": "6", "i": "1", "e": "c", "m": "rn", "a": "o"}

# ---------------------------------------------------------------------------
# Text normalisation (applied OUTSIDE the sklearn pipeline, so the joblib file has
# no custom-function pickling dependency)
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"\b\d[\d,./-]*\b")
_PUNCT_RE = re.compile(r"[^a-z0-9&\- ]+")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: Optional[str]) -> str:
    """Lowercase, drop pure-number tokens (invoice nos, qty, dates), strip punctuation.

    Alphanumeric model codes such as 'a4' or 'm2' are kept (they are not pure numbers).
    """
    if not text:
        return ""
    t = text.lower()
    t = _NUM_RE.sub(" ", t)
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def blocked_itc_tags(text: Optional[str]) -> List[str]:
    """Keyword/NLP tags suggesting CGST s.17(5) blocked credit."""
    if not text:
        return []
    return [tag for tag, pat in ITC_TAG_PATTERNS.items() if pat.search(text)]


def build_ledger_text(vendor_name: Optional[str], line_descriptions: Sequence[str] = (),
                      raw_text: Optional[str] = None, max_raw_chars: int = 300) -> str:
    """Compose the classifier input for a whole invoice: vendor + items (+ a raw-text hint)."""
    parts: List[str] = []
    if vendor_name:
        parts.append(vendor_name)
    parts.extend(d for d in line_descriptions if d)
    if not parts and raw_text:
        parts.append(raw_text[:max_raw_chars])
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Synthetic dataset: ledger classifier
# ---------------------------------------------------------------------------
def _add_noise(text: str, rng: random.Random, p_noise: float = 0.35) -> str:
    """Simulate OCR damage + data-entry sloppiness."""
    if rng.random() < p_noise:
        out: List[str] = []
        for ch in text:
            r = rng.random()
            if r < 0.015:
                continue  # dropped character
            if r < 0.035 and ch.lower() in OCR_CONFUSIONS:
                out.append(OCR_CONFUSIONS[ch.lower()])  # look-alike confusion
            elif r < 0.045:
                out.append(ch * 2)  # doubled character
            else:
                out.append(ch)
        text = "".join(out)
    case = rng.random()
    if case < 0.25:
        text = text.upper()
    elif case < 0.45:
        text = text.title()
    return text


def _compose_description(spec: Dict[str, Any], rng: random.Random) -> str:
    vendor, item = rng.choice(spec["vendors"]), rng.choice(spec["items"])
    r = rng.random()
    if r < 0.30:
        text = f"{vendor} {item}"
    elif r < 0.50:
        text = item
    elif r < 0.65:
        text = f"{item} - {vendor}"
    elif r < 0.78:
        text = f"{item} {rng.choice(EXTRAS)}"
    elif r < 0.90:
        text = f"{vendor} {item} {rng.choice(EXTRAS)}"
    elif r < 0.95:
        text = f"{item} and {rng.choice(spec['items'])}"
    else:
        text = vendor  # vendor-only receipts (hard cases)
    return _add_noise(text, rng)


def load_real_expenses(path: Path = REAL_DATA_PATH) -> pd.DataFrame:
    """Optional user data: CSV with columns `description,category` (category must be a known ledger)."""
    if not path.exists():
        return pd.DataFrame(columns=["description", "category"])
    df = pd.read_csv(path)
    if not {"description", "category"}.issubset(df.columns):
        raise ValueError(f"{path} must contain 'description' and 'category' columns")
    df = df[df["category"].isin(CATEGORIES)].dropna(subset=["description"])
    log.info("Loaded %d real labelled expenses from %s", len(df), path)
    return df[["description", "category"]]


# Items/vendors that genuinely occur under several ledgers. Without these the synthetic task is
# trivially separable (100% accuracy) which would be a misleading result - real books are ambiguous.
AMBIGUOUS_POOL: List[Tuple[str, List[str]]] = [
    ("hotel bill", ["Travel & Conveyance", "Blocked: Food & Beverages"]),
    ("consulting services", ["Professional Fees", "Contractor Fees"]),
    ("printing charges", ["Office Supplies", "Marketing & Advertising"]),
    ("annual maintenance contract", ["Repairs & Maintenance", "IT Infrastructure"]),
    ("gift items", ["Marketing & Advertising", "Blocked: Personal & Club"]),
    ("mobile phone", ["IT Infrastructure", "Blocked: Personal & Club", "Office Supplies"]),
    ("amazon order", ["Office Supplies", "IT Infrastructure", "Blocked: Personal & Club"]),
    ("reliance retail bill", ["Office Supplies", "Blocked: Food & Beverages", "Blocked: Personal & Club"]),
    ("flipkart purchase", ["IT Infrastructure", "Office Supplies", "Blocked: Personal & Club"]),
    ("annual subscription", ["IT Infrastructure", "Utilities & Telecom", "Marketing & Advertising"]),
    ("rent and maintenance charges", ["Rent & Lease", "Repairs & Maintenance"]),
    ("courier and logistics charges", ["Office Supplies", "Contractor Fees"]),
    ("service charges", ["Repairs & Maintenance", "Professional Fees", "Contractor Fees"]),
    ("event management", ["Marketing & Advertising", "Contractor Fees", "Blocked: Food & Beverages"]),
    ("miscellaneous expenses", ["Office Supplies", "Blocked: Food & Beverages", "Travel & Conveyance"]),
    ("training and workshop fee", ["Professional Fees", "Marketing & Advertising"]),
    ("laptop bag and accessories", ["IT Infrastructure", "Office Supplies", "Blocked: Personal & Club"]),
]
AMBIGUOUS_FRACTION = 0.14
LABEL_NOISE = 0.02  # mislabelled rows, as in any manually-keyed ledger


def generate_ledger_dataset(n_samples: int = 5000, seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    cats = CATEGORIES
    weights = [CATEGORY_SPECS[c]["weight"] for c in cats]
    rows = []
    for cat in rng.choices(cats, weights=weights, k=n_samples):
        if rng.random() < AMBIGUOUS_FRACTION:
            item, possible = rng.choice(AMBIGUOUS_POOL)
            cat = rng.choice(possible)  # label is genuinely uncertain given the text
            extra = rng.choice(EXTRAS) if rng.random() < 0.5 else ""
            desc = _add_noise(f"{item} {extra}".strip(), rng)
        else:
            desc = _compose_description(CATEGORY_SPECS[cat], rng)
        if rng.random() < LABEL_NOISE:
            cat = rng.choice(cats)
        rows.append({"description": desc, "category": cat})
    df = pd.DataFrame(rows)
    real = load_real_expenses()
    if len(real):
        df = pd.concat([df, real], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ledger classifier: build / train / evaluate
# ---------------------------------------------------------------------------
def build_ledger_pipeline(C: float = 8.0) -> Pipeline:
    """Word n-grams capture semantics ('team lunch'); char n-grams make it robust to OCR typos
    ('restaurnt', 'S wiggy'). class_weight='balanced' re-weights the loss ~ 1/class_frequency."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode",
                                 token_pattern=r"(?u)\b[a-z][a-z0-9&\-]+\b", min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
    ])
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    return Pipeline([("tfidf", features), ("clf", clf)])


def _confidence_analysis(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> Dict[str, Any]:
    pred = classes[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    correct = pred == y_true
    table = []
    for thr in (0.0, 0.4, 0.55, 0.7, 0.85, 0.95):
        mask = conf >= thr
        table.append({
            "threshold": thr,
            "coverage": round(float(mask.mean()), 4),
            "accuracy": round(float(correct[mask].mean()), 4) if mask.any() else None,
        })
    return {
        "mean_confidence_correct": round(float(conf[correct].mean()), 4),
        "mean_confidence_incorrect": round(float(conf[~correct].mean()), 4) if (~correct).any() else None,
        "selective_prediction": table,
    }


def train_ledger_classifier(n_samples: int = 5000, seed: int = SEED, verbose: bool = True) -> Dict[str, Any]:
    df = generate_ledger_dataset(n_samples, seed)
    X = df["description"].map(normalize_text).to_numpy()
    y = df["category"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    # --- tiny hyper-parameter search on the training split, scored by macro-F1 (imbalance-aware)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    search: List[Dict[str, float]] = []
    best_c, best_f1 = 8.0, -1.0
    for C in (2.0, 8.0, 32.0):
        scores = cross_val_score(build_ledger_pipeline(C), X_tr, y_tr, cv=cv, scoring="f1_macro")
        search.append({"C": C, "macro_f1_mean": round(float(scores.mean()), 4), "std": round(float(scores.std()), 4)})
        if scores.mean() > best_f1:
            best_c, best_f1 = C, float(scores.mean())

    # --- honest hold-out evaluation
    eval_model = build_ledger_pipeline(best_c).fit(X_tr, y_tr)
    proba = eval_model.predict_proba(X_te)
    classes = eval_model.classes_
    y_pred = classes[proba.argmax(axis=1)]
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, y_pred, labels=list(classes))

    # --- final model: refit on ALL data (more data -> better production model)
    final = build_ledger_pipeline(best_c).fit(X, y)
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    metrics = {
        "model": "ledger_classifier",
        "version": version,
        "sklearn": sklearn.__version__,
        "n_samples": int(len(df)),
        "class_distribution": {k: int(v) for k, v in df["category"].value_counts().items()},
        "algorithm": "TF-IDF(word 1-2gram + char_wb 2-5gram) -> LogisticRegression(multinomial, class_weight=balanced)",
        "best_C": best_c,
        "hyperparameter_search": search,
        "holdout": {
            "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_te, y_pred, average="macro")), 4),
            "weighted_f1": round(float(f1_score(y_te, y_pred, average="weighted")), 4),
            "per_class": {k: {m: round(float(v[m]), 4) for m in ("precision", "recall", "f1-score")}
                          | {"support": int(v["support"])}
                          for k, v in report.items() if k in classes},
        },
        "confusion_matrix": {"labels": [str(c) for c in classes], "matrix": cm.tolist()},
        "confidence": _confidence_analysis(y_te, proba, classes),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "meta": {"version": version, "categories": [str(c) for c in classes],
                                            "best_C": best_c}}, LEDGER_PATH, compress=3)
    if verbose:
        print(f"[ledger] n={len(df)}  best C={best_c}  hold-out acc={metrics['holdout']['accuracy']}  "
              f"macro-F1={metrics['holdout']['macro_f1']}")
        print(classification_report(y_te, y_pred, zero_division=0))
    return metrics


# ---------------------------------------------------------------------------
# Anomaly detector: feature engineering
# ---------------------------------------------------------------------------
ANOMALY_FEATURES = ["log_total", "gst_rate", "tax_ratio_dev", "cat_z", "vendor_z",
                    "round_flag", "weekend_flag", "near_threshold"]


def _to_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _to_date(x: Any) -> Optional[date]:
    if x is None or x == "":
        return None
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    s = str(x).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _robust_center_scale(values: np.ndarray, min_scale: float = 0.20) -> Tuple[float, float]:
    """Median / MAD (x1.4826 ~ sigma for normal data). Robust to the outliers we are hunting."""
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) * 1.4826
    return med, max(mad, min_scale)


def build_anomaly_features(invoice: Dict[str, Any], vendor_history: Optional[Sequence[float]],
                           category_stats: Dict[str, Tuple[float, float]]
                           ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Turn one invoice into the model's feature vector (+ context used for explanations)."""
    total = _to_float(invoice.get("total_amount"))
    if total is None or total <= 0:
        return None, {}
    rate = _to_float(invoice.get("gst_rate"))
    base = _to_float(invoice.get("base_amount"))
    gst = _to_float(invoice.get("gst_amount"))
    if base is None and rate is not None:
        base = total / (1 + rate / 100.0)
    if gst is None and base is not None:
        gst = max(total - base, 0.0)

    eff_rate = (gst / base * 100.0) if (gst is not None and base and base > 0) else None
    if rate is None and eff_rate is not None:
        rate_feat = min(GST_SLABS, key=lambda s: abs(s - eff_rate))
    else:
        rate_feat = rate if rate is not None else 18.0
    if eff_rate is None:
        tax_dev = 0.0
    elif rate is not None:
        tax_dev = abs(eff_rate - rate)
    else:
        tax_dev = min(abs(eff_rate - s) for s in GST_SLABS)

    log_total = math.log1p(total)
    cat = invoice.get("ledger_category")
    c_med, c_scale = category_stats.get(cat, category_stats.get("__global__", (log_total, 1.0)))
    cat_z = (log_total - c_med) / c_scale

    vendor_median = None
    if vendor_history is not None and len(vendor_history) >= 3:
        logs = np.log1p(np.asarray([h for h in vendor_history if h and h > 0], dtype=float))
        if len(logs) >= 3:
            v_med, v_scale = _robust_center_scale(logs)
            vendor_z = (log_total - v_med) / v_scale
            vendor_median = float(np.expm1(v_med))
        else:
            vendor_z = 0.0
    else:
        vendor_z = 0.0

    nearest500 = round(total / 500.0) * 500.0
    round_flag = 1.0 if (total >= 5000 and abs(total - nearest500) < 0.01) else 0.0
    d = _to_date(invoice.get("invoice_date"))
    weekend = 1.0 if (d is not None and d.weekday() >= 5) else 0.0
    near_thr = 1.0 if any(0.90 * t <= total < t for t in STATUTORY_THRESHOLDS) else 0.0

    vec = np.array([log_total, rate_feat, tax_dev, cat_z, vendor_z, round_flag, weekend, near_thr], dtype=float)
    ctx = {"total": total, "base": base, "gst": gst, "eff_rate": eff_rate, "rate": rate, "category": cat,
           "vendor_median": vendor_median, "cat_median": float(np.expm1(c_med)), "date": d}
    return vec, ctx


# ---------------------------------------------------------------------------
# Synthetic dataset: transactions with injected anomalies (for evaluation)
# ---------------------------------------------------------------------------
def generate_transaction_dataset(n: int = 6000, anomaly_rate: float = 0.03, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    py = random.Random(seed)

    vendors: List[Dict[str, Any]] = []
    for cat, spec in CATEGORY_SPECS.items():
        for v in spec["vendors"]:
            vendors.append({
                "vendor": v, "category": cat, "rate_choices": spec["gst_rates"],
                "median": spec["median"] * float(rng.lognormal(0, 0.35)),
                "sigma": float(rng.uniform(0.18, 0.40)),  # a vendor's own spend is tighter than the category's
                "w": spec["weight"] / len(spec["vendors"]),
            })
    probs = np.array([v["w"] for v in vendors], dtype=float)
    probs /= probs.sum()
    start = date(2024, 4, 1)

    rows: List[Dict[str, Any]] = []
    for _ in range(n):
        v = vendors[int(rng.choice(len(vendors), p=probs))]
        rate = float(py.choice(v["rate_choices"]))
        base = float(v["median"] * math.exp(rng.normal(0, v["sigma"])))
        d = start + timedelta(days=int(rng.integers(0, 365)))
        if d.weekday() >= 5 and rng.random() < 0.6:  # businesses mostly transact on weekdays
            d += timedelta(days=int(7 - d.weekday()))
        gst = base * rate / 100.0
        label = "normal"
        if rng.random() < anomaly_rate:
            label = py.choice(["spike", "tax_mismatch", "threshold_hugging", "round_weekend"])
            if label == "spike":
                base *= float(rng.uniform(6, 20))
                gst = base * rate / 100.0
            elif label == "tax_mismatch":
                gst *= float(rng.uniform(0.4, 0.8)) if rng.random() < 0.5 else float(rng.uniform(1.3, 2.0))
            elif label == "threshold_hugging":
                total_t = py.choice([10_000, 30_000, 50_000]) * float(rng.uniform(0.93, 0.995))
                base = total_t / (1 + rate / 100.0)
                gst = base * rate / 100.0
            else:  # round_weekend
                total_t = float(py.choice([25_000, 50_000, 75_000, 100_000]))
                base = total_t / (1 + rate / 100.0)
                gst = base * rate / 100.0
                d += timedelta(days=(6 - d.weekday()) % 7)  # push to a Sunday
        rows.append({
            "vendor": v["vendor"], "ledger_category": v["category"], "invoice_date": d,
            "base_amount": round(base, 2), "gst_rate": rate, "gst_amount": round(gst, 2),
            "total_amount": round(base + gst, 2), "label": label, "is_anomaly": int(label != "normal"),
        })
    return pd.DataFrame(rows)


def _featurize_dataset(df: pd.DataFrame, category_stats: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Vendor history = every OTHER transaction of the same vendor (leave-one-out)."""
    X = np.zeros((len(df), len(ANOMALY_FEATURES)))
    by_vendor = {v: g["total_amount"].to_numpy() for v, g in df.groupby("vendor")}
    index_in_vendor = df.groupby("vendor").cumcount().to_numpy()
    records = df.to_dict("records")
    for i, rec in enumerate(records):
        hist = np.delete(by_vendor[rec["vendor"]], index_in_vendor[i])
        vec, _ = build_anomaly_features(rec, hist, category_stats)
        X[i] = vec
    return X


def _fit_category_stats(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for cat, g in df.groupby("ledger_category"):
        stats[str(cat)] = _robust_center_scale(np.log1p(g["total_amount"].to_numpy()), min_scale=0.30)
    stats["__global__"] = _robust_center_scale(np.log1p(df["total_amount"].to_numpy()), min_scale=0.30)
    return stats


def train_anomaly_detector(n: int = 6000, anomaly_rate: float = 0.03, seed: int = SEED,
                           verbose: bool = True) -> Dict[str, Any]:
    df = generate_transaction_dataset(n, anomaly_rate, seed)
    cat_stats = _fit_category_stats(df)
    X = _featurize_dataset(df, cat_stats)
    y = df["is_anomaly"].to_numpy()
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=y, random_state=seed)

    def make_forest() -> IsolationForest:
        # contamination='auto' -> we pick the decision threshold ourselves from the score sweep below.
        return IsolationForest(n_estimators=300, max_samples=256, contamination="auto",
                               random_state=seed, n_jobs=-1)

    forest = make_forest().fit(X[idx_tr])  # unsupervised: labels never touch the fit
    s_tr, s_te = -forest.score_samples(X[idx_tr]), -forest.score_samples(X[idx_te])  # higher = more anomalous

    # --- contamination sweep: contamination only moves the threshold (a quantile of training scores)
    sweep = []
    for c in (0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.12):
        thr = float(np.quantile(s_tr, 1 - c))
        p, r, f, _ = precision_recall_fscore_support(y[idx_tr], (s_tr >= thr).astype(int),
                                                     average="binary", zero_division=0)
        sweep.append({"contamination": c, "threshold": round(thr, 5), "precision": round(float(p), 4),
                      "recall": round(float(r), 4), "f1": round(float(f), 4)})
    best = max(sweep, key=lambda d: (d["f1"], -d["contamination"]))
    thr = best["threshold"]
    pred_te = (s_te >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y[idx_te], pred_te, average="binary", zero_division=0)
    per_type = {}
    for lab in sorted(set(df["label"]) - {"normal"}):
        m = (df["label"].to_numpy()[idx_te] == lab)
        per_type[lab] = round(float(pred_te[m].mean()), 4) if m.any() else None  # recall by anomaly type

    # --- final detector: refit on all data, reuse the chosen threshold *quantile*
    final = make_forest().fit(X)
    s_all = -final.score_samples(X)
    final_thr = float(np.quantile(s_all, 1 - best["contamination"]))
    score_grid = np.quantile(s_all, np.linspace(0, 1, 501))
    feat_mean, feat_std = X.mean(axis=0), X.std(axis=0) + 1e-9

    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    metrics = {
        "model": "anomaly_detector", "version": version, "sklearn": sklearn.__version__,
        "n_samples": int(len(df)), "injected_anomaly_rate": anomaly_rate,
        "algorithm": "IsolationForest(n_estimators=300, max_samples=256) on 8 engineered features",
        "features": ANOMALY_FEATURES,
        "chosen_contamination": best["contamination"],
        "holdout": {"roc_auc": round(float(roc_auc_score(y[idx_te], s_te)), 4),
                    "average_precision": round(float(average_precision_score(y[idx_te], s_te)), 4),
                    "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4),
                    "recall_by_anomaly_type": per_type},
        "contamination_sweep_train": sweep,
    }
    joblib.dump({
        "model": final, "features": ANOMALY_FEATURES, "category_stats": cat_stats,
        "threshold": final_thr, "score_grid": score_grid, "feat_mean": feat_mean, "feat_std": feat_std,
        "meta": {"version": version, "contamination": best["contamination"]},
    }, ANOMALY_PATH, compress=3)
    if verbose:
        print(f"[anomaly] n={len(df)}  contamination={best['contamination']}  AUC={metrics['holdout']['roc_auc']}  "
              f"P={metrics['holdout']['precision']}  R={metrics['holdout']['recall']}  F1={metrics['holdout']['f1']}")
        print(f"[anomaly] recall by type: {per_type}")
    return metrics


def train_all(n_ledger: int = 5000, n_txn: int = 6000, seed: int = SEED, verbose: bool = True) -> Dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {"ledger": train_ledger_classifier(n_ledger, seed, verbose),
               "anomaly": train_anomaly_detector(n_txn, 0.03, seed, verbose)}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    global _BUNDLE
    with _LOCK:
        _BUNDLE = None  # force reload of freshly trained artifacts
    if verbose:
        print(f"Artifacts written to {MODEL_DIR}")
    return metrics


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@dataclass
class ModelBundle:
    ledger_pipeline: Pipeline
    ledger_meta: Dict[str, Any]
    anomaly: Dict[str, Any]

    @property
    def ledger_version(self) -> str:
        return str(self.ledger_meta.get("version", ""))

    @property
    def anomaly_version(self) -> str:
        return str(self.anomaly["meta"].get("version", ""))


_BUNDLE: Optional[ModelBundle] = None
_LOCK = threading.Lock()


def load_models(auto_train: bool = True) -> ModelBundle:
    """Thread-safe lazy loader. Trains from synthetic data on first use if artifacts are missing."""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    with _LOCK:
        if _BUNDLE is not None:
            return _BUNDLE
        if not (LEDGER_PATH.exists() and ANOMALY_PATH.exists()):
            if not auto_train:
                raise FileNotFoundError("Model artifacts missing - run `python ml_engine.py train`")
            log.warning("Model artifacts missing - training on synthetic data (one-off)...")
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            m = {"ledger": train_ledger_classifier(verbose=False), "anomaly": train_anomaly_detector(verbose=False)}
            METRICS_PATH.write_text(json.dumps(m, indent=2))
        led = joblib.load(LEDGER_PATH)
        ano = joblib.load(ANOMALY_PATH)
        _BUNDLE = ModelBundle(led["pipeline"], led["meta"], ano)
        return _BUNDLE


def load_metrics() -> Dict[str, Any]:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


def predict_ledger_batch(texts: Sequence[str], top_k: int = 3) -> List[Dict[str, Any]]:
    b = load_models()
    cleaned = [normalize_text(t) for t in texts]
    proba = b.ledger_pipeline.predict_proba(cleaned)
    classes = b.ledger_pipeline.classes_
    out: List[Dict[str, Any]] = []
    for text, row in zip(texts, proba):
        order = np.argsort(row)[::-1]
        best = str(classes[order[0]])
        conf = float(row[order[0]])
        tags = blocked_itc_tags(text)
        out.append({
            "category": best,
            "confidence": round(conf, 4),
            "confidence_pct": round(conf * 100, 1),
            "label": f"{best} \u2013 {conf * 100:.1f}% confidence",
            "top_k": [{"category": str(classes[i]), "confidence": round(float(row[i]), 4)} for i in order[:top_k]],
            "is_blocked_itc": best in BLOCKED_ITC_CATEGORIES,
            "itc_tags": tags,
            "tds_section": TDS_SECTION_BY_CATEGORY.get(best),
            "needs_review": conf < LOW_CONFIDENCE_THRESHOLD,
            "model_version": b.ledger_version,
        })
    return out


def predict_ledger(text: str, top_k: int = 3) -> Dict[str, Any]:
    """Predict the accounting ledger for one description with calibrated-ish softmax confidence."""
    if not normalize_text(text):
        return {"category": "Uncategorised", "confidence": 0.0, "confidence_pct": 0.0,
                "label": "Uncategorised \u2013 0.0% confidence", "top_k": [], "is_blocked_itc": False,
                "itc_tags": [], "tds_section": None, "needs_review": True,
                "model_version": load_models().ledger_version}
    return predict_ledger_batch([text], top_k)[0]


def detect_anomaly(invoice: Dict[str, Any], vendor_history: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Score one invoice with the Isolation Forest.

    `invoice` keys: total_amount, base_amount, gst_rate, gst_amount, invoice_date, ledger_category.
    `vendor_history`: past invoice totals for the same vendor (>=3 enables vendor-relative scoring).
    """
    b = load_models()
    a = b.anomaly
    vec, ctx = build_anomaly_features(invoice, vendor_history, a["category_stats"])
    if vec is None:
        return {"score": 0.0, "raw_score": 0.0, "threshold": float(a["threshold"]), "is_anomaly": False,
                "reasons": ["Insufficient data: total amount missing, anomaly check skipped"],
                "features": {}, "model_version": b.anomaly_version}
    raw = float(-a["model"].score_samples(vec.reshape(1, -1))[0])
    grid = a["score_grid"]
    score = float(np.interp(raw, grid, np.linspace(0, 1, len(grid))))  # percentile vs training distribution
    is_anom = raw >= float(a["threshold"])
    feats = dict(zip(a["features"], [round(float(v), 4) for v in vec]))
    reasons = _explain(feats, ctx, vendor_history) if is_anom else []
    if is_anom and not reasons:
        reasons = ["Unusual combination of amount, tax and timing features (multivariate outlier)"]
    return {"score": round(score, 4), "raw_score": round(raw, 5), "threshold": round(float(a["threshold"]), 5),
            "is_anomaly": bool(is_anom), "reasons": reasons, "features": feats, "model_version": b.anomaly_version}


def _explain(f: Dict[str, float], ctx: Dict[str, Any], vendor_history: Optional[Sequence[float]]) -> List[str]:
    """Human-readable reason codes from the feature values that drive isolation."""
    out: List[str] = []
    total = ctx["total"]
    if f["vendor_z"] > 3.0 and ctx.get("vendor_median"):
        out.append(f"Amount \u20b9{total:,.0f} is {total / ctx['vendor_median']:.1f}x this vendor's usual "
                   f"(median \u20b9{ctx['vendor_median']:,.0f})")
    elif f["vendor_z"] < -3.5 and ctx.get("vendor_median"):
        out.append(f"Amount \u20b9{total:,.0f} is far below this vendor's usual (median \u20b9{ctx['vendor_median']:,.0f})")
    if f["cat_z"] > 2.5:
        out.append(f"Amount \u20b9{total:,.0f} is unusually high for ledger '{ctx['category']}' "
                   f"(typical \u20b9{ctx['cat_median']:,.0f})")
    if f["tax_ratio_dev"] > 1.0 and ctx.get("eff_rate") is not None:
        stated = f"stated {ctx['rate']:g}%" if ctx.get("rate") is not None else "any standard GST slab"
        out.append(f"Effective tax rate {ctx['eff_rate']:.1f}% does not match {stated}")
    if f["near_threshold"] >= 1:
        thr = min(t for t in STATUTORY_THRESHOLDS if 0.90 * t <= total < t)
        out.append(f"Amount \u20b9{total:,.0f} sits just below the \u20b9{thr:,} statutory threshold (possible bill-splitting)")
    if f["round_flag"] >= 1 and total >= 10_000:
        out.append(f"Round-figure invoice (\u20b9{total:,.0f}) - genuine bills rarely total exactly")
    if f["weekend_flag"] >= 1 and total >= 25_000:
        out.append("High-value invoice dated on a weekend")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _demo() -> None:
    samples = ["Swiggy team lunch order", "Dell Latitude 5440 laptop - Redington India", "OLA cab airport transfer",
               "Kapoor & Associates statutory audit fee", "Cult Fit gym membership", "Sai Construction labour contract",
               "xyz traders misc", "REST4URANT B1LL"]
    for s in samples:
        r = predict_ledger(s)
        print(f"{s!r:55} -> {r['label']:55} blocked={r['is_blocked_itc']} tds={r['tds_section']} "
              f"review={r['needs_review']}")
    print("\nAnomaly checks:")
    normal = {"total_amount": 2124.0, "base_amount": 1800.0, "gst_rate": 18, "gst_amount": 324.0,
              "invoice_date": "2025-01-14", "ledger_category": "Office Supplies"}
    spike = {**normal, "total_amount": 49500.0, "base_amount": 41949.15, "gst_amount": 7550.85}
    bad_tax = {**normal, "gst_amount": 120.0}
    split = {**normal, "total_amount": 9850.0, "base_amount": 8347.46, "gst_amount": 1502.54}
    hist = [1900, 2100, 1750, 2300, 2050, 1980]
    for name, inv in (("normal", normal), ("spike", spike), ("tax-mismatch", bad_tax), ("bill-splitting", split)):
        r = detect_anomaly(inv, hist)
        print(f"  {name:15} anomaly={r['is_anomaly']!s:5} score={r['score']:.2f}  {r['reasons']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FinAudit-AI ML engine")
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="generate data, train, evaluate and save both models")
    t.add_argument("--samples", type=int, default=5000)
    t.add_argument("--txns", type=int, default=6000)
    t.add_argument("--seed", type=int, default=SEED)
    p = sub.add_parser("predict", help="classify one expense description")
    p.add_argument("text")
    a = sub.add_parser("anomaly", help="score one invoice")
    a.add_argument("--total", type=float, required=True)
    a.add_argument("--base", type=float)
    a.add_argument("--gst-amount", type=float)
    a.add_argument("--rate", type=float)
    a.add_argument("--date")
    a.add_argument("--category")
    a.add_argument("--history", type=float, nargs="*")
    sub.add_parser("metrics", help="print stored evaluation metrics")
    sub.add_parser("demo", help="run sample predictions")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "train":
        train_all(args.samples, args.txns, args.seed)
    elif args.cmd == "predict":
        print(json.dumps(predict_ledger(args.text), indent=2, ensure_ascii=False))
    elif args.cmd == "anomaly":
        inv = {"total_amount": args.total, "base_amount": args.base, "gst_amount": args.gst_amount,
               "gst_rate": args.rate, "invoice_date": args.date, "ledger_category": args.category}
        print(json.dumps(detect_anomaly(inv, args.history), indent=2, ensure_ascii=False))
    elif args.cmd == "metrics":
        print(json.dumps(load_metrics(), indent=2))
    else:
        _demo()


if __name__ == "__main__":
    main()
