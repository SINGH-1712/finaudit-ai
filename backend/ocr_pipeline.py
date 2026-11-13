"""ocr_pipeline.py - Computer-vision + OCR + structured-extraction pipeline for Indian receipts.

Stages
------
1. ImagePreprocessor : resize -> grayscale -> shadow removal -> adaptive denoise -> CLAHE ->
                       projection-profile deskew -> Otsu/adaptive binarisation (chosen from image stats)
2. OCR engines       : EasyOCR (deep-learning, runs on gray) and/or Tesseract (runs on binary).
                       Every pass reports per-token confidence; the best pass is selected by a
                       quality score (mean confidence x coverage).
3. Line reconstruction from token boxes (y-clustering) so tabular invoices keep their rows.
4. FieldExtractor    : deterministic regex (GSTIN, dates, amounts, taxes) + OCR-confusion repair +
                       checksum-guided GSTIN correction + rapidfuzz vendor canonicalisation +
                       optional LLM JSON fallback (Ollama or OpenAI-compatible) for missing fields.
5. GSTIN validation  : format regex, state-code table and the official modulo-36 check character.

Public API
----------
    process_receipt(path, known_vendors=None) -> dict   (matches schemas.ProcessingResult)
    extract_from_text(text, known_vendors=None) -> dict (no image needed: tests / pasted text)
    validate_gstin(gstin) -> GSTINResult
    compute_gstin_check_char(first14) / generate_valid_gstin(state, pan)

CLI:
    python ocr_pipeline.py receipt.jpg [--debug-dir out/]
    python ocr_pipeline.py --selftest
"""
from __future__ import annotations

import argparse
import difflib
import json
import logging
import math
import os
import re
import shutil
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:  # rapidfuzz is the intended engine; difflib keeps the module importable without it
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz import process as _rf_process

    def fuzzy_best(query: str, choices: Sequence[str], cutoff: float = 90.0) -> Optional[Tuple[str, float]]:
        hit = _rf_process.extractOne(query, choices, scorer=_rf_fuzz.token_set_ratio, score_cutoff=cutoff)
        return (hit[0], float(hit[1])) if hit else None

except ImportError:  # pragma: no cover
    def fuzzy_best(query: str, choices: Sequence[str], cutoff: float = 90.0) -> Optional[Tuple[str, float]]:
        best: Optional[Tuple[str, float]] = None
        for c in choices:
            r = difflib.SequenceMatcher(None, query.lower(), c.lower()).ratio() * 100
            if r >= cutoff and (best is None or r > best[1]):
                best = (c, r)
        return best

log = logging.getLogger("finaudit.ocr")

MIN_TOKEN_CONF = 0.25  # tokens below this are dropped as noise
LOW_CONF = 0.60  # tokens below this count towards `low_conf_ratio`

# ===========================================================================
# 1. GSTIN: format + state code + modulo-36 checksum
# ===========================================================================
GSTIN_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_REGEX = re.compile(r"^(0[1-9]|[12][0-9]|3[0-8]|97|99)[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
STATE_CODES = {
    "01": "Jammu & Kashmir", "02": "Himachal Pradesh", "03": "Punjab", "04": "Chandigarh", "05": "Uttarakhand",
    "06": "Haryana", "07": "Delhi", "08": "Rajasthan", "09": "Uttar Pradesh", "10": "Bihar", "11": "Sikkim",
    "12": "Arunachal Pradesh", "13": "Nagaland", "14": "Manipur", "15": "Mizoram", "16": "Tripura",
    "17": "Meghalaya", "18": "Assam", "19": "West Bengal", "20": "Jharkhand", "21": "Odisha",
    "22": "Chhattisgarh", "23": "Madhya Pradesh", "24": "Gujarat", "25": "Daman & Diu",
    "26": "Dadra & Nagar Haveli and Daman & Diu", "27": "Maharashtra", "28": "Andhra Pradesh (old)",
    "29": "Karnataka", "30": "Goa", "31": "Lakshadweep", "32": "Kerala", "33": "Tamil Nadu", "34": "Puducherry",
    "35": "Andaman & Nicobar Islands", "36": "Telangana", "37": "Andhra Pradesh", "38": "Ladakh",
    "97": "Other Territory", "99": "Centre Jurisdiction",
}
_PAN_REGEX = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")


@dataclass
class GSTINResult:
    gstin: str
    is_valid_format: bool
    is_valid_checksum: bool
    state_code: Optional[str] = None
    state_name: Optional[str] = None
    pan: Optional[str] = None
    expected_check_char: Optional[str] = None
    reason: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.is_valid_format and self.is_valid_checksum


def compute_gstin_check_char(first14: str) -> str:
    """Official GSTIN check digit (Luhn-mod-36 variant).

    Walk the first 14 characters left->right; value = index in 0-9A-Z; multiply by 1 for
    odd positions and 2 for even positions; fold the product as (p // 36) + (p % 36); sum;
    check = (36 - sum % 36) % 36.
    """
    total = 0
    for i, ch in enumerate(first14.upper()):
        p = GSTIN_CHARSET.index(ch) * (1 if i % 2 == 0 else 2)
        total += p // 36 + p % 36
    return GSTIN_CHARSET[(36 - total % 36) % 36]


def validate_gstin(gstin: Optional[str]) -> GSTINResult:
    g = re.sub(r"[\s\-]", "", (gstin or "")).upper()
    if len(g) != 15 or any(c not in GSTIN_CHARSET for c in g):
        return GSTINResult(g, False, False, reason="GSTIN must be exactly 15 alphanumeric characters")
    if not GSTIN_REGEX.match(g):
        sc = g[:2]
        reason = (f"Invalid state code '{sc}'" if sc not in STATE_CODES
                  else "GSTIN structure invalid (expected 2 digits + PAN + entity + 'Z' + check char)")
        return GSTINResult(g, False, False, state_code=sc if sc in STATE_CODES else None, reason=reason)
    expected = compute_gstin_check_char(g[:14])
    ok = expected == g[14]
    return GSTINResult(
        g, True, ok, state_code=g[:2], state_name=STATE_CODES.get(g[:2]), pan=g[2:12], expected_check_char=expected,
        reason=None if ok else f"Checksum mismatch: check character should be '{expected}', found '{g[14]}'",
    )


def generate_valid_gstin(state_code: str, pan: str, entity_no: str = "1") -> str:
    """Build a checksum-valid GSTIN (used by the demo-data seeder and unit tests)."""
    if state_code not in STATE_CODES or not _PAN_REGEX.match(pan):
        raise ValueError("invalid state code or PAN")
    body = f"{state_code}{pan}{entity_no}Z"
    return body + compute_gstin_check_char(body)


# --- OCR-damage repair --------------------------------------------------------
_TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
_TO_ALPHA = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}
_CONFUSION = {"0": "OD", "O": "0", "D": "0", "1": "IL", "I": "1", "L": "1", "5": "S", "S": "5",
              "8": "B", "B": "8", "2": "Z", "Z": "2", "6": "G", "G": "6", "U": "V", "V": "U"}
# position classes: D = digit, A = alpha, N = alphanumeric
_GSTIN_SHAPE = "DDAAAAADDDDANZN"


def _positional_repair(window: str) -> str:
    out = []
    for ch, kind in zip(window, _GSTIN_SHAPE):
        if kind == "D" and not ch.isdigit():
            ch = _TO_DIGIT.get(ch, ch)
        elif kind == "A" and not ch.isalpha():
            ch = _TO_ALPHA.get(ch, ch)
        out.append(ch)
    out[13] = "Z" if out[13] in ("2", "Z") else out[13]
    return "".join(out)


def repair_gstin(raw: str) -> Tuple[str, str]:
    """Return (best_guess, method). Method in {'exact','positional','checksum','none'}.

    1) positional repair: digits where digits belong (O->0 ...), letters where letters belong;
    2) if the checksum still fails, accept a single-character confusion swap ONLY when exactly one
       such swap produces a checksum-valid GSTIN (error correction guided by the checksum).
    """
    g = re.sub(r"[^0-9A-Z]", "", raw.upper())
    if len(g) != 15:
        return g, "none"
    if validate_gstin(g).is_valid:
        return g, "exact"
    rep = _positional_repair(g)
    if validate_gstin(rep).is_valid:
        return rep, "positional"
    found = set()
    for i, ch in enumerate(rep):
        for alt in _CONFUSION.get(ch, ""):
            cand = rep[:i] + alt + rep[i + 1:]
            if validate_gstin(cand).is_valid:
                found.add(cand)
    if len(found) == 1:
        return found.pop(), "checksum"
    return rep, "positional" if rep != g else "none"


# ===========================================================================
# 2. Image preprocessing
# ===========================================================================
@dataclass
class PreprocessReport:
    original_size: Tuple[int, int] = (0, 0)
    processed_size: Tuple[int, int] = (0, 0)
    scale_factor: float = 1.0
    blur_score: float = 0.0  # variance of Laplacian: < ~60 means blurry
    contrast: float = 0.0  # std-dev of grayscale
    noise_sigma: float = 0.0
    background_variation: float = 0.0
    skew_angle: float = 0.0
    binarization: str = ""
    denoise: str = "none"
    is_blurry: bool = False
    steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ImagePreprocessor:
    """Receipt-oriented OpenCV pipeline. Returns an *enhanced grayscale* (best for neural OCR,
    which dislikes hard binarisation artefacts) and a *binary* image (best for Tesseract)."""

    def __init__(self, target_width: int = 1600, max_dim: int = 3400, blur_threshold: float = 60.0):
        self.target_width = target_width
        self.max_dim = max_dim
        self.blur_threshold = blur_threshold

    @staticmethod
    def load(path: str | Path) -> np.ndarray:
        data = np.fromfile(str(path), dtype=np.uint8)  # robust to unicode paths on macOS
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # honours EXIF orientation
        if img is None:
            raise ValueError(f"Cannot decode image: {path}")
        return img

    @staticmethod
    def _estimate_noise(gray: np.ndarray) -> float:
        """Immerkaer's fast noise-sigma estimator (Laplacian-difference kernel)."""
        h, w = gray.shape
        k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
        conv = cv2.filter2D(gray.astype(np.float32), -1, k)[1:-1, 1:-1]
        return float(np.sum(np.abs(conv)) * math.sqrt(0.5 * math.pi) / (6.0 * (w - 2) * (h - 2)))

    @staticmethod
    def _estimate_skew(binary_inv: np.ndarray, limit: float = 10.0) -> float:
        """Projection-profile search: the correct angle maximises row-sum variance (text rows align)."""
        h, w = binary_inv.shape
        scale = 600.0 / max(w, 1)
        small = cv2.resize(binary_inv, (600, max(int(h * scale), 1)), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape
        centre = (sw / 2, sh / 2)

        def score(angle: float) -> float:
            m = cv2.getRotationMatrix2D(centre, angle, 1.0)
            rot = cv2.warpAffine(small, m, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
            return float(np.var(rot.sum(axis=1)))

        coarse = max(np.arange(-limit, limit + 0.01, 0.5), key=score)
        fine = max(np.arange(coarse - 0.5, coarse + 0.51, 0.1), key=score)
        return float(fine)

    def process(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, PreprocessReport]:
        rep = PreprocessReport(original_size=(bgr.shape[1], bgr.shape[0]))

        # 1. scale: OCR accuracy collapses below ~20px x-height; upscale small phone crops
        h, w = bgr.shape[:2]
        scale = 1.0
        if w < self.target_width:
            scale = self.target_width / w
        elif max(h, w) > self.max_dim:
            scale = self.max_dim / max(h, w)
        if abs(scale - 1.0) > 1e-3:
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=interp)
            rep.steps.append(f"resize x{scale:.2f}")
        rep.scale_factor = round(scale, 3)

        # 2. grayscale + quality telemetry
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rep.blur_score = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
        rep.contrast = round(float(gray.std()), 2)
        rep.is_blurry = rep.blur_score < self.blur_threshold
        if rep.is_blurry:
            rep.warnings.append(f"Image looks blurry (Laplacian variance {rep.blur_score}); OCR confidence may be low")
        if rep.contrast < 35:
            rep.warnings.append("Very low contrast image; consider re-scanning")
        rep.steps.append("grayscale")

        # 3. illumination flattening: estimate background with dilate+median, divide it out
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        background = cv2.medianBlur(cv2.dilate(gray, kernel), 21)
        rep.background_variation = round(float(cv2.GaussianBlur(background, (0, 0), 25).std()), 2)
        flat = cv2.normalize(255 - cv2.absdiff(gray, background), None, 0, 255, cv2.NORM_MINMAX)
        rep.steps.append("shadow-removal")

        # 4. adaptive denoise: NLM is slow & blurs fine strokes, so only apply when noise warrants it
        rep.noise_sigma = round(self._estimate_noise(flat), 2)
        if rep.noise_sigma > 6.0:
            h_strength = float(min(15.0, 1.2 * rep.noise_sigma))
            flat = cv2.fastNlMeansDenoising(flat, None, h=h_strength, templateWindowSize=7, searchWindowSize=21)
            rep.denoise = f"nlm(h={h_strength:.1f})"
        elif rep.noise_sigma > 3.0:
            flat = cv2.bilateralFilter(flat, d=5, sigmaColor=40, sigmaSpace=40)
            rep.denoise = "bilateral"
        rep.steps.append(f"denoise:{rep.denoise}")

        # 5. CLAHE: local contrast boost for faded thermal-paper prints
        gray_enh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(flat)
        rep.steps.append("clahe")

        # 6. deskew
        _, otsu_inv = cv2.threshold(gray_enh, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        angle = self._estimate_skew(otsu_inv)
        rep.skew_angle = round(angle, 2)
        if abs(angle) >= 0.3:
            hh, ww = gray_enh.shape
            m = cv2.getRotationMatrix2D((ww / 2, hh / 2), angle, 1.0)
            gray_enh = cv2.warpAffine(gray_enh, m, (ww, hh), flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
            rep.steps.append(f"deskew {angle:+.1f}deg")

        # 7. binarise. Otsu = one global threshold: sharp & fast, but fails on uneven light.
        #    Adaptive Gaussian = local threshold: robust to gradients, but amplifies noise in blank areas.
        #    Because step 3 already flattened illumination, Otsu is preferred unless residual
        #    background variation is large.
        if rep.background_variation < 10.0:
            _, binary = cv2.threshold(gray_enh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            rep.binarization = "otsu"
        else:
            block = max(15, int(min(gray_enh.shape) / 40) // 2 * 2 + 1)
            binary = cv2.adaptiveThreshold(gray_enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block, 12)
            rep.binarization = f"adaptive(block={block})"
        rep.steps.append(f"binarize:{rep.binarization}")
        rep.processed_size = (gray_enh.shape[1], gray_enh.shape[0])
        return gray_enh, binary, rep


# ===========================================================================
# 3. OCR engines
# ===========================================================================
@dataclass
class OCRToken:
    text: str
    conf: float  # 0..1
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2.0

    @property
    def height(self) -> float:
        return float(max(self.box[3] - self.box[1], 1))


@dataclass
class OCRLine:
    text: str
    conf: float
    y: float
    tokens: List[OCRToken] = field(default_factory=list)


class BaseOCREngine:
    name = "base"
    preferred_image = "gray"  # "gray" | "binary"

    def available(self) -> bool:
        raise NotImplementedError

    def read(self, image: np.ndarray) -> List[OCRToken]:
        raise NotImplementedError


class EasyOCREngine(BaseOCREngine):
    name = "easyocr"
    preferred_image = "gray"
    _reader = None

    def available(self) -> bool:
        try:
            import easyocr  # noqa: F401
            return True
        except ImportError:
            return False

    def read(self, image: np.ndarray) -> List[OCRToken]:
        if EasyOCREngine._reader is None:
            import easyocr
            EasyOCREngine._reader = easyocr.Reader(["en"], gpu=False, verbose=False)  # CPU: portable on macOS
        out: List[OCRToken] = []
        for pts, text, conf in EasyOCREngine._reader.readtext(image, detail=1, paragraph=False):
            xs, ys = [int(p[0]) for p in pts], [int(p[1]) for p in pts]
            text = text.strip()
            if text:
                out.append(OCRToken(text, float(conf), (min(xs), min(ys), max(xs), max(ys))))
        return out


class TesseractEngine(BaseOCREngine):
    name = "tesseract"
    preferred_image = "binary"

    def __init__(self, psm: int = 6):
        self.psm = psm  # 6 = single uniform block of text: best for receipts

    def available(self) -> bool:
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            return False
        return shutil.which("tesseract") is not None

    def read(self, image: np.ndarray) -> List[OCRToken]:
        import pytesseract
        data = pytesseract.image_to_data(image, config=f"--oem 3 --psm {self.psm} -c preserve_interword_spaces=1",
                                         output_type=pytesseract.Output.DICT)
        out: List[OCRToken] = []
        for i, text in enumerate(data["text"]):
            text = (text or "").strip()
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if not text or conf < 0:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            out.append(OCRToken(text, conf / 100.0, (x, y, x + w, y + h)))
        return out


ENGINE_REGISTRY = {"easyocr": EasyOCREngine, "tesseract": TesseractEngine}


def available_engines(names: Optional[Sequence[str]] = None) -> List[BaseOCREngine]:
    names = names or [n.strip() for n in os.getenv("FINAUDIT_OCR_ENGINES", "easyocr,tesseract").split(",") if n.strip()]
    engines = []
    for n in names:
        cls = ENGINE_REGISTRY.get(n)
        if cls:
            eng = cls()
            if eng.available():
                engines.append(eng)
            else:
                log.info("OCR engine '%s' not available - skipping", n)
    return engines


def _quality(tokens: List[OCRToken]) -> Tuple[float, float]:
    """(mean confidence of kept tokens, pass-selection quality). Coverage term stops a tiny
    high-confidence fragment from beating a fuller, slightly less confident read."""
    if not tokens:
        return 0.0, 0.0
    mean = float(np.mean([t.conf for t in tokens]))
    chars = sum(len(t.text) for t in tokens)
    return mean, mean * (1.0 - math.exp(-chars / 250.0))


def run_ocr(gray: np.ndarray, binary: np.ndarray, engines: List[BaseOCREngine]
            ) -> Tuple[List[OCRToken], str, List[Dict[str, Any]]]:
    if not engines:
        raise RuntimeError("No OCR engine available. Install one: `pip install easyocr` or `brew install tesseract` "
                           "+ `pip install pytesseract`.")
    passes: List[Tuple[str, str, List[OCRToken]]] = []
    for eng in engines:
        imgs = [eng.preferred_image]
        if len(engines) == 1:  # only one engine: try both image variants and keep the better one
            imgs.append("binary" if eng.preferred_image == "gray" else "gray")
        for kind in imgs:
            try:
                toks = [t for t in eng.read(gray if kind == "gray" else binary) if t.conf >= MIN_TOKEN_CONF]
            except Exception as exc:  # engine failure must not kill the whole request
                log.warning("OCR pass %s/%s failed: %s", eng.name, kind, exc)
                continue
            passes.append((eng.name, kind, toks))
    if not passes:
        raise RuntimeError("All OCR passes failed")
    summary, best, best_q = [], None, -1.0
    for name, kind, toks in passes:
        mean, q = _quality(toks)
        summary.append({"engine": name, "image": kind, "tokens": len(toks),
                        "mean_confidence": round(mean, 4), "quality": round(q, 4)})
        if q > best_q:
            best, best_q = (name, kind, toks), q
    assert best is not None
    return best[2], f"{best[0]}:{best[1]}", summary


def tokens_to_lines(tokens: List[OCRToken]) -> List[OCRLine]:
    """Cluster tokens into text lines by vertical centre; keep column gaps as double spaces."""
    if not tokens:
        return []
    med_h = statistics.median(t.height for t in tokens)
    tol = max(med_h * 0.6, 4.0)
    rows: List[List[OCRToken]] = []
    for t in sorted(tokens, key=lambda k: k.cy):
        if rows and abs(t.cy - float(np.mean([x.cy for x in rows[-1]]))) <= tol:
            rows[-1].append(t)
        else:
            rows.append([t])
    lines: List[OCRLine] = []
    for row in rows:
        row.sort(key=lambda k: k.box[0])
        parts = [row[0].text]
        for prev, cur in zip(row, row[1:]):
            gap = cur.box[0] - prev.box[2]
            parts.append(("  " if gap > 1.8 * med_h else " ") + cur.text)
        weights = [max(len(t.text), 1) for t in row]
        conf = float(np.average([t.conf for t in row], weights=weights))
        lines.append(OCRLine("".join(parts), conf, float(np.mean([t.cy for t in row])), row))
    return lines


# ===========================================================================
# 4. Field extraction
# ===========================================================================
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                       "nov", "dec"], start=1)}
_DATE_PATTERNS = [
    (re.compile(r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})(?!\d)"), "dmy"),
    (re.compile(r"(?<!\d)(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})(?!\d)"), "ymd"),
    (re.compile(r"(?<!\d)(\d{1,2})[\s\-]*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,\-]*(\d{4}|\d{2})(?!\d)",
                re.I), "dMy"),
    (re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})(?!\d)", re.I), "Mdy"),
]
_NUM_TOKEN = re.compile(r"(?<![\w/.\-])(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d{1,2})?(?![\w/%]|\.\d)")
_TOTAL_KEY = re.compile(r"grand\s*total|total\s*amount|net\s*(?:amount\s*)?payable|amount\s*payable|invoice\s*total|"
                        r"total\s*invoice|bill\s*amount|amount\s*due|net\s*total|\btotal\b", re.I)
_TOTAL_EXCLUDE = re.compile(r"sub\s*-?\s*total|taxable|total\s*tax|total\s*gst|total\s*qty|total\s*items|"
                            r"total\s*discount|in\s*words|round", re.I)
_BASE_KEY = re.compile(r"taxable\s*(?:value|amount|amt)|sub\s*-?\s*total|basic\s*(?:amount|value)|net\s*amount|"
                       r"amount\s*before\s*tax|total\s*before\s*tax|taxable", re.I)
_BUYER_KEY = re.compile(r"bill(?:ed)?\s*to|buyer|customer|consignee|ship(?:ped)?\s*to|recipient|purchaser|sold\s*to", re.I)
_VENDOR_SUFFIX = re.compile(r"\b(pvt|private|ltd|limited|llp|llc|inc|traders?|trading|enterprises?|solutions|services|"
                            r"stores?|industries|associates|co|corporation|infotech|technologies|mart|agency|"
                            r"agencies|works|hotel|restaurant|cafe|chambers|consultants?|electronics|foods?|retail)\b|&",
                            re.I)
_HEADER_SKIP = re.compile(r"tax\s*invoice|invoice|bill\s*of\s*supply|cash\s*memo|receipt|original|duplicate|gstin|gst\s*no|"
                          r"\bpan\b|phone|\btel\b|mob|e-?mail|www\.|http|@|\bcin\b|\bdate\b|\bstate\b|\bcode\b|"
                          r"address|\bph[:.]|welcome|thank", re.I)
_INV_NO = re.compile(r"(?:tax\s*invoice|invoice|inv|bill|receipt|voucher|memo)\s*(?:no|number|num|#)\.?\s*[:\-#]?\s*"
                     r"([A-Z0-9][A-Z0-9\-/]{2,24})", re.I)
_SLABS = (0.0, 0.25, 3.0, 5.0, 12.0, 18.0, 28.0)


def parse_amount(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def numbers_in_line(line: str) -> List[float]:
    vals = []
    for m in _NUM_TOKEN.finditer(line):
        v = parse_amount(m.group(0))
        if v is not None:
            vals.append(v)
    return vals


def _parse_date_match(m: re.Match, kind: str) -> Optional[date]:
    g = m.groups()
    try:
        if kind == "dmy":
            d, mo, y = int(g[0]), int(g[1]), int(g[2])
        elif kind == "ymd":
            y, mo, d = int(g[0]), int(g[1]), int(g[2])
        elif kind == "dMy":
            d, mo, y = int(g[0]), _MONTHS[g[1][:3].lower()], int(g[2])
        else:
            mo, d, y = _MONTHS[g[0][:3].lower()], int(g[1]), int(g[2])
        if y < 100:
            y += 2000
        dt = date(y, mo, d)
    except (ValueError, KeyError):
        return None
    return dt if 2000 <= dt.year <= date.today().year + 1 else None


@dataclass
class FieldValue:
    value: Any
    conf: float
    source: str  # regex | fuzzy | derived | llm | repaired


@dataclass
class ExtractionResult:
    fields: Dict[str, Any] = field(default_factory=dict)
    confidences: Dict[str, float] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    all_gstins: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def put(self, key: str, fv: Optional[FieldValue]) -> None:
        if fv is not None and fv.value is not None:
            self.fields[key] = fv.value
            self.confidences[key] = round(max(0.0, min(1.0, fv.conf)), 4)
            self.sources[key] = fv.source


class FieldExtractor:
    """Hybrid extractor: deterministic regex first, fuzzy matching second, LLM last."""

    def __init__(self, known_vendors: Optional[Sequence[str]] = None):
        self.known_vendors = [v for v in (known_vendors or []) if v]

    # ---- GSTIN -------------------------------------------------------------
    def _find_gstins(self, lines: List[OCRLine]) -> List[Dict[str, Any]]:
        found: Dict[str, Dict[str, Any]] = {}
        for idx, ln in enumerate(lines):
            compact = re.sub(r"[^0-9A-Z]", "", ln.text.upper())
            context = " ".join(l.text for l in lines[max(0, idx - 1): idx + 1])
            is_buyer = bool(_BUYER_KEY.search(context))
            for i in range(0, max(len(compact) - 14, 0)):
                window = compact[i:i + 15]
                rep, method = repair_gstin(window)
                if len(rep) == 15 and GSTIN_REGEX.match(rep):
                    res = validate_gstin(rep)
                    if rep not in found:
                        found[rep] = {"gstin": rep, "raw": window, "line": idx, "repair": method,
                                      "checksum_valid": res.is_valid, "state_code": res.state_code,
                                      "role": "buyer" if is_buyer else "seller", "line_conf": ln.conf}
        return sorted(found.values(), key=lambda d: d["line"])

    # ---- Date --------------------------------------------------------------
    def _find_date(self, lines: List[OCRLine]) -> Optional[FieldValue]:
        cands: List[Tuple[int, date, OCRLine]] = []
        for ln in lines:
            for pat, kind in _DATE_PATTERNS:
                for m in pat.finditer(ln.text):
                    d = _parse_date_match(m, kind)
                    if d:
                        pri = 0 if re.search(r"\b(date|dated|dt|invoice\s*date|bill\s*date)\b", ln.text, re.I) else 1
                        if re.search(r"due|expir|valid", ln.text, re.I):
                            pri += 2
                        cands.append((pri, d, ln))
        if not cands:
            return None
        pri, d, ln = min(cands, key=lambda c: c[0])
        return FieldValue(d.isoformat(), ln.conf * (1.0 if pri == 0 else 0.85), "regex")

    # ---- Invoice number ----------------------------------------------------
    def _find_invoice_no(self, lines: List[OCRLine]) -> Optional[FieldValue]:
        for ln in lines:
            m = _INV_NO.search(ln.text)
            if not m:
                continue
            val = m.group(1).strip("-/")
            if re.search(r"\d", val) and not any(p.fullmatch(val) for p, _ in _DATE_PATTERNS):
                return FieldValue(val.upper(), ln.conf, "regex")
        return None

    # ---- Money -------------------------------------------------------------
    def _find_total(self, lines: List[OCRLine]) -> Optional[FieldValue]:
        best: Optional[Tuple[int, int, float, OCRLine]] = None
        for idx, ln in enumerate(lines):
            if _TOTAL_EXCLUDE.search(ln.text) or not _TOTAL_KEY.search(ln.text):
                continue
            nums = numbers_in_line(ln.text)
            if not nums:
                continue
            strong = 1 if re.search(r"grand|amount\s*payable|net\s*payable|invoice\s*total|amount\s*due", ln.text, re.I) else 0
            cand = (strong, idx, nums[-1], ln)
            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                best = cand
        if best:
            return FieldValue(best[2], best[3].conf, "regex")
        return None

    def _find_base(self, lines: List[OCRLine]) -> Optional[FieldValue]:
        best: Optional[Tuple[int, float, OCRLine]] = None
        for idx, ln in enumerate(lines):
            if _BASE_KEY.search(ln.text) and not re.search(r"total\s*tax|grand", ln.text, re.I):
                nums = numbers_in_line(ln.text)
                if nums:
                    best = (idx, nums[-1], ln)  # last occurrence wins (summary block sits at the bottom)
        return FieldValue(best[1], best[2].conf, "regex") if best else None

    def _find_taxes(self, lines: List[OCRLine]) -> Dict[str, Tuple[Optional[float], Optional[float], float]]:
        """name -> (rate %, amount, line confidence)."""
        out: Dict[str, Tuple[Optional[float], Optional[float], float]] = {}
        for ln in lines:
            for name in ("cgst", "sgst", "utgst", "igst"):
                if re.search(rf"\b{name}\b", ln.text, re.I):
                    pct = re.search(r"(\d{1,2}(?:\.\d+)?)\s*%", ln.text)
                    nums = [n for n in numbers_in_line(ln.text)
                            if not (pct and abs(n - float(pct.group(1))) < 1e-9 and len(numbers_in_line(ln.text)) > 1)]
                    amt = nums[-1] if nums else None
                    rate = float(pct.group(1)) if pct else None
                    if rate is None and amt is not None and len(nums) >= 2 and nums[0] in _SLABS + (2.5, 6.0, 9.0, 14.0):
                        rate = nums[0]  # "CGST 9 810.00" (percent sign lost by OCR)
                    out[name] = (rate, amt, ln.conf)
        return out

    def _find_generic_gst(self, lines: List[OCRLine]) -> Optional[Tuple[float, Optional[float], float]]:
        for ln in lines:
            if re.search(r"\bgst\b", ln.text, re.I) and not re.search(r"gstin|[cs]gst|igst|utgst", ln.text, re.I):
                pct = re.search(r"(\d{1,2}(?:\.\d+)?)\s*%", ln.text)
                if pct:
                    nums = [n for n in numbers_in_line(ln.text) if abs(n - float(pct.group(1))) > 1e-9]
                    return float(pct.group(1)), (nums[-1] if nums else None), ln.conf
        return None

    # ---- Payment mode ------------------------------------------------------
    def _find_payment_mode(self, lines: List[OCRLine]) -> Optional[FieldValue]:
        patterns = [
            ("CASH", re.compile(r"\bcash\b(?!\s*(?:/\s*credit\s*)?(?:memo|invoice))", re.I)),
            ("UPI", re.compile(r"\b(upi|gpay|google\s*pay|phonepe|paytm|bhim)\b", re.I)),
            ("CARD", re.compile(r"\b(card|visa|mastercard|rupay|pos)\b", re.I)),
            ("BANK_TRANSFER", re.compile(r"\b(neft|rtgs|imps|bank\s*transfer|net\s*banking)\b", re.I)),
            ("CHEQUE", re.compile(r"\b(cheque|chq|check\s*no)\b", re.I)),
        ]
        for ln in lines:  # prefer an explicit "payment / mode / paid / tender" line
            if re.search(r"payment|mode|paid|tender|settled", ln.text, re.I):
                for mode, pat in patterns:
                    if pat.search(ln.text):
                        return FieldValue(mode, ln.conf, "regex")
        for ln in lines:
            for mode, pat in patterns:
                if pat.search(ln.text):
                    return FieldValue(mode, ln.conf * 0.8, "regex")
        return None

    # ---- Vendor ------------------------------------------------------------
    def _find_vendor(self, lines: List[OCRLine]) -> Optional[FieldValue]:
        best: Optional[Tuple[float, OCRLine]] = None
        for idx, ln in enumerate(lines[:8]):
            text = re.sub(r"\s{2,}", " ", ln.text).strip()
            letters = sum(c.isalpha() for c in text)
            if letters < 4 or _HEADER_SKIP.search(text):
                continue
            digits = sum(c.isdigit() for c in text)
            score = letters / max(len(text), 1) - 0.4 * digits / max(len(text), 1)
            score += 1.0 if _VENDOR_SUFFIX.search(text) else 0.0
            score += 0.3 if (text.isupper() or text.istitle()) else 0.0
            score += max(0.0, 0.4 - 0.08 * idx)
            if best is None or score > best[0]:
                best = (score, ln)
        if not best:
            return None
        name = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9.)&]+$", "", re.sub(r"\s{2,}", " ", best[1].text)).strip()
        conf = best[1].conf
        source = "regex"
        hit = fuzzy_best(name, self.known_vendors, cutoff=90.0) if self.known_vendors and len(name) >= 4 else None
        if hit:
            name, conf, source = hit[0], conf * (hit[1] / 100.0), "fuzzy"
        return FieldValue(name, conf, source)

    # ---- Line items --------------------------------------------------------
    _ITEM_FULL = re.compile(r"^\s*(?:\d{1,2}[.)]?\s+)?(?P<desc>[A-Za-z][^\n]*?[A-Za-z0-9)])\s+(?:(?P<hsn>\d{4,8})\s+)?"
                            r"(?P<qty>\d+(?:\.\d+)?)\s*(?:nos?|pcs?|units?|kgs?|ltr|x)?\s+(?P<rate>\d[\d,]*\.\d{1,2})\s+"
                            r"(?P<amt>\d[\d,]*\.\d{1,2})\s*$", re.I)
    _ITEM_SIMPLE = re.compile(r"^\s*(?:\d{1,2}[.)]?\s+)?(?P<desc>[A-Za-z][^\n]{2,60}?)\s+(?P<amt>\d[\d,]*\.\d{2})\s*$")
    _ITEM_SKIP = re.compile(r"total|tax|gst|discount|round|invoice|date|gstin|phone|change|cash|upi|card|balance|"
                            r"amount|payable|paid|tender|thank", re.I)

    def _find_line_items(self, lines: List[OCRLine]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for ln in lines:
            text = re.sub(r"\s{2,}", "  ", ln.text).replace("  ", " ")
            if self._ITEM_SKIP.search(text):
                continue
            m = self._ITEM_FULL.match(text)
            if m:
                d = m.groupdict()
                items.append({"description": d["desc"].strip(), "hsn_sac": d.get("hsn"),
                              "quantity": float(d["qty"]), "rate": parse_amount(d["rate"]),
                              "amount": parse_amount(d["amt"]), "confidence": round(ln.conf, 4)})
                continue
            m = self._ITEM_SIMPLE.match(text)
            if m and sum(c.isalpha() for c in m.group("desc")) >= 3:
                items.append({"description": m.group("desc").strip(), "hsn_sac": None, "quantity": None,
                              "rate": None, "amount": parse_amount(m.group("amt")),
                              "confidence": round(ln.conf, 4)})
        return items

    # ---- Orchestration -----------------------------------------------------
    def extract(self, lines: List[OCRLine]) -> ExtractionResult:
        res = ExtractionResult()

        gstins = self._find_gstins(lines)
        res.all_gstins = gstins
        sellers = [g for g in gstins if g["role"] == "seller"]
        buyers = [g for g in gstins if g["role"] == "buyer"]
        chosen = next((g for g in sellers if g["checksum_valid"]), None) or (sellers[0] if sellers else None)
        if chosen:
            v = validate_gstin(chosen["gstin"])
            factor = {"exact": 1.0, "positional": 0.85, "checksum": 0.75, "none": 0.7}[chosen["repair"]]
            conf = chosen["line_conf"] * factor * (1.0 if v.is_valid else 0.6)
            res.put("vendor_gstin", FieldValue(chosen["gstin"], conf, "regex" if chosen["repair"] == "exact" else "repaired"))
            res.fields["gstin_valid"] = v.is_valid
            res.fields["gstin_state_code"] = v.state_code
            res.fields["gstin_state_name"] = v.state_name
            if not v.is_valid:
                res.warnings.append(f"Vendor GSTIN {chosen['gstin']} failed validation: {v.reason}")
            if chosen["repair"] in ("positional", "checksum"):
                res.warnings.append(f"GSTIN auto-corrected from OCR text '{chosen['raw']}' ({chosen['repair']} repair)")
        if buyers:
            res.put("buyer_gstin", FieldValue(buyers[0]["gstin"], buyers[0]["line_conf"], "regex"))

        res.put("invoice_date", self._find_date(lines))
        res.put("invoice_no", self._find_invoice_no(lines))
        res.put("vendor_name", self._find_vendor(lines))
        res.put("payment_mode", self._find_payment_mode(lines))
        res.put("total_amount", self._find_total(lines))
        res.put("base_amount", self._find_base(lines))

        taxes = self._find_taxes(lines)
        rate_sum: Optional[float] = None
        for name in ("cgst", "sgst", "igst"):
            if name in taxes and taxes[name][1] is not None:
                res.put(f"{name}_amount", FieldValue(taxes[name][1], taxes[name][2], "regex"))
        if "cgst" in taxes and "sgst" in taxes and taxes["cgst"][0] is not None and taxes["sgst"][0] is not None:
            rate_sum = taxes["cgst"][0] + taxes["sgst"][0]
            res.put("gst_rate", FieldValue(rate_sum, min(taxes["cgst"][2], taxes["sgst"][2]), "regex"))
        elif "igst" in taxes and taxes["igst"][0] is not None:
            res.put("gst_rate", FieldValue(taxes["igst"][0], taxes["igst"][2], "regex"))
        else:
            gen = self._find_generic_gst(lines)
            if gen:
                res.put("gst_rate", FieldValue(gen[0], gen[2], "regex"))
                if gen[1] is not None:
                    res.put("gst_amount", FieldValue(gen[1], gen[2], "regex"))

        res.line_items = self._find_line_items(lines)
        self.reconcile(res)
        return res

    @staticmethod
    def reconcile(res: ExtractionResult) -> None:
        """Fill ONLY missing values from arithmetic. Never overwrite OCR'd numbers: a mismatch between
        printed figures is a compliance finding for the audit engine, not something to silently fix."""
        f, c = res.fields, res.confidences
        parts = [f.get(k) for k in ("cgst_amount", "sgst_amount", "igst_amount") if f.get(k) is not None]

        def derive(key: str, value: float, basis: Sequence[str]) -> None:
            f[key] = round(value, 2)
            c[key] = round(0.6 * min([c.get(b, 0.6) for b in basis] or [0.6]), 4)
            res.sources[key] = "derived"

        if f.get("gst_amount") is None and parts:
            derive("gst_amount", sum(parts), [k for k in ("cgst_amount", "sgst_amount", "igst_amount") if k in f])
        if f.get("base_amount") is None and f.get("total_amount") is not None and f.get("gst_amount") is not None:
            derive("base_amount", f["total_amount"] - f["gst_amount"], ["total_amount", "gst_amount"])
        if f.get("gst_amount") is None and f.get("base_amount") is not None and f.get("total_amount") is not None:
            derive("gst_amount", max(f["total_amount"] - f["base_amount"], 0.0), ["total_amount", "base_amount"])
        if f.get("total_amount") is None and f.get("base_amount") is not None and f.get("gst_amount") is not None:
            derive("total_amount", f["base_amount"] + f["gst_amount"], ["base_amount", "gst_amount"])
        if f.get("gst_rate") is None and f.get("base_amount") and f.get("gst_amount") is not None:
            eff = f["gst_amount"] / f["base_amount"] * 100.0
            snapped = min(_SLABS, key=lambda s: abs(s - eff))
            if abs(snapped - eff) <= 0.75:
                derive("gst_rate", snapped, ["base_amount", "gst_amount"])
        if f.get("total_amount") is None:
            res.warnings.append("Total amount could not be identified")
        if f.get("invoice_date") is None:
            res.warnings.append("Invoice date could not be identified")


# ---- Optional LLM structured-JSON fallback ---------------------------------
LLM_FIELDS = ("vendor_name", "vendor_gstin", "invoice_no", "invoice_date", "base_amount", "gst_rate",
              "gst_amount", "total_amount", "payment_mode")
_LLM_PROMPT = (
    "You extract fields from OCR text of an Indian GST invoice/receipt. The text may contain OCR errors. "
    "Return ONLY a JSON object with exactly these keys: vendor_name (string), vendor_gstin (15-char string), "
    "invoice_no (string), invoice_date (YYYY-MM-DD), base_amount (number, pre-tax), gst_rate (number, percent), "
    "gst_amount (number), total_amount (number), payment_mode (one of CASH, UPI, CARD, BANK_TRANSFER, CHEQUE). "
    "Use null when a value is not present. Do not invent values.\n\nOCR TEXT:\n"
)


def _llm_call(raw_text: str) -> Optional[Dict[str, Any]]:
    """Provider selected by env: FINAUDIT_LLM_PROVIDER=ollama|openai. Uses stdlib urllib only."""
    provider = os.getenv("FINAUDIT_LLM_PROVIDER", "").lower()
    if provider not in ("ollama", "openai"):
        return None
    prompt = _LLM_PROMPT + raw_text[:4000]
    timeout = float(os.getenv("FINAUDIT_LLM_TIMEOUT", "45"))
    try:
        if provider == "ollama":
            url = os.getenv("FINAUDIT_LLM_URL", "http://localhost:11434") + "/api/generate"
            body = {"model": os.getenv("FINAUDIT_LLM_MODEL", "llama3.2"), "prompt": prompt, "stream": False,
                    "format": "json", "options": {"temperature": 0}}
            headers = {"Content-Type": "application/json"}
            req = urllib.request.Request(url, json.dumps(body).encode(), headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(json.loads(r.read())["response"])
        url = os.getenv("FINAUDIT_LLM_URL", "https://api.openai.com/v1") + "/chat/completions"
        body = {"model": os.getenv("FINAUDIT_LLM_MODEL", "gpt-4o-mini"), "temperature": 0,
                "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": prompt}]}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('FINAUDIT_LLM_API_KEY', '')}"}
        req = urllib.request.Request(url, json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(json.loads(r.read())["choices"][0]["message"]["content"])
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
        log.warning("LLM fallback failed: %s", exc)
        return None


def llm_fill_missing(res: ExtractionResult, raw_text: str) -> bool:
    """Fill ONLY fields the deterministic pass missed; validate before accepting. Returns True if used."""
    critical = ("vendor_name", "invoice_no", "invoice_date", "total_amount")
    if all(k in res.fields for k in critical):
        return False
    data = _llm_call(raw_text)
    if not data:
        if os.getenv("FINAUDIT_LLM_PROVIDER"):
            res.warnings.append("LLM fallback was configured but returned no usable result")
        return False
    used = False
    for key in LLM_FIELDS:
        if key in res.fields or data.get(key) in (None, "", "null"):
            continue
        val = data[key]
        try:
            if key in ("base_amount", "gst_rate", "gst_amount", "total_amount"):
                val = float(val)
            elif key == "invoice_date":
                val = datetime.strptime(str(val)[:10], "%Y-%m-%d").date().isoformat()
            elif key == "vendor_gstin":
                val = re.sub(r"\s", "", str(val)).upper()
                if not GSTIN_REGEX.match(val):
                    continue
                v = validate_gstin(val)
                res.fields["gstin_valid"], res.fields["gstin_state_code"] = v.is_valid, v.state_code
                res.fields["gstin_state_name"] = v.state_name
            elif key == "payment_mode":
                val = str(val).upper()
                if val not in {"CASH", "UPI", "CARD", "BANK_TRANSFER", "CHEQUE"}:
                    continue
            else:
                val = str(val).strip()
        except (ValueError, TypeError):
            continue
        res.put(key, FieldValue(val, 0.5, "llm"))  # fixed low prior: the UI shows these as "needs review"
        used = True
    if used:
        FieldExtractor.reconcile(res)
        res.warnings.append("Some fields were filled by the LLM fallback - please verify")
    return used


# ===========================================================================
# 5. Public API
# ===========================================================================
def _result_dict(lines: List[OCRLine], tokens: List[OCRToken], engine_name: str, passes: List[Dict[str, Any]],
                 prep: PreprocessReport, ext: ExtractionResult, ms: int) -> Dict[str, Any]:
    confs = [t.conf for t in tokens] or [0.0]
    _, quality = _quality(tokens)
    extracted = {k: ext.fields.get(k) for k in (
        "vendor_name", "vendor_gstin", "gstin_valid", "gstin_state_code", "gstin_state_name", "buyer_gstin",
        "invoice_no", "invoice_date", "base_amount", "gst_rate", "cgst_amount", "sgst_amount", "igst_amount",
        "gst_amount", "total_amount", "payment_mode")}
    warnings = list(prep.warnings) + ext.warnings
    mean_conf = float(np.mean(confs))
    if mean_conf < 0.6:
        warnings.append(f"Low overall OCR confidence ({mean_conf:.0%}) - verify every field manually")
    return {
        "ocr": {
            "engine": engine_name, "mean_confidence": round(mean_conf, 4),
            "median_confidence": round(float(np.median(confs)), 4),
            "low_conf_ratio": round(float(np.mean([c < LOW_CONF for c in confs])), 4),
            "token_count": len(tokens), "quality": round(quality, 4),
            "preprocess": asdict(prep), "passes": passes,
        },
        "raw_text": "\n".join(l.text for l in lines),
        "lines": [{"text": l.text, "confidence": round(l.conf, 4)} for l in lines],
        "extracted": extracted,
        "field_confidences": ext.confidences,
        "field_sources": ext.sources,
        "line_items": [{k: v for k, v in it.items() if k != "confidence"} for it in ext.line_items],
        "all_gstins": ext.all_gstins,
        "warnings": warnings,
        "processing_ms": ms,
    }


def process_receipt(image_path: str | Path, known_vendors: Optional[Sequence[str]] = None,
                    use_llm_fallback: bool = True, debug_dir: Optional[str | Path] = None,
                    engines: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Image file -> structured invoice dict + OCR telemetry. Raises RuntimeError if no OCR engine exists."""
    t0 = time.perf_counter()
    pre = ImagePreprocessor()
    bgr = pre.load(image_path)
    gray, binary, prep = pre.process(bgr)
    if debug_dir:
        d = Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        cv2.imwrite(str(d / f"{stem}_gray.png"), gray)
        cv2.imwrite(str(d / f"{stem}_binary.png"), binary)

    tokens, engine_name, passes = run_ocr(gray, binary, available_engines(engines))
    lines = tokens_to_lines(tokens)
    ext = FieldExtractor(known_vendors).extract(lines)
    if use_llm_fallback:
        llm_fill_missing(ext, "\n".join(l.text for l in lines))
    return _result_dict(lines, tokens, engine_name, passes, prep, ext, int((time.perf_counter() - t0) * 1000))


def extract_from_text(text: str, known_vendors: Optional[Sequence[str]] = None,
                      use_llm_fallback: bool = False) -> Dict[str, Any]:
    """Run only the NLP stage on plain text (every line treated as OCR confidence 1.0)."""
    lines = [OCRLine(t, 1.0, float(i)) for i, t in enumerate(text.splitlines()) if t.strip()]
    ext = FieldExtractor(known_vendors).extract(lines)
    if use_llm_fallback:
        llm_fill_missing(ext, text)
    return _result_dict(lines, [OCRToken(l.text, 1.0, (0, i * 10, 10, i * 10 + 8)) for i, l in enumerate(lines)],
                        "text", [], PreprocessReport(), ext, 0)


# ===========================================================================
# Self-test / CLI
# ===========================================================================
_SAMPLE_RECEIPT = """\
SHARMA OFFICE SOLUTIONS PVT LTD
Plot 14, Sector 62, Noida, Uttar Pradesh
GSTIN: {gstin}
TAX INVOICE
Invoice No: INV/2024-25/0412        Date: 14/08/2024
Bill To: Acme Analytics LLP
GSTIN: 29AAGCB7383J1Z4
1  A4 Paper 75gsm Ream    4816  10  300.00  3000.00
2  Laser Toner Cartridge  8443  2   2500.00  5000.00
Taxable Value 8000.00
CGST @ 9% 720.00
SGST @ 9% 720.00
Grand Total 9440.00
Payment Mode: Cash
"""


def selftest() -> None:
    # 1. checksum algorithm against well-known valid GSTINs
    for g in ("27AAPFU0939F1ZV", "29AAGCB7383J1Z4"):
        assert validate_gstin(g).is_valid, f"{g} should be valid"
    assert not validate_gstin("27AAPFU0939F1ZX").is_valid
    assert not validate_gstin("99ZZZZZ9999Z1Z1").is_valid
    assert not validate_gstin("27AAPFU0939F1Z").is_valid_format  # wrong length
    assert validate_gstin("27AAPFU0939F1ZV").state_name == "Maharashtra"
    gen = generate_valid_gstin("09", "AABCS1234K")
    assert validate_gstin(gen).is_valid and gen[:2] == "09"
    # 2. OCR-damage repair (O/0, S/5, B/8 swaps)
    assert repair_gstin("Z7AAPFUO939F1ZV") == ("27AAPFU0939F1ZV", "positional")
    assert repair_gstin("27AAPFU0939F1ZV") == ("27AAPFU0939F1ZV", "exact")
    fixed, _ = repair_gstin("27AAPFU0939F1ZX")  # genuinely wrong check char: must stay INVALID (never "fixed")
    assert not validate_gstin(fixed).is_valid
    # 3. end-to-end text extraction
    seller = generate_valid_gstin("09", "AABCS1234K")
    out = extract_from_text(_SAMPLE_RECEIPT.format(gstin=seller), known_vendors=["Sharma Office Solutions Pvt Ltd"])
    e = out["extracted"]
    assert e["vendor_gstin"] == seller and e["gstin_valid"] is True, e
    assert e["invoice_no"] == "INV/2024-25/0412", e
    assert e["invoice_date"] == "2024-08-14", e
    assert e["total_amount"] == 9440.0 and e["base_amount"] == 8000.0 and e["gst_amount"] == 1440.0, e
    assert e["gst_rate"] == 18.0 and e["payment_mode"] == "CASH", e
    assert e["buyer_gstin"] == "29AAGCB7383J1Z4", e
    assert e["vendor_name"] == "Sharma Office Solutions Pvt Ltd", e
    assert len(out["line_items"]) == 2, out["line_items"]
    print("selftest passed:")
    print(json.dumps(e, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description="FinAudit-AI receipt OCR + extraction")
    ap.add_argument("image", nargs="?")
    ap.add_argument("--debug-dir")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.selftest:
        selftest()
    elif args.image:
        out = process_receipt(args.image, debug_dir=args.debug_dir, use_llm_fallback=not args.no_llm)
        print(json.dumps(out, indent=2, default=str, ensure_ascii=False))
    else:
        ap.error("provide an image path or --selftest")


if __name__ == "__main__":
    main()
