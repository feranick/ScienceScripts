#!/usr/bin/env python3
"""
sem_edx_summary.py
==================

Automatically build a per-field SEM/EDX summary PDF from a set of exported images
(one SEM micrograph, one EDX spectrum, and one elemental map per element).

What it does automatically
--------------------------
* Discovers the SEM image, the spectrum, and the elemental maps in an input folder.
* Parses the element name and map resolution from the map filenames
  (e.g. "..._map_Titanium__480x258_points_.png" -> Titanium, 480x258).
* Detects the active map frame (from the union of all maps) and crops the maps to it.
* Computes per-element "lit-pixel coverage" and relative map brightness (Ti=100 if Ti
  present, otherwise normalised to the strongest element).
* Estimates a data-quality flag from map population and shows a coloured banner.
* Builds a peak-identification table from a built-in X-ray line reference and flags
  known line overlaps (e.g. Si Kalpha / W Malpha).
* Renders a clean multi-page PDF (parameters, SEM, spectrum, peak table, map grid,
  coverage table, standard caveats).

What it does NOT do
-------------------
* It does not write the analytical discussion / interpretation -- that is left to you.
  (Pass free-text notes via the params JSON "observations" key if you want a section.)
* It does not read the burned-in acquisition banner unless you enable optional OCR
  (--ocr, needs pytesseract + tesseract). Otherwise supply those values via --params.

Usage
-----
    python sem_edx_summary.py --input ./field_folder --output summary.pdf
    python sem_edx_summary.py --input ./field_folder --params params.json
    python sem_edx_summary.py --input ./field_folder --ocr        # best-effort banner OCR

params.json (all keys optional):
{
  "sample": "20260626_TiO2_run11_crystal",
  "field_label": "faceted-grain field",
  "magnification": "4700 x", "field_width": "147 um", "hv": "15 kV",
  "wd": "5.036 mm", "vacuum": "0.10 Pa (low-vacuum)", "scale_bar": "30 um",
  "detector": "Mixed SE/BSE (Mix 50%)", "date": "2026-06-26 16:16",
  "observations": ["Free-text bullet one.", "Free-text bullet two."]
}

Requires: numpy, pillow, reportlab. (pytesseract optional, for --ocr.)
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak, HRFlowable)
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --------------------------------------------------------------------------- #
# Reference data
# --------------------------------------------------------------------------- #

# Full element name (as used in export filenames) -> chemical symbol.
NAME_TO_SYMBOL = {
    "carbon": "C", "oxygen": "O", "nitrogen": "N", "sodium": "Na",
    "magnesium": "Mg", "aluminum": "Al", "aluminium": "Al", "silicon": "Si",
    "phosphorus": "P", "sulfur": "S", "sulphur": "S", "chlorine": "Cl",
    "potassium": "K", "calcium": "Ca", "titanium": "Ti", "vanadium": "V",
    "chromium": "Cr", "manganese": "Mn", "iron": "Fe", "cobalt": "Co",
    "nickel": "Ni", "copper": "Cu", "zinc": "Zn", "zirconium": "Zr",
    "niobium": "Nb", "molybdenum": "Mo", "tin": "Sn", "tungsten": "W",
    "platinum": "Pt", "gold": "Au", "lead": "Pb",
}

# Symbol -> (primary line label, primary energy keV, optional secondary text).
XRAY_LINES = {
    "C":  ("K\u03b1", 0.277, ""),
    "N":  ("K\u03b1", 0.392, ""),
    "O":  ("K\u03b1", 0.525, ""),
    "Na": ("K\u03b1", 1.041, ""),
    "Mg": ("K\u03b1", 1.254, ""),
    "Al": ("K\u03b1", 1.486, ""),
    "Si": ("K\u03b1", 1.740, ""),
    "P":  ("K\u03b1", 2.013, ""),
    "S":  ("K\u03b1", 2.307, ""),
    "Cl": ("K\u03b1", 2.622, ""),
    "K":  ("K\u03b1", 3.314, ""),
    "Ca": ("K\u03b1", 3.692, ""),
    "Ti": ("K\u03b1 / K\u03b2", 4.511, "K\u03b2 ~4.93"),
    "V":  ("K\u03b1", 4.952, ""),
    "Cr": ("K\u03b1", 5.415, ""),
    "Mn": ("K\u03b1", 5.899, ""),
    "Fe": ("K\u03b1", 6.404, ""),
    "Ni": ("K\u03b1", 7.478, ""),
    "Cu": ("K\u03b1", 8.048, ""),
    "Zn": ("K\u03b1", 8.639, ""),
    "Zr": ("L\u03b1", 2.042, ""),
    "Mo": ("L\u03b1", 2.293, ""),
    "Sn": ("L\u03b1", 3.444, ""),
    "W":  ("M\u03b1 / L\u03b1", 1.775, "L\u03b1 ~8.40"),
    "Pt": ("M\u03b1", 2.051, ""),
    "Au": ("M\u03b1 / L\u03b1", 2.123, "L\u03b1 ~9.71"),
    "Pb": ("M\u03b1", 2.342, ""),
}

# Known close line overlaps worth flagging (symbols, human note).
KNOWN_OVERLAPS = [
    ({"Si", "W"},  "Si K\u03b1 (1.74 keV) and W M\u03b1 (1.77 keV) overlap; W is best confirmed by its L line (~8.4 keV)."),
    ({"Au", "P"},  "Au M\u03b1 (2.12 keV) and P K\u03b1 (2.01 keV) sit close together; check for an Au coating."),
    ({"Ti", "N"},  "Ti L\u03b1 (~0.45 keV) and N K\u03b1 (0.39 keV) can overlap at low energy."),
    ({"Ti", "Ba"}, "Ti K\u03b1 (4.51 keV) and Ba L\u03b1 (4.47 keV) overlap."),
]

# Per-element colour for the little legend square next to each map title.
ELEMENT_SWATCH = {
    "Ti": "#b5651d", "O": "#1f8a70", "C": "#d4791f", "Si": "#cf7a2a",
    "W": "#ccb400", "Al": "#3aa0d0", "P": "#c9b03a", "K": "#caa83a",
    "Fe": "#b03a2e", "Ca": "#6a8f3a", "Na": "#8a6fb0",
}

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #

ACCENT = colors.HexColor("#1f6f8b")
DARK = colors.HexColor("#22303c")
LIGHT = colors.HexColor("#eef3f5")
GREY = colors.HexColor("#5a6b75")

DEJAVU_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/DejaVuSans.ttf",
]


def register_unicode_font():
    """Register a font that contains the micro sign; fall back to Helvetica."""
    for path in DEJAVU_CANDIDATES:
        if os.path.exists(path):
            pdfmetrics.registerFont(TTFont("DejaVu", path))
            return "DejaVu"
    return "Helvetica"


# --------------------------------------------------------------------------- #
# File discovery & parsing
# --------------------------------------------------------------------------- #

IMAGE_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")

# Lenient: just need "map" + an element-like token. Resolution is parsed separately.
MAP_RE = re.compile(r"map[_\-\s]*([A-Za-z]{1,14})", re.IGNORECASE)
RES_RE = re.compile(r"(\d+)\s*[x\u00d7]\s*(\d+)", re.IGNORECASE)


def _list_images(input_dir):
    """List image files in a folder, case-insensitive on extension."""
    if not os.path.isdir(input_dir):
        return []
    out = []
    for fn in sorted(os.listdir(input_dir)):
        if fn.startswith("."):
            continue
        if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
            out.append(os.path.join(input_dir, fn))
    return out


def discover_files(input_dir, verbose=False):
    """Return (sem_path, spectrum_path, [(symbol, name, w, h, path), ...], diagnostics)."""
    files = _list_images(input_dir)
    maps, sem, spectrum, diag = [], None, None, []
    for p in files:
        base = os.path.basename(p)
        low = base.lower()
        m = MAP_RE.search(base)
        # A map must contain "map<elem>" AND either a known element name or a WxH token,
        # so we don't misread an arbitrary image that merely contains the substring "map".
        res = RES_RE.search(base)
        is_known = m and m.group(1).lower() in NAME_TO_SYMBOL
        if m and (is_known or res):
            name = m.group(1)
            sym = NAME_TO_SYMBOL.get(name.lower(), name[:2].capitalize())
            w, h = (int(res.group(1)), int(res.group(2))) if res else (0, 0)
            maps.append((sym, name, w, h, p))
            diag.append((base, f"MAP -> {sym}" + (f" {w}x{h}" if w else " (no resolution in name)")))
        elif "spectrum" in low:
            spectrum = p
            diag.append((base, "SPECTRUM"))
        elif "map" not in low and "analysis" not in low:
            if sem is None:
                sem = p
                diag.append((base, "SEM micrograph"))
            else:
                diag.append((base, "ignored (extra non-map image)"))
        else:
            diag.append((base, "ignored (looks map-related but no element/resolution recognised)"))
    if verbose:
        for b, why in diag:
            print(f"  {b}  ->  {why}")
    return sem, spectrum, maps, diag


# --------------------------------------------------------------------------- #
# Image analysis
# --------------------------------------------------------------------------- #

def _lum(path):
    return np.asarray(PILImage.open(path).convert("RGB")).astype(float).sum(axis=2)


def detect_frame(map_paths):
    """Detect the active map rectangle from the union of all maps.

    Returns (left, top, right, bottom) suitable for PIL.crop.
    """
    union = None
    for p in map_paths:
        b = _lum(p)
        b[b.shape[0] - 70:, :] = 0          # mask the bottom info banner
        union = b if union is None else np.maximum(union, b)
    mask = union > 25
    rows = np.where(mask.sum(axis=1) > 4)[0]
    cols = np.where(mask.sum(axis=0) > 4)[0]
    if len(rows) == 0 or len(cols) == 0:    # fallback: whole image minus banner
        h, w = union.shape
        return (0, 0, w, h - 70)
    return (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)


def coverage_stats(maps, box):
    """Return list of dicts with coverage% and mean brightness per element."""
    left, top, right, bottom = box
    out = []
    for sym, name, w, h, path in maps:
        b = _lum(path)[top:bottom, left:right]
        out.append({
            "symbol": sym, "name": name,
            "res": f"{w}\u00d7{h}" if w else "\u2014",
            "lit": float((b > 45).mean() * 100.0),
            "mean": float(b.mean()),
        })
    return out


def data_quality(stats):
    """Crude data-quality flag from the strongest element's map population."""
    if not stats:
        return ("unknown", "")
    top = max(s["lit"] for s in stats)
    if top < 2:
        return ("very low",
                "Very low map population (&lt;2% lit pixels): the EDX is count-starved and "
                "results are noise-limited; treat as qualitative only and consider a longer-dwell re-acquisition.")
    if top < 12:
        return ("low / moderate",
                "Low-to-moderate map population: maps are interpretable but counts are well below a high-statistics "
                "acquisition; relative numbers are indicative.")
    return ("good", "")


# --------------------------------------------------------------------------- #
# Optional OCR of the burned-in banner
# --------------------------------------------------------------------------- #

def ocr_banner(sem_path):
    """Best-effort parse of the SEM banner. Returns a params dict (possibly empty)."""
    try:
        import pytesseract
    except Exception:
        sys.stderr.write("[ocr] pytesseract not available; skipping banner OCR.\n")
        return {}
    try:
        img = PILImage.open(sem_path).convert("RGB")
        w, h = img.size
        strip = img.crop((0, int(h * 0.93), w, h))          # bottom banner strip
        text = pytesseract.image_to_string(strip)
    except Exception as e:
        sys.stderr.write(f"[ocr] failed: {e}\n")
        return {}
    p = {}
    def grab(pattern, key):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            p[key] = m.group(1).strip()
    grab(r"([\d.]+\s*[x\u00d7])", "magnification")
    grab(r"([\d.]+\s*kV)", "hv")
    grab(r"([\d.]+\s*mm)", "wd")
    grab(r"([\d.]+\s*Pa)", "vacuum")
    grab(r"(\d{4}-\d{2}-\d{2}[\s\d:]*)", "date")
    grab(r"([0-9]{8}_\S+)", "sample")
    return p


# --------------------------------------------------------------------------- #
# PDF building
# --------------------------------------------------------------------------- #

def build_pdf(out_path, sem, spectrum, maps, stats, params, font):
    styles = getSampleStyleSheet()
    H1 = ParagraphStyle('H1', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=20, textColor=DARK, spaceAfter=2, leading=23)
    SUB = ParagraphStyle('SUB', parent=styles['Normal'], fontSize=10.5, textColor=GREY, spaceAfter=2)
    H2 = ParagraphStyle('H2', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=13, textColor=ACCENT, spaceBefore=10, spaceAfter=5)
    BODY = ParagraphStyle('BODY', parent=styles['Normal'], fontSize=9.7, textColor=DARK, leading=13.5, spaceAfter=5)
    CAP = ParagraphStyle('CAP', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8.3, textColor=GREY, leading=11, spaceBefore=3, alignment=TA_CENTER)
    BULLET = ParagraphStyle('BULLET', parent=BODY, leftIndent=12, spaceAfter=3)
    TINY = ParagraphStyle('TINY', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=7.8, textColor=GREY, leading=10)
    PC = ParagraphStyle('PC', parent=styles['Normal'], fontName=font, fontSize=8.5, textColor=DARK)

    usable = letter[0] - 2 * 0.7 * inch

    def fit(path, maxw):
        iw, ih = PILImage.open(path).size
        return Image(path, width=maxw, height=maxw * ih / iw)

    story = []

    # --- header ---
    sample = params.get("sample", "(sample)")
    field = params.get("field_label", "")
    date = params.get("date", "")
    line = f"Sample: <b>{sample}</b>"
    if field:
        line += f" &nbsp;|&nbsp; {field}"
    if date:
        line += f" &nbsp;|&nbsp; Acquired {date}"
    story.append(Paragraph("SEM / EDX Analysis Summary", H1))
    story.append(Paragraph("Scanning electron microscopy with energy-dispersive X-ray mapping", SUB))
    story.append(Paragraph(line, SUB))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT, spaceAfter=8))

    # --- data-quality banner (only if not "good") ---
    flag, note = data_quality(stats)
    if note:
        warn = colors.HexColor("#9c5a00")
        WARNP = ParagraphStyle('WARNP', parent=BODY, fontSize=9.2, textColor=warn, fontName='Helvetica-Bold', leading=12.5)
        banner = Table([[Paragraph(f"DATA QUALITY ({flag}) \u2014 {note}", WARNP)]], colWidths=[usable])
        banner.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fdf3e3")),
            ('BOX', (0, 0), (-1, -1), 0.75, warn),
            ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(banner)
        story.append(Spacer(1, 4))

    # --- parameters table ---
    res = stats[0]["res"] if stats else params.get("map_resolution", "\u2014")
    elems = ", ".join(s["symbol"] for s in stats) if stats else "\u2014"
    story.append(Paragraph("Acquisition Parameters", H2))
    rows = [
        ["Magnification", params.get("magnification", "\u2014"), "Accel. voltage (HV)", params.get("hv", "\u2014")],
        ["Field width (FW)", Paragraph(params.get("field_width", "\u2014"), PC), "Working distance (WD)", params.get("wd", "\u2014")],
        ["Scale bar", Paragraph(params.get("scale_bar", "\u2014"), PC), "Chamber vacuum", params.get("vacuum", "\u2014")],
        ["Detector", params.get("detector", "Mixed SE/BSE"), "Acquisition mode", "EDX map"],
        ["Map resolution", res + " points", "Mapped elements", elems],
    ]
    pt = Table(rows, colWidths=[1.35 * inch, 1.55 * inch, 1.55 * inch, 1.55 * inch])
    pt.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 8.8),
        ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 8.8), ('FONT', (2, 0), (2, -1), 'Helvetica-Bold', 8.8),
        ('TEXTCOLOR', (0, 0), (0, -1), GREY), ('TEXTCOLOR', (2, 0), (2, -1), GREY),
        ('TEXTCOLOR', (1, 0), (1, -1), DARK), ('TEXTCOLOR', (3, 0), (3, -1), DARK),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [LIGHT, colors.white]), ('GRID', (0, 0), (-1, -1), 0.5, colors.white),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(pt)

    # --- SEM ---
    if sem:
        story.append(Paragraph("Secondary / Backscattered Electron Micrograph", H2))
        story.append(fit(sem, usable))
        story.append(Paragraph("Figure 1. SEM micrograph; the marked rectangle (if present) is the EDX-mapped region.", CAP))

    # --- spectrum + peak table ---
    if spectrum:
        story.append(PageBreak())
        story.append(Paragraph("EDX Spectrum", H2))
        story.append(fit(spectrum, usable * 0.92))
        story.append(Paragraph("Figure 2. Area EDX spectrum (counts vs. X-ray energy, keV).", CAP))
        story.append(Spacer(1, 6))

    story.append(Paragraph("Peak Identification (reference lines)", H2))
    head = ["Element", "Line(s)", "Approx. energy", "Map coverage"]
    prows = [head]
    for s in sorted(stats, key=lambda x: -x["lit"]):
        sym = s["symbol"]
        line_lbl, energy, extra = XRAY_LINES.get(sym, ("\u2014", None, ""))
        ene = f"~{energy:.2f} keV" + (f"; {extra}" if extra else "") if energy else "\u2014"
        prows.append([f"{sym} ({s['name'].lower()})", line_lbl, ene, f"{s['lit']:.0f}%"])
    ptab = Table(prows, colWidths=[1.6 * inch, 1.1 * inch, 2.0 * inch, 1.0 * inch])
    ptab.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 8.6), ('FONT', (0, 1), (-1, -1), 'Helvetica', 8.6),
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]), ('TEXTCOLOR', (0, 1), (-1, -1), DARK),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d4dde1")),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(ptab)

    present = {s["symbol"] for s in stats}
    overlaps = [note for symset, note in KNOWN_OVERLAPS if symset <= present]
    if overlaps:
        story.append(Paragraph("Possible line overlaps: " + " ".join(overlaps), TINY))

    # --- map grid ---
    if maps:
        story.append(PageBreak())
        story.append(Paragraph("Elemental Distribution Maps", H2))
        mapw = (usable - 0.18 * inch) / 2

        def cell(s):
            col = ELEMENT_SWATCH.get(s["symbol"], "#7a7a7a")
            lab = Paragraph(f'<font color="{col}"><b>&#9632;</b></font> &nbsp;<b>{s["name"]} ({s["symbol"]})</b>',
                            ParagraphStyle('ml', parent=BODY, fontSize=9.5, spaceAfter=2))
            return [lab, fit(s["crop"], mapw)]

        ordered = sorted(stats, key=lambda x: -x["lit"])
        grid_rows = []
        for i in range(0, len(ordered), 2):
            left = cell(ordered[i])
            right = cell(ordered[i + 1]) if i + 1 < len(ordered) else [Paragraph("", BODY)]
            grid_rows.append([left, right])
        grid = Table(grid_rows, colWidths=[mapw + 8, mapw + 8])
        grid.setStyle(TableStyle([
            ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(grid)
        story.append(Paragraph("Figure 3. EDX maps, cropped to the mapped region and ordered by coverage.", CAP))

        # coverage table
        story.append(Paragraph("Map Signal Density (within mapped region)", H2))
        base = max((s["mean"] for s in stats), default=1.0) or 1.0
        ref = next((s["mean"] for s in stats if s["symbol"] == "Ti"), base)
        ref = ref or base
        ref_label = "Ti = 100" if any(s["symbol"] == "Ti" for s in stats) else "strongest = 100"
        crows = [["Element", "Lit-pixel coverage", f"Relative map brightness ({ref_label})"]]
        for s in ordered:
            crows.append([f"{s['name']} ({s['symbol']})", f"{s['lit']:.0f}%", f"{s['mean'] / ref * 100:.0f}"])
        ctab = Table(crows, colWidths=[1.7 * inch, 2.0 * inch, 2.6 * inch])
        ctab.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 8.8), ('FONT', (0, 1), (-1, -1), 'Helvetica', 8.8),
            ('BACKGROUND', (0, 0), (-1, 0), DARK), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]), ('TEXTCOLOR', (0, 1), (-1, -1), DARK),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d4dde1")),
            ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6), ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(ctab)
        story.append(Paragraph(
            "Coverage and brightness are image-pixel statistics indicating spatial coverage only. EDX maps are "
            "auto-contrast-stretched per element, so brightness is not directly comparable between elements for "
            "quantification \u2014 use the spectrum peak heights as the abundance guide.", TINY))

    # --- optional user observations (no auto-interpretation) ---
    obs = params.get("observations") or []
    if obs:
        story.append(Paragraph("Observations", H2))
        for t in obs:
            story.append(Paragraph("\u2022 " + t, BULLET))

    # --- footer ---
    story.append(Spacer(1, 3))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#c5d0d5"), spaceAfter=5))
    story.append(Paragraph(
        "Note: Auto-generated qualitative compilation of the supplied micrograph, spectrum, and maps. Peak energies "
        "are reference values; relative abundances are not quantified. Quantitative wt%/at% would require "
        "standards-based or standardless EDX quantification with matrix (ZAF) correction.", TINY))

    doc = SimpleDocTemplate(out_path, pagesize=letter, topMargin=0.55 * inch, bottomMargin=0.5 * inch,
                            leftMargin=0.7 * inch, rightMargin=0.7 * inch, title="SEM/EDX Summary")
    doc.build(story)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Auto-build a SEM/EDX summary PDF for one field.")
    ap.add_argument("--input", "-i", required=True, help="Folder with the SEM image, spectrum, and element maps.")
    ap.add_argument("--output", "-o", default=None, help="Output PDF path (default: <input>/SEM_EDX_summary.pdf).")
    ap.add_argument("--params", "-p", default=None, help="Optional JSON file with acquisition params and observations.")
    ap.add_argument("--ocr", action="store_true", help="Try OCR of the SEM banner (needs pytesseract + tesseract).")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print how each file in the folder was classified.")
    args = ap.parse_args()

    if not os.path.isdir(args.input):
        sys.exit(f"Input folder not found: {args.input}  (cwd: {os.getcwd()})")

    params = {}
    if args.params:
        with open(args.params) as fh:
            params = json.load(fh)

    sem, spectrum, maps, diag = discover_files(args.input, verbose=args.verbose)
    if not maps:
        all_files = sorted(os.listdir(args.input))
        msg = ["No elemental maps found in: " + os.path.abspath(args.input),
               "",
               "Expected map files named like:  ..._map_<Element>__480x264_points_.png",
               "(needs the substring 'map' + an element name, e.g. 'map_Titanium').",
               ""]
        if not all_files:
            msg.append("The folder is EMPTY. Check the path / that the files are really in here.")
        else:
            msg.append("Files actually in the folder (%d):" % len(all_files))
            for fn in all_files:
                tag = ""
                if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS and not fn.startswith("."):
                    tag = "   <- not a recognised image extension"
                msg.append("   " + fn + tag)
            if diag:
                msg.append("")
                msg.append("How each image was classified:")
                for b, why in diag:
                    msg.append(f"   {b}  ->  {why}")
        sys.exit("\n".join(msg))

    print(f"SEM: {os.path.basename(sem) if sem else '(none)'} | "
          f"spectrum: {os.path.basename(spectrum) if spectrum else '(none)'} | "
          f"maps: {', '.join(s for s, *_ in maps)}")

    if args.ocr and sem:
        ocr = ocr_banner(sem)
        for k, v in ocr.items():
            params.setdefault(k, v)

    box = detect_frame([m[4] for m in maps])
    print(f"Detected map frame (l,t,r,b): {box}")
    stats = coverage_stats(maps, box)

    # crop each map to the frame into a temp dir next to the output
    out_path = args.output or os.path.join(args.input, "SEM_EDX_summary.pdf")
    tmpdir = os.path.join(os.path.dirname(os.path.abspath(out_path)), "_map_crops")
    os.makedirs(tmpdir, exist_ok=True)
    for s, (sym, name, w, h, path) in zip(stats, maps):
        crop_path = os.path.join(tmpdir, f"crop_{sym}.png")
        PILImage.open(path).convert("RGB").crop(box).save(crop_path)
        s["crop"] = crop_path

    font = register_unicode_font()
    build_pdf(out_path, sem, spectrum, maps, stats, params, font)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
