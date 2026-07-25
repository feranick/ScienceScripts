import os
import io
import re
import json
import zipfile
import threading
import urllib.parse
import urllib.request
import webbrowser
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

# Embed Matplotlib into Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks

# HDF5 support (HORIBA LabSpec 6 .h5). Optional so the app still launches
# if h5py is not installed; the user is told how to add it on first use.
try:
    import h5py
    H5_AVAILABLE = True
except ImportError:
    H5_AVAILABLE = False

# ==========================================
# GLOBAL CONFIGURATIONS & CONSTANTS
# ==========================================
VERSION_TAG = "raman-v2026.07.25.1"

# RRUFF reference database (open Raman spectra of minerals).
# Data are distributed as per-quality zip archives of two-column .txt files.
RRUFF_BASE_URL = "https://rruff.info/zipped_data_files/raman/"
RRUFF_DATASETS = [
    "excellent_unoriented", "excellent_oriented",
    "fair_unoriented", "fair_oriented",
    "poor_unoriented", "poor_oriented",
    "ignore_unoriented",
]
RRUFF_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".raman_plotter_rruff")

# ROD -- Raman Open Database (https://solsa.crystallography.net/rod/).
# The Raman sibling of the COD, CC0, ~1100 entries. Two access modes:
#   * offline: one or more baked .h5 libraries from build_rod_library.py
#              (the browser build can only do this -- ROD's CORS header is
#              pinned to a single third-party origin)
#   * online:  the documented REST endpoints, queried on demand
ROD_BASE_URL = "https://solsa.crystallography.net/rod"
ROD_SEARCH_URL = ROD_BASE_URL + "/result"
ROD_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".raman_plotter_rod")
ROD_UA = ("Mozilla/5.0 (compatible; Raman-Plotter/1.0; "
          "+https://solsa.crystallography.net/rod/)")
COD_PAGE_URL = "https://www.crystallography.net/cod/{cid}.html"

# SDBS -- Spectral Database for Organic Compounds (AIST, Japan).
# ~34k organic compounds including laser Raman. Deliberately NOT automated:
# SDBS prohibits automated retrieval and rate-limits access, so the app only
# opens the search page in a browser and lets the user download by hand. The
# resulting JCAMP-DX file imports through the normal file open dialog.
SDBS_SEARCH_URL = "https://sdbs.db.aist.go.jp/sdbs/cgi-bin/cre_index.cgi"
SDBS_HOME_URL = "https://sdbs.db.aist.go.jp/"


# ==========================================
# MATHEMATICAL FUNCTION CODES
# ==========================================

def gaussian_profile(x, amp, cent, wid):
    """Single Gaussian band: amp * exp(-((x-cent)/wid)^2)."""
    return amp * np.exp(-((x - cent) / wid) ** 2)

def multi_gaussian_composite(x, *params):
    """Sum of several Gaussian bands (params flattened as amp, cent, wid triplets)."""
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        amp = params[i]
        cent = params[i+1]
        wid = params[i+2]
        y += gaussian_profile(x, amp, cent, wid)
    return y

def detect_reference_peaks(x, y, max_peaks=40, min_prominence=0.04):
    """Detects the most prominent band positions (cm-1) in a reference spectrum.
    Intensities are min-max normalized so the prominence threshold is scale-free."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 5:
        return np.array([])
    rng = np.ptp(y)
    if rng <= 0:
        return np.array([])
    yn = (y - y.min()) / rng
    peaks, props = find_peaks(yn, prominence=min_prominence, distance=3)
    if len(peaks) == 0:
        return np.array([])
    proms = props.get('prominences', np.ones(len(peaks)))
    order = np.argsort(proms)[::-1][:max_peaks]
    return np.sort(x[peaks[order]])


def peak_match_score(reference_peaks, experimental_peaks, tolerance):
    """Figure-of-merit for how well a reference's peaks explain the marked peaks.
    Returns (score_percent, average_closeness, matched_count)."""
    ref = np.asarray(reference_peaks, dtype=float)
    exp = list(experimental_peaks)
    if not exp or ref.size == 0:
        return 0.0, float(tolerance), 0
    matched = 0
    cumulative = 0.0
    for ex in exp:
        delta = np.min(np.abs(ref - ex))
        if delta <= tolerance:
            matched += 1
            cumulative += delta
        else:
            cumulative += tolerance  # out-of-tolerance penalty
    score = (matched / len(exp)) * 100.0
    avg = cumulative / len(exp)
    return score, avg, matched


def snip_background(y, iterations=40):
    """Estimates a smooth baseline (fluorescence/background) using the SNIP algorithm."""
    bg = np.array(y, dtype=float)
    n = len(bg)
    max_iter = min(iterations, int(n / 2) - 1)
    if max_iter < 1:
        return np.zeros_like(bg)
    for p in range(1, max_iter + 1):
        temp = np.copy(bg)
        bg[p:-p] = np.minimum(bg[p:-p], (temp[:-2*p] + temp[2*p:]) / 2.0)
    return bg


# ==========================================
# PARSING ENGINE CORE LOGIC
# ==========================================

def _decode(val):
    """Normalizes an HDF5 attribute (bytes / numpy bytes / array) to a clean str."""
    if isinstance(val, bytes):
        return val.decode('latin1').rstrip('\x00').strip()
    if isinstance(val, np.ndarray):
        return _decode(val.tolist())
    if isinstance(val, (list, tuple)) and val:
        return _decode(val[0])
    return str(val).strip()


def load_raman_data(file_path):
    """
    Parses a Raman spectrum file and returns a list of spectra, each as a dict
    {'x': wavenumber array, 'y': intensity array, 'label': str}.

    Supported LabSpec 6 exports (and generic text):
      * .h5           -> HORIBA LabSpec 6 HDF5 (all 1-D 'Spectrum' datasets)
      * .xml          -> HORIBA LabSpec 6 "LSX" XML export (single or multi-row)
      * .txt / .csv / .dat / .asc -> two-column (Raman shift, Intensity),
                         including RRUFF reference files with '##' headers
      * .jdx / .dx    -> JCAMP-DX, the interchange format used by ROD, SDBS
                         and most instrument vendors for exported spectra
    """
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    base_id = os.path.splitext(filename)[0]

    if ext in ('.h5', '.hdf5'):
        return _load_labspec_h5(file_path, base_id)
    elif ext == '.xml':
        return _load_labspec_xml(file_path, base_id)
    elif ext in ('.jdx', '.dx', '.jcm'):
        return _load_jcamp(file_path, base_id)
    elif ext in ('.txt', '.csv', '.dat', '.asc', '.spc'):
        return _load_two_column(file_path, base_id)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def _load_jcamp(file_path, base_id):
    """Loads a JCAMP-DX spectrum exported from SDBS, ROD, or an instrument.

    The label prefers the file's own ##TITLE over the filename, since SDBS
    exports are named by record number. A non-Raman ##DATA TYPE (an FT-IR
    trace, say) is kept but flagged in the label -- the axes are still
    wavenumbers, and users do overlay IR deliberately.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    x, y, header = parse_jcamp_dx(text)

    title = (header.get("TITLE") or "").strip()
    label = title or base_id
    dtype = (header.get("DATA TYPE") or "").strip().upper()
    if dtype and "RAMAN" not in dtype:
        label = f"{label} [{dtype.title()}]"
    return [{'x': x, 'y': y, 'label': label}]


def _load_labspec_xml(file_path, base_id):
    """
    Parses a HORIBA LabSpec 6 'LSX' XML export.
      * X axis (Raman shift) = the Format="6" numeric array inside <LSX_Tree>.
      * Y intensity          = each <LSX_Row> inside <LSX_Matrix> (one row per spectrum).
    """
    try:
        root = ET.parse(file_path).getroot()
    except ET.ParseError as e:
        raise ValueError(f"Malformed XML: {e}")

    # Title: first Format="8" string that is not a file path; else the filename.
    title = None
    for el in root.iter('LSX'):
        if el.attrib.get('Format') == '8' and el.text:
            v = el.text.strip()
            if ':\\' not in v and '/' not in v:
                title = v
                break
    if not title:
        title = base_id

    # X axis: first Format="6" numeric array (lives in the <LSX_Tree> metadata).
    x_axis = None
    for el in root.iter('LSX'):
        if el.attrib.get('Format') == '6' and el.text and el.text.strip():
            try:
                x_axis = np.array(el.text.split(), dtype=float)
            except ValueError:
                x_axis = None
            break

    # Y rows: every <LSX_Row> under <LSX_Matrix>.
    rows = []
    for mat in root.iter('LSX_Matrix'):
        for row in mat.iter('LSX_Row'):
            if row.text and row.text.strip():
                try:
                    rows.append(np.array(row.text.split(), dtype=float))
                except ValueError:
                    continue

    if not rows:
        raise ValueError("No spectral data (LSX_Matrix rows) found in this XML.")

    spectra = []
    for i, y in enumerate(rows):
        x = x_axis
        if x is None or len(x) != len(y):
            x = np.arange(len(y), dtype=float)
        label = title if len(rows) == 1 else f"{title} [{i}]"
        spectra.append({'x': x, 'y': y, 'label': label})
    return spectra


def _load_labspec_h5(file_path, base_id):
    if not H5_AVAILABLE:
        raise ImportError("Reading .h5 files requires the 'h5py' package.\n"
                          "Install it with:  pip install h5py")
    spectra = []
    with h5py.File(file_path, 'r') as f:
        if 'Datas' not in f:
            raise ValueError("Not a recognized LabSpec .h5 file ('Datas' group missing).")
        grp = f['Datas']
        # Keep natural ordering Data1, Data2, ... rather than lexical Data1, Data10, ...
        def sort_key(name):
            digits = ''.join(c for c in name if c.isdigit())
            return int(digits) if digits else 0
        data_names = sorted(
            [k for k in grp.keys() if k.startswith('Data')
             and not k.startswith('DataInfo')],
            key=sort_key
        )
        used_labels = {}
        for name in data_names:
            ds = grp[name]
            attrs = ds.attrs
            dtype = _decode(attrs.get('DataType', b''))
            # Only 1-D spectra; skip the optical image (DataType='Video', 3-D)
            if dtype != 'Spectrum' or ds.ndim != 1:
                continue
            if 'Axis1' not in attrs:
                continue
            x = np.asarray(attrs['Axis1'], dtype=float)
            y = np.asarray(ds[:], dtype=float)
            if x.shape[0] != y.shape[0]:
                continue
            title = _decode(attrs.get('Title', name)) or name
            # De-duplicate identical titles (LabSpec allows repeats)
            if title in used_labels:
                used_labels[title] += 1
                label = f"{title} ({used_labels[title]})"
            else:
                used_labels[title] = 1
                label = title
            spectra.append({'x': x, 'y': y, 'label': label})
    if not spectra:
        raise ValueError("No 1-D Raman spectra found inside this .h5 file.")
    return spectra


def _load_two_column(file_path, base_id):
    """Parses a two-column (Raman shift, Intensity) text file.

    Handles plain LabSpec .txt exports and RRUFF reference files, whose
    metadata lines start with '##' (e.g. ##NAMES=Quartz, ##RRUFFID=R040031).
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    x, y, label = _parse_two_column_text(text, base_id)
    if len(x) == 0:
        raise ValueError("Could not parse two numeric columns (Raman shift, Intensity).")
    return [{'x': x, 'y': y, 'label': label}]


def _parse_two_column_text(text, base_id):
    """Shared parser for two-column / RRUFF text. Returns (x, y, label)."""
    x_list, y_list = [], []
    rruff_name, rruff_id = None, None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('##') or line.startswith('#'):
            m = re.match(r'##?\s*([A-Za-z ]+)\s*=\s*(.*)', line)
            if m:
                key = m.group(1).strip().upper(); val = m.group(2).strip()
                if key == 'NAMES' and val:
                    rruff_name = val
                elif key == 'RRUFFID' and val:
                    rruff_id = val
            continue
        for sep in (',', '\t', ';'):
            if sep in line:
                parts = line.split(sep)
                break
        else:
            parts = line.split()
        if len(parts) < 2:
            continue
        try:
            xv = float(parts[0]); yv = float(parts[1])
        except ValueError:
            continue  # stray header/comment row
        x_list.append(xv); y_list.append(yv)

    if rruff_name:
        label = f"RRUFF: {rruff_name}" + (f" ({rruff_id})" if rruff_id else "")
    else:
        label = base_id
    return np.array(x_list), np.array(y_list), label


# ==========================================
# RRUFF REFERENCE DATABASE (download / cache / search)
# ==========================================

def rruff_dataset_dir(dataset):
    return os.path.join(RRUFF_CACHE_DIR, dataset)


def rruff_is_cached(dataset):
    d = rruff_dataset_dir(dataset)
    return os.path.isdir(d) and any(fn.lower().endswith('.txt') for fn in os.listdir(d))


def rruff_download_dataset(dataset, progress_cb=None):
    """Downloads and extracts a RRUFF raman zip archive into the local cache.
    Returns the number of .txt spectra extracted. Network access required."""
    if dataset not in RRUFF_DATASETS:
        raise ValueError(f"Unknown RRUFF dataset '{dataset}'.")
    os.makedirs(RRUFF_CACHE_DIR, exist_ok=True)
    dest = rruff_dataset_dir(dataset)
    os.makedirs(dest, exist_ok=True)
    url = f"{RRUFF_BASE_URL}{dataset}.zip"
    if progress_cb:
        progress_cb(f"Downloading {dataset}.zip ...")
    req = urllib.request.Request(url, headers={"User-Agent": "raman-plotter/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        blob = resp.read()
    if progress_cb:
        progress_cb("Extracting archive ...")
    count = 0
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for member in zf.namelist():
            if not member.lower().endswith('.txt'):
                continue
            out_name = os.path.join(dest, os.path.basename(member))
            with zf.open(member) as src, open(out_name, 'wb') as out:
                out.write(src.read())
            count += 1
    return count


def rruff_search_cached(dataset, query):
    """Searches cached RRUFF .txt files for minerals whose name/ID/filename
    matches the query. Returns a list of dicts: {name, id, path}."""
    d = rruff_dataset_dir(dataset)
    results = []
    if not os.path.isdir(d):
        return results
    q = query.strip().lower()
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith('.txt'):
            continue
        path = os.path.join(d, fn)
        # RRUFF filenames look like: Quartz__R040031__Raman__..._532.txt
        parts = fn.split('__')
        name = parts[0] if parts else fn
        rid = parts[1] if len(parts) > 1 else ''
        hay = f"{name} {rid} {fn}".lower()
        if not q or q in hay:
            results.append({'name': name, 'id': rid, 'path': path})
    return results


def load_rruff_h5_library(path):
    """Reads a consolidated RRUFF library .h5 (built by build_rruff_library.py).
    Returns {'path', 'entries': [{group, name, id, quality, peaks(np.array)}]}.
    Spectra x/y are read lazily from the file when a reference is overlaid."""
    if not H5_AVAILABLE:
        raise ImportError("Reading .h5 libraries requires 'h5py' (pip install h5py).")
    entries = []
    with h5py.File(path, 'r') as f:
        if 'spectra' not in f:
            raise ValueError("Not a RRUFF library file ('spectra' group missing).")
        sp = f['spectra']
        for gname in sp:
            g = sp[gname]
            a = g.attrs
            peaks = np.asarray(a['peaks'], dtype=float) if 'peaks' in a else np.array([])
            entries.append({
                'group': gname,
                'name': _decode(a.get('name', gname)),
                'id': _decode(a.get('rruff_id', '')),
                'quality': _decode(a.get('quality', '')),
                'peaks': peaks,
            })
    if not entries:
        raise ValueError("Library contains no spectra.")
    return {'path': path, 'entries': entries}


def rruff_url(name, rid):
    """Link to the RRUFF page for a sample (by ID) or mineral (by name)."""
    if rid and re.match(r'^R\d+', str(rid)):
        return f"https://rruff.info/{rid}"
    if name:
        return "https://rruff.info/" + str(name).strip().lower()
    return None


# ==========================================
# ROD -- RAMAN OPEN DATABASE (offline .h5 libraries + online REST)
# ==========================================

def rod_url(rod_id):
    """Information-card page for a ROD entry."""
    if rod_id in (None, "", "\\N"):
        return None
    return f"{ROD_BASE_URL}/{rod_id}.html"


def load_rod_h5_library(path):
    """Reads a baked reference library .h5 built by build_rod_library.py or
    build_openspecy_library.py.

    Both use the same schema as the RRUFF/COD libraries -- /spectra/<id> with
    x,y datasets and precomputed peaks, so matching is instant -- and are
    distinguished only by the `source` attribute, which drives labels and
    links. Peaks come from the dataset when present, else the attribute. x/y
    stay on disk and are read lazily when a reference is actually overlaid.

    Returns {'path', 'source', 'entries': [{group, name, id, source, cod_id,
             formula, mineral, collection, laser, url, cod_url, peaks}]}.
    """
    if not H5_AVAILABLE:
        raise ImportError("Reading .h5 libraries requires 'h5py' (pip install h5py).")
    entries = []
    with h5py.File(path, 'r') as f:
        if 'spectra' not in f:
            raise ValueError("Not a reference library file ('spectra' group missing).")
        file_source = _decode(f.attrs.get('source', '')) or 'ROD'
        sp = f['spectra']
        for gname in sp:
            g = sp[gname]
            a = g.attrs
            if 'peaks' in g:                       # dataset (preferred)
                peaks = np.asarray(g['peaks'][:], dtype=float)
            elif 'peaks' in a:                     # attribute (fallback)
                peaks = np.asarray(a['peaks'], dtype=float)
            else:
                peaks = np.array([])
            source = _decode(a.get('source', '')) or file_source
            rid = (_decode(a.get('rod_id', '')) or _decode(a.get('rruff_id', ''))
                   or gname)
            cid = _decode(a.get('cod_id', ''))
            url = _decode(a.get('url', ''))
            if not url and source.upper() == 'ROD':
                url = rod_url(rid)
            entries.append({
                'group': gname,
                'name': _decode(a.get('name', gname)),
                'id': rid,
                'source': source,
                'cod_id': cid,
                'formula': _decode(a.get('formula', '')),
                'mineral': _decode(a.get('mineral', '')),
                'collection': _decode(a.get('collection', '')),
                'laser': _decode(a.get('laser_nm', '')),
                'url': url,
                'cod_url': _decode(a.get('cod_url', '')) or (COD_PAGE_URL.format(cid=cid) if cid else ''),
                'peaks': peaks,
            })
    if not entries:
        raise ValueError("Library contains no spectra.")
    return {'path': path, 'source': file_source, 'entries': entries}


def _rod_http_get(url, timeout=45, retries=2):
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ROD_UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:                     # noqa: BLE001 - retry anything
            last = e
    raise last


def rod_search_params(query, field="auto"):
    """Builds ROD `result` query parameters for a user query.

    `field` selects how the query is interpreted:
      'text'      free text over names and bibliography
      'elements'  comma/space list -> el1..el8 (all must be present)
      'formula'   empirical formula in Hill notation, e.g. 'C8 H10 N4 O2'
      'id'        one or more ROD ids ('%' wildcard allowed)
      'auto'      ids if it looks numeric, otherwise text
    """
    q = str(query or "").strip()
    if not q:
        return {"id": "%"}
    if field == "auto":
        field = "id" if re.fullmatch(r"[\d,\s%]+", q) else "text"

    if field == "id":
        return {"id": re.sub(r"[\s,]+", ",", q)}
    if field == "elements":
        els = [t for t in re.split(r"[,\s]+", q) if t][:8]
        return {f"el{i}": el for i, el in enumerate(els, start=1)} or {"text": q}
    if field == "formula":
        return {"formula": q}
    return {"text": q}


def rod_search_online(query, field="auto", limit=300, timeout=45):
    """Queries the ROD REST search endpoint.

    Result formats are tried in decreasing order of parse-friendliness --
    `csv` and `json` carry names, `lst` is ids only.

    Returns a list of {'name','id','formula','url','online':True}.
    """
    params = rod_search_params(query, field)

    for fmt in ("csv", "json", "lst"):
        url = f"{ROD_SEARCH_URL}?{urllib.parse.urlencode(dict(params, format=fmt))}"
        try:
            text = _rod_http_get(url, timeout=timeout)
        except Exception:
            continue
        hits = _rod_parse_results(text, fmt)
        if hits:
            return hits[:limit]
    return []


def _rod_parse_results(text, fmt):
    """Turns a ROD search payload into hit dicts. Tolerant by design: the CSV
    column set has changed across ROD releases, so we locate columns by header
    name rather than position and fall back to bare id extraction."""
    hits = []
    if not text:
        return hits

    if fmt == "json":
        try:
            data = json.loads(text)
        except Exception:
            data = None
        rows = data if isinstance(data, list) else (data or {}).get("results", [])
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("file") or row.get("id") or row.get("rod_id") or "").strip()
            if not rid:
                continue
            hits.append({'name': (row.get("mineral") or row.get("chemname")
                                  or row.get("commonname") or row.get("formula")
                                  or f"ROD {rid}"),
                         'id': rid,
                         'formula': str(row.get("formula") or "").replace("-", " ").strip(),
                         'url': rod_url(rid), 'online': True})
        if hits:
            return hits

    if fmt == "csv":
        import csv as _csv
        try:
            rows = list(_csv.reader(io.StringIO(text)))
        except Exception:
            rows = []
        if rows:
            head = [h.strip().lower() for h in rows[0]]

            def col(*cands):
                for c in cands:
                    if c in head:
                        return head.index(c)
                return None

            i_id = col("file", "id", "rod_id", "codid")
            i_min = col("mineral", "mineral name", "chemname", "commonname")
            i_for = col("formula", "calcformula", "cellformula")
            if i_id is not None:
                for r in rows[1:]:
                    if len(r) <= i_id:
                        continue
                    rid = r[i_id].strip()
                    if not re.fullmatch(r"\d{6,9}", rid):
                        continue
                    name = (r[i_min].strip() if i_min is not None and len(r) > i_min else "")
                    # COD/ROD wrap formulae in dashes: "- Fe H2 Na O2 S2 -"
                    formula = (" ".join(r[i_for].replace("-", " ").split())
                               if i_for is not None and len(r) > i_for else "")
                    hits.append({'name': name or formula or f"ROD {rid}", 'id': rid,
                                 'formula': formula, 'url': rod_url(rid), 'online': True})
                if hits:
                    return hits

    seen = set()
    for m in re.finditer(r"\b(\d{6,9})\b", text):
        rid = m.group(1)
        if rid in seen:
            continue
        seen.add(rid)
        hits.append({'name': f"ROD {rid}", 'id': rid, 'formula': '',
                     'url': rod_url(rid), 'online': True})
    return hits


def rod_fetch_spectrum(rod_id, timeout=45, use_cache=True):
    """Downloads (and caches) one ROD spectrum. Returns (x, y, meta).

    The spectrum comes from the JCAMP-DX serialization, which is compact and
    unambiguous; the CIF is fetched alongside only for the display name."""
    os.makedirs(ROD_CACHE_DIR, exist_ok=True)
    jdx_path = os.path.join(ROD_CACHE_DIR, f"{rod_id}.jdx")

    text = None
    if use_cache and os.path.exists(jdx_path) and os.path.getsize(jdx_path) > 0:
        with open(jdx_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    if text is None:
        text = _rod_http_get(f"{ROD_BASE_URL}/{rod_id}.jdx", timeout=timeout)
        try:
            with open(jdx_path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except Exception:
            pass

    x, y, header = parse_jcamp_dx(text)

    meta = {'id': str(rod_id), 'url': rod_url(rod_id),
            'name': header.get("TITLE", "") or f"ROD {rod_id}",
            'formula': header.get("MOLFORM", ""), 'cod_id': '', 'cod_url': ''}

    cif_path = os.path.join(ROD_CACHE_DIR, f"{rod_id}.rod")
    cif = None
    try:
        if use_cache and os.path.exists(cif_path) and os.path.getsize(cif_path) > 0:
            with open(cif_path, "r", encoding="utf-8", errors="replace") as fh:
                cif = fh.read()
        else:
            cif = _rod_http_get(f"{ROD_BASE_URL}/{rod_id}.rod", timeout=timeout)
            try:
                with open(cif_path, "w", encoding="utf-8") as fh:
                    fh.write(cif)
            except Exception:
                pass
    except Exception:
        cif = None

    if cif:
        mineral = _rod_cif_value(cif, "_chemical_name_mineral")
        formula = _rod_cif_value(cif, "_chemical_formula_sum")
        for m in re.finditer(r"^\s*(_\S*cod\S*)\s+(\d{7,8})\s*$", cif, re.M):
            meta['cod_id'] = m.group(2)
            meta['cod_url'] = COD_PAGE_URL.format(cid=m.group(2))
            break
        if mineral:
            meta['name'] = mineral
        if formula:
            meta['formula'] = formula.replace("-", " ").strip()

    return x, y, meta


def _rod_cif_value(cif_text, tag):
    """Single-line or `;`-delimited value for one CIF tag."""
    m = re.search(r"^\s*" + re.escape(tag) + r"\s+(.+?)\s*$", cif_text, re.M | re.I)
    if m:
        v = m.group(1).strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        return "" if v in ("?", ".") else v
    m = re.search(r"^\s*" + re.escape(tag) + r"\s*\n;\s*(.*?)\n;", cif_text, re.M | re.S | re.I)
    if m:
        return " ".join(m.group(1).split())
    return ""


def parse_jcamp_dx(text):
    """Parses a JCAMP-DX spectrum. Returns (x, y, header).

    Handles the (XY..XY) point-pair form ROD emits and the equidistant
    (X++(Y..Y)) form some depositors use. Shared by the ROD online fetch and
    the generic .jdx file importer, so anything the user downloads by hand from
    another database (SDBS included) opens the same way.
    """
    header, data_lines = {}, []
    block, collecting = None, False
    for line in text.splitlines():
        m = re.match(r"^\s*##\s*([^=]+?)\s*=\s*(.*)$", line)
        if m:
            key, val = m.group(1).strip().upper(), m.group(2).strip()
            header[key] = val
            if key in ("XYPOINTS", "XYDATA", "PEAK TABLE", "DATA TABLE"):
                block, collecting = (key, val.upper()), True
            else:
                collecting = False       # trailing headers close the data run
            continue
        if collecting and line.strip():
            data_lines.append(line)

    if not data_lines or block is None:
        raise ValueError("no data block found in JCAMP-DX")

    num = r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?"
    kind, form = block
    if kind in ("XYPOINTS", "PEAK TABLE", "DATA TABLE") or "XY..XY" in form:
        nums = [float(t.replace("D", "E").replace("d", "e"))
                for t in re.findall(num, " ".join(data_lines))]
        if len(nums) < 4:
            raise ValueError("too few data points")
        if len(nums) % 2:
            nums = nums[:-1]
        arr = np.asarray(nums, dtype=float).reshape(-1, 2)
        x, y = arr[:, 0], arr[:, 1]
    else:
        ys = []
        for line in data_lines:
            toks = re.findall(num, line)
            if len(toks) > 1:
                ys.extend(float(t.replace("D", "E")) for t in toks[1:])
        if len(ys) < 4:
            raise ValueError("too few data points")
        y = np.asarray(ys, dtype=float)
        x = np.linspace(float(header.get("FIRSTX", 0.0)),
                        float(header.get("LASTX", len(y) - 1)), len(y))

    x = x * float(header.get("XFACTOR", 1.0) or 1.0)
    y = y * float(header.get("YFACTOR", 1.0) or 1.0)
    order = np.argsort(x)                          # some deposits run high->low
    x, y = x[order], y[order]
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 5:
        raise ValueError("fewer than 5 finite points")
    return x, y, header


# ==========================================
# GUI & EMBEDDED PLOTTING INTERFACE
# ==========================================

class RamanPlotterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Raman Spectra Analysis Toolkit")
        self.root.geometry("1240x900")
        self.root.minsize(1000, 640)

        style = ttk.Style()
        style.theme_use('clam')

        # In-memory session state
        self.active_datasets = {}
        self.peak_guesses = []
        self.guess_lines_artists = []
        self.fitted_curves_artists = []
        self.target_checkbox_vars = {}
        self.history_stack = []

        self.fitting_mode_active = False
        self.normalization_mode_active = False
        self.cursor_line = None

        # Interactive wheel-adjust (LabSpec-style offset / scale)
        self.adjust_mode = None          # None | 'offset' | 'scale'
        self.adjust_armed = False        # one history snapshot per wheel session
        self.line_map = {}               # dataset key -> Line2D (for fast updates)
        self._adjust_key_by_label = {}

        # --- Left Sidebar Panel Layout (scrollable so nothing is ever clipped) ---
        sidebar_container = ttk.Frame(root)
        sidebar_container.pack(side="left", fill="y", padx=5, pady=5)
        sidebar_canvas = tk.Canvas(sidebar_container, borderwidth=0, highlightthickness=0, width=300)
        sidebar_vscroll = ttk.Scrollbar(sidebar_container, orient="vertical", command=sidebar_canvas.yview)
        sidebar_canvas.configure(yscrollcommand=sidebar_vscroll.set)
        sidebar_vscroll.pack(side="right", fill="y")
        sidebar_canvas.pack(side="left", fill="both", expand=True)

        sidebar_frame = ttk.Frame(sidebar_canvas, padding=12)
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar_frame, anchor="nw")
        sidebar_frame.bind("<Configure>", lambda e: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")))
        sidebar_canvas.bind("<Configure>", lambda e: sidebar_canvas.itemconfig(sidebar_window, width=e.width))

        def _sidebar_wheel(event):
            if event.num == 5 or event.delta < 0:
                sidebar_canvas.yview_scroll(1, "units")
            elif event.num == 4 or event.delta > 0:
                sidebar_canvas.yview_scroll(-1, "units")

        # Only capture the wheel while the pointer is over the sidebar, so it
        # doesn't interfere with the plot's own wheel-adjust behaviour.
        sidebar_canvas.bind("<Enter>", lambda e: (
            sidebar_canvas.bind_all("<MouseWheel>", _sidebar_wheel),
            sidebar_canvas.bind_all("<Button-4>", _sidebar_wheel),
            sidebar_canvas.bind_all("<Button-5>", _sidebar_wheel)))
        sidebar_canvas.bind("<Leave>", lambda e: (
            sidebar_canvas.unbind_all("<MouseWheel>"),
            sidebar_canvas.unbind_all("<Button-4>"),
            sidebar_canvas.unbind_all("<Button-5>")))

        ttk.Label(sidebar_frame, text="🔬 Raman Spectra Analyzer", font=("Helvetica", 12, "bold")).pack(side="top", anchor="w", pady=(0, 10))

        ttk.Button(sidebar_frame, text="📁 Select File(s)", command=self.select_and_plot_files).pack(side="top", fill="x", pady=3)
        ttk.Button(sidebar_frame, text="✂️ Crop to View", command=self.crop_to_current_view).pack(side="top", fill="x", pady=3)
        ttk.Button(sidebar_frame, text="✨ Subtract Baseline", command=self.subtract_background_profile).pack(side="top", fill="x", pady=3)
        ttk.Button(sidebar_frame, text="🧪 Subtract Reference Scan", command=self.open_blank_subtraction_dialog).pack(side="top", fill="x", pady=3)

        # Savitzky-Golay smoothing row
        smooth_row = ttk.Frame(sidebar_frame)
        smooth_row.pack(side="top", fill="x", pady=3)
        ttk.Button(smooth_row, text="🍃 Smooth Noise", command=self.smooth_active_profiles).pack(side="left", fill="x", expand=True)
        ttk.Label(smooth_row, text="Win:", font=("Helvetica", 9)).pack(side="left", padx=(4, 1))
        self.ent_smooth_win = ttk.Entry(smooth_row, width=4)
        self.ent_smooth_win.insert(0, "11")
        self.ent_smooth_win.pack(side="left", padx=1)

        # Wavenumber (x) calibration shift row
        shift_row = ttk.Frame(sidebar_frame)
        shift_row.pack(side="top", fill="x", pady=3)
        ttk.Button(shift_row, text="📐 Shift Raman shift", command=self.apply_shift).pack(side="left", fill="x", expand=True)
        ttk.Label(shift_row, text="Δ:", font=("Helvetica", 9)).pack(side="left", padx=(4, 1))
        self.ent_shift_val = ttk.Entry(shift_row, width=5)
        self.ent_shift_val.insert(0, "0.0")
        self.ent_shift_val.pack(side="left", padx=1)
        ttk.Label(shift_row, text="cm⁻¹", font=("Helvetica", 9)).pack(side="left", padx=(1, 2))

        # Normalization row
        norm_row = ttk.Frame(sidebar_frame)
        norm_row.pack(side="top", fill="x", pady=3)
        self.btn_normalize_toggle = ttk.Button(norm_row, text="⚖️ Normalize to Peak", command=self.toggle_normalization_mode)
        self.btn_normalize_toggle.pack(side="left", fill="x", expand=True)
        ttk.Label(norm_row, text="±", font=("Helvetica", 10)).pack(side="left", padx=(5, 1))
        self.ent_norm_span = ttk.Entry(norm_row, width=5)
        self.ent_norm_span.insert(0, "10")
        self.ent_norm_span.pack(side="left", padx=1)
        ttk.Label(norm_row, text="cm⁻¹", font=("Helvetica", 10)).pack(side="left", padx=(1, 2))

        # Interactive wheel-adjust panel (LabSpec-style add / multiply)
        adjust_frame = ttk.LabelFrame(sidebar_frame, text=" 🎚️ Interactive Adjust (mouse wheel) ", padding=(8, 6))
        adjust_frame.pack(side="top", fill="x", pady=4)
        t_row = ttk.Frame(adjust_frame)
        t_row.pack(fill="x")
        ttk.Label(t_row, text="Target:", font=("Helvetica", 8, "bold")).pack(side="left")
        self.combo_adjust_target = ttk.Combobox(t_row, state="readonly", width=16, values=[])
        self.combo_adjust_target.pack(side="left", fill="x", expand=True, padx=(3, 0))
        m_row = ttk.Frame(adjust_frame)
        m_row.pack(fill="x", pady=(4, 0))
        self.btn_adjust_offset = ttk.Button(m_row, text="➕ Offset", command=lambda: self.set_adjust_mode('offset'))
        self.btn_adjust_offset.pack(side="left", fill="x", expand=True)
        self.btn_adjust_scale = ttk.Button(m_row, text="✖ Scale", command=lambda: self.set_adjust_mode('scale'))
        self.btn_adjust_scale.pack(side="left", fill="x", expand=True, padx=(4, 0))
        s_row = ttk.Frame(adjust_frame)
        s_row.pack(fill="x", pady=(4, 0))
        ttk.Label(s_row, text="Δ+:", font=("Helvetica", 8)).pack(side="left")
        self.ent_offset_step = ttk.Entry(s_row, width=7)
        self.ent_offset_step.insert(0, "100")
        self.ent_offset_step.pack(side="left", padx=(1, 8))
        ttk.Label(s_row, text="×%:", font=("Helvetica", 8)).pack(side="left")
        self.ent_scale_step = ttk.Entry(s_row, width=5)
        self.ent_scale_step.insert(0, "5")
        self.ent_scale_step.pack(side="left", padx=1)
        # Step buttons: click to nudge (no scroll wheel required, e.g. on a MacBook)
        step_btn_row = ttk.Frame(adjust_frame)
        step_btn_row.pack(fill="x", pady=(4, 0))
        ttk.Button(step_btn_row, text="➖ Down", command=lambda: self.adjust_step_button(-1)).pack(side="left", fill="x", expand=True)
        ttk.Button(step_btn_row, text="Up ➕", command=lambda: self.adjust_step_button(1)).pack(side="left", fill="x", expand=True, padx=(4, 0))
        ttk.Label(adjust_frame, text="Pick a target and mode, then use ➖ / ➕ (or scroll the wheel / two-finger scroll over the plot).",
                  font=("Helvetica", 8), foreground="#555555", wraplength=250).pack(anchor="w", pady=(3, 0))

        self.btn_fit_toggle = ttk.Button(sidebar_frame, text="🎯 Peak Selection: OFF", command=self.toggle_fitting_mode)
        self.btn_fit_toggle.pack(side="top", fill="x", pady=3)

        self.btn_run_fit = ttk.Button(sidebar_frame, text="⚡ Fit", command=self.run_peak_optimization, state="disabled")
        self.btn_run_fit.pack(side="top", fill="x", pady=3)

        ttk.Button(sidebar_frame, text="📥 Export to CSV", command=self.export_active_data_to_csv).pack(side="top", fill="x", pady=3)
        ttk.Button(sidebar_frame, text="🗑️ Clear Canvas", command=self.clear_canvas).pack(side="top", fill="x", pady=3)
        self.btn_undo = ttk.Button(sidebar_frame, text="↩️ Undo Last Action", command=self.undo_last_action, state="disabled")
        self.btn_undo.pack(side="top", fill="x", pady=3)

        ttk.Separator(sidebar_frame, orient="horizontal").pack(side="top", fill="x", pady=10)

        # Status badge
        self.status_var = tk.StringVar(value="Active spectra loaded: 0")
        lbl_status = ttk.Label(sidebar_frame, textvariable=self.status_var, font=("Helvetica", 9, "bold"), background="#cff4fc", foreground="#055160", relief="solid", borderwidth=1, padding=6, anchor="center")
        lbl_status.pack(side="top", fill="x", pady=2)

        # --- RRUFF Reference Database Panel ---
        panel_rruff = ttk.LabelFrame(sidebar_frame, text=" 🌐 RRUFF Reference Database ", padding=(8, 6))
        panel_rruff.pack(side="top", fill="x", pady=5)

        ds_row = ttk.Frame(panel_rruff)
        ds_row.pack(fill="x", pady=(0, 4))
        ttk.Label(ds_row, text="Set:", font=("Helvetica", 8, "bold")).pack(side="left")
        self.combo_rruff_dataset = ttk.Combobox(ds_row, state="readonly", width=18, values=RRUFF_DATASETS)
        self.combo_rruff_dataset.current(0)
        self.combo_rruff_dataset.pack(side="left", fill="x", expand=True, padx=(3, 0))

        self.btn_rruff_download = ttk.Button(panel_rruff, text="⬇️ Download / Update Set", command=self.rruff_download_selected)
        self.btn_rruff_download.pack(fill="x", pady=2)
        ttk.Button(panel_rruff, text="📂 Use Local RRUFF Folder", command=self.rruff_pick_local_folder).pack(fill="x", pady=2)
        ttk.Button(panel_rruff, text="📚 Open RRUFF .h5 Library", command=self.rruff_open_library).pack(fill="x", pady=2)

        ttk.Label(panel_rruff, text="Search mineral / ID:", font=("Helvetica", 8, "bold")).pack(anchor="w", pady=(4, 0))
        search_row = ttk.Frame(panel_rruff)
        search_row.pack(fill="x", pady=(0, 4))
        self.ent_rruff_query = ttk.Entry(search_row)
        self.ent_rruff_query.pack(side="left", fill="x", expand=True)
        self.ent_rruff_query.bind("<Return>", lambda e: self.rruff_run_search())
        ttk.Button(search_row, text="🔍", width=3, command=self.rruff_run_search).pack(side="right", padx=(3, 0))

        self.rruff_results_list = tk.Listbox(panel_rruff, height=4, exportselection=False)
        self.rruff_results_list.pack(fill="x", pady=(0, 3))
        self.rruff_results_list.bind("<Double-1>", self._rruff_open_selected_page)
        self.rruff_search_hits = []

        ttk.Button(panel_rruff, text="➕ Overlay Selected Reference", command=self.rruff_overlay_selected).pack(fill="x", pady=(0, 2))

        ttk.Separator(panel_rruff, orient="horizontal").pack(fill="x", pady=4)
        tol_row = ttk.Frame(panel_rruff)
        tol_row.pack(fill="x", pady=(0, 2))
        ttk.Label(tol_row, text="Match tol. ±", font=("Helvetica", 8, "bold")).pack(side="left")
        self.ent_match_tol = ttk.Entry(tol_row, width=5)
        self.ent_match_tol.insert(0, "12")
        self.ent_match_tol.pack(side="left", padx=(2, 1))
        ttk.Label(tol_row, text="cm⁻¹", font=("Helvetica", 8)).pack(side="left")
        self.btn_rruff_match = ttk.Button(panel_rruff, text="🎯 Match by Selected Peaks", command=self.rruff_match_by_peaks)
        self.btn_rruff_match.pack(fill="x", pady=(0, 2))

        self.rruff_status_var = tk.StringVar(value="RRUFF: no set cached.")
        ttk.Label(panel_rruff, textvariable=self.rruff_status_var, font=("Helvetica", 8), foreground="#555555", wraplength=250).pack(anchor="w")

        self.rruff_local_dir = None
        self.rruff_lib = None
        self._refresh_rruff_status()

        # --- ROD Reference Database Panel (offline .h5 libraries + online REST) ---
        panel_rod = ttk.LabelFrame(sidebar_frame,
                                   text=" 🔬 Reference Libraries (ROD / Open Specy) ",
                                   padding=(8, 6))
        panel_rod.pack(side="top", fill="x", pady=5)

        self.rod_libs = []                 # [{name, path, entries, active, var}]
        self.rod_search_hits = []
        self.rod_cfg_path = os.path.join(os.path.expanduser("~"), ".raman_plotter_rod_libraries.json")
        self.rod_mode_var = tk.StringVar(value="h5")

        mode_row = ttk.Frame(panel_rod)
        mode_row.pack(fill="x", pady=(0, 3))
        ttk.Label(mode_row, text="Source:", font=("Helvetica", 8, "bold")).pack(side="left")
        ttk.Radiobutton(mode_row, text="Offline .h5", value="h5", variable=self.rod_mode_var,
                        command=self._rod_update_status).pack(side="left", padx=(4, 0))
        ttk.Radiobutton(mode_row, text="Online (ROD)", value="online", variable=self.rod_mode_var,
                        command=self._rod_update_status).pack(side="left", padx=(4, 0))

        self.rod_libs_frame = ttk.Frame(panel_rod)
        self.rod_libs_frame.pack(fill="x", pady=(0, 2))
        ttk.Button(panel_rod, text="📚 Add .h5 Librar(ies)…",
                   command=self.rod_add_library).pack(fill="x", pady=2)

        ttk.Label(panel_rod, text="Search name / formula / ID:",
                  font=("Helvetica", 8, "bold")).pack(anchor="w", pady=(4, 0))
        rod_search_row = ttk.Frame(panel_rod)
        rod_search_row.pack(fill="x", pady=(0, 3))
        self.ent_rod_query = ttk.Entry(rod_search_row)
        self.ent_rod_query.pack(side="left", fill="x", expand=True)
        self.ent_rod_query.bind("<Return>", lambda e: self.rod_run_search())
        ttk.Button(rod_search_row, text="🔍", width=3, command=self.rod_run_search).pack(side="right", padx=(3, 0))

        # How the query is interpreted. Offline libraries always substring-match;
        # online this picks which ROD search key is used (text / el1..el8 /
        # formula / id), which is what makes a bounded online match possible.
        rod_field_row = ttk.Frame(panel_rod)
        rod_field_row.pack(fill="x", pady=(0, 3))
        ttk.Label(rod_field_row, text="as", font=("Helvetica", 8)).pack(side="left")
        self.combo_rod_field = ttk.Combobox(rod_field_row, state="readonly", width=9,
                                            values=("auto", "text", "elements", "formula", "id"))
        self.combo_rod_field.current(0)
        self.combo_rod_field.pack(side="left", padx=(3, 0))
        ttk.Label(rod_field_row, text="max online:", font=("Helvetica", 8)).pack(side="left", padx=(8, 0))
        self.ent_rod_scan_cap = ttk.Entry(rod_field_row, width=5)
        self.ent_rod_scan_cap.insert(0, "100")
        self.ent_rod_scan_cap.pack(side="left", padx=(3, 0))

        self.rod_results_list = tk.Listbox(panel_rod, height=4, exportselection=False)
        self.rod_results_list.pack(fill="x", pady=(0, 3))
        self.rod_results_list.bind("<Double-1>", self._rod_open_selected_page)

        ttk.Button(panel_rod, text="➕ Overlay Selected Reference",
                   command=self.rod_overlay_selected).pack(fill="x", pady=(0, 2))

        ttk.Separator(panel_rod, orient="horizontal").pack(fill="x", pady=4)
        rod_tol_row = ttk.Frame(panel_rod)
        rod_tol_row.pack(fill="x")
        ttk.Label(rod_tol_row, text="Match tol (cm⁻¹):", font=("Helvetica", 8)).pack(side="left")
        self.ent_rod_match_tol = ttk.Entry(rod_tol_row, width=6)
        self.ent_rod_match_tol.insert(0, "12")
        self.ent_rod_match_tol.pack(side="left", padx=(4, 0))

        self.btn_rod_match = ttk.Button(panel_rod, text="🎯 Match by Selected Peaks (ROD)",
                                        command=self.rod_match_by_peaks)
        self.btn_rod_match.pack(fill="x", pady=(2, 2))

        self._rod_scan_cancel = False      # set by the Stop button mid online scan
        self.rod_status_var = tk.StringVar(value="ROD: add a baked .h5 library, or switch to Online.")
        ttk.Label(panel_rod, textvariable=self.rod_status_var, font=("Helvetica", 8),
                  foreground="#555555", wraplength=250).pack(anchor="w")

        self._rod_load_config()

        # --- SDBS (external lookup only) ---
        # SDBS prohibits automated retrieval, so there is no pull here by
        # design: the button opens their search page, you download a spectrum
        # by hand, and 📂 Load Spectra imports the .jdx like any other file.
        panel_sdbs = ttk.LabelFrame(sidebar_frame, text=" 🔗 SDBS (organic compounds) ", padding=(8, 6))
        panel_sdbs.pack(side="top", fill="x", pady=5)
        ttk.Label(panel_sdbs,
                  text="Opens SDBS in your browser. Download a Raman spectrum "
                       "there, then load the .jdx normally.",
                  font=("Helvetica", 8), foreground="#555555", wraplength=250).pack(anchor="w")
        sdbs_row = ttk.Frame(panel_sdbs)
        sdbs_row.pack(fill="x", pady=(4, 0))
        self.ent_sdbs_query = ttk.Entry(sdbs_row)
        self.ent_sdbs_query.pack(side="left", fill="x", expand=True)
        self.ent_sdbs_query.bind("<Return>", lambda e: self.sdbs_lookup())
        ttk.Button(sdbs_row, text="🔗", width=3, command=self.sdbs_lookup).pack(side="right", padx=(3, 0))
        ttk.Button(panel_sdbs, text="🔗 Look up in SDBS",
                   command=self.sdbs_lookup).pack(fill="x", pady=(3, 0))

        # --- Active Layers Control Panel ---
        self.panel_fit_targets = ttk.LabelFrame(sidebar_frame, text=" 📋 Plotted Spectra Layers ", padding=(8, 6))
        self.panel_fit_targets.pack(side="top", fill="x", pady=8, expand=True)
        self.lbl_no_targets = ttk.Label(self.panel_fit_targets, text="No Spectra Loaded", font=("Helvetica", 9, "italic"), foreground="#888888")
        self.lbl_no_targets.pack(side="top", anchor="w", padx=4)

        lbl_version = ttk.Label(sidebar_frame, text=VERSION_TAG, font=("Helvetica", 8), foreground="#888888")
        lbl_version.pack(side="bottom", pady=2)

        # --- Main Viewport ---
        self.main_container = ttk.PanedWindow(root, orient="vertical")
        self.main_container.pack(side="right", fill="both", expand=True, padx=10, pady=5)

        self.plot_frame = ttk.Frame(self.main_container, padding=5, relief="groove")
        self.main_container.add(self.plot_frame, weight=3)

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.configure_axis_labels()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side="top", fill="x")

        self.cursor_var = tk.StringVar(value="Cursor Position: Raman shift = --")
        ttk.Label(self.plot_frame, textvariable=self.cursor_var, font=("Consolas", 10, "bold"), background="#e9ecef", relief="solid", borderwidth=1, padding=5).pack(side="bottom", fill="x", pady=(4, 0))

        # --- Bottom Results Dashboard ---
        self.table_frame = ttk.LabelFrame(self.main_container, text=" 📊 Peak Fit Results ", padding=5)
        self.main_container.add(self.table_frame, weight=1)

        self.result_table = ttk.Treeview(self.table_frame, columns=("Dataset", "Peak", "Center", "Amplitude", "FWHM"), show="headings", height=5)
        self.result_table.heading("Dataset", text="Dataset / Spectrum")
        self.result_table.heading("Peak", text="Peak Index")
        self.result_table.heading("Center", text="Center (cm⁻¹)")
        self.result_table.heading("Amplitude", text="Amplitude (counts)")
        self.result_table.heading("FWHM", text="FWHM (cm⁻¹)")

        self.result_table.column("Dataset", width=160, anchor="w")
        self.result_table.column("Peak", width=110, anchor="center")
        self.result_table.column("Center", width=130, anchor="center")
        self.result_table.column("Amplitude", width=130, anchor="center")
        self.result_table.column("FWHM", width=130, anchor="center")
        self.result_table.pack(fill="both", expand=True)

        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

    def configure_axis_labels(self):
        self.ax.set_xlabel(r"Raman shift (cm$^{-1}$)", fontsize=10, fontweight='bold')
        self.ax.set_ylabel("Intensity (counts)", fontsize=10, fontweight='bold')
        self.ax.set_title("Raman Spectra", fontsize=11, fontweight='bold', pad=8)
        self.ax.grid(True, linestyle="--", alpha=0.5)

    def save_to_history(self):
        if len(self.history_stack) >= 25:
            self.history_stack.pop(0)
        tree_cache = []
        for row in self.result_table.get_children():
            tree_cache.append(self.result_table.item(row)['values'])
        snapshot = {
            'active_datasets': {k: {
                'angles': np.copy(v['angles']),
                'intensities': np.copy(v['intensities']),
                'label': v['label'],
                'rruff_name': v.get('rruff_name'),
                'rruff_id': v.get('rruff_id')
            } for k, v in self.active_datasets.items()},
            'peak_guesses': list(self.peak_guesses),
            'table_data': tree_cache
        }
        self.history_stack.append(snapshot)
        self.btn_undo.config(state="normal")

    def undo_last_action(self):
        if not self.history_stack:
            return
        snapshot = self.history_stack.pop()
        for line in self.fitted_curves_artists:
            try: line.remove()
            except Exception: pass
        for line in self.guess_lines_artists:
            try: line.remove()
            except Exception: pass
        self.fitted_curves_artists = []
        self.guess_lines_artists = []
        self.active_datasets = snapshot['active_datasets']
        self.peak_guesses = snapshot['peak_guesses']
        for row in self.result_table.get_children():
            self.result_table.delete(row)
        for values in snapshot['table_data']:
            self.result_table.insert("", "end", values=values)
        self.replot_and_refresh_canvas()
        if not self.history_stack:
            self.btn_undo.config(state="disabled")

    def refresh_checkbox_targets_panel(self):
        for child in self.panel_fit_targets.winfo_children():
            child.destroy()
        if not self.active_datasets:
            self.lbl_no_targets = ttk.Label(self.panel_fit_targets, text="No Spectra Loaded", font=("Helvetica", 9, "italic"), foreground="#888888")
            self.lbl_no_targets.pack(side="top", anchor="w", padx=4)
            return
        for key, data in list(self.active_datasets.items()):
            row_frame = ttk.Frame(self.panel_fit_targets)
            row_frame.pack(side="top", fill="x", pady=2, expand=True)
            if not key.startswith("__fit_") and not key.startswith("__ref_"):
                if key not in self.target_checkbox_vars:
                    self.target_checkbox_vars[key] = tk.BooleanVar(value=True)
                cb = ttk.Checkbutton(row_frame, text=data['label'],
                                     variable=self.target_checkbox_vars[key],
                                     command=self.redraw_plot)
                cb.pack(side="left", anchor="w")
            else:
                # ROD/COD-baked references carry an explicit URL; RRUFF ones are derived.
                url = None
                if key.startswith("__ref_"):
                    url = data.get('ref_url') or rruff_url(data.get('rruff_name'), data.get('rruff_id'))
                if url:
                    lbl = ttk.Label(row_frame, text=data['label'], font=("Helvetica", 9, "underline"),
                                    foreground="#0d6efd", cursor="hand2")
                    lbl.bind("<Button-1>", lambda e, u=url: webbrowser.open_new_tab(u))
                else:
                    lbl = ttk.Label(row_frame, text=data['label'], font=("Helvetica", 9, "italic"), foreground="#555555")
                lbl.pack(side="left", anchor="w", padx=4)
            btn_del = ttk.Button(row_frame, text="❌", width=2, command=lambda k=key: self.remove_specific_dataset(k))
            btn_del.pack(side="right", anchor="e")

    def remove_specific_dataset(self, key_to_remove):
        self.save_to_history()
        if key_to_remove in self.active_datasets:
            del self.active_datasets[key_to_remove]
        for k in list(self.active_datasets.keys()):
            if k.endswith(f"_{key_to_remove}"):
                del self.active_datasets[k]
        self.replot_and_refresh_canvas()

    def toggle_normalization_mode(self):
        if not self.active_datasets:
            messagebox.showwarning("Execution Halted", "Load spectra before normalization.")
            return
        self.normalization_mode_active = not self.normalization_mode_active
        if self.normalization_mode_active:
            if self.fitting_mode_active: self.toggle_fitting_mode()
            self.btn_normalize_toggle.config(text="⚖️ Mode: SELECT PEAK")
            self.status_var.set("Left-click near a band to scale all active spectra.")
        else:
            self.btn_normalize_toggle.config(text="⚖️ Normalize to Peak")

    def toggle_fitting_mode(self):
        if not self.active_datasets:
            messagebox.showwarning("Execution Halted", "Load spectra before fitting.")
            return
        self.fitting_mode_active = not self.fitting_mode_active
        if self.fitting_mode_active:
            if self.normalization_mode_active: self.toggle_normalization_mode()
            self.btn_fit_toggle.config(text="🎯 Peak Selection: ACTIVE")
            self.btn_run_fit.config(state="normal")
        else:
            self.btn_fit_toggle.config(text="🎯 Peak Selection: OFF")
            self.btn_run_fit.config(state="disabled")

    def on_mouse_move(self, event):
        if event.inaxes == self.ax and self.active_datasets:
            x = event.xdata
            self.cursor_var.set(f"Cursor Position: Raman shift = {x:.2f} cm⁻¹")
            if self.cursor_line is None:
                self.cursor_line = self.ax.axvline(x, color='red', linestyle='--', linewidth=1.0, alpha=0.5)
            else:
                self.cursor_line.set_xdata([x, x])
                self.cursor_line.set_visible(True)
            self.canvas.draw_idle()
        else:
            if self.cursor_line is not None:
                self.cursor_line.set_visible(False)
                self.canvas.draw_idle()
            self.cursor_var.set("Cursor Position: Raman shift = --")

    def on_canvas_click(self, event):
        if event.inaxes == self.ax:
            if self.normalization_mode_active and event.button == 1:
                x_click = event.xdata
                try:
                    window_span = float(self.ent_norm_span.get().strip())
                except ValueError:
                    window_span = 10.0
                    self.ent_norm_span.delete(0, tk.END)
                    self.ent_norm_span.insert(0, "10")
                data_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_")]
                will_normalize = False
                for key in data_keys:
                    angles = self.active_datasets[key]['angles']
                    intensities = self.active_datasets[key]['intensities']
                    mask = (angles >= x_click - window_span) & (angles <= x_click + window_span)
                    if np.any(mask):
                        global_max = np.max(intensities)
                        local_peak_max = np.max(intensities[mask])
                        if local_peak_max > 0 and local_peak_max >= (0.05 * global_max):
                            will_normalize = True
                            break
                if will_normalize:
                    self.save_to_history()
                normalized_any = False
                for key in data_keys:
                    angles = self.active_datasets[key]['angles']
                    intensities = self.active_datasets[key]['intensities']
                    mask = (angles >= x_click - window_span) & (angles <= x_click + window_span)
                    if np.any(mask):
                        global_max = np.max(intensities)
                        local_peak_max = np.max(intensities[mask])
                        if local_peak_max > 0 and local_peak_max >= (0.05 * global_max):
                            self.active_datasets[key]['intensities'] = intensities / local_peak_max
                            normalized_any = True
                if normalized_any:
                    self.clear_fitted_artists()
                    self.replot_and_refresh_canvas()
                    self.status_var.set(f"Spectra normalized to band near {x_click:.1f} cm⁻¹.")
                self.normalization_mode_active = False
                self.btn_normalize_toggle.config(text="⚖️ Normalize to Peak")
                return

            elif self.fitting_mode_active and event.button == 3:
                self.save_to_history()
                x_guess = event.xdata
                self.peak_guesses.append(x_guess)
                guess_line = self.ax.axvline(x_guess, color='#d63384', linestyle=':', linewidth=1.5)
                self.guess_lines_artists.append(guess_line)
                self.canvas.draw_idle()

    def subtract_background_profile(self):
        if not self.active_datasets:
            messagebox.showwarning("No Data", "No active spectra to baseline-correct.")
            return
        data_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_")]
        if not data_keys: return
        self.save_to_history()
        self.clear_fitted_artists()
        for file_path in data_keys:
            data = self.active_datasets[file_path]
            intensities = data['intensities']
            if len(intensities) < 3: continue
            bg = snip_background(intensities, iterations=40)
            data['intensities'] = intensities - bg
        self.replot_and_refresh_canvas()

    def open_blank_subtraction_dialog(self):
        raw_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_") and not k.startswith("__ref_")]
        if len(raw_keys) < 2:
            messagebox.showwarning("Insufficient Data", "You need at least two spectra loaded to subtract.")
            return
        pop = tk.Toplevel(self.root)
        pop.title("Reference Spectrum Subtraction")
        pop.geometry("460x320")
        pop.transient(self.root)
        pop.grab_set()
        ttk.Label(pop, text="Select Reference / Substrate Spectrum:", font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12, pady=(12, 2))
        combo_blank = ttk.Combobox(pop, state="readonly", width=55)
        combo_blank['values'] = [self.active_datasets[k]['label'] for k in raw_keys]
        combo_blank.current(0)
        combo_blank.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Label(pop, text="Select Target Spectrum(s) to subtract from:", font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12, pady=(4, 2))
        frame_list = ttk.Frame(pop)
        frame_list.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        scroll = ttk.Scrollbar(frame_list)
        scroll.pack(side="right", fill="y")
        listbox_targets = tk.Listbox(frame_list, selectmode="multiple", yscrollcommand=scroll.set, exportselection=False)
        for k in raw_keys:
            listbox_targets.insert(tk.END, self.active_datasets[k]['label'])
        listbox_targets.pack(fill="both", expand=True, side="left")
        scroll.config(command=listbox_targets.yview)
        for idx in range(1, len(raw_keys)):
            listbox_targets.select_set(idx)

        def run_reference_subtraction():
            blank_idx = combo_blank.current()
            selected_targets = listbox_targets.curselection()
            if not selected_targets:
                messagebox.showwarning("Void Bounds", "Please pick at least one target spectrum.")
                return
            blank_key = raw_keys[blank_idx]
            blank_angles = self.active_datasets[blank_key]['angles']
            blank_intensities = self.active_datasets[blank_key]['intensities']
            self.save_to_history()
            self.clear_fitted_artists()
            for idx in selected_targets:
                target_key = raw_keys[idx]
                if target_key == blank_key:
                    continue
                target_angles = self.active_datasets[target_key]['angles']
                target_intensities = self.active_datasets[target_key]['intensities']
                blank_profile_interp = np.interp(target_angles, blank_angles, blank_intensities)
                self.active_datasets[target_key]['intensities'] = target_intensities - blank_profile_interp
            pop.destroy()
            self.replot_and_refresh_canvas()
            self.status_var.set(f"Subtracted reference spectrum: '{self.active_datasets[blank_key]['label']}'.")
        ttk.Button(pop, text="Subtract Reference", command=run_reference_subtraction).pack(pady=8)

    def smooth_active_profiles(self):
        if not self.active_datasets:
            messagebox.showwarning("No Data", "No active spectra to smooth.")
            return
        try:
            window = int(self.ent_smooth_win.get().strip())
            if window < 3: raise ValueError
            if window % 2 == 0: window += 1
        except ValueError:
            window = 11
            self.ent_smooth_win.delete(0, tk.END)
            self.ent_smooth_win.insert(0, "11")
        self.save_to_history()
        self.clear_fitted_artists()
        data_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_") and not k.startswith("__ref_")]
        smoothed_count = 0
        for key in data_keys:
            y = self.active_datasets[key]['intensities']
            if len(y) > window:
                self.active_datasets[key]['intensities'] = savgol_filter(y, window, polyorder=2)
                smoothed_count += 1
        if smoothed_count > 0:
            self.replot_and_refresh_canvas()
            self.status_var.set(f"Smoothed {smoothed_count} spectra (Savgol window={window}).")

    def apply_shift(self):
        """Linearly shifts the Raman-shift axis to calibrate a zero-offset."""
        if not self.active_datasets:
            messagebox.showwarning("No Data", "No active spectra to calibrate.")
            return
        try:
            shift_val = float(self.ent_shift_val.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Value", "Please enter a valid numeric shift (cm⁻¹).")
            return
        if shift_val == 0.0:
            return
        self.save_to_history()
        self.clear_fitted_artists()
        data_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_") and not k.startswith("__ref_")]
        for key in data_keys:
            self.active_datasets[key]['angles'] = self.active_datasets[key]['angles'] + shift_val
        self.replot_and_refresh_canvas()
        self.status_var.set(f"Applied a rigid Raman-shift calibration of {shift_val} cm⁻¹.")

    def replot_and_refresh_canvas(self):
        """Full refresh: redraw the plot and rebuild the sidebar panels."""
        self.redraw_plot()
        self.refresh_checkbox_targets_panel()
        self.refresh_adjust_targets()

    def redraw_plot(self):
        """Redraws only the plot from current data + visibility, leaving the
        sidebar layer panel untouched (so its scroll position is preserved)."""
        self.ax.clear()
        self.configure_axis_labels()
        self.cursor_line = None
        self.fitted_curves_artists = []
        self.guess_lines_artists = []
        self.line_map = {}
        for file_path, data in self.active_datasets.items():
            if file_path.startswith("__fit_overall_composite"):
                line, = self.ax.plot(data['angles'], data['intensities'], color='#000000', linestyle='-', linewidth=2.0, label=data['label'])
                self.fitted_curves_artists.append(line)
            elif file_path.startswith("__fit_"):
                line, = self.ax.plot(data['angles'], data['intensities'], linestyle='--', linewidth=1.2, label=data['label'])
                self.fitted_curves_artists.append(line)
            elif file_path.startswith("__ref_"):
                line, = self.ax.plot(data['angles'], data['intensities'], linestyle='-.', linewidth=1.5, alpha=0.8, label=data['label'])
                self.line_map[file_path] = line
            else:
                # Honor the layer checkbox: unchecked spectra are hidden.
                var = self.target_checkbox_vars.get(file_path)
                if var is not None and not var.get():
                    continue
                line, = self.ax.plot(data['angles'], data['intensities'], label=data['label'], linewidth=1.2)
                self.line_map[file_path] = line
        for g_x in self.peak_guesses:
            guess_line = self.ax.axvline(g_x, color='#d63384', linestyle=':', linewidth=1.5)
            self.guess_lines_artists.append(guess_line)
        handles, _labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc="upper right", frameon=True, fontsize=8)
        self.ax.relim(); self.ax.autoscale_view(); self.canvas.draw()

    def run_peak_optimization(self):
        if not self.peak_guesses:
            messagebox.showwarning("Missing Inputs", "Right-click on the plot to specify band center guesses first.")
            return
        keys_to_fit = [k for k, v in self.target_checkbox_vars.items() if v.get() and k in self.active_datasets]
        if not keys_to_fit:
            messagebox.showwarning("Selection Missing", "Please select at least one spectrum to fit.")
            return
        self.save_to_history()
        for line in self.fitted_curves_artists:
            try: line.remove()
            except Exception: pass
        self.fitted_curves_artists.clear()
        for k in list(self.active_datasets.keys()):
            if k.startswith("__fit_"): del self.active_datasets[k]
        for row in self.result_table.get_children(): self.result_table.delete(row)

        fit_errors = []
        for key in keys_to_fit:
            x_data = self.active_datasets[key]['angles']
            y_data = self.active_datasets[key]['intensities']
            label_base = self.active_datasets[key]['label']
            p0 = []; bounds_min = []; bounds_max = []
            for g_x in self.peak_guesses:
                idx = np.argmin(np.abs(x_data - g_x))
                amp_guess = float(y_data[idx])
                # Raman bands are typically a few to tens of cm-1 wide
                p0.extend([amp_guess, g_x, 10.0])
                bounds_min.extend([0.0, g_x - 25.0, 0.5])
                bounds_max.extend([float(np.max(y_data)) * 2.0, g_x + 25.0, 200.0])
            try:
                p_opt, _ = curve_fit(multi_gaussian_composite, x_data, y_data, p0=p0, bounds=(bounds_min, bounds_max))
                y_fit_total = multi_gaussian_composite(x_data, *p_opt)
                total_fit_line, = self.ax.plot(x_data, y_fit_total, linestyle='-', linewidth=2.2, label=f"{label_base} Fit Total")
                self.fitted_curves_artists.append(total_fit_line)
                peak_counter = 1
                for i in range(0, len(p_opt), 3):
                    amp, cent, wid = p_opt[i], p_opt[i+1], p_opt[i+2]
                    y_peak = gaussian_profile(x_data, amp, cent, wid)
                    pk_line, = self.ax.plot(x_data, y_peak, linestyle='--', linewidth=1.2, label=f"{label_base} Pk {peak_counter}")
                    self.fitted_curves_artists.append(pk_line)
                    fwhm = 2.0 * np.sqrt(np.log(2)) * wid
                    self.result_table.insert("", "end", values=(label_base, f"Peak {peak_counter}", f"{cent:.2f}", f"{amp:.1f}", f"{fwhm:.2f}"))
                    self.active_datasets[f"__fit_peak_{peak_counter}_{key}"] = {'angles': x_data, 'intensities': y_peak, 'label': f"{label_base} Pk {peak_counter} Fit"}
                    peak_counter += 1
                self.active_datasets[f"__fit_overall_composite_{key}"] = {'angles': x_data, 'intensities': y_fit_total, 'label': f"{label_base} Overall Fit"}
            except Exception as e:
                fit_errors.append(f"{label_base}: {e}")
        self.ax.legend(loc="upper right", frameon=True, fontsize=8); self.canvas.draw()
        if fit_errors: messagebox.showerror("Fitting Errors Encountered", "\n".join(fit_errors))

    def select_and_plot_files(self):
        files = filedialog.askopenfilenames(
            title="Select Raman Data Files",
            filetypes=[("Raman Datasets", ("*.h5", "*.hdf5", "*.xml", "*.txt", "*.csv",
                                           "*.dat", "*.asc", "*.jdx", "*.dx", "*.jcm")),
                       ("HORIBA LabSpec HDF5", ("*.h5", "*.hdf5")),
                       ("HORIBA LabSpec XML", "*.xml"),
                       ("Text / CSV / RRUFF", ("*.txt", "*.csv", "*.dat", "*.asc")),
                       ("JCAMP-DX (SDBS / ROD / vendor)", ("*.jdx", "*.dx", "*.jcm")),
                       ("All Files", "*.*")]
        )
        if not files: return
        self.save_to_history()
        loaded_count = 0; error_logs = []
        for file_path in files:
            try:
                spectra = load_raman_data(file_path)
            except Exception as e:
                error_logs.append(f"{os.path.basename(file_path)}: {str(e)}")
                continue
            for i, spec in enumerate(spectra):
                # Unique key per spectrum (a single .h5 holds many)
                key = f"{file_path}::{i}"
                if key in self.active_datasets:
                    continue
                self.active_datasets[key] = {'angles': spec['x'], 'intensities': spec['y'], 'label': spec['label']}
                loaded_count += 1
        if loaded_count > 0:
            self.replot_and_refresh_canvas()
            raw_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_")]
            self.status_var.set(f"Active spectra loaded: {len(raw_keys)}")
        if error_logs: messagebox.showwarning("Import Errors", "\n".join(error_logs))

    def crop_to_current_view(self):
        if not self.active_datasets: return
        xmin, xmax = self.ax.get_xlim()
        self.save_to_history()
        self.clear_fitted_artists()
        for f_path, data in list(self.active_datasets.items()):
            if f_path.startswith("__fit_"):
                del self.active_datasets[f_path]
                continue
            ang, intset = data['angles'], data['intensities']
            mask = (ang >= xmin) & (ang <= xmax)
            data['angles'] = ang[mask]
            data['intensities'] = intset[mask]
        self.replot_and_refresh_canvas()

    def export_active_data_to_csv(self):
        if not self.active_datasets: return
        out_dir = filedialog.askdirectory(title="Select Output Folder")
        if not out_dir: return
        success_count = 0
        for path_key, data in self.active_datasets.items():
            try:
                raw_name = data.get('label', path_key)
                b_name = "".join(c if (c.isalnum() or c in "-.") else "_" for c in raw_name).strip("_")
                if not b_name:
                    b_name = "spectrum"
                out_path = os.path.join(out_dir, f"raman_{b_name}.csv")
                header = "Raman shift (cm-1),Intensity (counts)\n"
                with open(out_path, "w", encoding="utf-8") as fo:
                    fo.write(header)
                    for xv, yv in zip(data['angles'], data['intensities']):
                        fo.write(f"{xv:.6f},{yv:.4f}\n")
                success_count += 1
            except Exception as e:
                print(f"Exception saving file: {e}")
        messagebox.showinfo("Export Complete", f"Successfully saved {success_count} spectra.")

    def remove_fitted_only_artists(self):
        for line in self.fitted_curves_artists:
            try: line.remove()
            except Exception: pass
        self.fitted_curves_artists.clear()
        for row in self.result_table.get_children(): self.result_table.delete(row)

    def clear_fitted_artists(self):
        self.remove_fitted_only_artists()
        for line in self.guess_lines_artists:
            try: line.remove()
            except Exception: pass
        self.guess_lines_artists.clear()
        self.peak_guesses.clear()

    # ---------- Interactive wheel-adjust (offset / scale) ----------
    def refresh_adjust_targets(self):
        if not hasattr(self, 'combo_adjust_target'):
            return
        keys = [k for k in self.active_datasets.keys()
                if not k.startswith("__fit_")]
        self._adjust_key_by_label = {self.active_datasets[k]['label']: k for k in keys}
        labels = [self.active_datasets[k]['label'] for k in keys]
        current = self.combo_adjust_target.get()
        self.combo_adjust_target['values'] = labels
        if current in labels:
            self.combo_adjust_target.set(current)
        elif labels:
            self.combo_adjust_target.set(labels[0])
        else:
            self.combo_adjust_target.set('')

    def set_adjust_mode(self, mode):
        if not self.active_datasets:
            messagebox.showwarning("No Data", "Load spectra first.")
            return
        if self.adjust_mode == mode:
            self.adjust_mode = None
        else:
            self.adjust_mode = mode
            if self.fitting_mode_active:
                self.toggle_fitting_mode()
            if self.normalization_mode_active:
                self.toggle_normalization_mode()
        self.adjust_armed = False
        self.btn_adjust_offset.config(text=("➕ Offset ✓" if self.adjust_mode == 'offset' else "➕ Offset"))
        self.btn_adjust_scale.config(text=("✖ Scale ✓" if self.adjust_mode == 'scale' else "✖ Scale"))
        if self.adjust_mode:
            self.status_var.set(f"Wheel-adjust '{self.adjust_mode}' armed — scroll over the plot.")
        else:
            self.status_var.set("Wheel-adjust off.")

    def on_scroll(self, event):
        # Two-finger trackpad scroll (where the backend forwards it) or a mouse wheel.
        if self.adjust_mode is None or event.inaxes != self.ax:
            return
        step_val = getattr(event, 'step', 0) or 0
        if step_val == 0:
            step_val = 1 if getattr(event, 'button', None) == 'up' else -1
        self._apply_adjust(1.0 if step_val > 0 else -1.0)

    def adjust_step_button(self, direction):
        """Discrete +/- step from the on-screen buttons (no wheel needed)."""
        if not self.active_datasets:
            messagebox.showwarning("No Data", "Load spectra first.")
            return
        if self.adjust_mode is None:
            # Default to Offset so the buttons work without arming a mode first.
            self.set_adjust_mode('offset')
        self._apply_adjust(float(direction))

    def _apply_adjust(self, direction):
        """Applies one offset/scale step to the selected target spectrum."""
        label = self.combo_adjust_target.get()
        key = self._adjust_key_by_label.get(label)
        if key is None or key not in self.active_datasets:
            return
        if not self.adjust_armed:
            self.save_to_history()
            self.adjust_armed = True

        y = self.active_datasets[key]['intensities']
        if self.adjust_mode == 'offset':
            try:
                step = float(self.ent_offset_step.get().strip())
            except ValueError:
                step = 100.0
            y = y + step * direction
        else:  # scale (multiply)
            try:
                pct = float(self.ent_scale_step.get().strip())
            except ValueError:
                pct = 5.0
            factor = (1.0 + pct / 100.0) ** direction
            y = y * factor
        self.active_datasets[key]['intensities'] = y

        line = self.line_map.get(key)
        if line is not None:
            line.set_ydata(y)
            self.canvas.draw_idle()
        else:
            self.replot_and_refresh_canvas()

    # ---------- RRUFF reference database ----------
    def _refresh_rruff_status(self):
        ds = self.combo_rruff_dataset.get()
        if self.rruff_lib:
            self.rruff_status_var.set(f"RRUFF library: {len(self.rruff_lib['entries'])} spectra (precomputed peaks).")
            return
        if self.rruff_local_dir:
            n = len([f for f in os.listdir(self.rruff_local_dir) if f.lower().endswith('.txt')])
            self.rruff_status_var.set(f"RRUFF: local folder ({n} spectra).")
        elif rruff_is_cached(ds):
            n = len([f for f in os.listdir(rruff_dataset_dir(ds)) if f.lower().endswith('.txt')])
            self.rruff_status_var.set(f"RRUFF: '{ds}' cached ({n} spectra).")
        else:
            self.rruff_status_var.set(f"RRUFF: '{ds}' not downloaded yet.")

    def rruff_download_selected(self):
        ds = self.combo_rruff_dataset.get()
        self.btn_rruff_download.config(state="disabled")
        self.rruff_status_var.set(f"RRUFF: preparing to download '{ds}' ...")

        def worker():
            try:
                def prog(msg): self.root.after(0, lambda: self.rruff_status_var.set(f"RRUFF: {msg}"))
                count = rruff_download_dataset(ds, progress_cb=prog)
                self.rruff_local_dir = None
                self.root.after(0, lambda: self.rruff_status_var.set(f"RRUFF: '{ds}' ready ({count} spectra). Search above."))
            except Exception as e:
                self.root.after(0, lambda err=str(e): messagebox.showerror(
                    "RRUFF Download Failed",
                    f"Could not download '{ds}'.\n\n{err}\n\n"
                    f"You can also download archives manually from\n{RRUFF_BASE_URL}\n"
                    f"and point the tool at the folder with 'Use Local RRUFF Folder'."))
                self.root.after(0, self._refresh_rruff_status)
            finally:
                self.root.after(0, lambda: self.btn_rruff_download.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def rruff_pick_local_folder(self):
        d = filedialog.askdirectory(title="Select a folder of RRUFF .txt spectra")
        if not d:
            return
        if not any(fn.lower().endswith('.txt') for fn in os.listdir(d)):
            messagebox.showwarning("No Spectra", "That folder contains no .txt spectra.")
            return
        self.rruff_local_dir = d
        self.rruff_lib = None  # folder takes precedence when chosen
        self._refresh_rruff_status()

    def rruff_open_library(self):
        path = filedialog.askopenfilename(
            title="Open a consolidated RRUFF .h5 library",
            filetypes=[("RRUFF library", ("*.h5", "*.hdf5")), ("All Files", "*.*")])
        if not path:
            return
        try:
            self.rruff_lib = load_rruff_h5_library(path)
        except Exception as e:
            messagebox.showerror("Library Error", f"Could not open library:\n{e}")
            return
        self.rruff_local_dir = None  # library takes precedence
        n = len(self.rruff_lib['entries'])
        self.rruff_status_var.set(f"RRUFF library: {n} spectra (precomputed peaks). Search or Match.")

    def _read_reference_xy(self, hit):
        """Returns (x, y, label) for a search/match hit, from the .h5 library
        (lazy read of the group) or a two-column file, as appropriate."""
        if hit.get('group') is not None and self.rruff_lib:
            with h5py.File(self.rruff_lib['path'], 'r') as f:
                g = f['spectra'][hit['group']]
                x = np.array(g['x'][:], dtype=float)
                y = np.array(g['y'][:], dtype=float)
            label = f"RRUFF: {hit['name']}" + (f" ({hit['id']})" if hit['id'] else "")
            return x, y, label
        with open(hit['path'], 'r', encoding='utf-8', errors='ignore') as f:
            x, y, label = _parse_two_column_text(f.read(), hit['name'])
        if not label.startswith("RRUFF"):
            label = f"RRUFF: {hit['name']}" + (f" ({hit['id']})" if hit['id'] else "")
        return x, y, label

    def _add_reference(self, x, y, label, key, rruff_name=None, rruff_id=None, ref_url=None):
        """Scales a reference to the current data maximum and stores it.

        `ref_url` is an explicit database link (ROD/COD-baked references carry
        one); when absent the layers panel derives an RRUFF link from the name.
        """
        data_keys = [k for k in self.active_datasets.keys()
                     if not k.startswith("__fit_") and not k.startswith("__ref_")]
        if data_keys and np.max(y) > 0:
            max_scale = max(np.max(self.active_datasets[k]['intensities']) for k in data_keys)
            y = (y / np.max(y)) * max_scale
        self.active_datasets[key] = {'angles': x, 'intensities': y, 'label': label,
                                     'rruff_name': rruff_name, 'rruff_id': rruff_id,
                                     'ref_url': ref_url}

    def rruff_run_search(self):
        query = self.ent_rruff_query.get().strip()
        self.rruff_results_list.delete(0, tk.END)
        self.rruff_search_hits = []
        if self.rruff_lib:
            q = query.lower()
            hits = [{'name': e['name'], 'id': e['id'], 'group': e['group']}
                    for e in self.rruff_lib['entries']
                    if not q or q in f"{e['name']} {e['id']}".lower()]
        elif self.rruff_local_dir:
            hits = []
            q = query.lower()
            for fn in sorted(os.listdir(self.rruff_local_dir)):
                if not fn.lower().endswith('.txt'):
                    continue
                parts = fn.split('__')
                name = parts[0] if parts else fn
                rid = parts[1] if len(parts) > 1 else ''
                if not q or q in f"{name} {rid} {fn}".lower():
                    hits.append({'name': name, 'id': rid, 'path': os.path.join(self.rruff_local_dir, fn)})
        else:
            ds = self.combo_rruff_dataset.get()
            if not rruff_is_cached(ds):
                messagebox.showinfo("Download First", f"RRUFF set '{ds}' is not downloaded yet.\nUse 'Download / Update Set' or point to a local folder.")
                return
            hits = rruff_search_cached(ds, query)
        if not hits:
            self.rruff_status_var.set("RRUFF: no matches.")
            return
        self.rruff_search_hits = hits[:500]
        for h in self.rruff_search_hits:
            self.rruff_results_list.insert(tk.END, f"{h['name']} {('· ' + h['id']) if h['id'] else ''}")
        self.rruff_status_var.set(f"RRUFF: {len(hits)} match(es)" + (" (showing first 500)." if len(hits) > 500 else "."))

    def _rruff_open_selected_page(self, event=None):
        sel = self.rruff_results_list.curselection()
        for idx in sel:
            if idx < len(self.rruff_search_hits):
                h = self.rruff_search_hits[idx]
                url = rruff_url(h['name'], h.get('id'))
                if url:
                    webbrowser.open_new_tab(url)

    def rruff_overlay_selected(self):
        sel = self.rruff_results_list.curselection()
        if not sel or not self.rruff_search_hits:
            messagebox.showinfo("Nothing Selected", "Search, then select a RRUFF entry to overlay.")
            return
        self.save_to_history()
        added = 0
        for idx in sel:
            if idx >= len(self.rruff_search_hits):
                continue
            hit = self.rruff_search_hits[idx]
            try:
                x, y, label = self._read_reference_xy(hit)
                if len(x) == 0:
                    continue
            except Exception:
                continue
            key = f"__ref_{hit['name']}_{hit['id']}_{idx}"
            self._add_reference(x, y, label, key, rruff_name=hit['name'], rruff_id=hit['id'])
            added += 1
        if added:
            self.replot_and_refresh_canvas()
            self.rruff_status_var.set(f"RRUFF: overlaid {added} reference spectrum(s).")

    def _rruff_candidate_files(self, query=""):
        """Returns [(name, id, path)] for the active RRUFF source, optional filter."""
        if self.rruff_local_dir and os.path.isdir(self.rruff_local_dir):
            base = self.rruff_local_dir
        else:
            ds = self.combo_rruff_dataset.get()
            base = rruff_dataset_dir(ds) if ds else None
        out = []
        if not base or not os.path.isdir(base):
            return out
        q = (query or "").strip().lower()
        for fn in sorted(os.listdir(base)):
            if not fn.lower().endswith('.txt'):
                continue
            parts = fn.split('__')
            name = parts[0] if parts else fn
            rid = parts[1] if len(parts) > 1 else ''
            if q and q not in f"{name} {rid} {fn}".lower():
                continue
            out.append((name, rid, os.path.join(base, fn)))
        return out

    def rruff_match_by_peaks(self):
        """Rank RRUFF references by how well their peaks match the marked peaks."""
        if not self.peak_guesses:
            messagebox.showinfo(
                "Mark Peaks First",
                "Turn on '🎯 Peak Selection', then right-click on the plot to mark the "
                "peaks you want to match. Then run 'Match by Selected Peaks'.")
            return
        try:
            tolerance = float(self.ent_match_tol.get().strip())
        except ValueError:
            tolerance = 12.0
        exp_peaks = list(self.peak_guesses)

        # Fast path: consolidated .h5 library has precomputed peaks -> instant.
        if self.rruff_lib:
            q = self.ent_rruff_query.get().strip().lower()
            scored = []
            for e in self.rruff_lib['entries']:
                if q and q not in f"{e['name']} {e['id']}".lower():
                    continue
                score, avg, matched = peak_match_score(e['peaks'], exp_peaks, tolerance)
                if matched > 0:
                    scored.append({'score': score, 'avg': avg, 'matched': matched,
                                   'name': e['name'], 'id': e['id'], 'group': e['group']})
            scored.sort(key=lambda t: (-t['score'], t['avg']))
            self._show_rruff_match_results(scored[:100], len(exp_peaks), tolerance)
            return

        candidates = self._rruff_candidate_files(self.ent_rruff_query.get())
        if not candidates:
            messagebox.showinfo(
                "No RRUFF Data",
                "Open a RRUFF .h5 library, download a set, or point to a local RRUFF folder first.\n"
                "A search-box term, if present, restricts which references are scanned.")
            return

        self.btn_rruff_match.config(state="disabled")
        self.rruff_status_var.set(f"Matching {len(candidates)} references ...")

        def worker():
            scored = []
            total = len(candidates)
            for i, (name, rid, path) in enumerate(candidates):
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        x, y, _ = _parse_two_column_text(f.read(), name)
                    if len(x) < 5:
                        continue
                    ref_peaks = detect_reference_peaks(x, y)
                    score, avg, matched = peak_match_score(ref_peaks, exp_peaks, tolerance)
                    if matched > 0:
                        scored.append({'score': score, 'avg': avg, 'matched': matched,
                                       'name': name, 'id': rid, 'path': path})
                except Exception:
                    continue
                if (i % 200) == 0:
                    self.root.after(0, lambda i=i: self.rruff_status_var.set(
                        f"Matching {i}/{total} references ..."))
            scored.sort(key=lambda t: (-t['score'], t['avg']))
            self.root.after(0, lambda: self._show_rruff_match_results(scored[:100], len(exp_peaks), tolerance))
            self.root.after(0, lambda: self.btn_rruff_match.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _show_rruff_match_results(self, scored, n_exp, tolerance):
        if not scored:
            self.rruff_status_var.set("RRUFF: no references matched the marked peaks.")
            messagebox.showinfo("No Matches",
                                "No RRUFF reference had bands near your marked peaks.\n"
                                "Try a larger match tolerance or different peaks.")
            return
        self.rruff_status_var.set(f"RRUFF: {len(scored)} candidate(s) ranked (tol ±{tolerance:g}).")

        pop = tk.Toplevel(self.root)
        pop.title("RRUFF Search & Match — Candidate Ranking")
        pop.geometry("640x340")
        pop.transient(self.root)
        pop.grab_set()
        ttk.Label(pop, text=f"Ranked by alignment of RRUFF bands with your {n_exp} marked peak(s) "
                            f"(±{tolerance:g} cm⁻¹). Double-click a row to open its RRUFF page.",
                  font=("Helvetica", 9, "bold")).pack(pady=6, padx=8, anchor="w")

        frame = ttk.Frame(pop)
        frame.pack(fill="both", expand=True, padx=8, pady=4)
        scroll = ttk.Scrollbar(frame)
        scroll.pack(side="right", fill="y")
        tree = ttk.Treeview(frame, columns=("Score", "Mineral", "ID", "Matched"),
                            show="headings", yscrollcommand=scroll.set, height=10, selectmode="extended")
        tree.heading("Score", text="Match Score")
        tree.heading("Mineral", text="Mineral")
        tree.heading("ID", text="RRUFF ID")
        tree.heading("Matched", text="Peaks Matched")
        tree.column("Score", width=110, anchor="center")
        tree.column("Mineral", width=230, anchor="w")
        tree.column("ID", width=110, anchor="center")
        tree.column("Matched", width=120, anchor="center")
        tree.pack(fill="both", expand=True)
        scroll.config(command=tree.yview)

        row_map = {}
        for rec in scored:
            iid = tree.insert("", "end", values=(f"{rec['score']:.0f}%", rec['name'],
                                                  rec['id'], f"{rec['matched']}/{n_exp}"))
            row_map[iid] = rec

        def open_pages(event=None):
            sel = tree.selection()
            opened = 0
            for iid in sel:
                rec = row_map.get(iid)
                if not rec:
                    continue
                url = rruff_url(rec['name'], rec.get('id'))
                if url:
                    webbrowser.open_new_tab(url)
                    opened += 1
            if opened == 0:
                messagebox.showinfo("Nothing Selected", "Select a row, then open its RRUFF page.")
        tree.bind("<Double-1>", open_pages)

        def overlay_chosen():
            sel = tree.selection()
            if not sel:
                return
            pop.destroy()
            self.save_to_history()
            added = 0
            for iid in sel:
                hit = row_map[iid]
                try:
                    x, y, label = self._read_reference_xy(hit)
                    if len(x) == 0:
                        continue
                except Exception:
                    continue
                key = f"__ref_{hit['name']}_{hit['id']}_match"
                self._add_reference(x, y, label, key, rruff_name=hit['name'], rruff_id=hit['id'])
                added += 1
            if added:
                self.replot_and_refresh_canvas()
                self.rruff_status_var.set(f"RRUFF: overlaid {added} matched reference(s).")

        btn_row = ttk.Frame(pop)
        btn_row.pack(pady=8)
        ttk.Button(btn_row, text="🔗 Open RRUFF Page(s)", command=open_pages).pack(side="left", padx=4)
        ttk.Button(btn_row, text="➕ Overlay Selected Match(es)", command=overlay_chosen).pack(side="left", padx=4)

    # ---------- ROD reference database ----------
    # Two modes, chosen by the Source radio:
    #   'h5'     one or more baked libraries, searched and matched in memory
    #   'online' the ROD REST API, queried on demand (desktop only -- the
    #            browser build cannot, because ROD's CORS header names a
    #            single third-party origin)

    def rod_add_library(self):
        """Add one or more baked ROD .h5 libraries to the managed list."""
        paths = filedialog.askopenfilenames(
            title="Add ROD .h5 librar(ies) (from build_rod_library.py)",
            filetypes=[("ROD Raman library", ("*.h5", "*.hdf5")), ("All Files", "*.*")])
        added = 0
        for path in paths:
            if self._rod_load_library_path(path, active=True):
                added += 1
        if added:
            self.rod_mode_var.set("h5")
            self._rod_save_config()
        self._rod_refresh_libs_panel()
        self._rod_update_status()

    def _rod_load_library_path(self, path, active=True):
        """Load an .h5 into the managed list. Returns True on success."""
        if any(os.path.abspath(l['path']) == os.path.abspath(path) for l in self.rod_libs):
            self.rod_status_var.set(f"Already loaded: {os.path.basename(path)}")
            return False
        try:
            lib = load_rod_h5_library(path)
        except Exception as e:
            messagebox.showerror("Library Error",
                                 f"Could not open ROD library:\n{os.path.basename(path)}\n{e}")
            return False
        for e in lib['entries']:
            e['lib_path'] = path
        self.rod_libs.append({'name': os.path.basename(path), 'path': path,
                              'entries': lib['entries'], 'active': bool(active),
                              'var': tk.BooleanVar(value=bool(active))})
        return True

    def _rod_remove_library(self, idx):
        if 0 <= idx < len(self.rod_libs):
            del self.rod_libs[idx]
            self._rod_save_config()
            self._rod_refresh_libs_panel()
            self._rod_update_status()

    def _rod_toggle_library(self, idx):
        if 0 <= idx < len(self.rod_libs):
            self.rod_libs[idx]['active'] = bool(self.rod_libs[idx]['var'].get())
            self.rod_mode_var.set("h5")
            self._rod_save_config()
            self._rod_update_status()

    def _rod_refresh_libs_panel(self):
        for child in self.rod_libs_frame.winfo_children():
            child.destroy()
        for idx, lib in enumerate(self.rod_libs):
            row = ttk.Frame(self.rod_libs_frame)
            row.pack(fill="x", pady=1)
            lib.setdefault('var', tk.BooleanVar(value=lib['active']))
            lib['var'].set(lib['active'])
            ttk.Checkbutton(row, text=f"{lib['name']} ({len(lib['entries'])})",
                            variable=lib['var'],
                            command=lambda i=idx: self._rod_toggle_library(i)).pack(side="left", anchor="w")
            ttk.Button(row, text="❌", width=2,
                       command=lambda i=idx: self._rod_remove_library(i)).pack(side="right")

    def _rod_save_config(self):
        try:
            with open(self.rod_cfg_path, "w", encoding="utf-8") as fh:
                json.dump([{'path': l['path'], 'active': l['active']} for l in self.rod_libs],
                          fh, indent=2)
        except Exception:
            pass

    def _rod_load_config(self):
        """On launch, re-load remembered libraries (skipping any now-missing files)."""
        if os.path.exists(self.rod_cfg_path):
            try:
                with open(self.rod_cfg_path, "r", encoding="utf-8") as fh:
                    saved = json.load(fh)
            except Exception:
                saved = []
            for item in saved or []:
                p = item.get('path')
                if p and os.path.exists(p):
                    self._rod_load_library_path(p, active=item.get('active', True))
        self._rod_refresh_libs_panel()
        self._rod_update_status()

    def _rod_active_entries(self):
        out = []
        for lib in self.rod_libs:
            if lib['active']:
                out.extend(lib['entries'])
        return out

    def _rod_update_status(self):
        if self.rod_mode_var.get() == "online":
            self.rod_status_var.set(
                "ROD online: queries solsa.crystallography.net. Matching needs a "
                "narrowing query (name / formula / elements) and downloads up to "
                "'max online' spectra.")
            return
        active = [l for l in self.rod_libs if l['active']]
        total = sum(len(l['entries']) for l in active)
        if not self.rod_libs:
            self.rod_status_var.set(
                "Add a baked .h5 library (ROD or Open Specy), or switch to Online.")
        else:
            sources = sorted({e.get('source', 'ROD') for l in active for e in l['entries']})
            tag = "/".join(sources) if sources else "Libraries"
            self.rod_status_var.set(
                f"{tag}: {len(active)}/{len(self.rod_libs)} librar(ies) active · "
                f"{total:,} spectra. Search / Overlay / Match.")

    @staticmethod
    def _rod_haystack(e):
        return (f"{e.get('name','')} {e.get('id','')} {e.get('formula','')} "
                f"{e.get('mineral','')} {e.get('collection','')} "
                f"{e.get('cod_id','')}").lower()

    @staticmethod
    def _rod_label(hit, name=None):
        """'ROD: Quartz (1000076)' / 'OpenSpecy: Polyethylene (PE_01)'."""
        src = hit.get('source') or 'ROD'
        nm = name or hit.get('name') or f"{src} {hit.get('id','')}"
        rid = hit.get('id')
        return f"{src}: {nm}" + (f" ({rid})" if rid else "")

    def rod_run_search(self):
        query = self.ent_rod_query.get().strip()
        self.rod_results_list.delete(0, tk.END)
        self.rod_search_hits = []

        if self.rod_mode_var.get() == "online":
            field = self.combo_rod_field.get() or "auto"
            self.rod_status_var.set("ROD online: searching …")

            def worker():
                try:
                    hits = rod_search_online(query, field=field)
                    err = None
                except Exception as e:                       # noqa: BLE001
                    hits, err = [], e
                self.root.after(0, lambda: self._rod_show_search_hits(hits, err))

            threading.Thread(target=worker, daemon=True).start()
            return

        pool = self._rod_active_entries()
        if not pool:
            messagebox.showinfo(
                "No ROD Library",
                "Add a ROD .h5 library (built by build_rod_library.py), tick one in the "
                "list, or switch Source to Online.")
            return
        q = query.lower()
        hits = [e for e in pool if not q or q in self._rod_haystack(e)][:300]
        self._rod_show_search_hits(hits, None)

    def _rod_show_search_hits(self, hits, err):
        if err is not None:
            self.rod_status_var.set(f"ROD online search failed: {err}")
            return
        self.rod_search_hits = hits
        if not hits:
            self.rod_status_var.set("ROD: no matches.")
            return
        for h in hits:
            src = h.get('source') or 'ROD'
            bits = [h.get('name') or f"{src} {h.get('id')}", str(h.get('id', ''))]
            if h.get('formula'):
                bits.append(h['formula'])
            if h.get('collection'):
                bits.append(h['collection'])
            if h.get('laser'):
                bits.append(f"{h['laser']}nm")
            prefix = "" if src.upper() == "ROD" else f"[{src}] "
            self.rod_results_list.insert(tk.END, prefix + "  ·  ".join(b for b in bits if b))
        where = "online" if self.rod_mode_var.get() == "online" else "libraries"
        self.rod_status_var.set(
            f"ROD ({where}): {len(hits)} match(es). Select and overlay; double-click opens the ROD page.")

    def _rod_selected_hit(self):
        sel = self.rod_results_list.curselection()
        if not sel or sel[0] >= len(self.rod_search_hits):
            return None
        return self.rod_search_hits[sel[0]]

    def _rod_open_selected_page(self, event=None):
        hit = self._rod_selected_hit()
        if not hit:
            return
        url = hit.get('url') or rod_url(hit.get('id'))
        if url:
            webbrowser.open_new_tab(url)

    def _rod_read_xy(self, hit):
        """Returns (x, y, label, url) for a hit, from the .h5 group (lazy read)
        or by downloading it from ROD."""
        if hit.get('group') is not None and hit.get('lib_path'):
            with h5py.File(hit['lib_path'], 'r') as f:
                g = f['spectra'][hit['group']]
                x = np.array(g['x'][:], dtype=float)
                y = np.array(g['y'][:], dtype=float)
            return x, y, self._rod_label(hit), hit.get('url')

        # No group -> an online ROD hit; fetch it.
        x, y, meta = rod_fetch_spectrum(hit['id'])
        name = hit.get('name') or meta.get('name') or f"ROD {hit['id']}"
        return (x, y, self._rod_label(dict(hit, source='ROD'), name),
                (meta.get('url') or rod_url(hit['id'])))

    def rod_overlay_selected(self):
        hit = self._rod_selected_hit()
        if not hit:
            self.rod_status_var.set("ROD: select a search result first.")
            return

        def finish(x, y, label, url, err):
            if err is not None:
                self.rod_status_var.set(f"ROD: could not read that reference — {err}")
                return
            if len(x) == 0:
                self.rod_status_var.set("ROD: that entry has no spectrum data.")
                return
            self.save_to_history()
            self._add_reference(x, y, label, f"__ref_rod_{hit['id']}",
                                rruff_name=hit.get('name'), rruff_id=hit.get('id'),
                                ref_url=url)
            self.replot_and_refresh_canvas()
            self.rod_status_var.set(f"ROD: overlaid {hit.get('name')} ({hit['id']}).")

        if hit.get('group') is not None:
            try:
                x, y, label, url = self._rod_read_xy(hit)
                finish(x, y, label, url, None)
            except Exception as e:                           # noqa: BLE001
                finish(None, None, None, None, e)
            return

        self.rod_status_var.set(f"ROD: downloading {hit['id']} …")

        def worker():
            try:
                x, y, label, url = self._rod_read_xy(hit)
                self.root.after(0, lambda: finish(x, y, label, url, None))
            except Exception as e:                           # noqa: BLE001
                self.root.after(0, lambda e=e: finish(None, None, None, None, e))

        threading.Thread(target=worker, daemon=True).start()

    def rod_match_by_peaks(self):
        """Rank ROD references by how well their bands explain the marked peaks."""
        if not self.peak_guesses:
            messagebox.showinfo(
                "Mark Peaks First",
                "Turn on '🎯 Peak Selection', then right-click on the plot to mark the "
                "peaks you want to match. Then run 'Match by Selected Peaks (ROD)'.")
            return
        try:
            tolerance = float(self.ent_rod_match_tol.get().strip())
        except ValueError:
            tolerance = 12.0
        exp_peaks = list(self.peak_guesses)

        if self.rod_mode_var.get() == "online":
            self._rod_match_online(exp_peaks, tolerance)
            return

        pool = self._rod_active_entries()
        if not pool:
            messagebox.showinfo(
                "No ROD Library",
                "Peak matching in offline mode runs against baked libraries, which "
                "carry precomputed bands. Add a ROD .h5 library "
                "(build_rod_library.py) first, or switch Source to Online and give "
                "a narrowing query.")
            return
        q = self.ent_rod_query.get().strip().lower()

        scored = []
        for e in pool:
            if q and q not in self._rod_haystack(e):
                continue
            if e['peaks'].size == 0:
                continue
            score, avg, matched = peak_match_score(e['peaks'], exp_peaks, tolerance)
            if matched > 0:
                scored.append({'score': score, 'avg': avg, 'matched': matched,
                               'name': e['name'], 'id': e['id'], 'group': e['group'],
                               'source': e.get('source', 'ROD'),
                               'lib_path': e.get('lib_path'),
                               'formula': e.get('formula', '') or e.get('collection', ''),
                               'url': e.get('url'), 'cod_url': e.get('cod_url', '')})
        scored.sort(key=lambda t: (-t['score'], t['avg']))
        self._show_rod_match_results(scored[:100], len(exp_peaks), tolerance)

    # ---------- SDBS (external lookup) ----------

    def sdbs_lookup(self):
        """Opens the SDBS search page in the user's browser.

        No scraping and no automated download: SDBS's terms prohibit automated
        retrieval and impose a daily access limit, so the user drives it. The
        query falls back to the ROD search box, then to the selected layer's
        name, so the common case is one click.
        """
        query = self.ent_sdbs_query.get().strip()
        if not query:
            query = self.ent_rod_query.get().strip()
        if not query:
            query = self._first_reference_name() or ""

        webbrowser.open_new_tab(SDBS_SEARCH_URL)
        if query:
            self.root.clipboard_clear()
            self.root.clipboard_append(query)
            messagebox.showinfo(
                "SDBS Opened",
                f"SDBS search page opened in your browser.\n\n"
                f"'{query}' has been copied to your clipboard — paste it into the "
                f"compound name field.\n\n"
                f"SDBS does not permit automated downloads, so save the Raman "
                f"spectrum yourself (JCAMP-DX), then bring it in with "
                f"📂 Load Spectra.")
        else:
            messagebox.showinfo(
                "SDBS Opened",
                "SDBS search page opened in your browser. Search by compound "
                "name, formula, CAS number or SDBS number.\n\n"
                "Save the Raman spectrum as JCAMP-DX, then bring it in with "
                "📂 Load Spectra.")

    def _first_reference_name(self):
        """Best-guess compound name from the current plot, for the SDBS box."""
        for key, data in self.active_datasets.items():
            if key.startswith("__ref_") and data.get('rruff_name'):
                return data['rruff_name']
        for key, data in self.active_datasets.items():
            if not key.startswith("__fit_") and not key.startswith("__ref_"):
                return data.get('label')
        return None

    def _rod_stop_scan(self):
        self._rod_scan_cancel = True
        self.rod_status_var.set("ROD online: stopping …")

    def _rod_set_scanning(self, scanning):
        if scanning:
            self.btn_rod_match.config(text="⏹ Stop scanning", command=self._rod_stop_scan)
        else:
            self.btn_rod_match.config(text="🎯 Match by Selected Peaks (ROD)",
                                      command=self.rod_match_by_peaks)

    def _rod_match_online(self, exp_peaks, tolerance):
        """Bounded online match: narrow with a query, download only that subset,
        detect bands, and rank.

        Matching all of ROD live would mean downloading the whole database, so a
        narrowing query is required and the number of spectra actually fetched
        is capped. Downloads land in the shared cache, so re-running a scan over
        the same subset (or a wider tolerance) costs nothing.
        """
        query = self.ent_rod_query.get().strip()
        field = self.combo_rod_field.get() or "auto"
        if not query:
            messagebox.showinfo(
                "Narrow the Search First",
                "Online matching needs a narrowing query, because ranking every "
                "entry would mean downloading all of ROD.\n\n"
                "Type a mineral or compound name, a formula, or an element list "
                "in the search box and set 'as' accordingly — for example:\n"
                "    elements   Ti,O\n"
                "    formula    C8 H10 N4 O2\n"
                "    text       calcite\n\n"
                "Then run the match again.")
            return
        try:
            cap = max(1, int(float(self.ent_rod_scan_cap.get().strip())))
        except ValueError:
            cap = 100

        self.rod_status_var.set("ROD online: searching …")

        def worker():
            try:
                hits = rod_search_online(query, field=field, limit=10000)
            except Exception as e:                           # noqa: BLE001
                self.root.after(0, lambda e=e: self.rod_status_var.set(
                    f"ROD online search failed: {e}"))
                return
            self.root.after(0, lambda: self._rod_confirm_and_scan(
                hits, cap, exp_peaks, tolerance))

        threading.Thread(target=worker, daemon=True).start()

    def _rod_confirm_and_scan(self, hits, cap, exp_peaks, tolerance):
        if not hits:
            self.rod_status_var.set("ROD online: nothing matched that query.")
            messagebox.showinfo("No Entries",
                                "That query matched no ROD entries, so there is "
                                "nothing to download and rank.")
            return
        total = len(hits)
        if total > cap:
            proceed = messagebox.askyesno(
                "Narrow Further?",
                f"That query matches {total} ROD entries, more than the "
                f"cap of {cap}.\n\n"
                f"Scan the first {cap}? Choose No to narrow the query instead "
                f"(or raise 'max online' next to the search box).")
            if not proceed:
                self.rod_status_var.set(
                    f"ROD online: {total} entries matched — narrow the query or raise the cap.")
                return
            hits = hits[:cap]

        self._rod_scan_cancel = False
        self._rod_set_scanning(True)
        n = len(hits)

        def worker():
            scored, failed = [], 0
            for i, hit in enumerate(hits, 1):
                if self._rod_scan_cancel:
                    break
                try:
                    x, y, meta = rod_fetch_spectrum(hit['id'])
                    ref_peaks = detect_reference_peaks(x, y)
                    score, avg, matched = peak_match_score(ref_peaks, exp_peaks, tolerance)
                    if matched > 0:
                        scored.append({
                            'score': score, 'avg': avg, 'matched': matched,
                            'name': hit.get('name') or meta.get('name') or f"ROD {hit['id']}",
                            'id': hit['id'], 'group': None, 'lib_path': None,
                            'formula': hit.get('formula') or meta.get('formula', ''),
                            'url': hit.get('url') or meta.get('url'),
                            'cod_url': meta.get('cod_url', ''),
                        })
                except Exception:                            # noqa: BLE001
                    failed += 1
                if i % 5 == 0 or i == n:
                    self.root.after(0, lambda i=i: self.rod_status_var.set(
                        f"ROD online: scanned {i}/{n} …"))

            scored.sort(key=lambda t: (-t['score'], t['avg']))
            stopped = self._rod_scan_cancel

            def done():
                self._rod_set_scanning(False)
                self._rod_scan_cancel = False
                self._show_rod_match_results(scored[:100], len(exp_peaks), tolerance)
                note = " (stopped early)" if stopped else ""
                if failed:
                    note += f", {failed} unavailable"
                if scored:
                    self.rod_status_var.set(
                        f"ROD online: {len(scored)} of {n} scanned entr(ies) matched{note}.")
            self.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _show_rod_match_results(self, scored, n_exp, tolerance):
        if not scored:
            self.rod_status_var.set("ROD: no references matched the marked peaks.")
            messagebox.showinfo("No Matches",
                                "No ROD reference had bands near your marked peaks.\n"
                                "Try a larger match tolerance or different peaks.")
            return
        self.rod_status_var.set(f"ROD: {len(scored)} candidate(s) ranked (tol ±{tolerance:g} cm⁻¹).")

        pop = tk.Toplevel(self.root)
        pop.title("ROD Search & Match — Candidate Ranking")
        pop.geometry("720x360")
        pop.transient(self.root)
        pop.grab_set()
        ttk.Label(pop, text=f"Ranked by alignment of ROD bands with your {n_exp} marked peak(s) "
                            f"(±{tolerance:g} cm⁻¹). Double-click a row to open its ROD page.",
                  font=("Helvetica", 9, "bold")).pack(pady=6, padx=8, anchor="w")

        frame = ttk.Frame(pop)
        frame.pack(fill="both", expand=True, padx=8, pady=4)
        scroll = ttk.Scrollbar(frame)
        scroll.pack(side="right", fill="y")
        tree = ttk.Treeview(frame, columns=("Score", "Name", "ID", "Formula", "Matched"),
                            show="headings", yscrollcommand=scroll.set, height=10,
                            selectmode="extended")
        for col, head, width, anchor in (
                ("Score", "Match Score", 100, "center"),
                ("Name", "Name / Mineral", 220, "w"),
                ("ID", "ROD ID", 100, "center"),
                ("Formula", "Formula", 140, "w"),
                ("Matched", "Peaks Matched", 110, "center")):
            tree.heading(col, text=head)
            tree.column(col, width=width, anchor=anchor)
        tree.pack(fill="both", expand=True)
        scroll.config(command=tree.yview)

        row_map = {}
        for rec in scored:
            iid = tree.insert("", "end", values=(f"{rec['score']:.0f}%", rec['name'], rec['id'],
                                                 rec.get('formula', ''), f"{rec['matched']}/{n_exp}"))
            row_map[iid] = rec

        def open_pages(event=None):
            opened = 0
            for iid in tree.selection():
                rec = row_map.get(iid)
                url = rec and (rec.get('url') or rod_url(rec.get('id')))
                if url:
                    webbrowser.open_new_tab(url)
                    opened += 1
            if opened == 0:
                messagebox.showinfo("Nothing Selected", "Select a row, then open its ROD page.")
        tree.bind("<Double-1>", open_pages)

        def open_cod_pages():
            opened = 0
            for iid in tree.selection():
                rec = row_map.get(iid)
                if rec and rec.get('cod_url'):
                    webbrowser.open_new_tab(rec['cod_url'])
                    opened += 1
            if opened == 0:
                messagebox.showinfo("No Linked Structure",
                                    "None of the selected entries has a cross-linked "
                                    "COD structure.")

        def overlay_chosen():
            sel = tree.selection()
            if not sel:
                return
            pop.destroy()
            self.save_to_history()
            added = 0
            for iid in sel:
                hit = row_map[iid]
                try:
                    x, y, label, url = self._rod_read_xy(hit)
                    if len(x) == 0:
                        continue
                except Exception:
                    continue
                self._add_reference(x, y, label, f"__ref_rod_{hit['id']}_match",
                                    rruff_name=hit['name'], rruff_id=hit['id'], ref_url=url)
                added += 1
            if added:
                self.replot_and_refresh_canvas()
                self.rod_status_var.set(f"ROD: overlaid {added} matched reference(s).")

        btn_row = ttk.Frame(pop)
        btn_row.pack(pady=8)
        ttk.Button(btn_row, text="🔗 Open ROD Page(s)", command=open_pages).pack(side="left", padx=4)
        ttk.Button(btn_row, text="⬡ Open Linked COD Structure(s)",
                   command=open_cod_pages).pack(side="left", padx=4)
        ttk.Button(btn_row, text="➕ Overlay Selected Match(es)",
                   command=overlay_chosen).pack(side="left", padx=4)

    def clear_canvas(self):
        if self.active_datasets:
            self.save_to_history()
        self.active_datasets.clear(); self.clear_fitted_artists(); self.cursor_line = None
        self.ax.clear(); self.configure_axis_labels(); self.refresh_checkbox_targets_panel(); self.canvas.draw()
        self.fitting_mode_active = False; self.normalization_mode_active = False
        self.adjust_mode = None; self.adjust_armed = False; self.line_map = {}
        self.btn_adjust_offset.config(text="➕ Offset"); self.btn_adjust_scale.config(text="✖ Scale")
        self.refresh_adjust_targets()
        self.btn_fit_toggle.config(text="🎯 Peak Selection: OFF"); self.btn_normalize_toggle.config(text="⚖️ Normalize to Peak"); self.btn_run_fit.config(state="disabled")
        self.status_var.set("Active spectra loaded: 0"); self.cursor_var.set("Cursor Position: Raman shift = --")


if __name__ == "__main__":
    root = tk.Tk()
    app = RamanPlotterGUI(root)
    root.mainloop()
