import os
import io
import re
import zipfile
import threading
import json
import urllib.parse
import urllib.request
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

# Embed Matplotlib into Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks

# HDF5 support for the optional RRUFF .h5 reference library. Optional so the app still launches
# if h5py is not installed; the user is told how to add it on first use.
try:
    import h5py
    H5_AVAILABLE = True
except ImportError:
    H5_AVAILABLE = False

# ==========================================
# GLOBAL CONFIGURATIONS & CONSTANTS
# ==========================================
VERSION_TAG = "ftir-v2026.07.25.10"

# RRUFF reference database (open FTIR spectra of minerals).
# Data are distributed as per-quality zip archives of two-column .txt files.
RRUFF_BASE_URL = "https://www.rruff.net/zipped_data_files/infrared/"
RRUFF_DATASETS = [
    "infrared", "processed", "excellent", "fair", "poor",
]
RRUFF_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".ftir_plotter_rruff")

# Open Specy (https://openspecy.org, CC-BY) -- polymers, microplastics,
# pigments and organics. Its infrared holdings (FLOPP, FLOPP-e, Primpke,
# Cabernard) are if anything larger than its Raman ones. Used offline only,
# via .h5 libraries baked by build_openspecy_library.py --only ftir; the same
# files load in the browser build, which cannot fetch anything cross-origin.
OPENSPECY_URL = "https://openspecy.org"

# NIST Chemistry WebBook, SRD 69 -- ~16k IR spectra served as JCAMP-DX.
# Online only, on demand, cached locally: the data *compilation* is
# copyrighted by the U.S. Secretary of Commerce, so this app fetches
# individual spectra for your own use and never bakes a redistributable
# library out of them. NIST explicitly welcomes deep links to species pages.
NIST_BASE_URL = "https://webbook.nist.gov/cgi/cbook.cgi"
NIST_SPECIES_URL = NIST_BASE_URL + "?ID={sid}&Units=SI"
NIST_JCAMP_URL = NIST_BASE_URL + "?JCAMP={sid}&Index={index}&Type=IR"
NIST_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".ftir_plotter_nist")
NIST_UA = ("Mozilla/5.0 (compatible; FTIR-Plotter/1.0; "
           "+https://webbook.nist.gov/chemistry/)")

# SDBS -- Spectral Database for Organic Compounds (AIST, Japan), ~34k
# compounds including FT-IR. Deliberately NOT automated: SDBS prohibits
# automated retrieval and rate-limits access, so the app only opens the
# search page and lets the user download by hand. The resulting JCAMP-DX
# file imports through the normal file open dialog.
SDBS_SEARCH_URL = "https://sdbs.db.aist.go.jp/sdbs/cgi-bin/cre_index.cgi"


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
    Parses an FTIR spectrum file and returns a list of spectra, each as a dict
    {'x': wavenumber array, 'y': absorbance array, 'label': str}.

    Supported:
      * .jdx / .dx / .jcm  -> JCAMP-DX (e.g. Thermo Nicolet export)
      * .txt / .csv / .dat / .asc -> two-column (Wavenumber, Absorbance),
                         including RRUFF reference files with '##' headers
    """
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    base_id = os.path.splitext(filename)[0]

    if ext in ('.jdx', '.dx', '.jcm'):
        return _load_jcamp(file_path, base_id)
    elif ext in ('.txt', '.csv', '.dat', '.asc', '.spc'):
        return _load_two_column(file_path, base_id)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# --- JCAMP-DX ---------------------------------------------------------------
# One parser for every JCAMP source the app touches: instrument exports
# (Thermo Nicolet, Bruker), NIST WebBook downloads, SDBS downloads, and ROD.
# Handles both data forms and ASDF compression, which vendors and NIST both
# use and a naive whitespace split silently mis-reads.

_ASDF_SQZ = {'@': '0', 'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5',
             'F': '6', 'G': '7', 'H': '8', 'I': '9',
             'a': '-1', 'b': '-2', 'c': '-3', 'd': '-4', 'e': '-5',
             'f': '-6', 'g': '-7', 'h': '-8', 'i': '-9'}
_ASDF_DIF = {'%': '0', 'J': '1', 'K': '2', 'L': '3', 'M': '4', 'N': '5',
             'O': '6', 'P': '7', 'Q': '8', 'R': '9',
             'j': '-1', 'k': '-2', 'l': '-3', 'm': '-4', 'n': '-5',
             'o': '-6', 'p': '-7', 'q': '-8', 'r': '-9'}
_ASDF_DUP = {'S': '1', 'T': '2', 'U': '3', 'V': '4', 'W': '5', 'X': '6',
             'Y': '7', 'Z': '8', 's': '9'}


def _asdf_tokenize(line):
    """Splits a JCAMP data line into (kind, text) tokens.
    kind is 'num' (AFFN/PAC), 'dif' or 'dup'."""
    toks, cur, kind = [], '', None

    def flush():
        nonlocal cur, kind
        if cur not in ('', '-', '+'):
            toks.append((kind or 'num', cur))
        cur, kind = '', None

    for ch in line:
        if ch in _ASDF_SQZ:
            flush(); cur, kind = _ASDF_SQZ[ch], 'num'
        elif ch in _ASDF_DIF:
            flush(); cur, kind = _ASDF_DIF[ch], 'dif'
        elif ch in _ASDF_DUP:
            flush(); cur, kind = _ASDF_DUP[ch], 'dup'
        elif ch in '+-':
            flush(); cur, kind = ('' if ch == '+' else '-'), 'num'
        elif ch.isdigit() or ch == '.':
            if kind is None:
                kind = 'num'
            cur += ch
        elif ch in 'eE' and cur and (cur[-1].isdigit() or cur[-1] == '.'):
            cur += ch                      # exponent inside an AFFN number
        else:
            flush()
    flush()
    return toks


def _asdf_expand(tokens):
    """Reconstructs the Y values of one line. Returns (values, ended_in_dif)."""
    ys, last_dif, ended_dif = [], None, False
    for kind, text in tokens:
        try:
            val = float(text)
        except ValueError:
            continue
        if kind == 'dup':
            if not ys:
                continue
            for _ in range(int(val) - 1):
                ys.append(ys[-1] + last_dif if last_dif is not None else ys[-1])
        elif kind == 'dif':
            if not ys:
                continue
            last_dif = val
            ys.append(ys[-1] + val)
            ended_dif = True
        else:
            ys.append(val)
            last_dif, ended_dif = None, False
    return ys, ended_dif


def parse_jcamp_dx(text):
    """Parses a JCAMP-DX spectrum. Returns (x, y, header).

    Supports:
      * ##XYDATA=(X++(Y..Y))   equidistant ordinates, AFFN/PAC/SQZ/DIF/DUP
      * ##XYPOINTS=(XY..XY)    explicit point pairs (ROD, some SDBS exports)
    """
    header, data_lines = {}, []
    block, collecting = None, False
    for line in text.splitlines():
        m = re.match(r'^\s*##\s*([^=]+?)\s*=\s*(.*)$', line)
        if m:
            key, val = m.group(1).strip().upper(), m.group(2).strip()
            header[key] = val
            if key in ('XYDATA', 'XYPOINTS', 'PEAK TABLE', 'DATA TABLE'):
                block, collecting = (key, val.upper()), True
            else:
                collecting = False      # trailing headers close the data run
            continue
        if collecting and line.strip():
            data_lines.append(line)

    if not data_lines or block is None:
        raise ValueError("No ##XYDATA / ##XYPOINTS block found in JCAMP file.")

    def _num(k, default=None):
        try:
            return float(str(header[k]).replace('D', 'E'))
        except (KeyError, ValueError, TypeError):
            return default

    xfactor = _num('XFACTOR', 1.0) or 1.0
    yfactor = _num('YFACTOR', 1.0) or 1.0
    kind, form = block

    if kind == 'XYPOINTS' or 'XY..XY' in form:
        nums = [float(t.replace('D', 'E').replace('d', 'e'))
                for t in re.findall(r'[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?',
                                    " ".join(data_lines))]
        if len(nums) % 2:
            nums = nums[:-1]
        if len(nums) < 4:
            raise ValueError("Too few data points in JCAMP file.")
        arr = np.asarray(nums, dtype=float).reshape(-1, 2)
        x, y = arr[:, 0] * xfactor, arr[:, 1] * yfactor
    else:
        ys, prev_dif = [], False
        for raw in data_lines:
            toks = _asdf_tokenize(raw)
            if not toks:
                continue
            line_ys, ended_dif = _asdf_expand(toks[1:])   # toks[0] is the abscissa
            if prev_dif and ys and line_ys and abs(line_ys[0] - ys[-1]) < 1e-9:
                line_ys = line_ys[1:]      # DIF Y-check value, not a data point
            ys.extend(line_ys)
            prev_dif = ended_dif
        if len(ys) < 2:
            raise ValueError("No numeric data parsed from JCAMP file.")
        y = np.asarray(ys, dtype=float) * yfactor
        n = len(y)
        firstx, lastx, deltax = _num('FIRSTX'), _num('LASTX'), _num('DELTAX')
        # FIRSTX/LASTX are already in final units; only raw abscissas take XFACTOR.
        if firstx is not None and lastx is not None and n > 1:
            x = np.linspace(firstx, lastx, n)
        elif firstx is not None and deltax is not None:
            x = firstx + np.arange(n) * deltax
        else:
            x = np.arange(n, dtype=float)

    order = np.argsort(x)                  # IR is usually written high->low
    x, y = x[order], y[order]
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 2:
        raise ValueError("Fewer than 2 finite points in JCAMP file.")
    return x, y, header


def _load_jcamp(file_path, base_id):
    """Loads a JCAMP-DX spectrum from disk (instrument export, NIST, SDBS, ROD).
    Returns [{'x': wavenumber, 'y': absorbance, 'label': str}]."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    x, y, header = parse_jcamp_dx(text)
    label = (header.get('TITLE') or '').strip() or base_id
    dtype = (header.get('DATA TYPE') or '').strip().upper()
    if dtype and 'INFRARED' not in dtype and 'IR' not in dtype:
        label = f"{label} [{dtype.title()}]"
    return [{'x': x, 'y': y, 'label': label}]


def _load_two_column(file_path, base_id):
    """Parses a two-column (Wavenumber, Absorbance) text file.

    Handles plain FTIR .txt exports and RRUFF reference files, whose
    metadata lines start with '##' (e.g. ##NAMES=Quartz, ##RRUFFID=R040031).
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    x, y, label = _parse_two_column_text(text, base_id)
    if len(x) == 0:
        raise ValueError("Could not parse two numeric columns (Wavenumber, Intensity).")
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
        # RRUFF filenames look like: Quartz__R040031__FTIR__..._532.txt
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
                'url': _decode(a.get('url', '')),
                'quality': _decode(a.get('quality', '')),
                'peaks': peaks,
            })
    if not entries:
        raise ValueError("Library contains no spectra.")
    return {'path': path, 'entries': entries}


def rruff_url(name, rid, stored=None):
    """Link to the RRUFF page for a sample (stored URL, or by ID / mineral name)."""
    if stored:
        return stored
    if rid and re.match(r'^R\d+', str(rid)):
        return f"https://rruff.info/{rid}"
    if name:
        return "https://rruff.info/" + str(name).strip().lower()
    return None


# ==========================================
# BAKED REFERENCE LIBRARIES (.h5)
# ==========================================

def load_speclib_h5(path):
    """Reads a baked reference library .h5 (build_openspecy_library.py, or the
    ROD/COD builders -- they all share one schema).

    /spectra/<id> with x,y datasets and precomputed peaks, so matching is
    instant. The `source` attribute drives labels and links. Peaks come from
    the dataset when present, else the attribute. x/y stay on disk and are read
    lazily when a reference is actually overlaid.
    """
    if not H5_AVAILABLE:
        raise ImportError("Reading .h5 libraries requires 'h5py' (pip install h5py).")
    entries = []
    with h5py.File(path, 'r') as f:
        if 'spectra' not in f:
            raise ValueError("Not a reference library file ('spectra' group missing).")
        file_source = _decode(f.attrs.get('source', '')) or 'Library'
        technique = _decode(f.attrs.get('technique', ''))
        sp = f['spectra']
        for gname in sp:
            g = sp[gname]
            a = g.attrs
            if 'peaks' in g:
                peaks = np.asarray(g['peaks'][:], dtype=float)
            elif 'peaks' in a:
                peaks = np.asarray(a['peaks'], dtype=float)
            else:
                peaks = np.array([])
            source = _decode(a.get('source', '')) or file_source
            sid = (_decode(a.get('rruff_id', '')) or _decode(a.get('rod_id', ''))
                   or gname)
            entries.append({
                'group': gname,
                'name': _decode(a.get('name', gname)),
                'id': sid,
                'source': source,
                'formula': _decode(a.get('formula', '')),
                'collection': _decode(a.get('collection', '')),
                'spectrum_type': _decode(a.get('spectrum_type', '')),
                'url': _decode(a.get('url', '')),
                'peaks': peaks,
            })
    if not entries:
        raise ValueError("Library contains no spectra.")
    return {'path': path, 'source': file_source, 'technique': technique,
            'entries': entries}


# ==========================================
# NIST CHEMISTRY WEBBOOK (online, on demand)
# ==========================================

def _nist_http_get(url, timeout=45, retries=2, binary=False):
    last = None
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": NIST_UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            return raw if binary else raw.decode("utf-8", errors="replace")
        except Exception as e:                     # noqa: BLE001 - retry anything
            last = e
    raise last


def nist_species_url(sid):
    return NIST_SPECIES_URL.format(sid=sid)


def nist_search(query, field="auto", limit=100, timeout=45):
    """Searches the WebBook for species that have IR spectra.

    `field` is 'name', 'formula', 'cas', 'id', or 'auto'. A search that matches
    exactly one species redirects straight to its page, so we handle both the
    hit-list and single-species cases.

    Returns [{'name', 'id', 'formula', 'url'}]. Note these are *species*, not
    spectra -- a species commonly has several IR spectra (phases, instruments,
    resolutions), enumerated separately by nist_species_spectra().
    """
    q = str(query or "").strip()
    if not q:
        return []
    if field == "auto":
        if re.fullmatch(r"C?\d{2,7}-?\d{0,2}-?\d?", q) and any(c.isdigit() for c in q):
            field = "cas" if "-" in q else "id"
        elif re.fullmatch(r"[A-Z][A-Za-z0-9]*", q) and any(c.isdigit() for c in q):
            field = "formula"
        else:
            field = "name"

    if field == "id":
        sid = q if q.upper().startswith("C") else "C" + q
        params = {"ID": sid, "Units": "SI", "Mask": "80"}
    elif field == "cas":
        params = {"ID": "C" + re.sub(r"\D", "", q), "Units": "SI", "Mask": "80"}
    elif field == "formula":
        params = {"Formula": q, "NoIon": "on", "Units": "SI", "cIR": "on"}
    else:
        params = {"Name": q, "Units": "SI", "cIR": "on"}

    html = _nist_http_get(f"{NIST_BASE_URL}?{urllib.parse.urlencode(params)}",
                          timeout=timeout)
    return _nist_parse_search(html, limit=limit)


def _nist_parse_search(html, limit=100):
    """Pulls species out of a WebBook search response.

    Deliberately regex-based rather than a DOM parse: the WebBook markup is
    stable, simple, and this keeps the app dependency-free.
    """
    hits, seen = [], set()

    # Single-species page: <h1 id="Top">Name</h1> plus an ID in the links.
    m_title = re.search(r'<h1[^>]*id="Top"[^>]*>(.*?)</h1>', html, re.S | re.I)
    m_id = re.search(r'[?&](?:ID|Struct|Str2File)=(C\d+)', html)
    if m_title and m_id and 'Search Results' not in m_title.group(1):
        name = _strip_tags(m_title.group(1))
        formula = ""
        mf = re.search(r'Formula</a>[^:]*:</strong>\s*([^<\s]+)', html, re.I)
        if mf:
            formula = _strip_tags(mf.group(1))
        if name:
            return [{'name': name, 'id': m_id.group(1), 'formula': formula,
                     'url': nist_species_url(m_id.group(1))}]

    # Hit list: <a href="/cgi/cbook.cgi?ID=C71432&...">Name</a>
    for m in re.finditer(r'href="[^"]*[?&]ID=(C\d+)[^"]*"[^>]*>(.*?)</a>', html, re.S | re.I):
        sid, name = m.group(1), _strip_tags(m.group(2))
        if not name or sid in seen or len(name) > 120:
            continue
        seen.add(sid)
        hits.append({'name': name, 'id': sid, 'formula': '',
                     'url': nist_species_url(sid)})
        if len(hits) >= limit:
            break
    return hits


def _strip_tags(s):
    s = re.sub(r'<[^>]+>', '', s or '')
    s = (s.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
          .replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' '))
    return " ".join(s.split())


def nist_species_spectra(sid, timeout=45):
    """Lists the IR spectra available for one species.

    Returns [{'index', 'description'}] -- the description carries phase,
    instrument and resolution, which is what distinguishes the (often many)
    spectra of a single compound.
    """
    html = _nist_http_get(nist_species_url(sid) + "&Mask=80", timeout=timeout)
    out, seen = [], set()
    for m in re.finditer(
            r'href="[^"]*Type=IR-SPEC&(?:amp;)?Index=(\d+)[^"]*"[^>]*>(.*?)</a>',
            html, re.S | re.I):
        idx = int(m.group(1))
        if idx in seen:
            continue
        seen.add(idx)
        out.append({'index': idx, 'description': _strip_tags(m.group(2))})
    if not out:
        out = [{'index': 0, 'description': 'IR spectrum'}]
    return sorted(out, key=lambda d: d['index'])


def nist_fetch_ir(sid, index=0, timeout=45, use_cache=True):
    """Downloads (and caches) one NIST IR spectrum. Returns (x, y, header).

    Cached under ~/.ftir_plotter_nist/ so repeat use costs nothing and NIST
    isn't hit twice for the same spectrum.
    """
    os.makedirs(NIST_CACHE_DIR, exist_ok=True)
    path = os.path.join(NIST_CACHE_DIR, f"{sid}_{index}.jdx")
    text = None
    if use_cache and os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    if text is None:
        text = _nist_http_get(NIST_JCAMP_URL.format(sid=sid, index=index),
                              timeout=timeout)
        if "##" not in text:
            raise ValueError(f"NIST returned no JCAMP data for {sid} index {index}")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except Exception:
            pass
    return parse_jcamp_dx(text)


# ==========================================
# GUI & EMBEDDED PLOTTING INTERFACE
# ==========================================

class FTIRPlotterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FTIR Spectra Analysis Toolkit")
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

        # Interactive wheel-adjust (offset / scale)
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

        ttk.Label(sidebar_frame, text="🔬 FTIR Spectra Analyzer", font=("Helvetica", 12, "bold")).pack(side="top", anchor="w", pady=(0, 10))

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
        ttk.Button(shift_row, text="📐 Shift Wavenumber", command=self.apply_shift).pack(side="left", fill="x", expand=True)
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

        # Interactive wheel-adjust panel (add / multiply)
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
        # --- Unified Reference Databases panel -------------------------------
        # One search box over every enabled source: the RRUFF infrared folder /
        # dataset / .h5 library, any number of baked .h5 libraries (Open Specy
        # and friends), and NIST WebBook when its online mode is on. Results
        # merge into one list with a source tag, and Match ranks across all
        # enabled offline sources in a single pass.
        panel_db = ttk.LabelFrame(sidebar_frame, text=" 🔬 Reference Databases ", padding=(8, 6))
        panel_db.pack(side="top", fill="x", pady=5)

        db_search_row = ttk.Frame(panel_db)
        db_search_row.pack(fill="x", pady=(0, 3))
        self.ent_db_query = ttk.Entry(db_search_row)
        self.ent_db_query.pack(side="left", fill="x", expand=True)
        self.ent_db_query.bind("<Return>", lambda e: self.db_run_search())
        ttk.Button(db_search_row, text="🔍", width=3, command=self.db_run_search).pack(side="right", padx=(3, 0))

        self.db_sources_frame = ttk.Frame(panel_db)
        self.db_sources_frame.pack(fill="x", pady=(0, 3))

        self.db_results_list = tk.Listbox(panel_db, height=5, exportselection=False)
        self.db_results_list.pack(fill="x", pady=(0, 3))
        self.db_results_list.bind("<Double-1>", lambda e: self.db_overlay_selected())

        db_act = ttk.Frame(panel_db)
        db_act.pack(fill="x", pady=(0, 2))
        ttk.Button(db_act, text="➕ Overlay", command=self.db_overlay_selected).pack(side="left", fill="x", expand=True)
        self.btn_db_match = ttk.Button(db_act, text="🎯 Match", command=self.db_match_by_peaks)
        self.btn_db_match.pack(side="left", fill="x", expand=True, padx=(4, 4))
        self.ent_db_match_tol = ttk.Entry(db_act, width=5, justify="center")
        self.ent_db_match_tol.insert(0, "12")
        self.ent_db_match_tol.pack(side="left")

        self.db_status_var = tk.StringVar(value="Add a library under Manage sources, then search.")
        ttk.Label(panel_db, textvariable=self.db_status_var, font=("Helvetica", 8),
                  foreground="#555555", wraplength=250).pack(anchor="w")

        db_toggle_row = ttk.Frame(panel_db)
        db_toggle_row.pack(fill="x", pady=(4, 0))
        ttk.Label(db_toggle_row, text="Manage sources", font=("Helvetica", 8),
                  foreground="#6c757d").pack(side="left")
        self.btn_db_manage = ttk.Button(db_toggle_row, text="▾", width=3, command=self.db_toggle_manage)
        self.btn_db_manage.pack(side="right")

        self.db_manage_frame = ttk.Frame(panel_db)
        self._db_manage_open = False
        mf = self.db_manage_frame

        ttk.Button(mf, text="📚 Add .h5 librar(ies)…", command=self.lib_add).pack(fill="x", pady=2)

        ttk.Separator(mf, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(mf, text="RRUFF (infrared)", font=("Helvetica", 8, "bold")).pack(anchor="w")
        ds_row = ttk.Frame(mf)
        ds_row.pack(fill="x", pady=2)
        ttk.Label(ds_row, text="Set:", font=("Helvetica", 8)).pack(side="left")
        self.combo_rruff_dataset = ttk.Combobox(ds_row, state="readonly", width=16, values=RRUFF_DATASETS)
        self.combo_rruff_dataset.current(0)
        self.combo_rruff_dataset.pack(side="left", fill="x", expand=True, padx=(3, 0))
        self.btn_rruff_download = ttk.Button(mf, text="⬇️ Download / Update Set", command=self.rruff_download_selected)
        self.btn_rruff_download.pack(fill="x", pady=2)
        ttk.Button(mf, text="📂 Use Local RRUFF Folder", command=self.rruff_pick_local_folder).pack(fill="x", pady=2)
        ttk.Button(mf, text="📚 Open RRUFF .h5 Library", command=self.rruff_open_library).pack(fill="x", pady=2)

        ttk.Separator(mf, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(mf, text="NIST WebBook (online)", font=("Helvetica", 8, "bold")).pack(anchor="w")
        nist_mode_row = ttk.Frame(mf)
        nist_mode_row.pack(fill="x", pady=2)
        self.nist_mode_var = tk.StringVar(value="off")
        ttk.Radiobutton(nist_mode_row, text="Off", value="off", variable=self.nist_mode_var,
                        command=self.db_refresh).pack(side="left")
        ttk.Radiobutton(nist_mode_row, text="Include in search", value="on",
                        variable=self.nist_mode_var, command=self.db_refresh).pack(side="left", padx=(6, 0))
        nist_field_row = ttk.Frame(mf)
        nist_field_row.pack(fill="x", pady=2)
        ttk.Label(nist_field_row, text="as", font=("Helvetica", 8)).pack(side="left")
        self.combo_nist_field = ttk.Combobox(nist_field_row, state="readonly", width=8,
                                             values=("auto", "name", "formula", "cas", "id"))
        self.combo_nist_field.current(0)
        self.combo_nist_field.pack(side="left", padx=(3, 0))
        ttk.Label(nist_field_row, text="max:", font=("Helvetica", 8)).pack(side="left", padx=(8, 0))
        self.ent_nist_cap = ttk.Entry(nist_field_row, width=5)
        self.ent_nist_cap.insert(0, "25")
        self.ent_nist_cap.pack(side="left", padx=(3, 0))

        ttk.Separator(mf, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(mf, text="SDBS (organic compounds)", font=("Helvetica", 8, "bold")).pack(anchor="w")
        ttk.Label(mf, text="Opens SDBS in your browser; download an FT-IR spectrum there, "
                           "then load the .jdx normally. SDBS does not permit automated downloads.",
                  font=("Helvetica", 8), foreground="#555555", wraplength=240).pack(anchor="w")
        ttk.Button(mf, text="🔗 Look up in SDBS", command=self.sdbs_lookup).pack(fill="x", pady=2)

        # Aliases so the retained per-source code keeps working against the
        # shared widgets of the unified panel.
        self.ent_rruff_query = self.ent_db_query
        self.ent_lib_query = self.ent_db_query
        self.ent_nist_query = self.ent_db_query
        self.ent_sdbs_query = self.ent_db_query
        self.ent_lib_match_tol = self.ent_db_match_tol
        self.ent_match_tol = self.ent_db_match_tol
        self.rruff_status_var = self.db_status_var
        self.lib_status_var = self.db_status_var
        self.nist_status_var = self.db_status_var
        self.btn_rruff_match = self.btn_db_match
        self.btn_nist_match = self.btn_db_match
        self.rruff_results_list = self.db_results_list
        self.lib_results_list = self.db_results_list
        self.nist_results_list = self.db_results_list
        self.rruff_search_hits = []
        self.lib_search_hits = []
        self.nist_hits = []
        self.db_hits = []
        self.db_disabled = set()
        self.spec_libs = []
        self.lib_frame = self.db_sources_frame
        self.lib_cfg_path = os.path.join(os.path.expanduser("~"), ".ftir_plotter_libraries.json")
        self._nist_scan_cancel = False
        self.rruff_local_dir = None
        self.rruff_lib = None
        self._lib_load_config()
        self.db_refresh()
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

        self.cursor_var = tk.StringVar(value="Cursor Position: Wavenumber = --")
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
        self.ax.set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=10, fontweight='bold')
        self.ax.set_ylabel("Absorbance", fontsize=10, fontweight='bold')
        self.ax.set_title("FTIR Spectra", fontsize=11, fontweight='bold', pad=8)
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
                'rruff_id': v.get('rruff_id'),
                'rruff_url': v.get('rruff_url')
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
                url = rruff_url(data.get('rruff_name'), data.get('rruff_id'), data.get('rruff_url')) if key.startswith("__ref_") else None
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
            self.cursor_var.set(f"Cursor Position: Wavenumber = {x:.2f} cm⁻¹")
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
            self.cursor_var.set("Cursor Position: Wavenumber = --")

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
        """Linearly shifts the FTIR-shift axis to calibrate a zero-offset."""
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
        self.status_var.set(f"Applied a rigid FTIR-shift calibration of {shift_val} cm⁻¹.")

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
        self.ax.relim(); self.ax.autoscale_view()
        # FTIR convention: wavenumber decreases left-to-right.
        lo, hi = sorted(self.ax.get_xlim())
        self.ax.set_xlim(hi, lo)
        self.canvas.draw()

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
                # FTIR bands are typically a few to tens of cm-1 wide
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
            title="Select FTIR Data Files",
            filetypes=[("FTIR Datasets", ("*.jdx", "*.dx", "*.jcm", "*.csv", "*.txt", "*.dat", "*.asc")),
                       ("JCAMP-DX", ("*.jdx", "*.dx", "*.jcm")),
                       ("Text / CSV / RRUFF", ("*.csv", "*.txt", "*.dat", "*.asc")),
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
        xmin, xmax = sorted(self.ax.get_xlim())  # axis is reversed for FTIR
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
                header = "Wavenumber (cm-1),Absorbance\n"
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
        """Superseded by db_update_status; also picks up new RRUFF sources."""
        self.db_refresh()

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

    def _add_reference(self, x, y, label, key, rruff_name=None, rruff_id=None, rruff_url_=None):
        """Scales a reference to the current data maximum and stores it."""
        data_keys = [k for k in self.active_datasets.keys()
                     if not k.startswith("__fit_") and not k.startswith("__ref_")]
        if data_keys and np.max(y) > 0:
            max_scale = max(np.max(self.active_datasets[k]['intensities']) for k in data_keys)
            y = (y / np.max(y)) * max_scale
        self.active_datasets[key] = {'angles': x, 'intensities': y, 'label': label,
                                     'rruff_name': rruff_name, 'rruff_id': rruff_id,
                                     'rruff_url': rruff_url_}

    def rruff_run_search(self):
        query = self.ent_rruff_query.get().strip()
        self.rruff_results_list.delete(0, tk.END)
        self.rruff_search_hits = []
        if self.rruff_lib:
            q = query.lower()
            hits = [{'name': e['name'], 'id': e['id'], 'group': e['group'], 'url': e.get('url')}
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
                url = rruff_url(h['name'], h.get('id'), h.get('url'))
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
            self._add_reference(x, y, label, key, rruff_name=hit['name'], rruff_id=hit['id'], rruff_url_=hit.get('url'))
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
                                   'name': e['name'], 'id': e['id'], 'group': e['group'], 'url': e.get('url')})
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
                url = rruff_url(rec['name'], rec.get('id'), rec.get('url'))
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
                self._add_reference(x, y, label, key, rruff_name=hit['name'], rruff_id=hit['id'], rruff_url_=hit.get('url'))
                added += 1
            if added:
                self.replot_and_refresh_canvas()
                self.rruff_status_var.set(f"RRUFF: overlaid {added} matched reference(s).")

        btn_row = ttk.Frame(pop)
        btn_row.pack(pady=8)
        ttk.Button(btn_row, text="🔗 Open RRUFF Page(s)", command=open_pages).pack(side="left", padx=4)
        ttk.Button(btn_row, text="➕ Overlay Selected Match(es)", command=overlay_chosen).pack(side="left", padx=4)

    # ---------- Unified reference-database panel ----------
    # Dispatches over the per-source machinery below. A "source" is the RRUFF
    # library / folder / cached set, or one baked .h5 library.

    def db_toggle_manage(self):
        if self._db_manage_open:
            self.db_manage_frame.pack_forget()
            self.btn_db_manage.config(text="▾")
        else:
            self.db_manage_frame.pack(fill="x", pady=(4, 0))
            self.btn_db_manage.config(text="▴")
        self._db_manage_open = not self._db_manage_open

    def db_source_list(self):
        """[{key, label, kind, count, entries}] for every loaded source."""
        out = []
        if self.rruff_lib:
            out.append({'key': 'rruff-lib', 'label': 'RRUFF library', 'kind': 'rruff',
                        'count': len(self.rruff_lib['entries']),
                        'entries': self.rruff_lib['entries']})
        elif self.rruff_local_dir:
            files = [f for f in os.listdir(self.rruff_local_dir) if f.lower().endswith('.txt')]
            out.append({'key': 'rruff-folder', 'label': 'RRUFF folder', 'kind': 'folder',
                        'count': len(files), 'entries': None})
        else:
            ds = self.combo_rruff_dataset.get()
            if rruff_is_cached(ds):
                n = len([f for f in os.listdir(rruff_dataset_dir(ds)) if f.lower().endswith('.txt')])
                out.append({'key': 'rruff-set', 'label': f'RRUFF {ds}', 'kind': 'folder',
                            'count': n, 'entries': None})
        for i, lib in enumerate(self.spec_libs):
            src = (lib['entries'][0].get('source') if lib['entries'] else '') or 'Library'
            out.append({'key': f'lib-{i}', 'label': f"{src} — {lib['name']}", 'kind': 'lib',
                        'count': len(lib['entries']), 'entries': lib['entries'], 'lib_index': i})
        if self.nist_mode_var.get() == 'online':
            out.append({'key': 'nist-online', 'label': 'NIST (online)', 'kind': 'online',
                        'count': 0, 'entries': None})
        return out

    def db_enabled_sources(self):
        return [s for s in self.db_source_list() if s['key'] not in self.db_disabled]

    def db_refresh(self):
        for child in self.db_sources_frame.winfo_children():
            child.destroy()
        sources = self.db_source_list()
        if not sources:
            ttk.Label(self.db_sources_frame, text="No sources loaded — see Manage sources.",
                      font=("Helvetica", 8, "italic"), foreground="#888888").pack(anchor="w")
        for s in sources:
            row = ttk.Frame(self.db_sources_frame)
            row.pack(fill="x", pady=1)
            var = tk.BooleanVar(value=s['key'] not in self.db_disabled)
            label = s['label'] + (f" · {s['count']:,}" if s['count'] else "")
            ttk.Checkbutton(row, text=label, variable=var,
                            command=lambda k=s['key'], v=var, s=s: self._db_toggle_source(k, v, s)
                            ).pack(side="left", anchor="w")
            if s['kind'] == 'lib':
                ttk.Button(row, text="❌", width=2,
                           command=lambda i=s['lib_index']: (self._lib_remove(i),
                                                             self.db_refresh())).pack(side="right")
            row._db_var = var          # keep a reference so Tk does not GC it
        self.db_update_status()

    def _db_toggle_source(self, key, var, source):
        if var.get():
            self.db_disabled.discard(key)
        else:
            self.db_disabled.add(key)
        if source['kind'] == 'lib':
            self.spec_libs[source['lib_index']]['active'] = var.get()
            self._lib_save_config()
        self.db_update_status()

    def db_update_status(self, msg=None):
        if msg:
            self.db_status_var.set(msg)
            return
        all_s = self.db_source_list()
        if not all_s:
            self.db_status_var.set("Add a library under Manage sources, then search.")
            return
        on = self.db_enabled_sources()
        total = sum(s['count'] for s in on)
        self.db_status_var.set(
            f"{len(on)}/{len(all_s)} source(s) enabled · {total:,} spectra. "
            f"Search, or mark peaks and Match.")

    @staticmethod
    def _db_haystack(kind, e):
        if kind == 'lib':
            return (f"{e.get('name','')} {e.get('id','')} {e.get('formula','')} "
                    f"{e.get('mineral','')} {e.get('collection','')}").lower()
        return f"{e.get('name','')} {e.get('id','')}".lower()

    @staticmethod
    def _db_tag(kind, e):
        if kind in ('rruff', 'folder'):
            return 'RRUFF'
        src = (e.get('source') if isinstance(e, dict) else None) or 'LIB'
        return 'SPECY' if src == 'OpenSpecy' else src.upper()[:6]

    def db_run_search(self):
        query = self.ent_db_query.get().strip()
        self.db_results_list.delete(0, tk.END)
        self.db_hits = []
        sources = self.db_enabled_sources()
        if not sources:
            messagebox.showinfo("No Sources",
                                "Enable a source, or add an .h5 library under Manage sources.")
            return
        q = query.lower()

        for s in sources:
            if s['kind'] == 'online':
                continue                      # handled asynchronously below
            if s['kind'] == 'folder':
                for name, rid, path in self._rruff_candidate_files(query):
                    self.db_hits.append({'kind': 'folder', 'tag': 'RRUFF',
                                         'rec': {'name': name, 'id': rid, 'path': path}})
                continue
            for e in s['entries']:
                if q and q not in self._db_haystack(s['kind'], e):
                    continue
                self.db_hits.append({'kind': s['kind'], 'tag': self._db_tag(s['kind'], e),
                                     'rec': e})
            if len(self.db_hits) >= 400:
                break

        self._db_render_hits()

        online = [s for s in sources if s['kind'] == 'online']
        if online and query:
            self.db_update_status("NIST: searching …")
            field = self.combo_nist_field.get() or "auto"

            def worker():
                try:
                    hits = nist_search(query, field=field)
                except Exception:                            # noqa: BLE001
                    hits = []
                self.root.after(0, lambda: self._db_append_online(hits))

            threading.Thread(target=worker, daemon=True).start()

    def _db_append_online(self, hits):
        for h in hits:
            self.db_hits.append({'kind': 'online', 'tag': 'NIST', 'rec': h})
        self._db_render_hits()

    def _db_render_hits(self):
        self.db_results_list.delete(0, tk.END)
        for h in self.db_hits:
            e = h['rec']
            extra = e.get('formula') or e.get('collection') or ''
            bits = [f"[{h['tag']}]", e.get('name', '')]
            if e.get('id'):
                bits.append(str(e['id']))
            if extra:
                bits.append(extra)
            self.db_results_list.insert(tk.END, "  ".join(b for b in bits if b))
        if self.db_hits:
            self.db_results_list.selection_set(0)
            self.db_update_status(f"{len(self.db_hits)} match(es). Select and overlay.")
        else:
            q = self.ent_db_query.get().strip()
            total = sum(s['count'] for s in self.db_enabled_sources())
            # "No matches." read like a dead button. Name the query, the size of
            # what was searched, and the fact that this filter also gates Match.
            self.db_update_status(
                f'No match for "{q}" in {total:,} entries — clear the box to see '
                f'everything (this filter also applies to Match).'
                if q else "The enabled sources are empty.")

    def _db_selected(self):
        sel = self.db_results_list.curselection()
        if not sel or sel[0] >= len(self.db_hits):
            return None
        return self.db_hits[sel[0]]

    def db_overlay_selected(self):
        h = self._db_selected()
        if not h:
            self.db_update_status("Select a search result first.")
            return
        rec, kind = h['rec'], h['kind']
        if kind == 'lib':
            hit = dict(rec)
            hit.setdefault('lib_path', rec.get('lib_path'))
            self.nist_hits = [hit]
            self.db_results_list_index = 0
            try:
                x, y = self._lib_read_xy(hit)
                label, url = self._lib_label(hit), hit.get('url')
            except Exception as e:                           # noqa: BLE001
                self.db_update_status(f"Could not read that reference — {e}")
                return
            self.save_to_history()
            self._add_reference(x, y, label, f"__ref_lib_{rec.get('id')}",
                                rruff_name=rec.get('name'), rruff_id=rec.get('id'), rruff_url_=url)
            self.replot_and_refresh_canvas()
            self.db_update_status(f"Overlaid {rec.get('name')}.")
            return
        if kind == 'online':
            self.nist_hits = [rec]
            self.db_results_list.selection_clear(0, tk.END)
            self._db_overlay_online(rec)
            return
        # RRUFF library entry or folder file
        try:
            x, y, label = self._read_reference_xy(rec)
        except Exception as e:                               # noqa: BLE001
            self.db_update_status(f"Could not read that reference — {e}")
            return
        self.save_to_history()
        self._add_reference(x, y, label, f"__ref_rruff_{rec.get('name')}_{rec.get('id')}",
                            rruff_name=rec.get('name'), rruff_id=rec.get('id'))
        self.replot_and_refresh_canvas()
        self.db_update_status(f"Overlaid {rec.get('name')}.")

    def _db_overlay_online(self, rec):
        self.db_update_status(f"NIST: downloading {rec['id']} …")

        def worker():
            try:
                x, y, _hdr = nist_fetch_ir(rec['id'], rec.get('index', 0))
                meta = {'url': rec.get('url') or nist_species_url(rec['id'])}
                err = None
            except Exception as e:                           # noqa: BLE001
                x = y = meta = None
                err = e

            def done():
                if err is not None:
                    self.db_update_status(f"NIST: could not fetch — {err}")
                    return
                name = rec.get('name') or meta.get('name')
                self.save_to_history()
                self._add_reference(x, y, f"NIST: {name} ({rec['id']})",
                                    f"__ref_nist_{rec['id']}",
                                    rruff_name=name, rruff_id=rec['id'],
                                    rruff_url_=meta.get('url'))
                self.replot_and_refresh_canvas()
                self.db_update_status(f"Overlaid {name}.")
            self.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def db_match_by_peaks(self):
        """Rank every enabled source against the marked bands in one pass."""
        if not self.peak_guesses:
            messagebox.showinfo(
                "Mark Peaks First",
                "Turn on '🎯 Peak Selection', then right-click on the plot to mark the "
                "peaks you want to match.")
            return
        sources = self.db_enabled_sources()
        if not sources:
            messagebox.showinfo("No Sources", "Enable or add at least one source first.")
            return
        try:
            tolerance = float(self.ent_db_match_tol.get().strip())
        except ValueError:
            tolerance = 12.0
        exp_peaks = list(self.peak_guesses)
        q = self.ent_db_query.get().strip().lower()

        scored, scanned = [], 0
        for s in sources:
            if s['kind'] == 'online':
                continue                       # bounded online scan stays separate
            if s['kind'] == 'folder':
                for name, rid, path in self._rruff_candidate_files(self.ent_db_query.get()):
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                            x, y, _ = _parse_two_column_text(fh.read(), name)
                        if len(x) < 5:
                            continue
                        peaks = detect_reference_peaks(x, y)
                    except Exception:                        # noqa: BLE001
                        continue
                    scanned += 1
                    score, avg, matched = peak_match_score(peaks, exp_peaks, tolerance)
                    if matched > 0:
                        scored.append({'score': score, 'avg': avg, 'matched': matched,
                                       'name': name, 'id': rid, 'tag': 'RRUFF',
                                       'kind': 'folder', 'path': path, 'detail': ''})
                continue
            for e in s['entries']:
                if q and q not in self._db_haystack(s['kind'], e):
                    continue
                peaks = e.get('peaks')
                if peaks is None or len(peaks) == 0:
                    continue
                scanned += 1
                score, avg, matched = peak_match_score(peaks, exp_peaks, tolerance)
                if matched > 0:
                    scored.append({'score': score, 'avg': avg, 'matched': matched,
                                   'name': e.get('name'), 'id': e.get('id'),
                                   'tag': self._db_tag(s['kind'], e), 'kind': s['kind'],
                                   'group': e.get('group'), 'lib_path': e.get('lib_path'),
                                   'url': e.get('url'), 'cod_url': e.get('cod_url', ''),
                                   'detail': e.get('formula') or e.get('collection') or ''})
        # Nothing scanned means the search box filtered every entry out, which is
        # a different problem from too tight a tolerance; "try a larger tolerance"
        # would send the user the wrong way.
        if not scanned and q:
            total = sum(src['count'] for src in sources)
            self.db_update_status(
                f'Nothing to match: "{q}" excludes all {total:,} entries.')
            messagebox.showinfo(
                "Nothing to Match",
                f'No reference matches the search box ("{q}"), so there was '
                f'nothing to rank.\n\nThe search text filters Match as well as the '
                f'result list. Clear the box to match against all {total:,} entries.')
            return
        scored.sort(key=lambda t: (-t['score'], t['avg']))
        self.db_update_status(
            f"Matched {len(scored)}/{scanned} references across "
            f"{len([s for s in sources if s['kind'] != 'online'])} source(s) (tol ±{tolerance:g}).")
        self._db_show_match_results(scored[:100], len(exp_peaks), tolerance)

    def _db_show_match_results(self, scored, n_exp, tolerance):
        if not scored:
            messagebox.showinfo(
                "No Matches",
                f"None of the {n_exp} marked peak(s) fell within "
                f"±{tolerance:g} cm⁻¹ of a reference band in the scanned "
                f"entries.\n\nTry a larger tolerance, or check the search box is "
                f"not filtering out the phases you expect.")
            return
        pop = tk.Toplevel(self.root)
        pop.title("Reference Databases — Candidate Ranking")
        pop.geometry("780x380")
        pop.transient(self.root)
        pop.grab_set()
        ttk.Label(pop, text=f"Ranked across every enabled source by alignment with your "
                            f"{n_exp} marked peak(s) (±{tolerance:g} cm⁻¹). "
                            f"Double-click a row to open its database page.",
                  font=("Helvetica", 9, "bold")).pack(pady=6, padx=8, anchor="w")

        frame = ttk.Frame(pop)
        frame.pack(fill="both", expand=True, padx=8, pady=4)
        scroll = ttk.Scrollbar(frame)
        scroll.pack(side="right", fill="y")
        tree = ttk.Treeview(frame, columns=("Score", "Src", "Name", "ID", "Detail", "Matched"),
                            show="headings", yscrollcommand=scroll.set, height=11,
                            selectmode="extended")
        for col, head, w, anchor in (("Score", "Match", 70, "center"),
                                     ("Src", "Source", 70, "center"),
                                     ("Name", "Name", 210, "w"),
                                     ("ID", "ID", 95, "center"),
                                     ("Detail", "Formula / Collection", 190, "w"),
                                     ("Matched", "Peaks", 70, "center")):
            tree.heading(col, text=head)
            tree.column(col, width=w, anchor=anchor)
        tree.pack(fill="both", expand=True)
        scroll.config(command=tree.yview)

        row_map = {}
        for rec in scored:
            iid = tree.insert("", "end", values=(f"{rec['score']:.0f}%", rec['tag'], rec['name'],
                                                 rec.get('id', ''), rec.get('detail', ''),
                                                 f"{rec['matched']}/{n_exp}"))
            row_map[iid] = rec

        def open_pages(event=None):
            opened = 0
            for iid in tree.selection():
                rec = row_map.get(iid)
                if not rec:
                    continue
                url = rec.get('url') or (rruff_url(rec['name'], rec.get('id'), rec.get('url'))
                                         if rec['tag'] == 'RRUFF' else None)
                if url:
                    webbrowser.open_new_tab(url)
                    opened += 1
            if opened == 0:
                messagebox.showinfo("Nothing Selected", "Select a row, then open its page.")
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
                    if hit['kind'] == 'lib':
                        x, y = self._lib_read_xy(hit)
                        label, url = self._lib_label(hit), hit.get('url')
                    else:
                        x, y, label = self._read_reference_xy(hit)
                        url = None
                    if len(x) == 0:
                        continue
                except Exception:                            # noqa: BLE001
                    continue
                self._add_reference(x, y, label,
                                    f"__ref_match_{hit['tag']}_{hit.get('id')}_{added}",
                                    rruff_name=hit['name'], rruff_id=hit.get('id'), rruff_url_=url)
                added += 1
            if added:
                self.replot_and_refresh_canvas()
                self.db_update_status(f"Overlaid {added} matched reference(s).")

        btn_row = ttk.Frame(pop)
        btn_row.pack(pady=8)
        ttk.Button(btn_row, text="🔗 Open Page(s)", command=open_pages).pack(side="left", padx=4)
        ttk.Button(btn_row, text="➕ Overlay Selected Match(es)",
                   command=overlay_chosen).pack(side="left", padx=4)

    # ---------- Reference libraries (.h5, multi-library) ----------

    def lib_add(self):
        paths = filedialog.askopenfilenames(
            title="Add reference .h5 librar(ies) (build_openspecy_library.py)",
            filetypes=[("Reference library", ("*.h5", "*.hdf5")), ("All Files", "*.*")])
        added = 0
        for path in paths:
            if self._lib_load_path(path, active=True):
                added += 1
        if added:
            self._lib_save_config()
        self._lib_refresh_panel()
        self._lib_update_status()

    def _lib_load_path(self, path, active=True):
        if any(os.path.abspath(l['path']) == os.path.abspath(path) for l in self.spec_libs):
            self.lib_status_var.set(f"Already loaded: {os.path.basename(path)}")
            return False
        try:
            lib = load_speclib_h5(path)
        except Exception as e:
            messagebox.showerror("Library Error",
                                 f"Could not open library:\n{os.path.basename(path)}\n{e}")
            return False
        for e in lib['entries']:
            e['lib_path'] = path
        self.spec_libs.append({'name': os.path.basename(path), 'path': path,
                               'entries': lib['entries'], 'active': bool(active),
                               'var': tk.BooleanVar(value=bool(active))})
        return True

    def _lib_remove(self, idx):
        if 0 <= idx < len(self.spec_libs):
            del self.spec_libs[idx]
            self._lib_save_config()
            self._lib_refresh_panel()
            self._lib_update_status()

    def _lib_toggle(self, idx):
        if 0 <= idx < len(self.spec_libs):
            self.spec_libs[idx]['active'] = bool(self.spec_libs[idx]['var'].get())
            self._lib_save_config()
            self._lib_update_status()

    def _lib_refresh_panel(self):
        """The unified panel owns the source list now, so defer to it."""
        self.db_refresh()

    def _lib_save_config(self):
        try:
            with open(self.lib_cfg_path, "w", encoding="utf-8") as fh:
                json.dump([{'path': l['path'], 'active': l['active']}
                           for l in self.spec_libs], fh, indent=2)
        except Exception:
            pass

    def _lib_load_config(self):
        if os.path.exists(self.lib_cfg_path):
            try:
                with open(self.lib_cfg_path, "r", encoding="utf-8") as fh:
                    saved = json.load(fh)
            except Exception:
                saved = []
            for item in saved or []:
                p = item.get('path')
                if p and os.path.exists(p):
                    self._lib_load_path(p, active=item.get('active', True))
        self._lib_refresh_panel()
        self._lib_update_status()

    def _lib_active_entries(self):
        out = []
        for lib in self.spec_libs:
            if lib['active']:
                out.extend(lib['entries'])
        return out

    def _lib_update_status(self):
        """Superseded by db_update_status, which counts every source."""
        self.db_update_status()

    @staticmethod
    def _lib_haystack(e):
        return (f"{e.get('name','')} {e.get('id','')} {e.get('formula','')} "
                f"{e.get('collection','')}").lower()

    @staticmethod
    def _lib_label(e):
        src = e.get('source') or 'Library'
        return f"{src}: {e.get('name','')}" + (f" ({e['id']})" if e.get('id') else "")

    def lib_run_search(self):
        self.lib_results_list.delete(0, tk.END)
        self.lib_search_hits = []
        pool = self._lib_active_entries()
        if not pool:
            messagebox.showinfo(
                "No Library",
                "Add a reference .h5 library first.\n\n"
                "Build one with:\n"
                "  python build_openspecy_library.py --only ftir --out openspecy_ftir.h5")
            return
        q = self.ent_lib_query.get().strip().lower()
        hits = [e for e in pool if not q or q in self._lib_haystack(e)][:300]
        self.lib_search_hits = hits
        if not hits:
            self.lib_status_var.set("Libraries: no matches.")
            return
        for h in hits:
            bits = [h.get('name', ''), h.get('id', '')]
            if h.get('collection'):
                bits.append(h['collection'])
            self.lib_results_list.insert(tk.END, "  ·  ".join(b for b in bits if b))
        self.lib_status_var.set(f"Libraries: {len(hits)} match(es). Select and overlay.")

    def _lib_read_xy(self, hit):
        with h5py.File(hit['lib_path'], 'r') as f:
            g = f['spectra'][hit['group']]
            x = np.array(g['x'][:], dtype=float)
            y = np.array(g['y'][:], dtype=float)
        return x, y

    def lib_overlay_selected(self):
        sel = self.lib_results_list.curselection()
        if not sel or sel[0] >= len(self.lib_search_hits):
            self.lib_status_var.set("Libraries: select a search result first.")
            return
        hit = self.lib_search_hits[sel[0]]
        try:
            x, y = self._lib_read_xy(hit)
        except Exception as e:
            self.lib_status_var.set(f"Libraries: could not read that reference — {e}")
            return
        self.save_to_history()
        self._add_reference(x, y, self._lib_label(hit), f"__ref_lib_{hit['id']}",
                            rruff_name=hit.get('name'), rruff_id=hit.get('id'),
                            rruff_url_=hit.get('url') or OPENSPECY_URL)
        self.replot_and_refresh_canvas()
        self.lib_status_var.set(f"Overlaid {hit.get('name')} ({hit.get('id')}).")

    def lib_match_by_peaks(self):
        if not self.peak_guesses:
            messagebox.showinfo(
                "Mark Peaks First",
                "Turn on '🎯 Peak Selection', then right-click on the plot to mark the "
                "bands you want to match. Then run 'Match by Selected Peaks'.")
            return
        pool = self._lib_active_entries()
        if not pool:
            messagebox.showinfo("No Library", "Add a reference .h5 library first.")
            return
        try:
            tolerance = float(self.ent_lib_match_tol.get().strip())
        except ValueError:
            tolerance = 12.0
        exp_peaks = list(self.peak_guesses)
        q = self.ent_lib_query.get().strip().lower()

        scored = []
        for e in pool:
            if q and q not in self._lib_haystack(e):
                continue
            if e['peaks'].size == 0:
                continue
            score, avg, matched = peak_match_score(e['peaks'], exp_peaks, tolerance)
            if matched > 0:
                scored.append({'score': score, 'avg': avg, 'matched': matched,
                               'name': e['name'], 'id': e['id'],
                               'source': e.get('source', 'Library'),
                               'detail': e.get('collection', '') or e.get('formula', ''),
                               'group': e['group'], 'lib_path': e.get('lib_path'),
                               'url': e.get('url') or OPENSPECY_URL})
        scored.sort(key=lambda t: (-t['score'], t['avg']))
        self._show_match_results(scored[:100], len(exp_peaks), tolerance,
                                 title="Reference Libraries", status_var=self.lib_status_var,
                                 reader=self._lib_read_xy, labeler=self._lib_label)

    # ---------- NIST WebBook (online, on demand) ----------

    def nist_run_search(self):
        query = self.ent_nist_query.get().strip()
        self.nist_results_list.delete(0, tk.END)
        self.nist_hits = []
        if not query:
            self.nist_status_var.set("NIST: type a compound name, formula or CAS number.")
            return
        field = self.combo_nist_field.get() or "auto"
        try:
            cap = max(1, int(float(self.ent_nist_cap.get().strip())))
        except ValueError:
            cap = 25
        self.nist_status_var.set("NIST: searching …")

        def worker():
            try:
                species = nist_search(query, field=field, limit=cap)
                # Expand each species into its individual IR spectra, so the
                # user picks a phase/instrument rather than a compound.
                hits = []
                for sp in species:
                    try:
                        for s in nist_species_spectra(sp['id']):
                            hits.append({**sp, 'index': s['index'],
                                         'detail': s['description']})
                    except Exception:
                        hits.append({**sp, 'index': 0, 'detail': 'IR spectrum'})
                    if len(hits) >= cap * 4:
                        break
                err = None
            except Exception as e:                       # noqa: BLE001
                hits, err = [], e
            self.root.after(0, lambda: self._nist_show_hits(hits, err))

        threading.Thread(target=worker, daemon=True).start()

    def _nist_show_hits(self, hits, err):
        if err is not None:
            self.nist_status_var.set(f"NIST search failed: {err}")
            return
        self.nist_hits = hits
        if not hits:
            self.nist_status_var.set("NIST: no species with IR spectra matched.")
            return
        for h in hits:
            detail = (h.get('detail') or '')[:70]
            self.nist_results_list.insert(
                tk.END, f"{h['name']}  ·  #{h.get('index', 0)}" +
                        (f"  ·  {detail}" if detail else ""))
        self.nist_status_var.set(
            f"NIST: {len(hits)} spectra across {len({h['id'] for h in hits})} species. "
            f"Select and overlay; double-click opens the NIST page.")

    def _nist_selected(self):
        sel = self.nist_results_list.curselection()
        if not sel or sel[0] >= len(self.nist_hits):
            return None
        return self.nist_hits[sel[0]]

    def _nist_open_selected_page(self, event=None):
        hit = self._nist_selected()
        if hit:
            webbrowser.open_new_tab(hit.get('url') or nist_species_url(hit['id']))

    @staticmethod
    def _nist_label(hit):
        return f"NIST: {hit['name']} (#{hit.get('index', 0)})"

    def nist_overlay_selected(self):
        hit = self._nist_selected()
        if not hit:
            self.nist_status_var.set("NIST: select a search result first.")
            return
        self.nist_status_var.set(f"NIST: downloading {hit['name']} …")

        def worker():
            try:
                x, y, _ = nist_fetch_ir(hit['id'], hit.get('index', 0))
                err = None
            except Exception as e:                       # noqa: BLE001
                x = y = None
                err = e

            def done():
                if err is not None:
                    self.nist_status_var.set(f"NIST: could not fetch that spectrum — {err}")
                    return
                self.save_to_history()
                self._add_reference(x, y, self._nist_label(hit),
                                    f"__ref_nist_{hit['id']}_{hit.get('index', 0)}",
                                    rruff_name=hit['name'], rruff_id=hit['id'],
                                    rruff_url_=hit.get('url') or nist_species_url(hit['id']))
                self.replot_and_refresh_canvas()
                self.nist_status_var.set(f"NIST: overlaid {hit['name']}.")
            self.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _nist_stop_scan(self):
        self._nist_scan_cancel = True
        self.nist_status_var.set("NIST: stopping …")

    def _nist_set_scanning(self, scanning):
        if scanning:
            self.btn_nist_match.config(text="⏹ Stop scanning", command=self._nist_stop_scan)
        else:
            self.btn_nist_match.config(text="🎯 Match by Selected Peaks (NIST)",
                                       command=self.nist_match_by_peaks)

    def nist_match_by_peaks(self):
        """Bounded online match: only the spectra currently listed are fetched.

        Matching all of NIST live is not on -- it would mean downloading the
        whole compilation. So the search results *are* the candidate set: run a
        search first, then rank what it returned.
        """
        if not self.peak_guesses:
            messagebox.showinfo(
                "Mark Peaks First",
                "Turn on '🎯 Peak Selection', then right-click on the plot to mark the "
                "bands you want to match.")
            return
        if not self.nist_hits:
            messagebox.showinfo(
                "Search First",
                "NIST matching ranks the spectra your search returned, because "
                "scanning all of NIST would mean downloading the entire "
                "compilation.\n\nSearch by name, formula or CAS first — then match. "
                "Downloads are cached, so re-running is instant.")
            return
        try:
            tolerance = float(self.ent_lib_match_tol.get().strip())
        except (ValueError, AttributeError):
            tolerance = 12.0
        exp_peaks = list(self.peak_guesses)
        hits = list(self.nist_hits)
        n = len(hits)
        self._nist_scan_cancel = False
        self._nist_set_scanning(True)

        def worker():
            scored, failed = [], 0
            for i, hit in enumerate(hits, 1):
                if self._nist_scan_cancel:
                    break
                try:
                    x, y, _ = nist_fetch_ir(hit['id'], hit.get('index', 0))
                    ref_peaks = detect_reference_peaks(x, y)
                    score, avg, matched = peak_match_score(ref_peaks, exp_peaks, tolerance)
                    if matched > 0:
                        scored.append({'score': score, 'avg': avg, 'matched': matched,
                                       'name': hit['name'], 'id': hit['id'],
                                       'source': 'NIST', 'index': hit.get('index', 0),
                                       'detail': (hit.get('detail') or '')[:60],
                                       'group': None, 'lib_path': None,
                                       'url': hit.get('url') or nist_species_url(hit['id'])})
                except Exception:                        # noqa: BLE001
                    failed += 1
                if i % 3 == 0 or i == n:
                    self.root.after(0, lambda i=i: self.nist_status_var.set(
                        f"NIST: scanned {i}/{n} …"))
            scored.sort(key=lambda t: (-t['score'], t['avg']))
            stopped = self._nist_scan_cancel

            def done():
                self._nist_set_scanning(False)
                self._nist_scan_cancel = False
                note = " (stopped early)" if stopped else ""
                if failed:
                    note += f", {failed} unavailable"
                self.nist_status_var.set(
                    f"NIST: {len(scored)} of {n} scanned matched{note}.")
                self._show_match_results(
                    scored[:100], len(exp_peaks), tolerance, title="NIST WebBook",
                    status_var=self.nist_status_var,
                    reader=lambda h: nist_fetch_ir(h['id'], h.get('index', 0))[:2],
                    labeler=self._nist_label)
            self.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    # ---------- Shared match-results window ----------

    def _show_match_results(self, scored, n_exp, tolerance, title, status_var,
                            reader, labeler):
        """Ranked candidate window, shared by the library and NIST matchers.

        `reader(hit) -> (x, y)` and `labeler(hit) -> str` are what differ
        between the two sources; everything else is identical.
        """
        if not scored:
            status_var.set(f"{title}: nothing matched the marked bands.")
            messagebox.showinfo("No Matches",
                                f"No {title} reference had bands near your marked "
                                f"peaks.\nTry a larger match tolerance or different peaks.")
            return

        pop = tk.Toplevel(self.root)
        pop.title(f"{title} — Candidate Ranking")
        pop.geometry("760x360")
        pop.transient(self.root)
        pop.grab_set()
        ttk.Label(pop, text=f"Ranked by alignment with your {n_exp} marked band(s) "
                            f"(±{tolerance:g} cm⁻¹). Double-click a row to open its page.",
                  font=("Helvetica", 9, "bold")).pack(pady=6, padx=8, anchor="w")

        frame = ttk.Frame(pop)
        frame.pack(fill="both", expand=True, padx=8, pady=4)
        scroll = ttk.Scrollbar(frame)
        scroll.pack(side="right", fill="y")
        tree = ttk.Treeview(frame, columns=("Score", "Name", "ID", "Detail", "Matched"),
                            show="headings", yscrollcommand=scroll.set, height=10,
                            selectmode="extended")
        for col, head, width, anchor in (
                ("Score", "Match Score", 95, "center"),
                ("Name", "Name", 210, "w"),
                ("ID", "ID", 95, "center"),
                ("Detail", "Collection / Detail", 240, "w"),
                ("Matched", "Bands Matched", 105, "center")):
            tree.heading(col, text=head)
            tree.column(col, width=width, anchor=anchor)
        tree.pack(fill="both", expand=True)
        scroll.config(command=tree.yview)

        row_map = {}
        for rec in scored:
            iid = tree.insert("", "end", values=(f"{rec['score']:.0f}%", rec['name'],
                                                 rec['id'], rec.get('detail', ''),
                                                 f"{rec['matched']}/{n_exp}"))
            row_map[iid] = rec

        def open_pages(event=None):
            opened = 0
            for iid in tree.selection():
                rec = row_map.get(iid)
                if rec and rec.get('url'):
                    webbrowser.open_new_tab(rec['url'])
                    opened += 1
            if opened == 0:
                messagebox.showinfo("Nothing Selected", "Select a row, then open its page.")
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
                    x, y = reader(hit)
                    if len(x) == 0:
                        continue
                except Exception:
                    continue
                self._add_reference(x, y, labeler(hit),
                                    f"__ref_match_{hit['source']}_{hit['id']}_{added}",
                                    rruff_name=hit['name'], rruff_id=hit['id'],
                                    rruff_url_=hit.get('url'))
                added += 1
            if added:
                self.replot_and_refresh_canvas()
                status_var.set(f"{title}: overlaid {added} matched reference(s).")

        btn_row = ttk.Frame(pop)
        btn_row.pack(pady=8)
        ttk.Button(btn_row, text="🔗 Open Page(s)", command=open_pages).pack(side="left", padx=4)
        ttk.Button(btn_row, text="➕ Overlay Selected Match(es)",
                   command=overlay_chosen).pack(side="left", padx=4)

    # ---------- SDBS (external lookup) ----------

    def sdbs_lookup(self):
        """Opens the SDBS search page in the user's browser.

        No scraping and no automated download: SDBS's terms prohibit automated
        retrieval and impose a daily access limit, so the user drives it.
        """
        query = self.ent_sdbs_query.get().strip()
        if not query:
            query = self.ent_nist_query.get().strip() or self.ent_lib_query.get().strip()

        webbrowser.open_new_tab(SDBS_SEARCH_URL)
        if query:
            self.root.clipboard_clear()
            self.root.clipboard_append(query)
            messagebox.showinfo(
                "SDBS Opened",
                f"SDBS search page opened in your browser.\n\n"
                f"'{query}' has been copied to your clipboard — paste it into the "
                f"compound name field.\n\n"
                f"SDBS does not permit automated downloads, so save the FT-IR "
                f"spectrum yourself (JCAMP-DX), then bring it in with 📂 Load Spectra.")
        else:
            messagebox.showinfo(
                "SDBS Opened",
                "SDBS search page opened in your browser. Search by compound name, "
                "formula, CAS number or SDBS number.\n\n"
                "Save the FT-IR spectrum as JCAMP-DX, then bring it in with "
                "📂 Load Spectra.")

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
        self.status_var.set("Active spectra loaded: 0"); self.cursor_var.set("Cursor Position: Wavenumber = --")


if __name__ == "__main__":
    root = tk.Tk()
    app = FTIRPlotterGUI(root)
    root.mainloop()
