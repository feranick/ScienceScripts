import os
import io
import re
import zipfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import threading
import webbrowser

# Embed Matplotlib into Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks

# Conditional Imports to ensure app stability if packages are absent on launch
try:
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from mp_api.client import MPRester
    MP_LIBRARIES_AVAILABLE = True
except ImportError:
    MP_LIBRARIES_AVAILABLE = False

# HDF5 support for the optional RRUFF powder .h5 reference library.
try:
    import h5py
    H5_AVAILABLE = True
except ImportError:
    H5_AVAILABLE = False

# ==========================================
# GLOBAL CONFIGURATIONS & CONSTANTS
# ==========================================
VERSION_TAG = "v2026.07.23.1"
KEY_FILE_NAME = "mp_api_key.txt"

# RRUFF powder reference library (patterns calculated for Cu radiation, i.e. the
# same CuKa convention used for Materials Project simulated patterns).
SYNTH_MIN, SYNTH_MAX, SYNTH_STEP, SYNTH_SIGMA = 5.0, 90.0, 0.02, 0.10
_REF_DATA_ROW = re.compile(r'^\d+\.\d+(?:\s+-?\d+){3,4}$')


# ==========================================
# MATHEMATICAL FUNCTION CODES
# ==========================================

def gaussian_profile(x, amp, cent, wid):
    """Calculates a discrete single Gaussian peak vector layout configuration."""
    return amp * np.exp(-((x - cent) / wid) ** 2)

def multi_gaussian_composite(x, *params):
    """Aggregates multiple mathematical Gaussian curves arrays summation."""
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        amp = params[i]
        cent = params[i+1]
        wid = params[i+2]
        y += gaussian_profile(x, amp, cent, wid)
    return y

def snip_background(y, iterations=40):
    """Calculates the baseline background profile using the SNIP algorithm."""
    bg = np.array(y, dtype=float)
    n = len(bg)
    max_iter = min(iterations, int(n / 2) - 1)
    if max_iter < 1:
        return np.zeros_like(bg)
    for p in range(1, max_iter + 1):
        temp = np.copy(bg)
        bg[p:-p] = np.minimum(bg[p:-p], (temp[:-2*p] + temp[2*p:]) / 2.0)
    return bg

def calculate_crystallographic_match_score(theoretical_angles, experimental_guesses, tolerance=0.18):
    """
    Computes a mathematical Figure of Merit (FOM) matching score based on proximity
    between database reflections and user-marked experimental peak coordinates.
    """
    if not experimental_guesses or len(theoretical_angles) == 0:
        return 0.0, 0.0
    
    matched_peaks_count = 0
    cumulative_proximity_delta = 0.0
    
    for exp_x in experimental_guesses:
        deltas = np.abs(theoretical_angles - exp_x)
        minimum_delta = np.min(deltas)
        
        if minimum_delta <= tolerance:
            matched_peaks_count += 1
            cumulative_proximity_delta += minimum_delta
        else:
            cumulative_proximity_delta += tolerance  # Out-of-bounds positioning penalty
            
    score_percentage = (matched_peaks_count / len(experimental_guesses)) * 100.0
    average_closeness_error = cumulative_proximity_delta / len(experimental_guesses)
    return score_percentage, average_closeness_error


# ==========================================
# RRUFF POWDER REFERENCE DATABASE (added alongside Materials Project)
# ==========================================

def detect_reference_peaks(x, y, max_peaks=60, min_prominence=0.02):
    """Detects prominent 2-theta band positions in a continuous pattern."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(y) < 5:
        return np.array([])
    rng = np.ptp(y)
    if rng <= 0:
        return np.array([])
    yn = (y - y.min()) / rng
    peaks, props = find_peaks(yn, prominence=min_prominence, distance=2)
    if len(peaks) == 0:
        return np.array([])
    proms = props.get('prominences', np.ones(len(peaks)))
    order = np.argsort(proms)[::-1][:max_peaks]
    return np.sort(x[peaks[order]])


def peak_match_score(reference_peaks, experimental_peaks, tolerance):
    """FOM for how well a reference's peaks explain the marked peaks.
    Returns (score_percent, average_closeness, matched_count)."""
    ref = np.asarray(reference_peaks, dtype=float)
    exp = list(experimental_peaks)
    if not exp or ref.size == 0:
        return 0.0, float(tolerance), 0
    matched = 0; cumulative = 0.0
    for ex in exp:
        delta = np.min(np.abs(ref - ex))
        if delta <= tolerance:
            matched += 1; cumulative += delta
        else:
            cumulative += tolerance
    return (matched / len(exp)) * 100.0, cumulative / len(exp), matched


def _synth_profile(tth, inten):
    x = np.arange(SYNTH_MIN, SYNTH_MAX + SYNTH_STEP, SYNTH_STEP)
    y = np.zeros_like(x)
    for c, h in zip(tth, inten):
        y += h * np.exp(-((x - c) / SYNTH_SIGMA) ** 2)
    if y.max() > 0:
        y = y / y.max() * 100.0
    return x, y


def parse_rruff_powder(text):
    """Parses any RRUFF powder collection -> (x, y, peaks_or_None, meta).
    XY profile -> continuous x,y (peaks None); DIF / refinement_data /
    refinement_output_data -> 2-theta peak list + a synthesized display profile."""
    meta = {}
    lines = text.splitlines()
    for raw in lines:
        line = raw.strip()
        m = re.match(r'#+\s*([A-Za-z ]+)\s*=\s*(.*)', line)
        if m:
            meta[m.group(1).strip().upper()] = m.group(2).strip()
        mw = re.search(r'X-?RAY WAVELENGTH\s*(?:#\d+\s*)?[:=]?\s*([\d.]+)', line, re.IGNORECASE)
        if mw:
            meta.setdefault('WAVELENGTH', mw.group(1))
    upper = text.upper()

    def clamp(a):
        return 1.0 <= a <= 170.0

    # DIF: "2-THETA INTENSITY D-SPACING H K L"
    if 'INTENSITY' in upper and re.search(r'2-?\s*THETA', upper):
        start = 0
        for i, raw in enumerate(lines):
            if re.search(r'2-?\s*THETA', raw, re.IGNORECASE) and re.search(r'INTENSITY', raw, re.IGNORECASE):
                start = i + 1; break
        tth, inten = [], []
        for raw in lines[start:]:
            s = raw.strip()
            if not s or set(s) <= set('='):
                if tth:
                    break
                continue
            parts = s.split()
            try:
                a = float(parts[0]); b = float(parts[1])
            except (ValueError, IndexError):
                if tth:
                    break
                continue
            if clamp(a):
                tth.append(a); inten.append(b)
        if tth:
            x, y = _synth_profile(tth, inten)
            return x, y, np.array(tth), meta

    # refinement_output_data (REFINE program listing)
    if 'PROGRAM REFINE' in upper or ('OBSERVED' in upper and 'CALCULATED' in upper):
        tth = []
        for raw in lines:
            parts = raw.split()
            if len(parts) < 6:
                continue
            try:
                a = float(parts[0])
            except ValueError:
                continue
            if clamp(a):
                tth.append(a)
        if tth:
            x, y = _synth_profile(tth, [100.0] * len(tth))
            return x, y, np.array(tth), meta

    # refinement_data ("2theta h k l [wave#]")
    ref_rows = [l.strip() for l in lines if _REF_DATA_ROW.match(l.strip())]
    if len(ref_rows) >= 3:
        tth = [float(r.split()[0]) for r in ref_rows if clamp(float(r.split()[0]))]
        if tth:
            x, y = _synth_profile(tth, [100.0] * len(tth))
            return x, y, np.array(tth), meta

    # XY continuous two-column
    xs, ys = [], []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith('#'):
            continue
        for sep in (',', '\t', ';'):
            if sep in s:
                parts = s.split(sep); break
        else:
            parts = s.split()
        if len(parts) < 2:
            continue
        try:
            xv = float(parts[0]); yv = float(parts[1])
        except ValueError:
            continue
        xs.append(xv); ys.append(yv)
    return np.array(xs), np.array(ys), None, meta


def rruff_meta_from_filename(fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    parts = stem.split('__')
    name = parts[0] if parts else stem
    rid = ''
    for p in parts[1:]:
        mm = re.match(r'(R\d{5,7})', p)
        if mm:
            rid = mm.group(1); break
    return name, rid


def rruff_url(name, rid, stored_url=None):
    if stored_url:
        return stored_url
    if rid and re.match(r'^R\d+', str(rid)):
        return f"https://rruff.info/{rid}"
    if name:
        return "https://rruff.info/" + str(name).strip().lower()
    return None


def load_rruff_powder_h5_library(path):
    """Reads a consolidated RRUFF powder library .h5 (from build_rruff_powder_library.py).
    Returns {'path', 'entries': [{group, name, id, url, peaks(np.array)}]}."""
    if not H5_AVAILABLE:
        raise ImportError("Reading .h5 libraries requires 'h5py' (pip install h5py).")
    entries = []
    with h5py.File(path, 'r') as f:
        if 'spectra' not in f:
            raise ValueError("Not a RRUFF powder library ('spectra' group missing).")
        sp = f['spectra']
        for gname in sp:
            a = sp[gname].attrs
            def dec(v):
                return v.decode('latin1') if isinstance(v, bytes) else str(v)
            peaks = np.asarray(a['peaks'], dtype=float) if 'peaks' in a else np.array([])
            entries.append({
                'group': gname,
                'name': dec(a.get('name', gname)),
                'id': dec(a.get('rruff_id', '')),
                'url': dec(a.get('url', '')),
                'peaks': peaks,
            })
    if not entries:
        raise ValueError("Library contains no patterns.")
    return {'path': path, 'entries': entries}


# ==========================================
# PARSING ENGINE CORE LOGIC
# ==========================================

def load_xrd_data(file_path):
    """Parses custom instrument-tagged XRD CSV, simple tabular CSV, or native XML XRDML files."""
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    sample_id = os.path.splitext(filename)[0]
    
    if ext == '.csv':
        skiprows = 0
        has_scan_points = False
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for idx, line in enumerate(f):
                    if 'Sample identification' in line:
                        parts = line.split(',')
                        if len(parts) > 1 and parts[1].strip():
                            sample_id = parts[1].strip()
                    if '[Scan points]' in line:
                        skiprows = idx + 1
                        has_scan_points = True
                        break
        except Exception as e:
            raise IOError(f"Error reading CSV structure headers: {e}")
            
        try:
            if has_scan_points:
                df = pd.read_csv(file_path, skiprows=skiprows)
            else:
                # Fall back immediately to a clean, direct column layout parse strategy
                df = pd.read_csv(file_path)
                
            df.columns = df.columns.str.strip()
            
            # Match columns dynamically based on variations of common shorthand syntax keys
            angle_col = [c for c in df.columns if 'angle' in c.lower() or '2theta' in c.lower() or '2θ' in c.lower()]
            intensity_col = [c for c in df.columns if 'intensity' in c.lower() or 'count' in c.lower()]
            
            if angle_col and intensity_col:
                return df[angle_col[0]].values, df[intensity_col[0]].values, sample_id
            elif len(df.columns) >= 2:
                # Positional fallback to indices 0 and 1 if tracking text headers are completely modified
                return df.iloc[:, 0].values, df.iloc[:, 1].values, sample_id
            else:
                raise ValueError("Missing required distinct Angle and Intensity coordinate streams.")
        except Exception as e:
            raise ValueError(f"Parsing index exception inside CSV array: {e}")

    elif ext == '.xrdml':
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            ns = {'x': 'http://www.xrdml.com/XRDMeasurement/2.2'}
            
            s_id = root.find('.//x:sample/x:id', ns)
            if s_id is not None and s_id.text:
                sample_id = s_id.text
                
            counts_el = root.find('.//x:counts', ns)
            if counts_el is None or not counts_el.text:
                raise ValueError("No intensity counts data blocks located.")
            counts = [float(c) for c in counts_el.text.split()]
            
            pos_el = root.find('.//x:positions[@axis="2Theta"]', ns)
            if pos_el is None:
                raise ValueError("No 2Theta scanning positions found.")
                
            list_pos = pos_el.find('x:listPosition', ns)
            if list_pos is not None and list_pos.text:
                angles = [float(a) for a in list_pos.text.split()]
            else:
                start_el = pos_el.find('x:startPosition', ns)
                end_el = pos_el.find('x:endPosition', ns)
                if start_el is not None and end_el is not None:
                    start_val, end_val = float(start_el.text), float(end_el.text)
                    angles = np.linspace(start_val, end_val, len(counts))
                else:
                    raise ValueError("Unable to determine coordinate limits bounds.")
            
            return np.array(angles), np.array(counts), sample_id
        except Exception as e:
            raise ValueError(f"XML Tree processing exception: {e}")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# ==========================================
# GUI & EMBEDDED PLOTTING INTERFACE
# ==========================================

class XRDPlotterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced XRD Peak Analysis Toolkit")
        self.root.geometry("1050x780")
        self.root.minsize(850, 600)
        
        style = ttk.Style()
        style.theme_use('clam')
        
        # In-memory arrays session states initialization
        self.active_datasets = {}
        self.peak_guesses = []
        self.rruff_lib = None
        self.rruff_local_dir = None
        self.rruff_search_hits = []
        self.guess_lines_artists = []
        self.fitted_curves_artists = []
        self.target_checkbox_vars = {} 
        self.history_stack = []
        
        self.fitting_mode_active = False
        self.normalization_mode_active = False
        self.cursor_line = None  
        
        # --- Left Sidebar Panel Layout ---
        sidebar_frame = ttk.Frame(root, padding=12, relief="flat")
        sidebar_frame.pack(side="left", fill="y", padx=5, pady=5)
        
        ttk.Label(sidebar_frame, text="🔬 XRD Data Analyzer", font=("Helvetica", 12, "bold")).pack(side="top", anchor="w", pady=(0, 10))
        
        ttk.Button(sidebar_frame, text="📁 Select File(s)", command=self.select_and_plot_files).pack(side="top", fill="x", pady=3)
        ttk.Button(sidebar_frame, text="✂️ Crop to View", command=self.crop_to_current_view).pack(side="top", fill="x", pady=3)
        ttk.Button(sidebar_frame, text="✨ Subtract Background", command=self.subtract_background_profile).pack(side="top", fill="x", pady=3)
        ttk.Button(sidebar_frame, text="🧪 Subtract Reference Scan", command=self.open_blank_subtraction_dialog).pack(side="top", fill="x", pady=3)
        
        # Savitzky-Golay Signal Noise Smoothing Layout Row
        smooth_row = ttk.Frame(sidebar_frame)
        smooth_row.pack(side="top", fill="x", pady=3)
        
        ttk.Button(smooth_row, text="🍃 Smooth Noise", command=self.smooth_active_profiles).pack(side="left", fill="x", expand=True)
        ttk.Label(smooth_row, text="Win:", font=("Helvetica", 9)).pack(side="left", padx=(4, 1))
        self.ent_smooth_win = ttk.Entry(smooth_row, width=4)
        self.ent_smooth_win.insert(0, "11")
        self.ent_smooth_win.pack(side="left", padx=1)
        
        # Rigid 2-Theta Zero-Offset Calibration Interface row
        shift_row = ttk.Frame(sidebar_frame)
        shift_row.pack(side="top", fill="x", pady=3)
        
        ttk.Button(shift_row, text="📐 Shift 2θ", command=self.apply_two_theta_shift).pack(side="left", fill="x", expand=True)
        ttk.Label(shift_row, text="Δ:", font=("Helvetica", 9)).pack(side="left", padx=(4, 1))
        self.ent_shift_val = ttk.Entry(shift_row, width=5)
        self.ent_shift_val.insert(0, "0.0")
        self.ent_shift_val.pack(side="left", padx=1)
        ttk.Label(shift_row, text="°", font=("Helvetica", 9)).pack(side="left", padx=(1, 2))
        
        # Inline Horizontal Configuration Row for Normalization and variable window field bounds
        norm_row = ttk.Frame(sidebar_frame)
        norm_row.pack(side="top", fill="x", pady=3)
        
        self.btn_normalize_toggle = ttk.Button(norm_row, text="⚖️ Normalize to Peak", command=self.toggle_normalization_mode)
        self.btn_normalize_toggle.pack(side="left", fill="x", expand=True)
        
        ttk.Label(norm_row, text="±", font=("Helvetica", 10)).pack(side="left", padx=(5, 1))
        self.ent_norm_span = ttk.Entry(norm_row, width=5)
        self.ent_norm_span.insert(0, "0.3")
        self.ent_norm_span.pack(side="left", padx=1)
        ttk.Label(norm_row, text="°", font=("Helvetica", 10)).pack(side="left", padx=(1, 2))
        
        self.btn_fit_toggle = ttk.Button(sidebar_frame, text="🎯 Peak Selection: OFF", command=self.toggle_fitting_mode)
        self.btn_fit_toggle.pack(side="top", fill="x", pady=3)
        
        self.btn_run_fit = ttk.Button(sidebar_frame, text="⚡ Fit", command=self.run_peak_optimization, state="disabled")
        self.btn_run_fit.pack(side="top", fill="x", pady=3)
        
        ttk.Button(sidebar_frame, text="📥 Export to CSV", command=self.export_active_data_to_csv).pack(side="top", fill="x", pady=3)
        ttk.Button(sidebar_frame, text="🗑️ Clear Canvas", command=self.clear_canvas).pack(side="top", fill="x", pady=3)
        self.btn_undo = ttk.Button(sidebar_frame, text="↩️ Undo Last Action", command=self.undo_last_action, state="disabled")
        self.btn_undo.pack(side="top", fill="x", pady=3)
        
        ttk.Separator(sidebar_frame, orient="horizontal").pack(side="top", fill="x", pady=10)
        
        # Status Active Profiles Badge
        self.status_var = tk.StringVar(value="Active profiles loaded: 0")
        lbl_status = ttk.Label(sidebar_frame, textvariable=self.status_var, font=("Helvetica", 9, "bold"), background="#cff4fc", foreground="#055160", relief="solid", borderwidth=1, padding=6, anchor="center")
        lbl_status.pack(side="top", fill="x", pady=2)
        
        # --- Materials Genome API Search Control Panel ---
        panel_search = ttk.LabelFrame(sidebar_frame, text=" 🌐 Materials Genome API Search ", padding=(8, 6))
        panel_search.pack(side="top", fill="x", pady=5)
        
        key_header_frame = ttk.Frame(panel_search)
        key_header_frame.pack(fill="x", pady=(0, 2))
        
        ttk.Label(key_header_frame, text="API Key:", font=("Helvetica", 8, "bold")).pack(side="left", anchor="w")
        
        lbl_hyperlink = ttk.Label(key_header_frame, text="Get Key ↗", foreground="#0d6efd", cursor="hand2", font=("Helvetica", 8, "underline"))
        lbl_hyperlink.pack(side="right", anchor="e")
        lbl_hyperlink.bind("<Button-1>", lambda e: webbrowser.open_new("https://next-gen.materialsproject.org/api"))
        
        key_entry_row = ttk.Frame(panel_search)
        key_entry_row.pack(fill="x", pady=(0, 4))
        
        self.ent_api_key = ttk.Entry(key_entry_row, show="*")
        self.ent_api_key.pack(side="left", fill="x", expand=True)
        
        btn_save_key = ttk.Button(key_entry_row, text="💾", width=3, command=self.save_api_key_locally)
        btn_save_key.pack(side="right", padx=(3, 0))
        
        ttk.Label(panel_search, text="Formula or System (e.g. TiO2 or W-O):", font=("Helvetica", 8, "bold")).pack(anchor="w")
        self.ent_formula = ttk.Entry(panel_search)
        self.ent_formula.pack(fill="x", pady=(0, 6))
        
        btn_search_text = ttk.Button(panel_search, text="🔍 Search by Formula", command=lambda: self.execute_database_search(mode="text"))
        btn_search_text.pack(fill="x", pady=2)
        
        btn_search_match = ttk.Button(panel_search, text="🎯 Search/Match by Peaks", command=lambda: self.execute_database_search(mode="peaks"))
        btn_search_match.pack(fill="x", pady=2)

        # --- RRUFF (powder) Reference Database Panel ---
        panel_rruff = ttk.LabelFrame(sidebar_frame, text=" 💎 RRUFF (powder) ", padding=(8, 6))
        panel_rruff.pack(side="top", fill="x", pady=5)
        rr_btns = ttk.Frame(panel_rruff)
        rr_btns.pack(fill="x")
        ttk.Button(rr_btns, text="📚 Open .h5 Library", command=self.rruff_open_library).pack(side="left", fill="x", expand=True)
        ttk.Button(rr_btns, text="📂 Folder", command=self.rruff_pick_local_folder).pack(side="left", fill="x", expand=True, padx=(3, 0))
        rr_search = ttk.Frame(panel_rruff)
        rr_search.pack(fill="x", pady=(4, 2))
        self.ent_rruff_query = ttk.Entry(rr_search)
        self.ent_rruff_query.pack(side="left", fill="x", expand=True)
        self.ent_rruff_query.bind("<Return>", lambda e: self.rruff_run_search())
        ttk.Button(rr_search, text="🔍", width=3, command=self.rruff_run_search).pack(side="right", padx=(3, 0))
        self.rruff_results_list = tk.Listbox(panel_rruff, height=3, exportselection=False)
        self.rruff_results_list.pack(fill="x", pady=(0, 3))
        self.rruff_results_list.bind("<Double-1>", self._rruff_open_selected_page)
        ttk.Button(panel_rruff, text="➕ Overlay Selected", command=self.rruff_overlay_selected).pack(fill="x", pady=(0, 2))
        rr_tol = ttk.Frame(panel_rruff)
        rr_tol.pack(fill="x")
        ttk.Label(rr_tol, text="Match tol. ±", font=("Helvetica", 8, "bold")).pack(side="left")
        self.ent_rruff_tol = ttk.Entry(rr_tol, width=5)
        self.ent_rruff_tol.insert(0, "0.2")
        self.ent_rruff_tol.pack(side="left", padx=(2, 1))
        ttk.Label(rr_tol, text="°2θ", font=("Helvetica", 8)).pack(side="left")
        ttk.Button(panel_rruff, text="🎯 Match by Selected Peaks (RRUFF)", command=self.rruff_match_by_peaks).pack(fill="x", pady=(2, 2))
        self.rruff_status_var = tk.StringVar(value="RRUFF: open an .h5 library or a folder of powder files.")
        ttk.Label(panel_rruff, textvariable=self.rruff_status_var, font=("Helvetica", 8), foreground="#555555", wraplength=250).pack(anchor="w")

        # --- Active Layers Control Panel Frame Container ---
        self.panel_fit_targets = ttk.LabelFrame(sidebar_frame, text=" 📋 Plotted Canvas Layers ", padding=(8, 6))
        self.panel_fit_targets.pack(side="top", fill="x", pady=8, expand=True)
        self.lbl_no_targets = ttk.Label(self.panel_fit_targets, text="No Scans Loaded", font=("Helvetica", 9, "italic"), foreground="#888888")
        self.lbl_no_targets.pack(side="top", anchor="w", padx=4)
        
        # Version tag pinned to the absolute sidebar bottom
        lbl_version = ttk.Label(sidebar_frame, text=VERSION_TAG, font=("Helvetica", 8), foreground="#888888")
        lbl_version.pack(side="bottom", pady=2)
        
        # --- Main Viewport Container Panel (Right side) ---
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

        self.cursor_var = tk.StringVar(value="Cursor Position: 2θ = --")
        ttk.Label(self.plot_frame, textvariable=self.cursor_var, font=("Consolas", 10, "bold"), background="#e9ecef", relief="solid", borderwidth=1, padding=5).pack(side="bottom", fill="x", pady=(4, 0))

        # --- Bottom Results Dashboard ---
        self.table_frame = ttk.LabelFrame(self.main_container, text=" 📊 Peak Optimization Results Dashboard ", padding=5)
        self.main_container.add(self.table_frame, weight=1)
        
        self.result_table = ttk.Treeview(self.table_frame, columns=("Dataset", "Peak", "Center", "Amplitude", "FWHM"), show="headings", height=5)
        self.result_table.heading("Dataset", text="Dataset / Sample")
        self.result_table.heading("Peak", text="Peak Identity Index")
        self.result_table.heading("Center", text="Center position (2θ°)")
        self.result_table.heading("Amplitude", text="Peak Amplitude (counts)")
        self.result_table.heading("FWHM", text="FWHM Line Width (deg)")
        
        self.result_table.column("Dataset", width=160, anchor="w")
        self.result_table.column("Peak", width=110, anchor="center")
        self.result_table.column("Center", width=130, anchor="center")
        self.result_table.column("Amplitude", width=130, anchor="center")
        self.result_table.column("FWHM", width=130, anchor="center")
        self.result_table.pack(fill="both", expand=True)

        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        self.load_api_key_on_launch()

    def configure_axis_labels(self):
        self.ax.set_xlabel(r"2$\theta$ Angle (degrees)", fontsize=10, fontweight='bold')
        self.ax.set_ylabel("Intensity (Normalized or Counts)", fontsize=10, fontweight='bold')
        self.ax.set_title("XRD Diffraction Pattern Analysis", fontsize=11, fontweight='bold', pad=8)
        self.ax.grid(True, linestyle="--", alpha=0.5)

    def save_to_history(self):
        """Generates a deep state matrix cache to enable instantaneous layout rollbacks."""
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
        """Pops the last layout footprint snapshot off the history array stack and restores the UI state."""
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

    def load_api_key_on_launch(self):
        if os.path.exists(KEY_FILE_NAME):
            try:
                with open(KEY_FILE_NAME, "r", encoding="utf-8") as f:
                    cached_key = f.read().strip()
                    if cached_key:
                        self.ent_api_key.insert(0, cached_key)
                        return
            except Exception:
                pass
        if os.environ.get("MP_API_KEY"):
            self.ent_api_key.insert(0, os.environ.get("MP_API_KEY"))

    def save_api_key_locally(self):
        key_to_save = self.ent_api_key.get().strip()
        if not key_to_save:
            messagebox.showwarning("Empty String", "Please type or paste a valid API key string value before saving.")
            return
        try:
            with open(KEY_FILE_NAME, "w", encoding="utf-8") as f:
                f.write(key_to_save)
            messagebox.showinfo("Success", "Materials Project API key saved locally!")
        except Exception as e:
            messagebox.showerror("IO Fault", f"Could not write configuration token: {e}")

    def refresh_checkbox_targets_panel(self):
        """Itemizes ALL loaded data, mathematical fit, and literature curves with target checkboxes and deletion gates."""
        for child in self.panel_fit_targets.winfo_children():
            child.destroy()
            
        if not self.active_datasets:
            self.lbl_no_targets = ttk.Label(self.panel_fit_targets, text="No Scans Loaded", font=("Helvetica", 9, "italic"), foreground="#888888")
            self.lbl_no_targets.pack(side="top", anchor="w", padx=4)
            return

        for key, data in list(self.active_datasets.items()):
            row_frame = ttk.Frame(self.panel_fit_targets)
            row_frame.pack(side="top", fill="x", pady=2, expand=True)
            
            if not key.startswith("__fit_") and not key.startswith("__ref_"):
                if key not in self.target_checkbox_vars:
                    self.target_checkbox_vars[key] = tk.BooleanVar(value=True)
                cb = ttk.Checkbutton(row_frame, text=data['label'], variable=self.target_checkbox_vars[key])
                cb.pack(side="left", anchor="w")
            else:
                url = rruff_url(data.get('rruff_name'), data.get('rruff_id'), data.get('rruff_url')) if key.startswith("__ref_rruff_") else None
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
            messagebox.showwarning("Execution Halted", "Load data profiles before normalization.")
            return
        self.normalization_mode_active = not self.normalization_mode_active
        if self.normalization_mode_active:
            if self.fitting_mode_active: self.toggle_fitting_mode()
            self.btn_normalize_toggle.config(text="⚖️ Mode: SELECT PEAK")
            self.status_var.set("Left-click near a peak to scale all active tracks.")
        else:
            self.btn_normalize_toggle.config(text="⚖️ Normalize to Peak")

    def toggle_fitting_mode(self):
        if not self.active_datasets:
            messagebox.showwarning("Execution Halted", "Load standard profiles before optimizing.")
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
            self.cursor_var.set(f"Cursor Position: 2θ = {x:.4f}°")
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
            self.cursor_var.set("Cursor Position: 2θ = --")

    def on_canvas_click(self, event):
        if event.inaxes == self.ax:
            if self.normalization_mode_active and event.button == 1:
                x_click = event.xdata
                
                try:
                    window_span = float(self.ent_norm_span.get().strip())
                except ValueError:
                    window_span = 0.3  
                    self.ent_norm_span.delete(0, tk.END)
                    self.ent_norm_span.insert(0, "0.3")
                
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
                    self.status_var.set(f"Profiles successfully normalized to peak near 2θ = {x_click:.3f}°.")
                    
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
            messagebox.showwarning("No Data", "No active profiles found to apply background corrections.")
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
            messagebox.showwarning("Insufficient Data", "You must have at least two scans loaded to execute subtraction.")
            return
            
        pop = tk.Toplevel(self.root)
        pop.title("Experimental Reference Matrix Subtraction")
        pop.geometry("460x320")
        pop.transient(self.root)
        pop.grab_set()
        
        ttk.Label(pop, text="Select Blank / Supporting Substrate Scan (e.g. Clay Blank):", font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12, pady=(12, 2))
        
        combo_blank = ttk.Combobox(pop, state="readonly", width=55)
        combo_blank['values'] = [self.active_datasets[k]['label'] for k in raw_keys]
        combo_blank.current(0)
        combo_blank.pack(fill="x", padx=12, pady=(0, 12))
        
        ttk.Label(pop, text="Select Target Scan(s) to isolate and filter background from:", font=("Helvetica", 9, "bold")).pack(anchor="w", padx=12, pady=(4, 2))
        
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
                messagebox.showwarning("Void Bounds", "Please pick at least one sample data scan target row.")
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
            self.status_var.set(f"Subtracted reference scan: '{self.active_datasets[blank_key]['label']}'.")
            
        ttk.Button(pop, text="Subtract Reference Blank", command=run_reference_subtraction).pack(pady=8)

    def smooth_active_profiles(self):
        """Applies a peak-preserving polynomial Savitzky-Golay smoothing kernel to active rows."""
        if not self.active_datasets:
            messagebox.showwarning("No Data", "No active experimental trace paths found to smooth.")
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
            self.status_var.set(f"Smoothed residual noise across {smoothed_count} layers (Savgol window={window}).")

    def apply_two_theta_shift(self):
        """Linearly shifts the independent 2-Theta coordinates vector array to calibrate zero-offset artifacts."""
        if not self.active_datasets:
            messagebox.showwarning("No Data", "No active profiles found to execute calibration translations.")
            return
            
        try:
            shift_val = float(self.ent_shift_val.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Value", "Please enter a valid numeric calculation factor for the 2θ shift.")
            return
            
        if shift_val == 0.0:
            return
            
        self.save_to_history()
        self.clear_fitted_artists()
        
        data_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_") and not k.startswith("__ref_")]
        for key in data_keys:
            self.active_datasets[key]['angles'] = self.active_datasets[key]['angles'] + shift_val
            
        self.replot_and_refresh_canvas()
        self.status_var.set(f"Applied a rigid 2θ shift calibration correction of {shift_val}°.")

    def replot_and_refresh_canvas(self):
        self.ax.clear()
        self.configure_axis_labels()
        self.cursor_line = None
        # ax.clear() detached every artist; rebuild the tracking lists from scratch
        # so they don't accumulate stale references across replots.
        self.fitted_curves_artists = []
        self.guess_lines_artists = []
        
        for file_path, data in self.active_datasets.items():
            if file_path.startswith("__fit_overall_composite"):
                line, = self.ax.plot(data['angles'], data['intensities'], color='#000000', linestyle='-', linewidth=2.0, label=data['label'])
                self.fitted_curves_artists.append(line)
            elif file_path.startswith("__fit_"):
                line, = self.ax.plot(data['angles'], data['intensities'], linestyle='--', linewidth=1.2, label=data['label'])
                self.fitted_curves_artists.append(line)
            elif file_path.startswith("__ref_"):
                self.ax.plot(data['angles'], data['intensities'], linestyle='-.', linewidth=1.5, alpha=0.8, label=data['label'])
            else:
                self.ax.plot(data['angles'], data['intensities'], label=data['label'], linewidth=1.2)
        
        # Redraw the peak-center guess markers so they survive a replot
        # (e.g. after removing a dataset), mirroring the web version's behaviour.
        for g_x in self.peak_guesses:
            guess_line = self.ax.axvline(g_x, color='#d63384', linestyle=':', linewidth=1.5)
            self.guess_lines_artists.append(guess_line)
                
        if self.active_datasets:
            self.ax.legend(loc="upper right", frameon=True, fontsize=8)
        self.ax.relim(); self.ax.autoscale_view(); self.refresh_checkbox_targets_panel(); self.canvas.draw()

    def execute_database_search(self, mode="text"):
        if not MP_LIBRARIES_AVAILABLE:
            messagebox.showinfo("Packages Missing", "To query reference data directly, run this command in your terminal folder:\npip install mp-api pymatgen")
            return
            
        api_key = self.ent_api_key.get().strip()
        formula = self.ent_formula.get().strip()
        
        if not formula:
            messagebox.showwarning("Missing Inputs", "Please specify a crystal target formula / system constraint.")
            return
        if not api_key:
            messagebox.showwarning("Missing Key", "Materials Project queries require an API key.")
            return
        if mode == "peaks" and not self.peak_guesses:
            messagebox.showwarning("No Peak Targets", "Right-click on the canvas map area to designate reference peaks first.")
            return
            
        self.status_var.set("Searching Materials Project...")
        threading.Thread(target=self._bg_search_worker, args=(api_key, formula, mode), daemon=True).start()

    def _bg_search_worker(self, api_key, query_str, mode):
        try:
            query_kwargs = {
                "fields": ["material_id", "structure", "symmetry", "energy_above_hull", "formula_pretty"]
            }
            if "-" in query_str:
                query_kwargs["chemsys"] = query_str
            else:
                query_kwargs["formula"] = query_str
                
            with MPRester(api_key) as mpr:
                docs = mpr.materials.summary.search(**query_kwargs)
                
            compiled_results = []
            calculator = XRDCalculator(wavelength="CuKa")
            
            for doc in docs:
                try:
                    if mode == "peaks":
                        pattern = calculator.get_pattern(doc.structure, two_theta_range=(5, 90))
                        theoretical_peaks = np.array(pattern.x)
                        score, avg_err = calculate_crystallographic_match_score(theoretical_peaks, self.peak_guesses)
                        compiled_results.append((score, avg_err, doc))
                    else:
                        compiled_results.append((0.0, 0.0, doc))
                except Exception:
                    continue
            
            if mode == "peaks":
                compiled_results.sort(key=lambda item: (-item[0], item[1]))
                
            self.root.after(0, self.show_polymorph_selection, compiled_results, mode)
        except Exception as e:
            self.root.after(0, lambda err=str(e): messagebox.showerror("API Connection Error", f"Network handshake fault: {err}"))
            raw_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_")]
            self.root.after(0, lambda: self.status_var.set(f"Active profiles loaded: {len(raw_keys)}"))

    def show_polymorph_selection(self, scored_docs, mode="text"):
        if not scored_docs:
            messagebox.showinfo("Empty Registry", "No matching thermodynamic crystal arrays found for that search query.")
            self.refresh_checkbox_targets_panel()
            return
            
        pop = tk.Toplevel(self.root)
        pop.title("Select Structural Polymorph" if mode == "text" else "Search & Match Candidate Ranking")
        pop.geometry("640x280") 
        pop.transient(self.root); pop.grab_set()
        
        lbl_msg = "Select a crystallographic reference entry to simulate:" if mode == "text" else "Crystallographic candidates ranked by experimental peak alignment FOM scores:"
        ttk.Label(pop, text=lbl_msg, font=("Helvetica", 9, "bold")).pack(pady=6)
        
        frame = ttk.Frame(pop)
        frame.pack(fill="both", expand=True, padx=8, pady=4)
        scroll = ttk.Scrollbar(frame)
        scroll.pack(side="right", fill="y")
        
        tree = ttk.Treeview(frame, columns=("Score", "Formula", "ID", "Symmetry", "Stability"), show="headings", yscrollcommand=scroll.set, height=6, selectmode="extended")
        tree.heading("Score", text="Match Score")
        tree.heading("Formula", text="Material / Formula")
        tree.heading("ID", text="Material ID")
        tree.heading("Symmetry", text="Space Group Symbol")
        tree.heading("Stability", text="E Above Hull (eV/atom)")
        
        tree.column("Score", width=95, anchor="center")
        tree.column("Formula", width=120, anchor="center")
        tree.column("ID", width=95, anchor="center")
        tree.column("Symmetry", width=135, anchor="center")
        tree.column("Stability", width=145, anchor="center")
        tree.pack(fill="both", expand=True)
        scroll.config(command=tree.yview)
        
        doc_map = {}
        for item in scored_docs:
            score, avg_err, doc = item
            m_id = str(getattr(doc, 'material_id', 'Unknown'))
            sym = getattr(doc, 'symmetry', None)
            sym_symbol = getattr(sym, 'symbol', 'Unknown') if sym else 'Unknown'
            e_hull = getattr(doc, 'energy_above_hull', 0.0)
            formula = getattr(doc, 'formula_pretty', 'Unknown')
            
            score_string = f"{score:.1f}%" if mode == "peaks" else "N/A"
            
            item_id = tree.insert("", "end", values=(score_string, formula, m_id, sym_symbol, f"{e_hull:.4f}"))
            doc_map[item_id] = doc
            
        def trigger_plot_conversion():
            sel = tree.selection()
            if not sel: return
            pop.destroy()
            self.save_to_history() 
            for item_id in sel:
                target_doc = doc_map[item_id]
                self.simulate_and_add_reference(target_doc)
            
        ttk.Button(pop, text="Plot Theoretical Diffractogram", command=trigger_plot_conversion).pack(pady=8)

    def simulate_and_add_reference(self, doc):
        structure = doc.structure
        mat_id = str(doc.material_id)
        sym_symbol = doc.symmetry.symbol if doc.symmetry else "Unknown"
        formula = doc.formula_pretty if getattr(doc, 'formula_pretty', None) else "Ref"
        
        xmin, xmax = self.ax.get_xlim()
        if xmin >= xmax or xmin < 0:
            xmin, xmax = 5.0, 90.0
        
        calculator = XRDCalculator(wavelength="CuKa")
        pattern = calculator.get_pattern(structure, two_theta_range=(xmin, xmax))
        
        angles_grid = np.linspace(xmin, xmax, 2000)
        intensities_grid = np.zeros_like(angles_grid)
        sigma = 0.12  
        
        for x, y in zip(pattern.x, pattern.y):
            intensities_grid += y * np.exp(-((angles_grid - x) / sigma) ** 2)
            
        if np.max(intensities_grid) > 0:
            max_scale = 100.0
            data_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_")]
            if data_keys:
                max_scale = max([np.max(self.active_datasets[k]['intensities']) for k in data_keys])
            intensities_grid = (intensities_grid / np.max(intensities_grid)) * max_scale
            
        label = f"Ref: {formula} ({sym_symbol})"
        key_handle = f"__ref_{mat_id}"
        
        self.active_datasets[key_handle] = {'angles': angles_grid, 'intensities': intensities_grid, 'label': label}
        self.ax.plot(angles_grid, intensities_grid, linestyle='-.', linewidth=1.5, alpha=0.8, label=label)
        self.ax.legend(loc="upper right", frameon=True, fontsize=8)
        self.refresh_checkbox_targets_panel()
        self.canvas.draw()
        
        raw_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_")]
        self.status_var.set(f"Active profiles loaded: {len(raw_keys)}")

    def run_peak_optimization(self):
        if not self.peak_guesses:
            messagebox.showwarning("Missing Inputs", "Right-click on the graph canvas to specify peak center guesses first.")
            return
            
        keys_to_fit = [k for k, v in self.target_checkbox_vars.items() if v.get() and k in self.active_datasets]
        if not keys_to_fit:
            messagebox.showwarning("Selection Missing", "Please select at least one check box pattern row to fit.")
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
                p0.extend([amp_guess, g_x, 0.08])
                bounds_min.extend([0.0, g_x - 0.4, 0.005])
                bounds_max.extend([float(np.max(y_data)) * 2.0, g_x + 0.4, 1.5])

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
                    self.result_table.insert("", "end", values=(label_base, f"Peak {peak_counter}", f"{cent:.4f}°", f"{amp:.1f}", f"{fwhm:.4f}°"))
                    self.active_datasets[f"__fit_peak_{peak_counter}_{key}"] = {'angles': x_data, 'intensities': y_peak, 'label': f"{label_base} Pk {peak_counter} Fit"}
                    peak_counter += 1
                self.active_datasets[f"__fit_overall_composite_{key}"] = {'angles': x_data, 'intensities': y_fit_total, 'label': f"{label_base} Overall Fit"}
            except Exception as e:
                fit_errors.append(f"{label_base}: {e}")
                
        self.ax.legend(loc="upper right", frameon=True, fontsize=8); self.canvas.draw()
        if fit_errors: messagebox.showerror("Fitting Errors Encountered", "\n".join(fit_errors))

    def select_and_plot_files(self):
        files = filedialog.askopenfilenames(
            title="Select XRD Data Files",
            filetypes=[("XRD Datasets", ("*.csv", "*.xrdml")), ("Spreadsheets", "*.csv"), ("XML Readouts", "*.xrdml"), ("All Files", "*.*")]
        )
        if not files: return
        
        self.save_to_history() 
        loaded_count = 0; error_logs = []
        
        for file_path in files:
            if file_path in self.active_datasets: continue
            try:
                angles, intensities, label = load_xrd_data(file_path)
                self.ax.plot(angles, intensities, label=label, linewidth=1.2)
                self.active_datasets[file_path] = {'angles': angles, 'intensities': intensities, 'label': label}
                loaded_count += 1
            except Exception as e:
                error_logs.append(f"{os.path.basename(file_path)}: {str(e)}")
                
        if loaded_count > 0:
            self.ax.legend(loc="upper right", frameon=True, fontsize=9)
            self.refresh_checkbox_targets_panel()
            self.canvas.draw()
            raw_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_")]
            self.status_var.set(f"Active profiles loaded: {len(raw_keys)}")
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
                if path_key.startswith("__fit_") or path_key.startswith("__ref_"):
                    # Synthetic curves have no real path; build a safe name from the label.
                    raw_name = data.get('label', path_key)
                    b_name = "".join(c if (c.isalnum() or c in "-.") else "_" for c in raw_name).strip("_")
                    if not b_name:
                        b_name = "curve"
                else:
                    b_name = os.path.splitext(os.path.basename(path_key))[0]
                out_path = os.path.join(out_dir, f"clean_{b_name}.csv")
                pd.DataFrame({'Angle': data['angles'], 'Intensity': data['intensities']}).to_csv(out_path, index=False)
                success_count += 1
            except Exception as e: print(f"Exception tracking file save: {e}")
        messagebox.showinfo("Export Complete", f"Successfully saved {success_count} data profiles.")

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

    # ==========================================
    # RRUFF (powder) reference database methods
    # ==========================================
    def rruff_open_library(self):
        path = filedialog.askopenfilename(
            title="Open a consolidated RRUFF powder .h5 library",
            filetypes=[("RRUFF powder library", ("*.h5", "*.hdf5")), ("All Files", "*.*")])
        if not path:
            return
        try:
            self.rruff_lib = load_rruff_powder_h5_library(path)
        except Exception as e:
            messagebox.showerror("Library Error", f"Could not open library:\n{e}")
            return
        self.rruff_local_dir = None
        self.rruff_status_var.set(f"RRUFF library: {len(self.rruff_lib['entries'])} patterns (precomputed peaks). Search or Match.")

    def rruff_pick_local_folder(self):
        d = filedialog.askdirectory(title="Select a folder of RRUFF powder .txt files")
        if not d:
            return
        if not any(fn.lower().endswith('.txt') for fn in os.listdir(d)):
            messagebox.showwarning("No Files", "That folder contains no .txt files.")
            return
        self.rruff_local_dir = d
        self.rruff_lib = None
        n = len([f for f in os.listdir(d) if f.lower().endswith('.txt')])
        self.rruff_status_var.set(f"RRUFF folder: {n} files. Search or Match by peaks.")

    def _rruff_read_reference(self, hit):
        """Returns (x, y, label) for a hit from the .h5 library or a powder file."""
        if hit.get('group') is not None and self.rruff_lib:
            with h5py.File(self.rruff_lib['path'], 'r') as f:
                g = f['spectra'][hit['group']]
                x = np.array(g['x'][:], dtype=float); y = np.array(g['y'][:], dtype=float)
        else:
            with open(hit['path'], 'r', encoding='utf-8', errors='ignore') as fh:
                x, y, _peaks, _meta = parse_rruff_powder(fh.read())
        label = f"RRUFF: {hit['name']}" + (f" ({hit['id']})" if hit['id'] else "")
        return x, y, label

    def _rruff_add_reference(self, x, y, label, key, name=None, rid=None, url=None):
        data_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_") and not k.startswith("__ref_")]
        if data_keys and np.max(y) > 0:
            max_scale = max(np.max(self.active_datasets[k]['intensities']) for k in data_keys)
            y = (y / np.max(y)) * max_scale
        self.active_datasets[key] = {'angles': x, 'intensities': y, 'label': label,
                                     'rruff_name': name, 'rruff_id': rid, 'rruff_url': url}

    def _rruff_candidates(self, query=""):
        """[(name, id, url_or_None, key_dict)] from the active RRUFF source."""
        q = (query or "").strip().lower()
        out = []
        if self.rruff_lib:
            for e in self.rruff_lib['entries']:
                if not q or q in f"{e['name']} {e['id']}".lower():
                    out.append({'name': e['name'], 'id': e['id'], 'url': e.get('url'),
                                'group': e['group'], 'peaks': e['peaks']})
        elif self.rruff_local_dir:
            for fn in sorted(os.listdir(self.rruff_local_dir)):
                if not fn.lower().endswith('.txt'):
                    continue
                name, rid = rruff_meta_from_filename(fn)
                if not q or q in f"{name} {rid} {fn}".lower():
                    out.append({'name': name, 'id': rid, 'url': None,
                                'path': os.path.join(self.rruff_local_dir, fn)})
        return out

    def rruff_run_search(self):
        self.rruff_results_list.delete(0, tk.END)
        self.rruff_search_hits = []
        if not self.rruff_lib and not self.rruff_local_dir:
            messagebox.showinfo("No RRUFF Source", "Open an .h5 library or a local folder first.")
            return
        hits = self._rruff_candidates(self.ent_rruff_query.get())
        if not hits:
            self.rruff_status_var.set("RRUFF: no matches.")
            return
        self.rruff_search_hits = hits[:500]
        for h in self.rruff_search_hits:
            self.rruff_results_list.insert(tk.END, f"{h['name']}" + (f" · {h['id']}" if h['id'] else ""))
        self.rruff_status_var.set(f"RRUFF: {len(hits)} match(es)" + (" (first 500)." if len(hits) > 500 else "."))

    def _rruff_open_selected_page(self, event=None):
        for idx in self.rruff_results_list.curselection():
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
                x, y, label = self._rruff_read_reference(hit)
                if len(x) == 0:
                    continue
            except Exception:
                continue
            key = f"__ref_rruff_{hit['name']}_{hit['id']}_{idx}"
            self._rruff_add_reference(x, y, label, key, name=hit['name'], rid=hit['id'], url=hit.get('url'))
            added += 1
        if added:
            self.replot_and_refresh_canvas()
            self.rruff_status_var.set(f"RRUFF: overlaid {added} reference(s).")

    def rruff_match_by_peaks(self):
        if not self.peak_guesses:
            messagebox.showinfo("Mark Peaks First",
                                "Turn on '🎯 Peak Selection', right-click on the plot to mark peaks, then Match.")
            return
        if not self.rruff_lib and not self.rruff_local_dir:
            messagebox.showinfo("No RRUFF Source", "Open an .h5 library or a local folder first.")
            return
        try:
            tolerance = float(self.ent_rruff_tol.get().strip())
        except ValueError:
            tolerance = 0.2
        exp = list(self.peak_guesses)
        candidates = self._rruff_candidates(self.ent_rruff_query.get())
        if not candidates:
            messagebox.showinfo("No Candidates", "No RRUFF references to match (check the search filter).")
            return

        self.rruff_status_var.set(f"Matching {len(candidates)} references ...")

        def worker():
            scored = []
            total = len(candidates)
            for i, hit in enumerate(candidates):
                try:
                    if hit.get('peaks') is not None:
                        ref_peaks = hit['peaks']  # precomputed (library)
                    else:
                        with open(hit['path'], 'r', encoding='utf-8', errors='ignore') as fh:
                            x, y, pk, _m = parse_rruff_powder(fh.read())
                        ref_peaks = pk if pk is not None else detect_reference_peaks(x, y)
                    score, avg, matched = peak_match_score(ref_peaks, exp, tolerance)
                    if matched > 0:
                        rec = dict(hit); rec.update({'score': score, 'avg': avg, 'matched': matched})
                        scored.append(rec)
                except Exception:
                    continue
                if (i % 200) == 0:
                    self.root.after(0, lambda i=i: self.rruff_status_var.set(f"Matching {i}/{total} ..."))
            scored.sort(key=lambda t: (-t['score'], t['avg']))
            self.root.after(0, lambda: self._show_rruff_match_results(scored[:100], len(exp), tolerance))

        threading.Thread(target=worker, daemon=True).start()

    def _show_rruff_match_results(self, scored, n_exp, tolerance):
        if not scored:
            self.rruff_status_var.set("RRUFF: no references matched the marked peaks.")
            messagebox.showinfo("No Matches", "No RRUFF reference had bands near your marked peaks.\nTry a larger tolerance.")
            return
        self.rruff_status_var.set(f"RRUFF: {len(scored)} candidate(s) ranked (tol ±{tolerance:g}°).")
        pop = tk.Toplevel(self.root)
        pop.title("RRUFF Powder — Search & Match")
        pop.geometry("640x340")
        pop.transient(self.root); pop.grab_set()
        ttk.Label(pop, text=f"Ranked by alignment of RRUFF bands with your {n_exp} marked peak(s) "
                            f"(±{tolerance:g}°). Double-click a row to open its RRUFF page.",
                  font=("Helvetica", 9, "bold")).pack(pady=6, padx=8, anchor="w")
        frame = ttk.Frame(pop)
        frame.pack(fill="both", expand=True, padx=8, pady=4)
        scroll = ttk.Scrollbar(frame); scroll.pack(side="right", fill="y")
        tree = ttk.Treeview(frame, columns=("Score", "Mineral", "ID", "Matched"), show="headings",
                            yscrollcommand=scroll.set, height=10, selectmode="extended")
        for c, t, w in (("Score", "Match Score", 100), ("Mineral", "Mineral", 240),
                        ("ID", "RRUFF ID", 110), ("Matched", "Peaks Matched", 120)):
            tree.heading(c, text=t); tree.column(c, width=w, anchor=("w" if c == "Mineral" else "center"))
        tree.pack(fill="both", expand=True); scroll.config(command=tree.yview)
        row_map = {}
        for rec in scored:
            iid = tree.insert("", "end", values=(f"{rec['score']:.0f}%", rec['name'], rec['id'], f"{rec['matched']}/{n_exp}"))
            row_map[iid] = rec

        def open_pages(event=None):
            opened = 0
            for iid in tree.selection():
                rec = row_map.get(iid)
                if rec:
                    url = rruff_url(rec['name'], rec.get('id'), rec.get('url'))
                    if url:
                        webbrowser.open_new_tab(url); opened += 1
            if opened == 0:
                messagebox.showinfo("Nothing Selected", "Select a row, then open its RRUFF page.")
        tree.bind("<Double-1>", open_pages)

        def overlay_chosen():
            sel = tree.selection()
            if not sel:
                return
            pop.destroy(); self.save_to_history(); added = 0
            for iid in sel:
                hit = row_map[iid]
                try:
                    x, y, label = self._rruff_read_reference(hit)
                    if len(x) == 0:
                        continue
                except Exception:
                    continue
                key = f"__ref_rruff_{hit['name']}_{hit['id']}_match"
                self._rruff_add_reference(x, y, label, key, name=hit['name'], rid=hit['id'], url=hit.get('url'))
                added += 1
            if added:
                self.replot_and_refresh_canvas()
                self.rruff_status_var.set(f"RRUFF: overlaid {added} matched reference(s).")

        btn_row = ttk.Frame(pop); btn_row.pack(pady=8)
        ttk.Button(btn_row, text="🔗 Open RRUFF Page(s)", command=open_pages).pack(side="left", padx=4)
        ttk.Button(btn_row, text="➕ Overlay Selected Match(es)", command=overlay_chosen).pack(side="left", padx=4)

    def clear_canvas(self):
        if self.active_datasets:
            self.save_to_history() 
        self.active_datasets.clear(); self.clear_fitted_artists(); self.cursor_line = None  
        self.ax.clear(); self.configure_axis_labels(); self.refresh_checkbox_targets_panel(); self.canvas.draw()
        self.fitting_mode_active = False; self.normalization_mode_active = False
        self.btn_fit_toggle.config(text="🎯 Peak Selection: OFF"); self.btn_normalize_toggle.config(text="⚖️ Normalize to Peak"); self.btn_run_fit.config(state="disabled")
        self.status_var.set("Active profiles loaded: 0"); self.cursor_var.set("Cursor Position: 2θ = --")


if __name__ == "__main__":
    root = tk.Tk()
    app = XRDPlotterGUI(root)
    root.mainloop()