import os
import io
import re
import zipfile
import threading
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
from scipy.signal import savgol_filter

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
VERSION_TAG = "raman-v2026.07.22.2"

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
    """
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    base_id = os.path.splitext(filename)[0]

    if ext in ('.h5', '.hdf5'):
        return _load_labspec_h5(file_path, base_id)
    elif ext == '.xml':
        return _load_labspec_xml(file_path, base_id)
    elif ext in ('.txt', '.csv', '.dat', '.asc', '.spc'):
        return _load_two_column(file_path, base_id)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


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


# ==========================================
# GUI & EMBEDDED PLOTTING INTERFACE
# ==========================================

class RamanPlotterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Raman Spectra Analysis Toolkit")
        self.root.geometry("1050x780")
        self.root.minsize(850, 600)

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

        # --- Left Sidebar Panel Layout ---
        sidebar_frame = ttk.Frame(root, padding=12, relief="flat")
        sidebar_frame.pack(side="left", fill="y", padx=5, pady=5)

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
        ttk.Label(adjust_frame, text="Pick a target, click Offset or Scale, then scroll the wheel over the plot.",
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

        ttk.Label(panel_rruff, text="Search mineral / ID:", font=("Helvetica", 8, "bold")).pack(anchor="w", pady=(4, 0))
        search_row = ttk.Frame(panel_rruff)
        search_row.pack(fill="x", pady=(0, 4))
        self.ent_rruff_query = ttk.Entry(search_row)
        self.ent_rruff_query.pack(side="left", fill="x", expand=True)
        self.ent_rruff_query.bind("<Return>", lambda e: self.rruff_run_search())
        ttk.Button(search_row, text="🔍", width=3, command=self.rruff_run_search).pack(side="right", padx=(3, 0))

        self.rruff_results_list = tk.Listbox(panel_rruff, height=4, exportselection=False)
        self.rruff_results_list.pack(fill="x", pady=(0, 3))
        self.rruff_search_hits = []

        ttk.Button(panel_rruff, text="➕ Overlay Selected Reference", command=self.rruff_overlay_selected).pack(fill="x", pady=(0, 2))

        self.rruff_status_var = tk.StringVar(value="RRUFF: no set cached.")
        ttk.Label(panel_rruff, textvariable=self.rruff_status_var, font=("Helvetica", 8), foreground="#555555", wraplength=250).pack(anchor="w")

        self.rruff_local_dir = None
        self._refresh_rruff_status()

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
                'label': v['label']
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
                cb = ttk.Checkbutton(row_frame, text=data['label'], variable=self.target_checkbox_vars[key])
                cb.pack(side="left", anchor="w")
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
                line, = self.ax.plot(data['angles'], data['intensities'], label=data['label'], linewidth=1.2)
                self.line_map[file_path] = line
        for g_x in self.peak_guesses:
            guess_line = self.ax.axvline(g_x, color='#d63384', linestyle=':', linewidth=1.5)
            self.guess_lines_artists.append(guess_line)
        if self.active_datasets:
            self.ax.legend(loc="upper right", frameon=True, fontsize=8)
        self.ax.relim(); self.ax.autoscale_view(); self.refresh_checkbox_targets_panel()
        self.refresh_adjust_targets(); self.canvas.draw()

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
            filetypes=[("Raman Datasets", ("*.h5", "*.hdf5", "*.xml", "*.txt", "*.csv", "*.dat", "*.asc")),
                       ("HORIBA LabSpec HDF5", ("*.h5", "*.hdf5")),
                       ("HORIBA LabSpec XML", "*.xml"),
                       ("Text / CSV / RRUFF", ("*.txt", "*.csv", "*.dat", "*.asc")),
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
        if self.adjust_mode is None or event.inaxes != self.ax:
            return
        label = self.combo_adjust_target.get()
        key = self._adjust_key_by_label.get(label)
        if key is None or key not in self.active_datasets:
            return
        step_val = getattr(event, 'step', 0) or 0
        if step_val == 0:
            step_val = 1 if getattr(event, 'button', None) == 'up' else -1
        direction = 1.0 if step_val > 0 else -1.0

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
        self._refresh_rruff_status()

    def rruff_run_search(self):
        query = self.ent_rruff_query.get().strip()
        self.rruff_results_list.delete(0, tk.END)
        self.rruff_search_hits = []
        if self.rruff_local_dir:
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
                with open(hit['path'], 'r', encoding='utf-8', errors='ignore') as f:
                    x, y, label = _parse_two_column_text(f.read(), hit['name'])
                if len(x) == 0:
                    continue
            except Exception:
                continue
            # Scale reference to the current data maximum for easy comparison.
            data_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_") and not k.startswith("__ref_")]
            if data_keys and np.max(y) > 0:
                max_scale = max(np.max(self.active_datasets[k]['intensities']) for k in data_keys)
                y = (y / np.max(y)) * max_scale
            if not label.startswith("RRUFF"):
                label = f"RRUFF: {hit['name']}" + (f" ({hit['id']})" if hit['id'] else "")
            key = f"__ref_{hit['name']}_{hit['id']}_{idx}"
            self.active_datasets[key] = {'angles': x, 'intensities': y, 'label': label}
            added += 1
        if added:
            self.replot_and_refresh_canvas()
            self.rruff_status_var.set(f"RRUFF: overlaid {added} reference spectrum(s).")

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
