import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import threading

# Embed Matplotlib into Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit

# Conditional Imports to ensure app stability if packages are absent on launch
try:
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from mp_api.client import MPRester
    MP_LIBRARIES_AVAILABLE = True
except ImportError:
    MP_LIBRARIES_AVAILABLE = False

# ==========================================
# GLOBAL CONFIGURATIONS & CONSTANTS
# ==========================================
VERSION_TAG = "v2026.06.09.1"
KEY_FILE_NAME = "mp_api_key.txt"


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


# ==========================================
# PARSING ENGINE CORE LOGIC
# ==========================================

def load_xrd_data(file_path):
    """Parses custom XRD CSV or native XML XRDML configuration files layout."""
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    sample_id = os.path.splitext(filename)[0]
    
    if ext == '.csv':
        skiprows = 0
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for idx, line in enumerate(f):
                    if 'Sample identification' in line:
                        parts = line.split(',')
                        if len(parts) > 1 and parts[1].strip():
                            sample_id = parts[1].strip()
                    if '[Scan points]' in line:
                        skiprows = idx + 1
                        break
        except Exception as e:
            raise IOError(f"Error reading CSV header: {e}")
            
        if skiprows == 0:
            raise ValueError("Could not locate '[Scan points]' tag in CSV data frame.")
            
        try:
            df = pd.read_csv(file_path, skiprows=skiprows)
            df.columns = df.columns.str.strip()
            if 'Angle' in df.columns and 'Intensity' in df.columns:
                return df['Angle'].values, df['Intensity'].values, sample_id
            else:
                raise ValueError("Missing required 'Angle' or 'Intensity' columns.")
        except Exception as e:
            raise ValueError(f"Parsing error inside CSV rows: {e}")

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
        self.guess_lines_artists = []
        self.fitted_curves_artists = []
        self.target_checkbox_vars = {} 
        
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
        
        self.btn_normalize_toggle = ttk.Button(sidebar_frame, text="⚖️ Normalize to Peak", command=self.toggle_normalization_mode)
        self.btn_normalize_toggle.pack(side="top", fill="x", pady=3)
        
        self.btn_fit_toggle = ttk.Button(sidebar_frame, text="🎯 Peak Fitting: OFF", command=self.toggle_fitting_mode)
        self.btn_fit_toggle.pack(side="top", fill="x", pady=3)
        
        self.btn_run_fit = ttk.Button(sidebar_frame, text="⚡ Optimize Fit", command=self.run_peak_optimization, state="disabled")
        self.btn_run_fit.pack(side="top", fill="x", pady=3)
        
        ttk.Button(sidebar_frame, text="📥 Export to CSV", command=self.export_active_data_to_csv).pack(side="top", fill="x", pady=3)
        ttk.Button(sidebar_frame, text="🗑️ Clear Canvas", command=self.clear_canvas).pack(side="top", fill="x", pady=3)
        
        ttk.Separator(sidebar_frame, orient="horizontal").pack(side="top", fill="x", pady=10)
        
        # Status Active Profiles Badge
        self.status_var = tk.StringVar(value="Active profiles loaded: 0")
        lbl_status = ttk.Label(sidebar_frame, textvariable=self.status_var, font=("Helvetica", 9, "bold"), background="#cff4fc", foreground="#055160", relief="solid", borderwidth=1, padding=6, anchor="center")
        lbl_status.pack(side="top", fill="x", pady=2)
        
        # --- Materials Genome API Search Control Panel ---
        panel_search = ttk.LabelFrame(sidebar_frame, text=" 🌐 Materials Genome API Search ", padding=(8, 6))
        panel_search.pack(side="top", fill="x", pady=5)
        
        ttk.Label(panel_search, text="API Key:", font=("Helvetica", 8, "bold")).pack(anchor="w")
        
        key_entry_row = ttk.Frame(panel_search)
        key_entry_row.pack(fill="x", pady=(0, 4))
        
        self.ent_api_key = ttk.Entry(key_entry_row, show="*")
        self.ent_api_key.pack(side="left", fill="x", expand=True)
        
        btn_save_key = ttk.Button(key_entry_row, text="💾", width=3, command=self.save_api_key_locally)
        btn_save_key.pack(side="right", padx=(3, 0))
        
        ttk.Label(panel_search, text="Material Formula (e.g. TiO2):", font=("Helvetica", 8, "bold")).pack(anchor="w")
        self.ent_formula = ttk.Entry(panel_search)
        self.ent_formula.pack(fill="x", pady=(0, 6))
        
        btn_search = ttk.Button(panel_search, text="🔍 Search & Fetch Reference", command=self.execute_database_search)
        btn_search.pack(fill="x")

        # --- Checkbox Selector Panel ---
        self.panel_fit_targets = ttk.LabelFrame(sidebar_frame, text=" 🎯 Targets for Fitting ", padding=(8, 6))
        self.panel_fit_targets.pack(side="top", fill="x", pady=8)
        self.lbl_no_targets = ttk.Label(self.panel_fit_targets, text="No Scans Loaded", font=("Helvetica", 9, "italic"), foreground="#888888")
        self.lbl_no_targets.pack(side="top", anchor="w", padx=4)
        
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
            messagebox.showerror("IO Fault", f"Could not write configuration token to file system maps: {e}")

    def refresh_checkbox_targets_panel(self):
        for child in self.panel_fit_targets.winfo_children():
            child.destroy()
            
        raw_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_")]
        
        if not raw_keys:
            self.lbl_no_targets = ttk.Label(self.panel_fit_targets, text="No Scans Loaded", font=("Helvetica", 9, "italic"), foreground="#888888")
            self.lbl_no_targets.pack(side="top", anchor="w", padx=4)
            return

        self.target_checkbox_vars = {k: v for k, v in self.target_checkbox_vars.items() if k in raw_keys}
        
        for key in raw_keys:
            if key not in self.target_checkbox_vars:
                self.target_checkbox_vars[key] = tk.BooleanVar(value=True)
                
            row_frame = ttk.Frame(self.panel_fit_targets)
            row_frame.pack(side="top", fill="x", pady=2, expand=True)
                
            cb = ttk.Checkbutton(
                row_frame, 
                text=self.active_datasets[key]['label'], 
                variable=self.target_checkbox_vars[key]
            )
            cb.pack(side="left", anchor="w")
            
            btn_del = ttk.Button(
                row_frame, 
                text="❌", 
                width=2, 
                command=lambda k=key: self.remove_specific_dataset(k)
            )
            btn_del.pack(side="right", anchor="e")

    def remove_specific_dataset(self, key_to_remove):
        if key_to_remove in self.active_datasets:
            del self.active_datasets[key_to_remove]
            
        for k in list(self.active_datasets.keys()):
            if k.endswith(f"_{key_to_remove}"):
                del self.active_datasets[k]
                
        self.ax.clear()
        self.configure_axis_labels()
        self.cursor_line = None  
        
        for file_path, data in self.active_datasets.items():
            if file_path.startswith("__fit_composite"):
                self.ax.plot(data['angles'], data['intensities'], color='#000000', linestyle='-', linewidth=2.0, label=data['label'])
            elif file_path.startswith("__fit_"):
                self.ax.plot(data['angles'], data['intensities'], linestyle='--', linewidth=1.2, label=data['label'])
            elif file_path.startswith("__ref_"):
                self.ax.plot(data['angles'], data['intensities'], linestyle='-.', linewidth=1.5, alpha=0.8, label=data['label'])
            else:
                self.ax.plot(data['angles'], data['intensities'], label=data['label'], linewidth=1.2)
                
        if self.active_datasets:
            self.ax.legend(loc="upper right", frameon=True, fontsize=8)
            
        self.ax.relim()
        self.ax.autoscale_view()
        self.refresh_checkbox_targets_panel()
        self.canvas.draw()
        
        raw_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_")]
        self.status_var.set(f"Active profiles loaded: {len(raw_keys)}")

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
            self.btn_fit_toggle.config(text="🎯 Fitting Mode: ACTIVE")
            self.btn_run_fit.config(state="normal")
        else:
            self.btn_fit_toggle.config(text="🎯 Peak Fitting: OFF")
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
                window_span = 0.3  
                data_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_")]
                normalized_any = False
                
                for key in data_keys:
                    angles = self.active_datasets[key]['angles']
                    intensities = self.active_datasets[key]['intensities']
                    mask = (angles >= x_click - window_span) & (angles <= x_click + window_span)
                    if np.any(mask):
                        local_peak_max = np.max(intensities[mask])
                        if local_peak_max > 0:
                            self.active_datasets[key]['intensities'] = intensities / local_peak_max
                            normalized_any = True
                            
                if normalized_any:
                    self.clear_fitted_artists()
                    self.ax.clear()
                    self.configure_axis_labels()
                    self.cursor_line = None
                    for file_path, data in self.active_datasets.items():
                        if file_path.startswith("__ref_"):
                            self.ax.plot(data['angles'], data['intensities'], linestyle='-.', linewidth=1.5, alpha=0.8, label=data['label'])
                        else:
                            self.ax.plot(data['angles'], data['intensities'], label=data['label'], linewidth=1.2)
                    self.ax.legend(loc="upper right", frameon=True, fontsize=9)
                    self.ax.relim(); self.ax.autoscale_view(); self.canvas.draw()
                    
                self.normalization_mode_active = False
                self.btn_normalize_toggle.config(text="⚖️ Normalize to Peak")
                return

            elif self.fitting_mode_active and event.button == 3:
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
        self.clear_fitted_artists()

        for file_path in data_keys:
            data = self.active_datasets[file_path]
            intensities = data['intensities']
            if len(intensities) < 3: continue
            bg = snip_background(intensities, iterations=40)
            data['intensities'] = intensities - bg

        self.ax.clear()
        self.configure_axis_labels()
        self.cursor_line = None  
        for file_path, data in self.active_datasets.items():
            if file_path.startswith("__ref_"):
                self.ax.plot(data['angles'], data['intensities'], linestyle='-.', linewidth=1.5, alpha=0.8, label=data['label'])
            else:
                self.ax.plot(data['angles'], data['intensities'], label=data['label'], linewidth=1.2)
        self.ax.legend(loc="upper right", frameon=True, fontsize=9)
        self.ax.relim(); self.ax.autoscale_view(); self.canvas.draw()

    def execute_database_search(self):
        if not MP_LIBRARIES_AVAILABLE:
            messagebox.showinfo("Packages Missing", "To query reference data directly, run this command in your terminal folder:\npip install mp-api pymatgen")
            return
            
        api_key = self.ent_api_key.get().strip()
        formula = self.ent_formula.get().strip()
        
        if not formula:
            messagebox.showwarning("Missing Inputs", "Please specify a crystal target formula matrix (e.g. Si, TiO2).")
            return
        if not api_key:
            messagebox.showwarning("Missing Key", "Materials Project queries require an API key.")
            return
            
        self.status_var.set("Searching Materials Project...")
        threading.Thread(target=self._bg_search_worker, args=(api_key, formula), daemon=True).start()

    def _bg_search_worker(self, api_key, formula):
        try:
            with MPRester(api_key) as mpr:
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=["material_id", "structure", "symmetry", "energy_above_hull", "formula_pretty"]
                )
            self.root.after(0, self.show_polymorph_selection, docs)
        except Exception as e:
            self.root.after(0, lambda err=str(e): messagebox.showerror("API Connection Error", f"Network handshake fault: {err}"))
            raw_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_")]
            self.root.after(0, lambda: self.status_var.set(f"Active profiles loaded: {len(raw_keys)}"))

    def show_polymorph_selection(self, docs):
        if not docs:
            messagebox.showinfo("Empty Registry", "No matching thermodynamic crystal arrays found for that formula.")
            self.refresh_checkbox_targets_panel()
            return
            
        pop = tk.Toplevel(self.root)
        pop.title("Select Structural Polymorph")
        pop.geometry("560x260") # Slightly wider window layout parameters
        pop.transient(self.root); pop.grab_set()
        
        ttk.Label(pop, text="Select a crystallographic reference entry to simulate:", font=("Helvetica", 9, "bold")).pack(pady=6)
        
        frame = ttk.Frame(pop)
        frame.pack(fill="both", expand=True, padx=8, pady=4)
        scroll = ttk.Scrollbar(frame)
        scroll.pack(side="right", fill="y")
        
        # ADDED: Added 'Formula' to column registry tracking rules
        tree = ttk.Treeview(frame, columns=("Formula", "ID", "Symmetry", "Stability"), show="headings", yscrollcommand=scroll.set, height=6)
        tree.heading("Formula", text="Material / Formula")
        tree.heading("ID", text="Material ID")
        tree.heading("Symmetry", text="Space Group Symbol")
        tree.heading("Stability", text="E Above Hull (eV/atom)")
        
        tree.column("Formula", width=120, anchor="center")
        tree.column("ID", width=100, anchor="center")
        tree.column("Symmetry", width=140, anchor="center")
        tree.column("Stability", width=150, anchor="center")
        tree.pack(fill="both", expand=True)
        scroll.config(command=tree.yview)
        
        doc_map = {}
        for idx, doc in enumerate(docs):
            m_id = str(getattr(doc, 'material_id', f"MP-Index-{idx}"))
            sym = getattr(doc, 'symmetry', None)
            sym_symbol = getattr(sym, 'symbol', 'Unknown') if sym else 'Unknown'
            e_hull = getattr(doc, 'energy_above_hull', 0.0)
            formula = getattr(doc, 'formula_pretty', 'Unknown')
            
            # Insert the parsed dynamic text directly out into the row collection element array
            item_id = tree.insert("", "end", values=(formula, m_id, sym_symbol, f"{e_hull:.4f}"))
            doc_map[item_id] = doc
            
        def trigger_plot_conversion():
            sel = tree.selection()
            if not sel: return
            target_doc = doc_map[sel[0]]
            pop.destroy()
            self.simulate_and_add_reference(target_doc)
            
        ttk.Button(pop, text="Plot Theoretical Diffractogram", command=trigger_plot_conversion).pack(pady=8)

    def simulate_and_add_reference(self, doc):
        structure = doc.structure
        mat_id = str(doc.material_id)
        sym_symbol = doc.symmetry.symbol if doc.symmetry else "Unknown"
        formula = doc.formula_pretty if getattr(doc, 'formula_pretty', None) else "Ref"
        
        calculator = XRDCalculator(wavelength="CuKa")
        pattern = calculator.get_pattern(structure, two_theta_range=(5, 90))
        
        angles_grid = np.linspace(5, 90, 2000)
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
            filetypes=[("XRD Datasets (*.csv, *.xrdml)", "*.csv;*.xrdml"), ("Spreadsheets (*.csv)", "*.csv"), ("XML Readouts (*.xrdml)", "*.xrdml"), ("All Files", "*.*")]
        )
        if not files: return
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
        self.clear_fitted_artists()

        for f_path, data in list(self.active_datasets.items()):
            if f_path.startswith("__fit_"):
                del self.active_datasets[f_path]
                continue
            ang, intset = data['angles'], data['intensities']
            mask = (ang >= xmin) & (ang <= xmax)
            data['angles'] = ang[mask]
            data['intensities'] = intset[mask]

        self.ax.clear(); self.configure_axis_labels(); self.cursor_line = None  
        for f_path, data in self.active_datasets.items():
            if f_path.startswith("__ref_"):
                self.ax.plot(data['angles'], data['intensities'], linestyle='-.', linewidth=1.5, alpha=0.8, label=data['label'])
            else:
                self.ax.plot(data['angles'], data['intensities'], label=data['label'], linewidth=1.2)
                
        self.ax.legend(loc="upper right", frameon=True, fontsize=9)
        self.ax.set_xlim(xmin, xmax); self.ax.relim(); self.ax.autoscale_view(scalex=False, scaley=True); self.refresh_checkbox_targets_panel(); self.canvas.draw()
        raw_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_")]
        self.status_var.set(f"Active profiles loaded: {len(raw_keys)}")

    def export_active_data_to_csv(self):
        if not self.active_datasets: return
        out_dir = filedialog.askdirectory(title="Select Output Folder")
        if not out_dir: return 
        success_count = 0
        for path_key, data in self.active_datasets.items():
            try:
                b_name = os.path.splitext(os.path.basename(path_key))[0] if not path_key.startswith("__fit_") and not path_key.startswith("__ref_") else path_key.strip("__")
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

    def clear_canvas(self):
        self.active_datasets.clear(); self.clear_fitted_artists(); self.cursor_line = None  
        self.ax.clear(); self.configure_axis_labels(); self.refresh_checkbox_targets_panel(); self.canvas.draw()
        self.fitting_mode_active = False; self.normalization_mode_active = False
        self.btn_fit_toggle.config(text="🎯 Peak Fitting: OFF"); self.btn_normalize_toggle.config(text="⚖️ Normalize to Peak"); self.btn_run_fit.config(state="disabled")
        self.status_var.set("Active profiles loaded: 0"); self.cursor_var.set("Cursor Position: 2θ = --")


if __name__ == "__main__":
    root = tk.Tk()
    app = XRDPlotterGUI(root)
    root.mainloop()
