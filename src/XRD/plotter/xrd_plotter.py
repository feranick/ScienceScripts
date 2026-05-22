import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# Embed Matplotlib into Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit

# ==========================================
# GLOBAL CONFIGURATIONS & CONSTANTS
# ==========================================
VERSION_TAG = "v2026.05.22.1"


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
    # Dynamically bound iterations to safely prevent index crashes on tightly cropped regions
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
        self.root.geometry("1000x780")
        self.root.minsize(800, 600)
        
        style = ttk.Style()
        style.theme_use('clam')
        
        # In-memory arrays session states initialization
        self.active_datasets = {}
        self.peak_guesses = []
        self.guess_lines_artists = []
        self.fitted_curves_artists = []
        
        self.fitting_mode_active = False
        self.cursor_line = None  
        
        # --- Top Control Bar Panel ---
        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(side="top", fill="x")
        
        ttk.Button(control_frame, text="📁 Select File(s)", command=self.select_and_plot_files).pack(side="left", padx=3)
        ttk.Button(control_frame, text="✂️ Crop to View", command=self.crop_to_current_view).pack(side="left", padx=3)
        ttk.Button(control_frame, text="✨ Subtract Background", command=self.subtract_background_profile).pack(side="left", padx=3)
        
        # Fitting Interface Mode Anchors
        self.btn_fit_toggle = ttk.Button(control_frame, text="🎯 Peak Fitting: OFF", command=self.toggle_fitting_mode)
        self.btn_fit_toggle.pack(side="left", padx=3)
        
        self.btn_run_fit = ttk.Button(control_frame, text="⚡ Optimize Fit", command=self.run_peak_optimization, state="disabled")
        self.btn_run_fit.pack(side="left", padx=3)
        
        ttk.Button(control_frame, text="📥 Export to CSV", command=self.export_active_data_to_csv).pack(side="left", padx=3)
        ttk.Button(control_frame, text="🗑️ Clear Canvas", command=self.clear_canvas).pack(side="left", padx=3)
        
        self.status_var = tk.StringVar(value="System initialized. Load files to map data.")
        ttk.Label(control_frame, textvariable=self.status_var, font=("Helvetica", 9, "italic")).pack(side="left", padx=15)
        
        ttk.Label(control_frame, text=VERSION_TAG, font=("Helvetica", 8), foreground="#888888").pack(side="right", padx=5)
        
        # --- Middle Layout Display Configuration Split ---
        self.main_container = ttk.PanedWindow(root, orient="vertical")
        self.main_container.pack(fill="both", expand=True, padx=10, pady=5)
        
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

        # --- Bottom Panel Table Dashboard ---
        self.table_frame = ttk.LabelFrame(self.main_container, text=" 📊 Peak Optimization Results Dashboard ", padding=5)
        self.main_container.add(self.table_frame, weight=1)
        
        self.result_table = ttk.Treeview(self.table_frame, columns=("Peak", "Center", "Amplitude", "FWHM"), show="headings", height=5)
        self.result_table.heading("Peak", text="Peak Identity Index")
        self.result_table.heading("Center", text="Center position (2θ°)")
        self.result_table.heading("Amplitude", text="Peak Amplitude (counts)")
        self.result_table.heading("FWHM", text="FWHM Line Width (deg)")
        
        self.result_table.column("Peak", width=120, anchor="center")
        self.result_table.column("Center", width=180, anchor="center")
        self.result_table.column("Amplitude", width=180, anchor="center")
        self.result_table.column("FWHM", width=180, anchor="center")
        self.result_table.pack(fill="both", expand=True)

        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

    def configure_axis_labels(self):
        self.ax.set_xlabel(r"2$\theta$ Angle (degrees)", fontsize=10, fontweight='bold')
        self.ax.set_ylabel("Intensity (counts)", fontsize=10, fontweight='bold')
        self.ax.set_title("XRD Diffraction Pattern Analysis", fontsize=11, fontweight='bold', pad=8)
        self.ax.grid(True, linestyle="--", alpha=0.5)

    def toggle_fitting_mode(self):
        if not self.active_datasets:
            messagebox.showwarning("Execution Halted", "Load standard experimental datasets profiles before entering optimization modes.")
            return
        self.fitting_mode_active = not self.fitting_mode_active
        if self.fitting_mode_active:
            self.btn_fit_toggle.config(text="🎯 Fitting Mode: ACTIVE")
            self.btn_run_fit.config(state="normal")
            self.status_var.set("Fitting Mode active. Right-click on the graph to map approximate peak center targets.")
        else:
            self.btn_fit_toggle.config(text="🎯 Peak Fitting: OFF")
            self.btn_run_fit.config(state="disabled")
            self.status_var.set("Fitting Mode deactivated.")

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
        if self.fitting_mode_active and event.inaxes == self.ax and event.button == 3:
            x_guess = event.xdata
            self.peak_guesses.append(x_guess)
            
            guess_line = self.ax.axvline(x_guess, color='#d63384', linestyle=':', linewidth=1.5, label="Peak Guess" if not self.guess_lines_artists else "")
            self.guess_lines_artists.append(guess_line)
            self.canvas.draw_idle()
            self.status_var.set(f"Added peak approximation target coordinate entry near 2θ = {x_guess:.3f}°.")

    def subtract_background_profile(self):
        """Executes automated iterative SNIP background stripping across loaded patterns."""
        if not self.active_datasets:
            messagebox.showwarning("No Data", "No active datasets found on canvas to apply baseline corrections.")
            return
            
        data_keys = [k for k in self.active_datasets.keys() if not k.startswith("__fit_")]
        if not data_keys: return

        # Wipe mathematical fits if a parameter changes baselines background profile
        self.clear_fitted_artists()

        for file_path in data_keys:
            data = self.active_datasets[file_path]
            intensities = data['intensities']
            if len(intensities) < 3: continue
            
            bg = snip_background(intensities, iterations=40)
            data['intensities'] = intensities - bg

        # Redraw core spectra signals tracks
        self.ax.clear()
        self.configure_axis_labels()
        self.cursor_line = None  
        
        for file_path, data in self.active_datasets.items():
            self.ax.plot(data['angles'], data['intensities'], label=data['label'], linewidth=1.2)
                
        self.ax.legend(loc="upper right", frameon=True, fontsize=9)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        self.status_var.set("Baseline background successfully stripped across profiles using the SNIP filter algorithm.")

    def run_peak_optimization(self):
        if not self.peak_guesses:
            messagebox.showwarning("Missing Inputs", "Right-click on the graph canvas to specify peak center guesses first.")
            return
            
        first_key = list(self.active_datasets.keys())[0]
        x_data = self.active_datasets[first_key]['angles']
        y_data = self.active_datasets[first_key]['intensities']
        
        for line in self.fitted_curves_artists: line.remove()
        self.fitted_curves_artists.clear()
        for row in self.result_table.get_children(): self.result_table.delete(row)

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
            total_fit_line, = self.ax.plot(x_data, y_fit_total, color='black', linestyle='-', linewidth=2.2, label="Overall Fit Line")
            self.fitted_curves_artists.append(total_fit_line)
            
            peak_counter = 1
            for i in range(0, len(p_opt), 3):
                amp, cent, wid = p_opt[i], p_opt[i+1], p_opt[i+2]
                y_peak = gaussian_profile(x_data, amp, cent, wid)
                
                pk_line, = self.ax.plot(x_data, y_peak, linestyle='--', linewidth=1.2, label=f"Peak {peak_counter} Trace")
                self.fitted_curves_artists.append(pk_line)
                
                fwhm = 2.0 * np.sqrt(np.log(2)) * wid
                self.result_table.insert("", "end", values=(f"Peak {peak_counter}", f"{cent:.4f}°", f"{amp:.1f}", f"{fwhm:.4f}°"))
                self.active_datasets[f"__fit_peak_{peak_counter}"] = {'angles': x_data, 'intensities': y_peak, 'label': f"Peak {peak_counter} Fit Result"}
                peak_counter += 1
                
            self.active_datasets["__fit_overall_composite"] = {'angles': x_data, 'intensities': y_fit_total, 'label': "Overall Deconvoluted Curve Fit"}
            self.ax.legend(loc="upper right", frameon=True, fontsize=8)
            self.canvas.draw()
            self.status_var.set("Nonlinear mathematical optimization routine completed successfully.")
        except Exception as e:
            messagebox.showerror("Optimization Convergence Exception", f"Optimization routine failed to converge: {e}")

    def select_and_plot_files(self):
        files = filedialog.askopenfilenames(
            title="Select XRD Data Files",
            filetypes=[("XRD Datasets (*.csv, *.xrdml)", "*.csv;*.xrdml"), ("Spreadsheets (*.csv)", "*.csv"), ("XML Readouts (*.xrdml)", "*.xrdml"), ("All Files", "*.*")]
        )
        if not files: return
            
        loaded_count = 0
        error_logs = []
        
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
            self.canvas.draw()
            self.status_var.set(f"Added {loaded_count} patterns. Active datasets array total count: {len(self.active_datasets)}.")
        if error_logs:
            messagebox.showwarning("Import Errors", "\n".join(error_logs))

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

        self.ax.clear()
        self.configure_axis_labels()
        self.cursor_line = None  
        
        for f_path, data in self.active_datasets.items():
            self.ax.plot(data['angles'], data['intensities'], label=data['label'], linewidth=1.2)
                
        self.ax.legend(loc="upper right", frameon=True, fontsize=9)
        self.ax.set_xlim(xmin, xmax)  
        self.ax.relim()
        self.ax.autoscale_view(scalex=False, scaley=True) 
        self.canvas.draw()
        self.status_var.set(f"Permanently sliced spectra arrays to bounds: {xmin:.2f}° to {xmax:.2f}°.")

    def export_active_data_to_csv(self):
        if not self.active_datasets: return
        out_dir = filedialog.askdirectory(title="Select Output Save Directory Target Folder")
        if not out_dir: return 
            
        success_count = 0
        for path_key, data in self.active_datasets.items():
            try:
                b_name = os.path.splitext(os.path.basename(path_key))[0] if not path_key.startswith("__fit_") else path_key.strip("__")
                out_path = os.path.join(out_dir, f"clean_{b_name}.csv")
                pd.DataFrame({'Angle': data['angles'], 'Intensity': data['intensities']}).to_csv(out_path, index=False)
                success_count += 1
            except Exception as e:
                print(f"Exception tracking file save outputs array indices block: {e}")
                
        messagebox.showinfo("Export Cycle Terminated", f"Exported {success_count} structural configuration data profiles safely into:\n{out_dir}")

    def clear_fitted_artists(self):
        for line in self.fitted_curves_artists: line.remove()
        for line in self.guess_lines_artists: line.remove()
        self.fitted_curves_artists.clear()
        self.guess_lines_artists.clear()
        self.peak_guesses.clear()
        for row in self.result_table.get_children(): self.result_table.delete(row)

    def clear_canvas(self):
        self.active_datasets.clear()
        self.clear_fitted_artists()
        self.cursor_line = None  
        self.ax.clear()
        self.configure_axis_labels()
        self.canvas.draw()
        self.fitting_mode_active = False
        self.btn_fit_toggle.config(text="🎯 Peak Fitting: OFF")
        self.btn_run_fit.config(state="disabled")
        self.status_var.set("Canvas matrix dropped. System ready to receive new scan files array.")
        self.cursor_var.set("Cursor Position: 2θ = --")


if __name__ == "__main__":
    root = tk.Tk()
    app = XRDPlotterGUI(root)
    root.mainloop()
