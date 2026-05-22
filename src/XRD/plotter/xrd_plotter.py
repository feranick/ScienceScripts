import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# Embed Matplotlib into Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# ==========================================
# PARSING ENGINE CORE LOGIC
# ==========================================

def load_xrd_data(file_path):
    """
    Parses a single .csv or .xrdml file, returning:
    (angles_array, intensities_array, sample_label)
    """
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    sample_id = os.path.splitext(filename)[0] # Fallback label
    
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
                    n = len(counts)
                    angles = np.linspace(start_val, end_val, n)
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
        self.root.title("Additive XRD Visualizer & Exporter")
        self.root.geometry("900x650")
        self.root.minsize(750, 500)
        
        style = ttk.Style()
        style.theme_use('clam')
        
        # In-memory dictionary to hold data matrices for exporting
        self.active_datasets = {}
        
        # --- Top Control Bar Panel ---
        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(side="top", fill="x")
        
        btn_select = ttk.Button(control_frame, text="Select File(s) to Plot", command=self.select_and_plot_files)
        btn_select.pack(side="left", padx=5, ipadx=3)
        
        btn_export = ttk.Button(control_frame, text="Export Plotted to CSV", command=self.export_active_data_to_csv)
        btn_export.pack(side="left", padx=5, ipadx=3)
        
        btn_clear = ttk.Button(control_frame, text="Clear Canvas", command=self.clear_canvas)
        btn_clear.pack(side="left", padx=5, ipadx=3)
        
        self.status_var = tk.StringVar(value="No data loaded. Select files to generate plot profiles.")
        lbl_status = ttk.Label(control_frame, textvariable=self.status_var, font=("Helvetica", 9, "italic"))
        lbl_status.pack(side="left", padx=15)
        
        # --- Main Interactive Plot Canvas Panel ---
        self.plot_frame = ttk.Frame(root, padding=5, relief="groove")
        self.plot_frame.pack(side="bottom", fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.configure_axis_labels()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.toolbar.pack(side="top", fill="x")

    def configure_axis_labels(self):
        """Sets axis labels and styling."""
        self.ax.set_xlabel(r"2$\theta$ Angle (degrees)", fontsize=10, fontweight='bold')
        self.ax.set_ylabel("Intensity (counts)", fontsize=10, fontweight='bold')
        self.ax.set_title("XRD Diffraction Pattern Analysis", fontsize=12, fontweight='bold', pad=10)
        self.ax.grid(True, linestyle="--", alpha=0.5)

    def select_and_plot_files(self):
        """Triggers file browser, layering files additively onto the canvas."""
        files = filedialog.askopenfilenames(
            title="Select XRD Data Files",
            filetypes=[
                ("XRD Files (*.csv, *.xrdml)", "*.csv;*.xrdml"),
                ("CSV Spreadsheet (*.csv)", "*.csv"),
                ("Native XML Format (*.xrdml)", "*.xrdml"),
                ("All File Variations", "*.*")
            ]
        )
        
        if not files:
            return
            
        loaded_count = 0
        error_logs = []
        
        for file_path in files:
            # Skip if file is already loaded and active on the canvas
            if file_path in self.active_datasets:
                continue
            try:
                angles, intensities, label = load_xrd_data(file_path)
                self.ax.plot(angles, intensities, label=label, linewidth=1.2)
                
                # Store structural variables inside the session cache map
                self.active_datasets[file_path] = {
                    'angles': angles,
                    'intensities': intensities,
                    'label': label
                }
                loaded_count += 1
            except Exception as e:
                error_logs.append(f"{os.path.basename(file_path)}: {str(e)}")
                
        if loaded_count > 0:
            self.ax.legend(loc="upper right", frameon=True, fontsize=9)
            self.canvas.draw()
            
            total_tracks = len(self.active_datasets)
            self.status_var.set(f"Added {loaded_count} track(s). Total active patterns on canvas: {total_tracks}.")
        
        if error_logs:
            err_msg = "Exceptions occurred while importing data:\n\n" + "\n".join(error_logs)
            messagebox.showwarning("File Execution Warnings", err_msg)

    def export_active_data_to_csv(self):
        """Iterates through current plot cache memory and saves clean individual CSV files."""
        if not self.active_datasets:
            messagebox.showwarning("Export Void", "There are no active dataset profiles on the canvas to export.")
            return
            
        # Prompt user to map onto an output save directory destination folder
        output_dir = filedialog.askdirectory(title="Select Output Folder for Clean CSV Files")
        if not output_dir:
            return # Cancelled by user
            
        success_count = 0
        for original_path, data in self.active_datasets.items():
            try:
                # Isolate root baseline name string to build clean export handle
                base_name = os.path.splitext(os.path.basename(original_path))[0]
                export_filename = f"clean_{base_name}.csv"
                export_path = os.path.join(output_dir, export_filename)
                
                # Construct clean frame dataframe object
                export_df = pd.DataFrame({
                    'Angle': data['angles'],
                    'Intensity': data['intensities']
                })
                
                export_df.to_csv(export_path, index=False)
                success_count += 1
            except Exception as e:
                messagebox.showerror("Export Exception Encounted", f"Could not extract {base_name}: {e}")
                
        messagebox.showinfo(
            "Export Operation Complete", 
            f"Successfully normalized and saved {success_count} profile dataset(s) into:\n{output_dir}"
        )

    def clear_canvas(self):
        """Wipes the plot canvas matrix and flushes session memory storage structures."""
        self.active_datasets.clear()
        self.ax.clear()
        self.configure_axis_labels()
        self.canvas.draw()
        self.status_var.set("Canvas matrix dropped. System ready to receive new scan files array.")


if __name__ == "__main__":
    root = tk.Tk()
    app = XRDPlotterGUI(root)
    root.mainloop()