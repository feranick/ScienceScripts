import os
import glob
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import xml.etree.ElementTree as ET

# ==========================================
# PARSING ENGINE CORE LOGIC
# ==========================================

def process_xrd_data(input_file, output_file):
    """Parses .csv or .xrdml file, isolating the core Angle and Intensity variables."""
    ext = os.path.splitext(input_file)[1].lower()
    
    if ext == '.csv':
        skiprows = 0
        try:
            with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
                for idx, line in enumerate(f):
                    if '[Scan points]' in line:
                        skiprows = idx + 1
                        break
            if skiprows == 0:
                return False, "Could not find explicit data points identifier block."
            df = pd.read_csv(input_file, skiprows=skiprows)
            df.columns = df.columns.str.strip()
            if 'Angle' in df.columns and 'Intensity' in df.columns:
                df[['Angle', 'Intensity']].to_csv(output_file, index=False)
                return True, "Success"
            return False, "Required structural columns were missing."
        except Exception as e:
            return False, str(e)

    elif ext == '.xrdml':
        try:
            tree = ET.parse(input_file)
            root = tree.getroot()
            ns = {'x': 'http://www.xrdml.com/XRDMeasurement/2.2'}
            
            counts_el = root.find('.//x:counts', ns)
            if counts_el is None or not counts_el.text:
                return False, "Missing counts elements."
            counts = [float(c) for c in counts_el.text.split()]
            
            pos_el = root.find('.//x:positions[@axis="2Theta"]', ns)
            if pos_el is None:
                return False, "Missing angular positions maps."
                
            list_pos = pos_el.find('x:listPosition', ns)
            if list_pos is not None and list_pos.text:
                angles = [float(a) for a in list_pos.text.split()]
            else:
                start_el = pos_el.find('x:startPosition', ns)
                end_el = pos_el.find('x:endPosition', ns)
                if start_el is not None and end_el is not None:
                    start_val, end_val = float(start_el.text), float(end_el.text)
                    n = len(counts)
                    angles = [start_val + i * ((end_val - start_val) / (n - 1)) for i in range(n)] if n > 1 else [start_val]
                else:
                    return False, "Unable to resolve coordinates boundaries."
                    
            if len(angles) != len(counts):
                return False, "Vector array lengths misaligned."
                
            pd.DataFrame({'Angle': angles, 'Intensity': counts}).to_csv(output_file, index=False)
            return True, "Success"
        except Exception as e:
            return False, str(e)
            
    return False, "Unsupported system extension type."

def extract_experimental_parameters(input_file):
    """Cross-parses structural file types to feed the telemetry interface."""
    ext = os.path.splitext(input_file)[1].lower()
    parameters = {}
    
    if ext == '.csv':
        target_keys = {
            "Sample identification": "Sample ID", "Anode material": "Anode Material",
            "K-Alpha1 wavelength": "K-Alpha1 Wavelength (Å)", "Generator voltage": "Generator Voltage (kV)",
            "Tube current": "Tube Current (mA)", "Scan range": "Scan Range (2θ)", "File date and time": "Scan Date/Time"
        }
        try:
            with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if '[Scan points]' in line: break
                    if ',' in line:
                        parts = [p.strip() for p in line.split(',')]
                        if parts[0] in target_keys:
                            parameters[target_keys[parts[0]]] = " to ".join(parts[1:]) if parts[0] == "Scan range" else ", ".join(parts[1:])
        except Exception: pass
            
    elif ext == '.xrdml':
        try:
            tree = ET.parse(input_file)
            root = tree.getroot()
            ns = {'x': 'http://www.xrdml.com/XRDMeasurement/2.2'}
            
            s_id = root.find('.//x:sample/x:id', ns)
            if s_id is not None: parameters['Sample ID'] = s_id.text
            anode = root.find('.//x:anodeMaterial', ns)
            if anode is not None: parameters['Anode Material'] = anode.text
            wl = root.find('.//x:kAlpha1', ns)
            if wl is not None: parameters['K-Alpha1 Wavelength (Å)'] = wl.text
            kv = root.find('.//x:tension', ns)
            if kv is not None: parameters['Generator Voltage (kV)'] = f"{kv.text} {kv.attrib.get('unit','kV')}"
            ma = root.find('.//x:current', ns)
            if ma is not None: parameters['Tube Current (mA)'] = f"{ma.text} {ma.attrib.get('unit','mA')}"
            dt = root.find('.//x:startTimeStamp', ns)
            if dt is not None: parameters['Scan Date/Time'] = dt.text
            start_p = root.find('.//x:positions[@axis="2Theta"]/x:startPosition', ns)
            end_p = root.find('.//x:positions[@axis="2Theta"]/x:endPosition', ns)
            if start_p is not None and end_p is not None: parameters['Scan Range (2θ)'] = f"{start_p.text} to {end_p.text}"
        except Exception: pass
            
    return parameters

# ==========================================
# GRAPHICAL INTERFACE PRESENTATION
# ==========================================

class XRDConverterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("XRD Multi-Format Converter")
        self.root.geometry("580x420")
        self.root.resizable(False, False)
        
        style = ttk.Style()
        style.theme_use('clam')

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self.setup_single_file_tab()
        self.setup_batch_folder_tab()

    def setup_single_file_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Single File Conversion ")

        lbl = ttk.Label(tab, text="Select a specific XRD Data File (.csv, .xrdml):", font=("Helvetica", 10, "bold"))
        lbl.pack(anchor="w", padx=15, pady=10)

        frame = ttk.Frame(tab)
        frame.pack(fill="x", padx=15)

        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(frame, textvariable=self.file_path_var, width=48)
        self.file_entry.pack(side="left", padx=(0, 10), ipady=3)

        btn_browse = ttk.Button(frame, text="Browse...", command=self.browse_file)
        btn_browse.pack(side="left")

        btn_convert = ttk.Button(tab, text="Convert Dataset", command=self.convert_single_file)
        btn_convert.pack(side="bottom", pady=15, ipadx=15, ipady=3)

        self.param_frame = ttk.LabelFrame(tab, text=" Experimental Parameters Summary Box ")
        self.param_frame.pack(fill="both", expand=True, padx=15, pady=(10, 0))

        self.param_text = tk.Text(self.param_frame, wrap="word", background="#f9f9f9", font=("Consolas", 10), height=10)
        self.param_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.param_text.insert("1.0", "Awaiting execution parameters array loading processing trigger...")
        self.param_text.config(state="disabled")

    def setup_batch_folder_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Batch Folder Conversion ")

        lbl = ttk.Label(tab, text="Convert all matching data profiles within directory:", font=("Helvetica", 10, "bold"))
        lbl.pack(anchor="w", padx=15, pady=15)

        frame = ttk.Frame(tab)
        frame.pack(fill="x", padx=15)

        self.folder_path_var = tk.StringVar()
        self.folder_entry = ttk.Entry(frame, textvariable=self.folder_path_var, width=48)
        self.folder_entry.pack(side="left", padx=(0, 10), ipady=3)

        btn_browse = ttk.Button(frame, text="Browse...", command=self.browse_folder)
        btn_browse.pack(side="left")

        btn_convert = ttk.Button(tab, text="Convert All Datasets", command=self.convert_batch_folder)
        btn_convert.pack(pady=40, ipadx=15, ipady=3)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Open XRD Track Target",
            filetypes=[("XRD Data Configurations", "*.csv;*.xrdml"), ("CSV Formats", "*.csv"), ("Native XRDML Xml", "*.xrdml"), ("All Files", "*.*")]
        )
        if filename: self.file_path_var.set(filename)

    def browse_folder(self):
        directory = filedialog.askdirectory(title="Locate Batch Run Repository Directory Target")
        if directory: self.folder_path_var.set(directory)

    def convert_single_file(self):
        file_path = self.file_path_var.get()
        if not file_path or not os.path.isfile(file_path):
            messagebox.showerror("System Error", "Please map onto a functional source data module first.")
            return

        dir_name = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(dir_name, f"clean_{base_name}.csv")

        success, msg = process_xrd_data(file_path, output_file)
        if success:
            params = extract_experimental_parameters(file_path)
            self.param_text.config(state="normal")
            self.param_text.delete("1.0", tk.END)
            
            summary_text = f"File processed: {os.path.basename(file_path)}\n" + "="*50 + "\n"
            if params:
                for k, v in params.items(): summary_text += f"{k:<25}: {v}\n"
            else:
                summary_text += "No index telemetry properties metadata sets isolated safely."
                
            self.param_text.insert("1.0", summary_text)
            self.param_text.config(state="disabled")
            messagebox.showinfo("Operation Complete", f"Data normalized successfully:\n{output_file}")
        else:
            messagebox.showerror("Parser Exception Encountered", msg)

    def convert_batch_folder(self):
        folder_path = self.folder_path_var.get()
        if not folder_path or not os.path.isdir(folder_path):
            messagebox.showerror("Validation Boundary Check Failure", "Verify search folder directories path correctness.")
            return

        files = []
        for ext in ["*.csv", "*.xrdml", "*.CSV", "*.XRDML"]:
            files.extend(glob.glob(os.path.join(folder_path, ext)))
        files = list(set(files))

        if not files:
            messagebox.showwarning("Empty Target", "No matching index formats encountered.")
            return

        output_folder = os.path.join(folder_path, "processed_xrd_data")
        os.makedirs(output_folder, exist_ok=True)

        success_count = 0
        for f in files:
            b_name = os.path.splitext(os.path.basename(f))[0]
            out = os.path.join(output_folder, f"clean_{b_name}.csv")
            if process_xrd_data(f, out)[0]: success_count += 1

        messagebox.showinfo("Processing Cycle Finish", f"Successfully structured {success_count} / {len(files)} records.\nLocation: {output_folder}")

if __name__ == "__main__":
    root = tk.Tk()
    app = XRDConverterGUI(root)
    root.mainloop()