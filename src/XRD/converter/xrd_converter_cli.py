import pandas as pd
import os
import glob
import argparse
import sys
import xml.etree.ElementTree as ET

def process_xrd_data(input_file, output_file):
    """
    Detects file type, parses either custom XRD CSV or native XRDML,
    isolates Angle and Intensity columns, and saves to a clean CSV.
    """
    ext = os.path.splitext(input_file)[1].lower()
    
    # --- CSV FORMAT ENGINGE ---
    if ext == '.csv':
        skiprows = 0
        try:
            with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
                for idx, line in enumerate(f):
                    if '[Scan points]' in line:
                        skiprows = idx + 1
                        break
        except Exception as e:
            return False, f"Error reading CSV file: {e}"

        if skiprows == 0:
            return False, "Error: '[Scan points]' data tag not found in CSV."

        try:
            df = pd.read_csv(input_file, skiprows=skiprows)
            df.columns = df.columns.str.strip()  # Clear whitespace
            if 'Angle' in df.columns and 'Intensity' in df.columns:
                df_simple = df[['Angle', 'Intensity']]
                df_simple.to_csv(output_file, index=False)
                return True, "Success"
            else:
                return False, "Missing required 'Angle' or 'Intensity' columns."
        except Exception as e:
            return False, f"Parsing error in CSV: {e}"

    # --- XRDML FORMAT ENGINE ---
    elif ext == '.xrdml':
        try:
            tree = ET.parse(input_file)
            root = tree.getroot()
            ns = {'x': 'http://www.xrdml.com/XRDMeasurement/2.2'}
            
            # Extract intensity counts
            counts_el = root.find('.//x:counts', ns)
            if counts_el is None or not counts_el.text:
                return False, "No intensity data found in XRDML structure."
            counts = [float(c) for c in counts_el.text.split()]
            
            # Extract 2Theta positional coordinates
            pos_el = root.find('.//x:positions[@axis="2Theta"]', ns)
            if pos_el is None:
                return False, "No 2Theta positions element located."
                
            list_pos = pos_el.find('x:listPosition', ns)
            if list_pos is not None and list_pos.text:
                angles = [float(a) for a in list_pos.text.split()]
            else:
                start_el = pos_el.find('x:startPosition', ns)
                end_el = pos_el.find('x:endPosition', ns)
                if start_el is not None and end_el is not None:
                    start_val = float(start_el.text)
                    end_val = float(end_el.text)
                    num_points = len(counts)
                    if num_points > 1:
                        step = (end_val - start_val) / (num_points - 1)
                        angles = [start_val + i * step for i in range(num_points)]
                    else:
                        angles = [start_val]
                else:
                    return False, "Could not compute scanning coordinate coordinates."
            
            if len(angles) != len(counts):
                return False, f"Data alignment array size mismatch: {len(angles)} positions vs {len(counts)} counts."
                
            df_simple = pd.DataFrame({'Angle': angles, 'Intensity': counts})
            df_simple.to_csv(output_file, index=False)
            return True, "Success"
        except Exception as e:
            return False, f"Parsing error in XRDML engine: {e}"
            
    else:
        return False, f"Unsupported file type extension: '{ext}'"

def batch_process_xrd(input_folder):
    """Finds and batch-processes all .csv and .xrdml files in a folder."""
    if not os.path.isdir(input_folder):
        print(f"Error: Directory '{input_folder}' does not exist.")
        return

    output_folder = os.path.join(input_folder, "processed_xrd_data")
    os.makedirs(output_folder, exist_ok=True)
    
    # Gather both file extensions
    xrd_files = []
    for ext in ["*.csv", "*.xrdml", "*.CSV", "*.XRDML"]:
        xrd_files.extend(glob.glob(os.path.join(input_folder, ext)))
    xrd_files = list(set(xrd_files)) # De-duplicate
    
    if not xrd_files:
        print(f"No matchable .csv or .xrdml files found in '{input_folder}'.")
        return
        
    print(f"Found {len(xrd_files)} datasets. Batch conversion started...\n")
    
    success_count = 0
    for file_path in xrd_files:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"clean_{base_name}.csv")
        
        success, err_msg = process_xrd_data(file_path, output_path)
        if success:
            print(f"  -> Successfully Processed: {filename} -> processed_xrd_data/clean_{base_name}.csv")
            success_count += 1
        else:
            print(f"  -> Failed Processing {filename}: {err_msg}")
            
    print(f"\nBatch processing complete! Cleaned {success_count}/{len(xrd_files)} files.")

def main():
    parser = argparse.ArgumentParser(description="Convert custom XRD CSV or native XML XRDML datasets into plotting-ready formats.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', type=str, help="Path to a single .csv or .xrdml file.")
    group.add_argument('-b', '--batch', type=str, help="Path to a folder containing files for batch processing.")
    args = parser.parse_args()

    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: The target path '{args.file}' is invalid.")
            sys.exit(1)
            
        dir_name = os.path.dirname(args.file)
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        output_file = os.path.join(dir_name, f"clean_{base_name}.csv") if dir_name else f"clean_{base_name}.csv"
        
        print(f"Processing: {args.file}")
        success, err_msg = process_xrd_data(args.file, output_file)
        if success:
            print(f"Success! Saved as: {output_file}")
        else:
            print(f"Failure: {err_msg}")

    elif args.batch:
        batch_process_xrd(args.batch)

if __name__ == "__main__":
    main()