#!/usr/bin/env python3
"""
build_rruff_ir_library.py
=========================

Downloads the RRUFF infrared (FTIR) reference spectra (zipped archives) and packs
them into a single consolidated HDF5 "library" file that the FTIR plotter tools can
open directly. Peak positions are pre-computed and stored per spectrum, so the
"Match by Selected Peaks" feature becomes near-instant.

Run it once in a while to refresh the library:

    python build_rruff_ir_library.py                    # all sets -> rruff_ir_library.h5
    python build_rruff_ir_library.py --out my_lib.h5 --refresh

Downloaded zip archives are cached (default ./rruff_ir_cache) so re-runs only fetch
what changed unless --refresh is given. Set names are auto-discovered from the RRUFF
infrared directory index when reachable.

Source: https://www.rruff.net/zipped_data_files/infrared/  (Lafuente B, Downs R T,
Yang H, Stone N, 2015, "The power of databases: the RRUFF project".)

Requires:  pip install numpy scipy h5py
"""

import os
import re
import io
import sys
import time
import argparse
import zipfile
import urllib.request
from datetime import datetime, timezone

import numpy as np
from scipy.signal import find_peaks
import h5py

# Primary host + a mirror. The script tries each in turn.
BASE_URLS = [
    "https://www.rruff.net/zipped_data_files/infrared/",
    "https://rruff.info/zipped_data_files/infrared/",
]

# Fallback set names if the directory index can't be listed (auto-discovery preferred).
DEFAULT_DATASETS = [
    "infrared", "processed", "excellent", "fair", "poor",
]

LIBRARY_FORMAT = "rruff-ir-library"
LIBRARY_VERSION = 1


# ---------------------------------------------------------------------------
# Peak detection (identical algorithm to the Raman plotter tools, so the
# precomputed peaks match what the tools would otherwise compute on the fly).
# ---------------------------------------------------------------------------
def detect_reference_peaks(x, y, max_peaks=40, min_prominence=0.04):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 5:
        return np.array([], dtype=np.float32)
    rng = np.ptp(y)
    if rng <= 0:
        return np.array([], dtype=np.float32)
    yn = (y - y.min()) / rng
    peaks, props = find_peaks(yn, prominence=min_prominence, distance=3)
    if len(peaks) == 0:
        return np.array([], dtype=np.float32)
    proms = props.get('prominences', np.ones(len(peaks)))
    order = np.argsort(proms)[::-1][:max_peaks]
    return np.sort(x[peaks[order]]).astype(np.float32)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_rruff_txt(text):
    """Returns (x, y, meta) from a RRUFF two-column .txt with '##' headers."""
    meta = {}
    xs, ys = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('#'):
            m = re.match(r'#+\s*([A-Za-z ]+)\s*=\s*(.*)', line)
            if m:
                meta[m.group(1).strip().upper()] = m.group(2).strip()
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
            continue
        xs.append(xv); ys.append(yv)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), meta


def meta_from_filename(fname):
    """Best-effort extraction of name / id / wavelength / orientation from the
    RRUFF filename, e.g. 'Quartz__R040031__Raman__532__unoriented__...txt'."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    parts = stem.split('__')
    name = parts[0] if parts else stem
    rid = ''
    wavelength = ''
    orientation = ''
    for p in parts[1:]:
        if re.fullmatch(r'R\d{5,7}', p):
            rid = p
        elif re.fullmatch(r'\d{3,4}', p):
            wavelength = p
        elif p.lower() in ('oriented', 'unoriented'):
            orientation = p.lower()
    return name, rid, wavelength, orientation


# ---------------------------------------------------------------------------
# Download / discovery
# ---------------------------------------------------------------------------
def _http_get(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "rruff-library-builder/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def discover_datasets(base_url):
    """Parse the directory index for *.zip links. Returns a list of set names
    (zip basenames without extension). Empty list on failure."""
    try:
        html = _http_get(base_url, timeout=60).decode('utf-8', errors='ignore')
    except Exception:
        return []
    names = re.findall(r'href="([^"?]+\.zip)"', html, flags=re.IGNORECASE)
    out = []
    for n in names:
        base = os.path.splitext(os.path.basename(n))[0]
        if base and base not in out:
            out.append(base)
    return out


def fetch_zip(dataset, cache_dir, refresh=False):
    """Returns the local path to the dataset zip, downloading if needed.
    Tries each base URL until one works."""
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, f"{dataset}.zip")
    if os.path.exists(dest) and not refresh and os.path.getsize(dest) > 0:
        print(f"  [cache] {dataset}.zip ({os.path.getsize(dest)//1024} KB)")
        return dest
    last_err = None
    for base in BASE_URLS:
        url = base + dataset + ".zip"
        try:
            print(f"  [download] {url}")
            blob = _http_get(url)
            with open(dest, 'wb') as f:
                f.write(blob)
            print(f"            saved {len(blob)//1024} KB")
            return dest
        except Exception as e:
            last_err = e
            print(f"            failed: {e}")
    raise IOError(f"Could not download {dataset}.zip: {last_err}")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def sanitize_group_name(stem, existing):
    g = re.sub(r'[^A-Za-z0-9._-]+', '_', stem).strip('_') or "ref"
    base = g
    i = 1
    while g in existing:
        i += 1
        g = f"{base}_{i}"
    existing.add(g)
    return g


def build_library(datasets, out_path, cache_dir, refresh=False,
                  max_peaks=40, min_prominence=0.04):
    # Resolve dataset list
    if not datasets:
        found = []
        for base in BASE_URLS:
            found = discover_datasets(base)
            if found:
                print(f"Discovered {len(found)} sets from {base}")
                break
        datasets = found or DEFAULT_DATASETS
        if not found:
            print("Directory listing unavailable; using default set names.")
    print(f"Datasets: {', '.join(datasets)}")

    total = 0
    skipped = 0
    used_names = set()
    t0 = time.time()

    with h5py.File(out_path, 'w') as h5:
        h5.attrs['Format'] = LIBRARY_FORMAT
        h5.attrs['Version'] = LIBRARY_VERSION
        h5.attrs['Created'] = datetime.now(timezone.utc).isoformat()
        h5.attrs['Source'] = BASE_URLS[0]
        grp_root = h5.create_group('spectra')

        for ds in datasets:
            try:
                zip_path = fetch_zip(ds, cache_dir, refresh=refresh)
            except Exception as e:
                print(f"  Skipping '{ds}': {e}")
                continue
            try:
                zf = zipfile.ZipFile(zip_path)
            except Exception as e:
                print(f"  Bad zip '{ds}': {e}")
                continue
            members = [m for m in zf.namelist() if m.lower().endswith('.txt')]
            print(f"  {ds}: {len(members)} spectra")
            for m in members:
                try:
                    text = zf.read(m).decode('utf-8', errors='ignore')
                except Exception:
                    skipped += 1
                    continue
                x, y, meta = parse_rruff_txt(text)
                if x.size < 5:
                    skipped += 1
                    continue
                fname_name, fname_id, wavelength, orientation = meta_from_filename(m)
                name = meta.get('NAMES') or fname_name
                rid = meta.get('RRUFFID') or fname_id
                peaks = detect_reference_peaks(x, y, max_peaks=max_peaks,
                                               min_prominence=min_prominence)

                gname = sanitize_group_name(os.path.splitext(os.path.basename(m))[0], used_names)
                g = grp_root.create_group(gname)
                g.create_dataset('x', data=x, compression='gzip', compression_opts=4)
                g.create_dataset('y', data=y, compression='gzip', compression_opts=4)
                g.attrs['name'] = name
                g.attrs['rruff_id'] = rid
                g.attrs['url'] = meta.get('URL', '')
                g.attrs['quality'] = ds
                g.attrs['source_file'] = os.path.basename(m)
                g.attrs['peaks'] = peaks
                total += 1
            zf.close()

        h5.attrs['Count'] = total
        h5.attrs['Datasets'] = np.array([str(d) for d in datasets], dtype=h5py.string_dtype())

    dt = time.time() - t0
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nDone: {total} spectra ({skipped} skipped) -> {out_path} "
          f"({size_mb:.1f} MB) in {dt:.1f}s")
    return total


def main():
    ap = argparse.ArgumentParser(description="Build a consolidated RRUFF Raman .h5 library.")
    ap.add_argument("--out", default="rruff_ir_library.h5", help="Output .h5 path")
    ap.add_argument("--datasets", default="", help="Comma-separated set names (default: auto-discover / all)")
    ap.add_argument("--cache-dir", default="rruff_ir_cache", help="Where to cache downloaded zips")
    ap.add_argument("--refresh", action="store_true", help="Re-download zips even if cached")
    ap.add_argument("--max-peaks", type=int, default=40, help="Max peaks stored per spectrum")
    ap.add_argument("--min-prominence", type=float, default=0.04, help="Peak prominence threshold (0-1)")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    try:
        build_library(datasets, args.out, args.cache_dir, refresh=args.refresh,
                      max_peaks=args.max_peaks, min_prominence=args.min_prominence)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
