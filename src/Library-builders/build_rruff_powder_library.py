#!/usr/bin/env python3
"""
build_rruff_powder_library.py
=============================

Downloads RRUFF powder X-ray diffraction reference patterns and packs them into
a single consolidated HDF5 "library" that the XRD plotter tools can open
directly. 2-theta peak positions are pre-computed and stored per pattern, so the
"Match by Selected Peaks (RRUFF)" feature is near-instant.

RRUFF ships powder data in two collections, both handled here automatically:
  * XY  — continuous "Xray_Data_XY_Processed" profiles (2-theta vs intensity)
  * DIF — discrete peak tables (2-theta / intensity / d-spacing / h k l)
Both are calculated for Cu radiation (CuKalpha), i.e. the same convention the
tools use for Materials Project patterns, so no wavelength conversion is needed.

Usage (run occasionally to refresh):
    python build_rruff_powder_library.py                         # XY collection -> rruff_powder_library.h5
    python build_rruff_powder_library.py --base-url https://rruff.info/zipped_data_files/dif/
    python build_rruff_powder_library.py --datasets LR   --out my_lib.h5 --refresh

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

# Primary host + mirror; the '/powder/' path is the XY collection, '/dif/' the DIF one.
DEFAULT_BASE = "https://rruff.info/zipped_data_files/powder/"
BASE_HOSTS = ["https://rruff.info", "https://www.rruff.net"]

# Fallback set names if the directory index cannot be listed (auto-discovery preferred).
DEFAULT_DATASETS = ["excellent", "fair", "poor", "unknown"]

LIBRARY_FORMAT = "rruff-powder-library"
LIBRARY_VERSION = 1

# Grid used to synthesize a display profile from a DIF peak list.
SYNTH_MIN, SYNTH_MAX, SYNTH_STEP, SYNTH_SIGMA = 5.0, 90.0, 0.02, 0.10


# ---------------------------------------------------------------------------
# Peak detection (continuous profiles) — identical to the XRD tool's algorithm.
# ---------------------------------------------------------------------------
def detect_reference_peaks(x, y, max_peaks=60, min_prominence=0.02):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 5:
        return np.array([], dtype=np.float32)
    rng = np.ptp(y)
    if rng <= 0:
        return np.array([], dtype=np.float32)
    yn = (y - y.min()) / rng
    peaks, props = find_peaks(yn, prominence=min_prominence, distance=2)
    if len(peaks) == 0:
        return np.array([], dtype=np.float32)
    proms = props.get('prominences', np.ones(len(peaks)))
    order = np.argsort(proms)[::-1][:max_peaks]
    return np.sort(x[peaks[order]]).astype(np.float32)


# ---------------------------------------------------------------------------
# Parsing — handles both RRUFF powder collections.
# ---------------------------------------------------------------------------
# Row of "2theta  h  k  l  [wave#]" (RRUFF refinement_data): a decimal 2-theta
# followed by 3 or 4 (signed) integers.
_REF_DATA_ROW = re.compile(r'^\d+\.\d+(?:\s+-?\d+){3,4}$')


def _synth_profile(tth, inten):
    """Gaussian-broadens a peak list onto a 2-theta grid for display/overlay."""
    x = np.arange(SYNTH_MIN, SYNTH_MAX + SYNTH_STEP, SYNTH_STEP)
    y = np.zeros_like(x)
    for c, h in zip(tth, inten):
        y += h * np.exp(-((x - c) / SYNTH_SIGMA) ** 2)
    if y.max() > 0:
        y = y / y.max() * 100.0
    return x.astype(np.float32), y.astype(np.float32)


def parse_rruff_powder(text):
    """Returns (x, y, peaks_or_None, meta) for any RRUFF powder collection.

    XY profile          -> x,y are the pattern; peaks is None (detect later).
    DIF table           -> peaks + intensities from the table; x,y synthesized.
    refinement_data     -> peaks (2-theta) with uniform intensity; x,y synthesized.
    refinement_output   -> peaks (observed 2-theta), uniform intensity; x,y synthesized.
    """
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

    # --- DIF: "2-THETA  INTENSITY  D-SPACING  H K L" ---------------------
    if 'INTENSITY' in upper and re.search(r'2-?\s*THETA', upper):
        start = 0
        for i, raw in enumerate(lines):
            if re.search(r'2-?\s*THETA', raw, re.IGNORECASE) and re.search(r'INTENSITY', raw, re.IGNORECASE):
                start = i + 1
                break
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
            return x, y, np.array(tth, dtype=np.float32), meta

    # --- refinement_output_data: REFINE program listing -----------------
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
            return x, y, np.array(tth, dtype=np.float32), meta

    # --- refinement_data: "2theta  h k l  [wave#]" ----------------------
    ref_rows = [l.strip() for l in lines if _REF_DATA_ROW.match(l.strip())]
    if len(ref_rows) >= 3:
        tth = [float(r.split()[0]) for r in ref_rows if clamp(float(r.split()[0]))]
        if tth:
            x, y = _synth_profile(tth, [100.0] * len(tth))
            return x, y, np.array(tth, dtype=np.float32), meta

    # --- XY: continuous two-column profile ------------------------------
    xs, ys = [], []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith('#'):
            continue
        for sep in (',', '\t', ';'):
            if sep in s:
                parts = s.split(sep)
                break
        else:
            parts = s.split()
        if len(parts) < 2:
            continue
        try:
            xv = float(parts[0]); yv = float(parts[1])
        except ValueError:
            continue  # e.g. the "X  Y" column header
        xs.append(xv); ys.append(yv)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), None, meta


def meta_from_filename(fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    parts = stem.split('__')
    name = parts[0] if parts else stem
    rid = ''
    for p in parts[1:]:
        if re.fullmatch(r'R\d{5,7}(-\d+)?', p):
            rid = re.match(r'(R\d{5,7})', p).group(1)
            break
    return name, rid


# ---------------------------------------------------------------------------
# Download / discovery
# ---------------------------------------------------------------------------
def _http_get(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "rruff-powder-builder/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _base_variants(base_url):
    """Yield the same directory path on each known host."""
    path = re.sub(r'^https?://[^/]+', '', base_url)
    seen = []
    for host in BASE_HOSTS:
        u = host + path
        if not u.endswith('/'):
            u += '/'
        if u not in seen:
            seen.append(u)
    if base_url not in seen:
        seen.insert(0, base_url if base_url.endswith('/') else base_url + '/')
    return seen


def discover_datasets(base_url):
    for b in _base_variants(base_url):
        try:
            html = _http_get(b, timeout=60).decode('utf-8', errors='ignore')
        except Exception:
            continue
        names = re.findall(r'href="([^"?]+\.zip)"', html, flags=re.IGNORECASE)
        out = []
        for n in names:
            stem = os.path.splitext(os.path.basename(n))[0]
            if stem and stem not in out:
                out.append(stem)
        if out:
            return out, b
    return [], None


def fetch_zip(dataset, base_url, cache_dir, refresh=False):
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, f"{dataset}.zip")
    if os.path.exists(dest) and not refresh and os.path.getsize(dest) > 0:
        print(f"  [cache] {dataset}.zip ({os.path.getsize(dest)//1024} KB)")
        return dest
    last_err = None
    for b in _base_variants(base_url):
        url = b + dataset + ".zip"
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


def build_library(datasets, base_url, out_path, cache_dir, refresh=False,
                  max_peaks=60, min_prominence=0.02):
    if not datasets:
        found, used_base = discover_datasets(base_url)
        if found:
            print(f"Discovered {len(found)} sets from {used_base}")
            datasets = found
        else:
            print("Directory listing unavailable; using default set names.")
            datasets = DEFAULT_DATASETS
    print(f"Datasets: {', '.join(datasets)}")

    total = skipped = 0
    used_names = set()
    t0 = time.time()

    with h5py.File(out_path, 'w') as h5:
        h5.attrs['Format'] = LIBRARY_FORMAT
        h5.attrs['Version'] = LIBRARY_VERSION
        h5.attrs['Created'] = datetime.now(timezone.utc).isoformat()
        h5.attrs['Source'] = base_url
        grp_root = h5.create_group('spectra')

        for ds in datasets:
            try:
                zip_path = fetch_zip(ds, base_url, cache_dir, refresh=refresh)
            except Exception as e:
                print(f"  Skipping '{ds}': {e}")
                continue
            try:
                zf = zipfile.ZipFile(zip_path)
            except Exception as e:
                print(f"  Bad zip '{ds}': {e}")
                continue
            members = [m for m in zf.namelist() if m.lower().endswith('.txt')]
            print(f"  {ds}: {len(members)} files")
            for m in members:
                try:
                    text = zf.read(m).decode('utf-8', errors='ignore')
                except Exception:
                    skipped += 1
                    continue
                x, y, peaks, meta = parse_rruff_powder(text)
                if x.size < 3:
                    skipped += 1
                    continue
                if peaks is None:
                    peaks = detect_reference_peaks(x, y, max_peaks=max_peaks,
                                                   min_prominence=min_prominence)
                fname_name, fname_id = meta_from_filename(m)
                name = meta.get('NAMES') or fname_name
                rid = meta.get('RRUFFID') or fname_id
                url = meta.get('URL', '')
                gname = sanitize_group_name(os.path.splitext(os.path.basename(m))[0], used_names)
                g = grp_root.create_group(gname)
                g.create_dataset('x', data=x, compression='gzip', compression_opts=4)
                g.create_dataset('y', data=y, compression='gzip', compression_opts=4)
                g.attrs['name'] = name
                g.attrs['rruff_id'] = rid
                g.attrs['url'] = url
                g.attrs['quality'] = ds
                g.attrs['wavelength'] = meta.get('WAVELENGTH', '')
                g.attrs['source_file'] = os.path.basename(m)
                g.attrs['peaks'] = np.asarray(peaks, dtype=np.float32)
                total += 1
            zf.close()

        h5.attrs['Count'] = total
        h5.attrs['Datasets'] = np.array([str(d) for d in datasets], dtype=h5py.string_dtype())

    dt = time.time() - t0
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nDone: {total} patterns ({skipped} skipped) -> {out_path} ({size_mb:.1f} MB) in {dt:.1f}s")
    return total


def main():
    ap = argparse.ArgumentParser(description="Build a consolidated RRUFF powder XRD .h5 library.")
    ap.add_argument("--out", default="rruff_powder_library.h5", help="Output .h5 path")
    ap.add_argument("--base-url", default=DEFAULT_BASE,
                    help="RRUFF collection URL (powder XY default; use .../dif/ for the DIF collection)")
    ap.add_argument("--datasets", default="", help="Comma-separated set names (default: auto-discover)")
    ap.add_argument("--cache-dir", default="rruff_powder_cache", help="Where to cache downloaded zips")
    ap.add_argument("--refresh", action="store_true", help="Re-download zips even if cached")
    ap.add_argument("--max-peaks", type=int, default=60, help="Max peaks stored per pattern")
    ap.add_argument("--min-prominence", type=float, default=0.02, help="Peak prominence threshold (0-1)")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    try:
        build_library(datasets, args.base_url, args.out, args.cache_dir, refresh=args.refresh,
                      max_peaks=args.max_peaks, min_prominence=args.min_prominence)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
