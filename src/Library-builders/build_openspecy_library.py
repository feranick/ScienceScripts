#!/usr/bin/env python3
"""
build_openspecy_library.py
==========================

Build a Raman reference library (.h5) from the Open Specy reference libraries,
for use with the Raman Peak Analysis Toolkit (raman_plotter.py /
raman_plotter.html).

WHY THIS EXISTS
---------------
Open Specy (https://openspecy.org, CC-BY) is the open collection that actually
covers the *organics* gap: polymers, microplastics, pigments, pharmaceutical
excipients, seized-drug references, and more -- several thousand Raman and
FT-IR spectra aggregated from a dozen contributing projects. It is the
practical, redistributable answer to "I want SDBS-like coverage", which SDBS
itself cannot be (their terms prohibit automated retrieval).

The catch is the distribution format: the libraries live on OSF as R `.rds`
files, which the R package pulls with `get_lib()`. This builder decodes them
from Python and bakes the Raman subset into the same `.h5` schema the apps
already read for RRUFF, COD and ROD -- so an Open Specy library loads in the
same panel, searches the same way, and peak-matches identically.

    1. DOWNLOAD the chosen library (cached; CloudFront mirror by default).
    2. DECODE the .rds with `rdata` (pure Python) or `pyreadr`.
    3. SELECT the Raman spectra (FT-IR is included in the same file).
    4. DETECT bands with the same prominence rule the apps use.
    5. BAKE to .h5.

WHICH LIBRARY TYPE
------------------
  raw          the rawest form -- real intensities. USE THIS for overlaying.
  nobaseline   baseline-corrected.
  derivative   absolute first derivative; good for matching, meaningless to
               overlay against a raw measurement.

`--type raw` is the default for that reason.

EXAMPLES
--------
  # The Raman half of the raw library -- the recommended build
  python build_openspecy_library.py --out openspecy_raman.h5

  # The FT-IR half, for the FTIR toolkit
  python build_openspecy_library.py --only ftir --out openspecy_ftir.h5

  # Everything, both techniques, baseline-corrected
  python build_openspecy_library.py --type nobaseline --only both \
        --out openspecy_all.h5

  # Only spectra whose metadata mentions polyethylene
  python build_openspecy_library.py --filter polyethylene --out pe.h5

  # Re-bake from the cached .rds, no network
  python build_openspecy_library.py --offline --step 2 --out small.h5

REQUIREMENTS
------------
  pip install h5py numpy scipy rdata

`rdata` is the pure-Python R-serialization reader. If it cannot decode the
file (Open Specy objects are data.table-based and the format has moved over
the years), the builder falls back to `pyreadr`, and failing that tells you how
to export a CSV from R in two lines -- see --from-csv below.

LICENSE NOTE
------------
Open Specy libraries are CC-BY: attribution is required, unlike ROD's CC0. The
builder stamps `license` and `source` on the file and every group, and carries
each spectrum's originating collection in the `collection` attribute. The
libraries aggregate work from many independent groups -- see the reference list
in the Open Specy package documentation and cite the contributing collections
you actually rely on.
"""

import argparse
import os
import sys
import time
import urllib.request

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("This tool needs h5py:  pip install h5py")

# Band detection is shared with the ROD builder so scores stay comparable.
try:
    from build_rod_library import detect_bands, normalize, crop, decimate
except ImportError:
    sys.exit("Place this next to build_rod_library.py (it reuses the band "
             "detector so match scores are consistent across libraries).")


OSF_URLS = {
    "raw": "https://osf.io/download/kzv3n/",
    "nobaseline": "https://osf.io/download/jy7zk/",
    "derivative": "https://osf.io/download/2qbkt/",
    "medoid_derivative": "https://osf.io/download/2dmwu/",
    "medoid_nobaseline": "https://osf.io/download/8f3sg/",
}
AWS_URLS = {
    "raw": "https://d2jrxerjcsjhs7.cloudfront.net/raw.rds",
    "nobaseline": "https://d2jrxerjcsjhs7.cloudfront.net/nobaseline.rds",
    "derivative": "https://d2jrxerjcsjhs7.cloudfront.net/derivative.rds",
    "medoid_derivative": "https://d2jrxerjcsjhs7.cloudfront.net/medoid_derivative.rds",
    "medoid_nobaseline": "https://d2jrxerjcsjhs7.cloudfront.net/medoid_nobaseline.rds",
}
OPENSPECY_URL = "https://openspecy.org"
DEFAULT_UA = ("Mozilla/5.0 (compatible; Raman-Toolkit-OpenSpecy-builder/1.0; "
              "+https://openspecy.org)")

# Metadata columns to try, in order, when naming a spectrum. Open Specy's
# schema is a union of many contributing projects, so most rows populate only
# a few of these.
NAME_COLUMNS = ("sample_name", "material_name", "polymer_class", "material_class",
                "spectrum_identity", "material_form", "product_name", "sample_id",
                "material", "polymer", "name")
COLLECTION_COLUMNS = ("collection_code", "collection", "organization",
                      "data_source", "source", "contributor")
TYPE_COLUMNS = ("spectrum_type", "type", "technique", "spectra_type")
ID_COLUMNS = ("sample_id", "spectrum_id", "id")


# ============================================================================
# Download / decode
# ============================================================================

def download_rds(lib_type, cache_dir, use_aws=True, offline=False, ua=DEFAULT_UA,
                 timeout=300):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{lib_type}.rds")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"Using cached {path} ({os.path.getsize(path)/1e6:.1f} MB)")
        return path
    if offline:
        sys.exit(f"Offline mode: {path} is not cached yet.")

    urls = AWS_URLS if use_aws else OSF_URLS
    if lib_type not in urls:
        sys.exit(f"Unknown library type '{lib_type}'. "
                 f"Choose from: {', '.join(sorted(urls))}")
    url = urls[lib_type]
    print(f"Downloading {lib_type} library from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(path, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            print(f"\r  {os.path.getsize(path)/1e6:7.1f} MB", end="", flush=True)
    print()
    return path


def decode_rds(path):
    """Decodes an .rds into plain Python. Tries `rdata`, then `pyreadr`."""
    errors = []
    try:
        import rdata
        obj = rdata.read_rds(path)
        return _plainify(obj)
    except Exception as e:                                  # noqa: BLE001
        errors.append(f"rdata: {e}")
    try:
        import pyreadr
        result = pyreadr.read_r(path)
        return {k: v for k, v in result.items()}
    except Exception as e:                                  # noqa: BLE001
        errors.append(f"pyreadr: {e}")

    sys.exit(
        "Could not decode the .rds with either reader.\n  " +
        "\n  ".join(errors) +
        "\n\nWorkaround -- export a CSV from R once, then use --from-csv:\n"
        "    R -e 'library(OpenSpecy); get_lib(\"raw\"); l <- load_lib(\"raw\");\n"
        "          write.csv(cbind(wavenumber = l$wavenumber, l$spectra),\n"
        "                    \"openspecy_spectra.csv\", row.names = FALSE);\n"
        "          write.csv(l$metadata, \"openspecy_metadata.csv\", row.names = FALSE)'\n"
        "    python build_openspecy_library.py --from-csv openspecy_spectra.csv \\\n"
        "           --metadata-csv openspecy_metadata.csv --out openspecy_raman.h5")


def _plainify(obj):
    """rdata returns nested dicts / numpy arrays / pandas frames already; this
    just normalizes the couple of container types it can hand back."""
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return obj
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {str(k): _plainify(v) for k, v in obj.items()}
    return obj


# ============================================================================
# Structure extraction
# ============================================================================

def _find_key(mapping, *names):
    if not isinstance(mapping, dict):
        return None
    lower = {str(k).lower(): k for k in mapping}
    for n in names:
        if n in lower:
            return lower[n]
    return None


def extract_library(obj):
    """Pulls (wavenumber, spectra_dict, metadata_rows) out of a decoded object.

    An OpenSpecy object is a named list of `wavenumber` (numeric) and `spectra`
    (a data.table, one column per spectrum), usually with `metadata`. Written
    defensively: the exact container types differ between rdata and pyreadr,
    and Open Specy's own schema has grown over releases.
    """
    if not isinstance(obj, dict):
        sys.exit(f"Unexpected .rds contents (got {type(obj).__name__}); "
                 "this does not look like an OpenSpecy library.")

    # pyreadr wraps everything one level deeper (filename -> frame)
    if len(obj) == 1 and _find_key(obj, "wavenumber") is None:
        inner = next(iter(obj.values()))
        if isinstance(inner, dict):
            obj = inner

    k_wav = _find_key(obj, "wavenumber", "wavenumbers", "x")
    k_spec = _find_key(obj, "spectra", "spectrum", "y")
    if k_wav is None or k_spec is None:
        sys.exit(f"Could not find 'wavenumber' and 'spectra' in the decoded "
                 f"object (keys: {sorted(map(str, obj))[:12]}).")

    wav = np.asarray(_column_values(obj[k_wav]), dtype=float).ravel()
    spectra = _as_columns(obj[k_spec])
    meta = obj.get(_find_key(obj, "metadata", "meta") or "", None)
    meta_rows = _as_rows(meta) if meta is not None else []
    return wav, spectra, meta_rows


def _column_values(v):
    try:
        import pandas as pd
        if isinstance(v, (pd.Series, pd.DataFrame)):
            return v.to_numpy().ravel()
    except ImportError:
        pass
    if isinstance(v, dict):
        return np.asarray(next(iter(v.values())))
    return np.asarray(v)


def _as_columns(tbl):
    """-> {column_name: 1-D array} for a data.table-ish object."""
    try:
        import pandas as pd
        if isinstance(tbl, pd.DataFrame):
            return {str(c): tbl[c].to_numpy() for c in tbl.columns}
    except ImportError:
        pass
    if isinstance(tbl, dict):
        return {str(k): np.asarray(v) for k, v in tbl.items()}
    arr = np.asarray(tbl)
    if arr.ndim == 2:
        return {str(i): arr[:, i] for i in range(arr.shape[1])}
    sys.exit("Could not interpret the 'spectra' table.")


def _as_rows(tbl):
    """-> [ {col: value}, ... ] for a data.table-ish object."""
    try:
        import pandas as pd
        if isinstance(tbl, pd.DataFrame):
            return tbl.to_dict("records")
    except ImportError:
        pass
    if isinstance(tbl, dict):
        cols = {str(k): list(np.asarray(v).ravel()) for k, v in tbl.items()}
        n = max((len(v) for v in cols.values()), default=0)
        return [{k: (v[i] if i < len(v) else None) for k, v in cols.items()}
                for i in range(n)]
    return []


def _clean(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "na", "none", "null", "") else s


def _pick(row, columns):
    for c in columns:
        for key in row:
            if str(key).lower() == c:
                val = _clean(row[key])
                if val:
                    return val
    return ""


# ============================================================================
# CSV fallback path
# ============================================================================

def load_from_csv(spectra_csv, metadata_csv):
    """Reads the two CSVs produced by the documented R export."""
    import csv as _csv
    with open(spectra_csv, newline="", encoding="utf-8-sig") as fh:
        rows = list(_csv.reader(fh))
    if len(rows) < 2:
        sys.exit(f"{spectra_csv} has no data rows.")
    head = [h.strip() for h in rows[0]]
    try:
        wcol = [h.lower() for h in head].index("wavenumber")
    except ValueError:
        wcol = 0
    data = np.array([[float(c) if c not in ("", "NA") else np.nan for c in r]
                     for r in rows[1:] if len(r) == len(head)], dtype=float)
    wav = data[:, wcol]
    spectra = {head[i]: data[:, i] for i in range(len(head)) if i != wcol}

    meta_rows = []
    if metadata_csv and os.path.exists(metadata_csv):
        with open(metadata_csv, newline="", encoding="utf-8-sig") as fh:
            meta_rows = list(_csv.DictReader(fh))
    return wav, spectra, meta_rows


# ============================================================================
# Build
# ============================================================================

def build(wav, spectra, meta_rows, args):
    names = list(spectra.keys())
    total = len(names)
    print(f"Decoded {total} spectra on a {len(wav)}-point axis.")

    # Metadata rows are positionally aligned with the spectra columns; index by
    # id as well, since some releases carry the column name in an id field.
    meta_by_id = {}
    for row in meta_rows:
        rid = _pick(row, ID_COLUMNS)
        if rid:
            meta_by_id[rid] = row

    kept = skipped_type = skipped_filter = skipped_bad = 0
    flt = (args.filter or "").lower()
    want = "both" if getattr(args, "include_ftir", False) else getattr(args, "only", "raman")

    with h5py.File(args.out, "w") as h5:
        h5.attrs["source"] = "OpenSpecy"
        h5.attrs["technique"] = want
        h5.attrs["database"] = "Open Specy reference library"
        h5.attrs["database_url"] = OPENSPECY_URL
        h5.attrs["library_type"] = args.type
        h5.attrs["license"] = "CC-BY-4.0"
        h5.attrs["attribution_required"] = "yes"
        h5.attrs["built"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h5.attrs["builder"] = "build_openspecy_library.py 1.0"
        h5.attrs["units_x"] = "cm-1"
        h5.attrs["storage"] = "curve+peaks"
        grp = h5.create_group("spectra")

        for n, col in enumerate(names, 1):
            row = meta_by_id.get(col) or (meta_rows[n - 1] if n - 1 < len(meta_rows) else {})
            stype = _pick(row, TYPE_COLUMNS).lower()

            # Open Specy tags technique as 'raman' / 'ftir' (occasionally 'ir').
            # Rows with no tag at all are kept: dropping them would silently
            # lose spectra whose metadata is merely incomplete.
            if stype and want != "both":
                is_raman = "raman" in stype
                is_ir = ("ftir" in stype) or ("ir" in stype and not is_raman)
                if (want == "raman" and not is_raman) or (want == "ftir" and not is_ir):
                    skipped_type += 1
                    continue

            name = _pick(row, NAME_COLUMNS) or col
            collection = _pick(row, COLLECTION_COLUMNS)
            if flt and flt not in f"{name} {collection} {col}".lower():
                skipped_filter += 1
                continue

            y = np.asarray(spectra[col], dtype=float)
            x = np.asarray(wav, dtype=float)
            if y.shape != x.shape:
                skipped_bad += 1
                continue
            good = np.isfinite(x) & np.isfinite(y)
            x, y = x[good], y[good]
            if x.size < 5:
                skipped_bad += 1
                continue
            order = np.argsort(x)
            x, y = x[order], y[order]

            x, y = crop(x, y, args.xmin, args.xmax)
            if x.size < 5:
                skipped_bad += 1
                continue
            x, y = decimate(x, y, args.step)
            y = normalize(y)
            px, py = detect_bands(x, y, max_peaks=args.max_peaks,
                                  min_prominence=args.prominence)
            if args.require_peaks and px.size == 0:
                skipped_bad += 1
                continue

            gid = str(col).replace("/", "_")
            if gid in grp:
                gid = f"{gid}_{n}"
            g = grp.create_group(gid)
            g.create_dataset("x", data=x.astype("float32"), compression="gzip")
            g.create_dataset("y", data=y.astype("float32"), compression="gzip")
            g.create_dataset("peaks", data=px.astype("float32"), compression="gzip")
            g.create_dataset("intensities", data=py.astype("float32"), compression="gzip")

            g.attrs["name"] = name
            g.attrs["rod_id"] = ""
            g.attrs["rruff_id"] = str(col)      # back-compat: loaders show this
            g.attrs["cod_id"] = ""
            g.attrs["url"] = OPENSPECY_URL
            g.attrs["cod_url"] = ""
            g.attrs["peaks"] = px.astype("float32")
            g.attrs["mineral"] = ""
            g.attrs["formula"] = ""
            g.attrs["collection"] = collection
            g.attrs["spectrum_type"] = stype or (want if want != "both" else "raman")
            g.attrs["laser_nm"] = _pick(row, ("laser_light_used", "laser", "excitation"))
            g.attrs["instrument"] = _pick(row, ("instrument_used", "instrument",
                                                "spectral_resolution"))
            g.attrs["quality"] = collection
            g.attrs["source"] = "OpenSpecy"
            g.attrs["license"] = "CC-BY-4.0"
            kept += 1

            if n % 250 == 0 or n == total:
                print(f"  ... {n}/{total} processed ({kept} kept)")

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nDone: {kept} spectra -> {args.out}  ({size_mb:.1f} MB)")
    if skipped_type:
        print(f"  {skipped_type} skipped as not '{want}' (use --only both to keep everything)")
    if skipped_filter:
        print(f"  {skipped_filter} skipped by --filter")
    if skipped_bad:
        print(f"  {skipped_bad} skipped as unusable (length mismatch / empty / no bands)")
    if kept == 0:
        print("WARNING: no spectra written. Try --only both, or relax --filter.")
    else:
        print("\nOpen Specy data are CC-BY: cite the contributing collections you use.")
    return kept


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build an Open Specy .h5 Raman reference library for the "
                    "Raman Peak Analysis Toolkit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Open Specy data are CC-BY. https://openspecy.org")

    src = p.add_argument_group("source")
    src.add_argument("--type", default="raw",
                     choices=sorted(set(OSF_URLS) | set(AWS_URLS)),
                     help="library variant (default raw -- real intensities, "
                          "the only one sensible to overlay)")
    src.add_argument("--osf", action="store_true",
                     help="download from OSF instead of the CloudFront mirror")
    src.add_argument("--cache", default="openspecy_cache",
                     help="directory for the downloaded .rds (default openspecy_cache)")
    src.add_argument("--offline", action="store_true",
                     help="use only the cached .rds; make no network requests")
    src.add_argument("--from-csv", dest="from_csv",
                     help="skip the .rds entirely and read a CSV exported from R "
                          "(wavenumber column + one column per spectrum)")
    src.add_argument("--metadata-csv", dest="metadata_csv",
                     help="companion metadata CSV for --from-csv")

    sel = p.add_argument_group("selection")
    sel.add_argument("--only", choices=("raman", "ftir", "both"), default="raman",
                     help="which technique to keep (default raman). Use 'ftir' to "
                          "build a library for the FTIR toolkit -- Open Specy's "
                          "infrared holdings are if anything larger than its Raman "
                          "ones (FLOPP, FLOPP-e, Primpke, Cabernard).")
    sel.add_argument("--include-ftir", dest="include_ftir", action="store_true",
                     help="deprecated alias for --only both")
    sel.add_argument("--filter",
                     help="substring filter over name / collection / column id")

    proc = p.add_argument_group("spectrum processing")
    proc.add_argument("--xmin", type=float, default=None, help="crop below (cm-1)")
    proc.add_argument("--xmax", type=float, default=None, help="crop above (cm-1)")
    proc.add_argument("--step", type=float, default=0.0,
                      help="resample onto a uniform grid of this step (0 = native)")
    proc.add_argument("--prominence", type=float, default=0.04,
                      help="band-detection prominence (default 0.04, matches the apps)")
    proc.add_argument("--max-peaks", dest="max_peaks", type=int, default=40,
                      help="keep at most N bands per spectrum (default 40)")
    proc.add_argument("--require-peaks", dest="require_peaks", action="store_true",
                      help="skip spectra where no band clears the threshold")

    p.add_argument("--out", default="openspecy_raman.h5", help="output .h5 path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.from_csv:
        wav, spectra, meta = load_from_csv(args.from_csv, args.metadata_csv)
    else:
        path = download_rds(args.type, args.cache, use_aws=not args.osf,
                            offline=args.offline)
        print("Decoding .rds ...")
        wav, spectra, meta = extract_library(decode_rds(path))

    build(wav, spectra, meta, args)



if __name__ == "__main__":
    main()
