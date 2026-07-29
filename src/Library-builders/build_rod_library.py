#!/usr/bin/env python3
"""
build_rod_library.py
====================

Build a Raman reference library (.h5) from the Raman Open Database (ROD),
for use with the Raman Peak Analysis Toolkit (raman_plotter.py /
raman_plotter.html).

WHY THIS EXISTS
---------------
ROD (https://solsa.crystallography.net/rod/) is the Raman sibling of the
Crystallography Open Database, run by the same group, with all data placed in
the public domain under CC0. Unlike COD it is small -- ~1100 entries -- so the
*entire* database fits comfortably in a single .h5 that both apps can load and
search offline. That matters because ROD's REST API currently sends
`Access-Control-Allow-Origin: https://thermott.ibt.lt`, i.e. CORS is enabled
but pinned to one origin, so the browser app cannot query it live.

Unlike COD (crystal structures -> computed powder patterns) ROD ships measured
spectra directly, so there is nothing to simulate: we download, parse, detect
bands, and bake.

    1. DISCOVER which ROD entries to include (whole DB by default, or a
       filtered subset via the ROD search endpoint).
    2. FETCH each spectrum as JCAMP-DX   (https://solsa.crystallography.net/rod/<id>.jdx)
       and its metadata as CIF2          (https://solsa.crystallography.net/rod/<id>.rod)
    3. DETECT reference bands with the same prominence rule the apps use, so
       "Match by Selected Peaks" scores identically against RRUFF, COD and ROD.
    4. BAKE everything into a compact .h5.

The output .h5 uses the SAME schema as the RRUFF/COD libraries the apps already
read, so it loads today through the existing "Open .h5 Library" buttons even
before any ROD-specific UI exists:

    /spectra/<rod_id>                (one group per spectrum)
        dataset  x                   Raman shift (cm-1)
        dataset  y                   intensity, normalized 0..100
        dataset  peaks               detected band positions (cm-1)
        dataset  intensities         relative heights at those bands (0..100)
        attrs:
            name                     display name (mineral / chemical / formula)
            rod_id                   ROD entry id
            rruff_id                 = rod_id  (back-compat: existing loaders
                                       show this field, so IDs stay visible)
            cod_id                   cross-linked COD structure id, if any
            url                      https://solsa.crystallography.net/rod/<id>.html
            peaks                    same as the dataset (attribute copy, for
                                       loaders that read peaks from attrs)
            formula, mineral, laser_nm, resolution_cm1, instrument,
            publication, year, quality, source="ROD", license="CC0-1.0"

EXAMPLES
--------
  # The whole database (recommended -- it is small)
  python build_rod_library.py --all --out rod_raman_library.h5

  # Just the carbonates, by free-text metadata search
  python build_rod_library.py --text calcite --out calcite.h5

  # Everything containing Ti and O
  python build_rod_library.py --elements Ti,O --out tio2.h5

  # A specific set of ROD ids
  python build_rod_library.py --ids 1000076,1000506 --out picks.h5

  # Re-bake from an existing download cache, no network at all
  python build_rod_library.py --all --offline --out rod_raman_library.h5

REQUIREMENTS
------------
  pip install h5py numpy scipy      (scipy optional; a fallback peak finder is
                                     used if it is missing)

LICENSE NOTE
------------
ROD data are CC0. The builder stamps `license` and `source` attributes into the
file and each group so attribution travels with the library. Users of the data
should still acknowledge the original depositors -- the `publication` attribute
carries the citation where ROD provides one.
"""

import argparse
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.error

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("This tool needs h5py:  pip install h5py")

try:
    from scipy.signal import find_peaks
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


ROD_BASE = "https://solsa.crystallography.net/rod"
ROD_SEARCH_URL = ROD_BASE + "/result"
ROD_JDX_URL = ROD_BASE + "/{rid}.jdx"
ROD_CIF_URL = ROD_BASE + "/{rid}.rod"
ROD_PAGE_URL = ROD_BASE + "/{rid}.html"
COD_PAGE_URL = "https://www.crystallography.net/cod/{cid}.html"

DEFAULT_UA = ("Mozilla/5.0 (compatible; Raman-Toolkit-ROD-builder/1.0; "
              "+https://solsa.crystallography.net/rod/)")


# ============================================================================
# HTTP with an on-disk cache
# ============================================================================


def _ds(group, name, data, gzip_min=1024):
    """Create a dataset, compressing only when the array is big enough to gain.

    gzip on a 60-value array is a net loss: HDF5 stores a chunk index and filter
    pipeline per dataset, which outweighs any compression of so few values. A
    195k-entry peaks library measured 1423 MB compressed and 597 MB not. Curves
    run to thousands of points and do compress, hence the size test rather than
    dropping compression outright.
    """
    arr = np.asarray(data)
    if arr.size >= gzip_min:
        return group.create_dataset(name, data=arr, compression="gzip")
    return group.create_dataset(name, data=arr)


def _cache_path(cache_dir, rid, ext):
    return os.path.join(cache_dir, f"{rid}.{ext}")


def http_get(url, cfg, binary=False):
    """GET with retries. Returns text (or bytes) or raises."""
    last = None
    for attempt in range(cfg["retries"] + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": cfg["user_agent"]})
            with urllib.request.urlopen(req, timeout=cfg["timeout"]) as resp:
                raw = resp.read()
            return raw if binary else raw.decode("utf-8", errors="replace")
        except Exception as e:                      # noqa: BLE001 - retry anything
            last = e
            if attempt < cfg["retries"]:
                time.sleep(cfg["pause"] * (attempt + 2))
    raise last


def fetch_entry_file(rid, ext, cfg):
    """Returns the text of <rid>.<ext>, using (and filling) the disk cache.
    In --offline mode only the cache is consulted."""
    path = _cache_path(cfg["cache"], rid, ext)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    if cfg["offline"]:
        raise FileNotFoundError(f"{path} not in cache (offline mode)")

    url = (ROD_JDX_URL if ext == "jdx" else ROD_CIF_URL).format(rid=rid)
    text = http_get(url, cfg)
    os.makedirs(cfg["cache"], exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    time.sleep(cfg["pause"])
    return text


# ============================================================================
# Entry discovery
# ============================================================================

# Search keys the ROD result endpoint understands that we expose on the CLI.
SEARCH_KEYS = ("text", "formula", "journal", "year", "doi",
               "strictmin", "strictmax", "vmin", "vmax")


def build_search_query(args):
    """Translates CLI filters into ROD `result` query parameters."""
    q = {}
    for key in SEARCH_KEYS:
        val = getattr(args, key, None)
        if val not in (None, ""):
            q[key] = str(val)
    if args.elements:
        for i, el in enumerate(_split_list(args.elements)[:8], start=1):
            q[f"el{i}"] = el
    if args.exclude_elements:
        for i, el in enumerate(_split_list(args.exclude_elements)[:4], start=1):
            q[f"nel{i}"] = el
    if args.all and not q:
        q["id"] = "%"          # wildcard: every entry
    return q


def _split_list(s):
    return [t for t in re.split(r"[,\s]+", str(s)) if t]


def _ids_from_text(text):
    """Pulls 6-9 digit ROD ids out of an lst/csv/json payload, order-preserving."""
    seen, out = set(), []
    for m in re.finditer(r"\b(\d{6,9})\b", text or ""):
        rid = m.group(1)
        if rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out


def discover_ids(args, cfg):
    """Resolves the requested selection to a list of ROD ids.

    Tries the documented result formats in order of how easy they are to parse.
    `lst` is a bare id list; `csv`/`json` carry metadata we do not need here
    (we read metadata per-entry from the CIF) but they are useful fallbacks if
    a format is disabled server-side."""
    explicit = []
    if args.ids:
        explicit += _split_list(args.ids)
    if args.ids_file:
        with open(args.ids_file) as fh:
            explicit += _split_list(fh.read())
    if explicit:
        seen, out = set(), []
        for r in explicit:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    query = build_search_query(args)
    if not query:
        sys.exit("Nothing selected. Use --all, a filter (--text/--elements/...), "
                 "or --ids / --ids-file.")

    if cfg["offline"]:
        ids = sorted({fn.split(".")[0] for fn in os.listdir(cfg["cache"])
                      if fn.endswith(".jdx")})
        if not ids:
            sys.exit(f"Offline mode: no cached .jdx files in {cfg['cache']}")
        print(f"Offline: using {len(ids)} cached entr(y/ies).")
        return ids

    for fmt in ("lst", "csv", "json"):
        q = dict(query, format=fmt)
        url = f"{ROD_SEARCH_URL}?{urllib.parse.urlencode(q)}"
        try:
            text = http_get(url, cfg)
        except Exception as e:                      # noqa: BLE001
            print(f"  [warn] search format={fmt} failed: {e}")
            continue
        ids = _ids_from_text(text)
        if ids:
            print(f"Search (format={fmt}) matched {len(ids)} entr(y/ies).")
            return ids
        print(f"  [warn] search format={fmt} returned no ids")

    sys.exit("ROD search returned nothing for that selection. Check the filter, "
             "or pass ids directly with --ids / --ids-file.")


# ============================================================================
# JCAMP-DX parsing (ROD serves ##XYPOINTS=(XY..XY))
# ============================================================================

_JDX_HEADER = re.compile(r"^\s*##\s*([^=]+?)\s*=\s*(.*)$")
_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?")


def parse_jcamp(text):
    """Parses a ROD JCAMP-DX spectrum.

    Returns (x, y, header_dict). Handles the (XY..XY) form ROD emits and the
    common (X++(Y..Y)) equidistant form as a courtesy, since depositors
    occasionally supply the latter.
    """
    header, data_lines = {}, []
    block = None          # which data-block tag opened the current run
    collecting = False
    for line in text.splitlines():
        m = _JDX_HEADER.match(line)
        if m:
            key = m.group(1).strip().upper()
            val = m.group(2).strip()
            header[key] = val
            if key in ("XYPOINTS", "XYDATA", "PEAK TABLE", "DATA TABLE"):
                block = (key, val.upper())
                collecting = True
            else:
                # Any other header (##END=, ##MOLFORM=, ...) closes the run.
                # ROD puts several of these *after* the data, so `block` must
                # survive; only collection stops.
                collecting = False
            continue
        if collecting and line.strip():
            data_lines.append(line)

    if not data_lines or block is None:
        raise ValueError("no data block found in JCAMP-DX")

    blob = " ".join(data_lines)
    kind, form = block

    if kind in ("XYPOINTS", "PEAK TABLE", "DATA TABLE") or "XY..XY" in form:
        nums = [float(t.replace("D", "E").replace("d", "e"))
                for t in _NUMBER.findall(blob)]
        if len(nums) < 4:
            raise ValueError("too few data points")
        if len(nums) % 2:
            nums = nums[:-1]
        arr = np.asarray(nums, dtype=float).reshape(-1, 2)
        x, y = arr[:, 0], arr[:, 1]
    else:
        # (X++(Y..Y)): first number of each line is X, the rest are Y on an
        # equidistant grid defined by FIRSTX/LASTX/NPOINTS.
        ys = []
        for line in data_lines:
            toks = _NUMBER.findall(line)
            if len(toks) > 1:
                ys.extend(float(t.replace("D", "E")) for t in toks[1:])
        if len(ys) < 4:
            raise ValueError("too few data points")
        y = np.asarray(ys, dtype=float)
        x0 = float(header.get("FIRSTX", 0.0))
        x1 = float(header.get("LASTX", len(y) - 1))
        x = np.linspace(x0, x1, len(y))

    xf = float(header.get("XFACTOR", 1.0) or 1.0)
    yf = float(header.get("YFACTOR", 1.0) or 1.0)
    x, y = x * xf, y * yf

    order = np.argsort(x)                       # some deposits run high->low
    x, y = x[order], y[order]

    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 5:
        raise ValueError("fewer than 5 finite points")
    return x, y, header


# ============================================================================
# CIF2 metadata (generic tag scrape -- robust to dictionary changes)
# ============================================================================

def parse_cif_tags(text):
    """Collects simple `_tag value` pairs from a CIF. Loop bodies are skipped:
    ROD stores the spectrum itself in a loop and we take that from the .jdx.

    Deliberately generic. ROD uses the cif_raman / cif_rod dictionaries whose
    exact item names have changed over the project's life, so downstream we
    match on substrings rather than hardcoded names.
    """
    tags, in_loop, pending = {}, False, None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        i += 1

        if not line or line.startswith("#"):
            continue
        low = line.lower()

        if low.startswith("loop_"):
            in_loop = True
            continue
        if in_loop:
            # A loop ends at the next non-tag, non-data construct. Tags that
            # start a new simple assignment (tag + value on one line) end it.
            if line.startswith("_") and len(line.split(None, 1)) > 1:
                in_loop = False
            elif line.startswith("_") or not line.startswith("_"):
                continue

        if line.startswith(";"):                     # multiline text field
            buf = [line[1:]]
            while i < len(lines) and not lines[i].startswith(";"):
                buf.append(lines[i])
                i += 1
            i += 1
            if pending:
                tags[pending] = " ".join(b.strip() for b in buf).strip()
                pending = None
            continue

        if line.startswith("_"):
            parts = line.split(None, 1)
            tag = parts[0].lower()
            if len(parts) == 1:
                pending = tag                        # value on following line(s)
            else:
                tags[tag] = _clean_cif_value(parts[1])
                pending = None
        elif pending:
            tags[pending] = _clean_cif_value(line)
            pending = None
    return tags


def _clean_cif_value(v):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        v = v[1:-1]
    v = re.sub(r"\(\d+\)$", "", v).strip()           # drop e.s.d. like 1.234(5)
    return "" if v in ("?", ".") else v


def _first_tag(tags, *substrings, numeric=False):
    """Returns the first tag value whose name contains all of `substrings`."""
    for tag, val in tags.items():
        if not val:
            continue
        if all(s in tag for s in substrings):
            if numeric and not _NUMBER.search(val):
                continue
            return val
    return ""


def extract_metadata(rid, cif_text, jdx_header):
    """Builds the attribute dict for one ROD entry."""
    tags = parse_cif_tags(cif_text) if cif_text else {}

    mineral = _first_tag(tags, "_chemical_name_mineral") or \
        _first_tag(tags, "chemical_name", "mineral")
    common = _first_tag(tags, "_chemical_name_common")
    systematic = _first_tag(tags, "_chemical_name_systematic")
    formula = (_first_tag(tags, "_chemical_formula_sum")
               or _first_tag(tags, "chemical_formula")
               or jdx_header.get("MOLFORM", ""))
    formula = re.sub(r"\s+", " ", str(formula).replace("-", " ")).strip()

    # COD cross-link: any tag mentioning cod whose value looks like a COD id.
    cod_id = ""
    for tag, val in tags.items():
        if "cod" in tag and re.fullmatch(r"\d{7,8}", str(val).strip()):
            cod_id = str(val).strip()
            break

    laser = _first_tag(tags, "wavelength", numeric=True) or \
        _first_tag(tags, "excitation", numeric=True)
    laser_nm = ""
    if laser:
        m = _NUMBER.search(laser)
        if m:
            v = float(m.group())
            # Some deposits give metres or micrometres; normalize to nm.
            if v < 1e-3:
                v *= 1e9
            elif v < 10:
                v *= 1000.0
            laser_nm = f"{v:g}"

    resolution = _first_tag(tags, "resolution", numeric=True) or \
        jdx_header.get("RESOLUTION", "")
    instrument = " ".join(p for p in (
        _first_tag(tags, "device", "company") or _first_tag(tags, "instrument", "make"),
        _first_tag(tags, "device", "model") or _first_tag(tags, "instrument", "model"),
    ) if p).strip()

    publication = _first_tag(tags, "_publ_section_title")
    year = _first_tag(tags, "_journal_year") or _first_tag(tags, "journal", "year")

    name = (mineral or common or systematic or formula
            or jdx_header.get("TITLE", "") or f"ROD {rid}")

    return {
        "name": str(name).strip(),
        "mineral": mineral,
        "formula": formula,
        "cod_id": cod_id,
        "laser_nm": laser_nm,
        "resolution_cm1": str(resolution),
        "instrument": instrument,
        "publication": publication,
        "year": str(year),
    }


# ============================================================================
# Band detection -- mirrors detect_reference_peaks() in raman_plotter.py
# ============================================================================

def detect_bands(x, y, max_peaks=40, min_prominence=0.04, distance=3):
    """Most prominent band positions, on min-max normalized intensity so the
    threshold is scale-free. Kept bit-for-bit consistent with the app's own
    detector, otherwise match scores would differ between a baked library and
    a folder of raw spectra."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 5:
        return np.array([]), np.array([])
    rng = np.ptp(y)
    if rng <= 0:
        return np.array([]), np.array([])
    yn = (y - y.min()) / rng

    if _HAVE_SCIPY:
        idx, props = find_peaks(yn, prominence=min_prominence, distance=distance)
        proms = props.get("prominences", np.ones(len(idx)))
    else:
        idx, proms = _fallback_peaks(yn, min_prominence, distance)

    if len(idx) == 0:
        return np.array([]), np.array([])
    order = np.argsort(proms)[::-1][:max_peaks]
    idx = idx[order]
    keep = np.argsort(x[idx])
    idx = idx[keep]
    return x[idx], yn[idx] * 100.0


def _fallback_peaks(yn, min_prominence, distance):
    """Crude local-maximum finder for installs without scipy. Prominence is
    approximated by the drop to the lower of the two neighbouring minima."""
    idx = []
    n = len(yn)
    for i in range(1, n - 1):
        if yn[i] > yn[i - 1] and yn[i] >= yn[i + 1]:
            idx.append(i)
    proms = []
    keep = []
    for i in idx:
        lo = i
        while lo > 0 and yn[lo - 1] <= yn[lo]:
            lo -= 1
        hi = i
        while hi < n - 1 and yn[hi + 1] <= yn[hi]:
            hi += 1
        prom = yn[i] - max(yn[lo], yn[hi])
        if prom >= min_prominence:
            keep.append(i)
            proms.append(prom)
    if not keep:
        return np.array([], dtype=int), np.array([])
    # enforce minimum separation, strongest first
    order = np.argsort(proms)[::-1]
    chosen = []
    for o in order:
        i = keep[o]
        if all(abs(i - c) >= distance for c in chosen):
            chosen.append(i)
    chosen = np.array(sorted(chosen), dtype=int)
    prom_map = dict(zip(keep, proms))
    return chosen, np.array([prom_map[i] for i in chosen])


# ============================================================================
# Post-processing
# ============================================================================

def normalize(y):
    """Scale to 0..100. The apps rescale references to the active spectrum on
    overlay anyway; this just keeps the stored file consistent and compressible."""
    y = np.asarray(y, dtype=float)
    lo = float(np.min(y))
    rng = float(np.ptp(y))
    if rng <= 0:
        return np.zeros_like(y)
    return (y - lo) / rng * 100.0


def crop(x, y, xmin, xmax):
    if xmin is None and xmax is None:
        return x, y
    m = np.ones(len(x), dtype=bool)
    if xmin is not None:
        m &= x >= xmin
    if xmax is not None:
        m &= x <= xmax
    return x[m], y[m]


def decimate(x, y, step):
    """Uniform resample onto a `step` cm-1 grid. Optional -- native grids are
    already only a few thousand points, so this is for size-sensitive uses."""
    if not step or step <= 0 or len(x) < 3:
        return x, y
    gx = np.arange(float(x[0]), float(x[-1]) + step, step)
    if len(gx) < 3:
        return x, y
    return gx, np.interp(gx, x, y)


# ============================================================================
# Build
# ============================================================================

def build_library(ids, args, cfg):
    total = len(ids)
    kept = failed = 0
    skipped_reasons = {}

    with h5py.File(args.out, "w") as h5:
        h5.attrs["source"] = "ROD"
        h5.attrs["database"] = "Raman Open Database"
        h5.attrs["database_url"] = ROD_BASE + "/"
        h5.attrs["license"] = "CC0-1.0"
        h5.attrs["built"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h5.attrs["builder"] = "build_rod_library.py 1.0"
        h5.attrs["units_x"] = "cm-1"
        h5.attrs["storage"] = "curve+peaks"
        spectra = h5.create_group("spectra")

        for n, rid in enumerate(ids, 1):
            try:
                jdx = fetch_entry_file(rid, "jdx", cfg)
                x, y, header = parse_jcamp(jdx)
            except Exception as e:                       # noqa: BLE001
                failed += 1
                reason = str(e).split("\n")[0][:80]
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                if args.verbose:
                    print(f"  [skip] ROD {rid}: {reason}")
                continue

            cif_text = ""
            if not args.no_metadata:
                try:
                    cif_text = fetch_entry_file(rid, "rod", cfg)
                except Exception as e:                   # noqa: BLE001
                    if args.verbose:
                        print(f"  [warn] ROD {rid}: metadata unavailable ({e})")

            meta = extract_metadata(rid, cif_text, header)

            x, y = crop(x, y, args.xmin, args.xmax)
            if len(x) < 5:
                failed += 1
                skipped_reasons["empty after crop"] = \
                    skipped_reasons.get("empty after crop", 0) + 1
                continue
            x, y = decimate(x, y, args.step)
            y = normalize(y)
            px, py = detect_bands(x, y, max_peaks=args.max_peaks,
                                  min_prominence=args.prominence)
            if args.require_peaks and px.size == 0:
                failed += 1
                skipped_reasons["no bands detected"] = \
                    skipped_reasons.get("no bands detected", 0) + 1
                continue

            g = spectra.create_group(str(rid))
            g.create_dataset("x", data=x.astype("float32"), compression="gzip")
            g.create_dataset("y", data=y.astype("float32"), compression="gzip")
            _ds(g, "peaks", px.astype("float32"))
            _ds(g, "intensities", py.astype("float32"))

            g.attrs["name"] = meta["name"]
            g.attrs["rod_id"] = str(rid)
            g.attrs["rruff_id"] = str(rid)      # back-compat: existing loaders
                                                # surface this as the entry id
            g.attrs["cod_id"] = meta["cod_id"]
            g.attrs["url"] = ROD_PAGE_URL.format(rid=rid)
            g.attrs["cod_url"] = COD_PAGE_URL.format(cid=meta["cod_id"]) if meta["cod_id"] else ""
            g.attrs["peaks"] = px.astype("float32")   # attribute copy for loaders
                                                      # that read peaks from attrs
            g.attrs["mineral"] = meta["mineral"]
            g.attrs["formula"] = meta["formula"]
            g.attrs["laser_nm"] = meta["laser_nm"]
            g.attrs["resolution_cm1"] = meta["resolution_cm1"]
            g.attrs["instrument"] = meta["instrument"]
            g.attrs["publication"] = meta["publication"]
            g.attrs["year"] = meta["year"]
            g.attrs["quality"] = meta["laser_nm"] and f"{meta['laser_nm']} nm" or ""
            g.attrs["source"] = "ROD"
            g.attrs["license"] = "CC0-1.0"
            kept += 1

            if n % 25 == 0 or n == total:
                print(f"  ... {n}/{total} processed ({kept} kept, {failed} skipped)")

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nDone: {kept} spectra -> {args.out}  ({size_mb:.1f} MB, {failed} skipped)")
    # --flat: hand the finished file to the repack converter rather than
    # duplicating it here, so the layout has one implementation and one test.
    if getattr(args, "flat", False):
        try:
            from .repack_library import flat_convert
        except ImportError:
            try:
                from repack_library import flat_convert
            except ImportError:
                print("  --flat needs repack_library.py alongside this script; "
                      "the ordinary library was written and is usable.")
                flat_convert = None
        if flat_convert is not None:
            before = os.path.getsize(args.out)
            # Verify a sample against the file we just wrote, while the
            # group-per-entry original still exists to compare with.
            try:
                flat_convert(args.out, verify=200)
            except Exception as e:
                print('  --flat skipped: %s' % e)
                print('  The ordinary library was written and is usable.')
                return kept
            after = os.path.getsize(args.out)
            print("  --flat: %.1f MB -> %.1f MB (%.1fx smaller)"
                  % (before / 1e6, after / 1e6, before / max(1, after)))

    if skipped_reasons:
        print("Skip reasons:")
        for reason, count in sorted(skipped_reasons.items(), key=lambda t: -t[1]):
            print(f"   {count:5d}  {reason}")
    if kept == 0:
        print("WARNING: no spectra written. Check the selection, cache, or network.")
    return kept




# ============================================================================
# CLI
# ============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build a Raman Open Database (ROD) .h5 reference library "
                    "for the Raman Peak Analysis Toolkit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="ROD data are CC0. https://solsa.crystallography.net/rod/")

    sel = p.add_argument_group("selection")
    sel.add_argument("--all", action="store_true",
                     help="every entry in ROD (~1100; the whole DB is only a few MB)")
    sel.add_argument("--ids", help="explicit comma/space separated ROD ids")
    sel.add_argument("--ids-file", dest="ids_file",
                     help="text file with ROD ids, one per line")
    sel.add_argument("--text", help="free-text metadata search (name, mineral, biblio)")
    sel.add_argument("--formula", help="empirical formula in Hill notation, e.g. 'C8 H10 N4 O2'")
    sel.add_argument("--elements", help="elements that must be present, e.g. 'Ti,O'")
    sel.add_argument("--exclude-elements", dest="exclude_elements",
                     help="elements that must be absent")
    sel.add_argument("--journal", help="journal name filter")
    sel.add_argument("--year", help="publication year filter")
    sel.add_argument("--doi", help="DOI filter")
    sel.add_argument("--strictmin", help="min number of distinct elements")
    sel.add_argument("--strictmax", help="max number of distinct elements")
    sel.add_argument("--vmin", help="min cell volume")
    sel.add_argument("--vmax", help="max cell volume")

    proc = p.add_argument_group("spectrum processing")
    proc.add_argument("--xmin", type=float, default=None,
                      help="crop below this Raman shift (cm-1)")
    proc.add_argument("--xmax", type=float, default=None,
                      help="crop above this Raman shift (cm-1)")
    proc.add_argument("--step", type=float, default=0.0,
                      help="resample onto a uniform grid of this step (0 = keep native)")
    proc.add_argument("--prominence", type=float, default=0.04,
                      help="band-detection prominence on 0..1 normalized intensity "
                           "(default 0.04, matches the apps)")
    proc.add_argument("--max-peaks", dest="max_peaks", type=int, default=40,
                      help="keep at most N bands per spectrum (default 40)")
    proc.add_argument("--require-peaks", dest="require_peaks", action="store_true",
                      help="skip entries where no band clears the prominence threshold")

    net = p.add_argument_group("fetching")
    net.add_argument("--cache", default="rod_cache",
                     help="directory for downloaded .jdx/.rod files (default rod_cache)")
    net.add_argument("--offline", action="store_true",
                     help="build only from the cache; make no network requests")
    net.add_argument("--no-metadata", dest="no_metadata", action="store_true",
                     help="skip the .rod CIF fetch (halves requests; names fall back "
                          "to the JCAMP TITLE/MOLFORM)")
    net.add_argument("--pause", type=float, default=0.34,
                     help="seconds between requests (default 0.34)")
    net.add_argument("--retries", type=int, default=2, help="retries per request")
    net.add_argument("--timeout", type=float, default=60.0, help="per-request timeout")
    net.add_argument("--user-agent", dest="user_agent", default=DEFAULT_UA)

    p.add_argument("--limit", type=int, default=0, help="cap number of entries")
    p.add_argument("--verbose", action="store_true", help="report every skip")
    p.add_argument("--flat", action="store_true",
                   help="write the consolidated layout (one concatenated peaks "
                        "array plus an offset index): ~16x smaller than one "
                        "group per entry, and read by the apps from "
                        "2026.07.28.1 onward")
    p.add_argument("--out", default="rod_raman_library.h5", help="output .h5 path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not _HAVE_SCIPY:
        print("Note: scipy not found; using the fallback peak finder. Band "
              "positions may differ slightly from the app's detector.")

    cfg = {"cache": args.cache, "offline": args.offline, "pause": args.pause,
           "retries": args.retries, "timeout": args.timeout,
           "user_agent": args.user_agent}
    os.makedirs(args.cache, exist_ok=True)

    ids = discover_ids(args, cfg)
    if args.limit:
        ids = ids[:args.limit]
    print(f"Building from {len(ids)} ROD entr(y/ies) -> {args.out}")
    build_library(ids, args, cfg)



if __name__ == "__main__":
    main()
