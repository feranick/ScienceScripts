#!/usr/bin/env python3
"""
build_cod_powder_library.py
===========================

Build an ad-hoc powder-XRD reference library (.h5) from the Crystallography
Open Database (COD), for use with the XRD Peak Analysis Toolkit
(xrd_plotter.py / xrd_plotter.html).

WHY THIS EXISTS
---------------
COD distributes crystal *structures* (CIF files), not ready-made diffraction
patterns, and the Profex COD SQLite (`cod-*.db3`) is a metadata-only *search
index* (unit cell, space group, formula, names, bibliography -- no atomic
coordinates, no patterns). So to get overlayable patterns we:

    1. SEARCH the Profex db3 offline to choose which COD entries we want
       (any chemistry, not just minerals).
    2. FETCH just those CIFs from COD  (https://www.crystallography.net/cod/<id>.cif).
    3. COMPUTE each powder pattern with pymatgen's XRDCalculator (Cu-Kalpha),
       the same engine the app already uses for Materials Project.
    4. BAKE everything into a compact .h5 that both apps load offline.

The output .h5 uses the SAME schema as the RRUFF powder library the apps
already read, so no changes to the loaders are required:

    /spectra/<group>            (one group per phase)
        dataset  x              2-theta grid (deg)
        dataset  y              broadened intensity profile (0..100)
        attrs:
            name                display name (mineral / chemical name / formula)
            cod_id              COD entry id
            rruff_id            "" (kept for loader compatibility)
            url                 https://www.crystallography.net/cod/<id>.html
            peaks               1-D array of reflection 2-theta positions (deg)
            intensities         1-D array of reflection intensities (0..100)
            formula, sg, wavelength, source="COD"

EXAMPLES
--------
  # Everything called "quartz", one representative per cell, into quartz.h5
  python build_cod_powder_library.py --db3 cod-260101.db3 \
        --name quartz --max-per-formula 1 --out quartz.h5

  # All anatase/rutile/brookite (TiO2 polymorphs)
  python build_cod_powder_library.py --db3 cod-260101.db3 \
        --elements Ti,O --only-elements --out tio2.h5

  # A specific set of COD ids (no db3 needed for selection)
  python build_cod_powder_library.py --ids 1011097,9008789 --out picks.h5

  # Space group 152 (P3121) silicates
  python build_cod_powder_library.py --db3 cod-260101.db3 \
        --elements Si,O --sg 152 --limit 50 --out sg152.h5

REQUIREMENTS
------------
  pip install pymatgen h5py numpy
  (Cu-Kalpha wavelength default; requires network access only for CIF fetch.)
"""

import argparse
import os
import re
import sys
import time
import sqlite3
import urllib.request
import urllib.error

import numpy as np

# --- Heavy / optional deps are imported lazily with a friendly message -------
try:
    import h5py
except ImportError:
    sys.exit("This tool needs h5py:  pip install h5py")

COD_CIF_URL = "https://www.crystallography.net/cod/{cid}.cif"
COD_PAGE_URL = "https://www.crystallography.net/cod/{cid}.html"
DEFAULT_UA = ("Mozilla/5.0 (compatible; XRD-Toolkit-COD-builder/1.0; "
              "+https://www.crystallography.net/)")

# Elements, for parsing COD formula strings like "- O2 Si -"
_ELEMENT_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


# ============================================================================
# Selection: query the Profex COD db3 (metadata search index)
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


def elements_in_formula(formula):
    """Return the set of chemical-element symbols found in a COD formula string.
    COD formulae look like '- O2 Si -' or '- C5 H17 Al N2 O8 P2 -'."""
    if not formula:
        return set()
    out = set()
    for tok in str(formula).replace("-", " ").split():
        m = _ELEMENT_TOKEN.match(tok)
        if m and m.group(1):
            out.add(m.group(1))
    return out


def _normalise_element_list(s):
    return [e.strip().capitalize() for e in re.split(r"[,\s]+", s) if e.strip()]


def select_from_db3(db3_path, args):
    """Return a list of candidate rows (dicts) from the Profex db3."""
    if not os.path.exists(db3_path):
        sys.exit(f"db3 not found: {db3_path}")
    con = sqlite3.connect(db3_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    where, params = [], []
    if args.mineral:
        where.append("mineral LIKE ?")
        params.append(f"%{args.mineral}%")
    if args.name:
        where.append("(mineral LIKE ? OR chemname LIKE ? OR commonname LIKE ?)")
        params += [f"%{args.name}%"] * 3
    if args.formula_contains:
        where.append("(formula LIKE ? OR calcformula LIKE ?)")
        params += [f"%{args.formula_contains}%"] * 2
    if args.sg is not None:
        where.append("sgNumber = ?")
        params.append(args.sg)
    if getattr(args, "minerals_only", False):
        where.append("mineral IS NOT NULL AND mineral != '' AND mineral != ?")
        params.append("\\N")

    sql = "SELECT file, a, b, c, alpha, beta, gamma, sg, sgNumber, " \
          "formula, calcformula, cellformula, mineral, chemname, commonname, Z " \
          "FROM data"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY file"

    rows = [dict(r) for r in cur.execute(sql, params).fetchall()]
    con.close()

    # Element-based filtering is done in Python (COD has no element columns).
    want = set(_normalise_element_list(args.elements)) if args.elements else set()
    excl = set(_normalise_element_list(args.exclude_elements)) if args.exclude_elements else set()
    if getattr(args, "inorganic", False):
        excl |= {"C", "H"}                          # inorganic = no carbon, no hydrogen
    need_pass = bool(want or excl or getattr(args, "organic", False))
    if need_pass:
        filtered = []
        for r in rows:
            els = elements_in_formula(r.get("calcformula") or r.get("formula"))
            if not els:
                continue
            if want:
                if args.only_elements:
                    if not (els <= want):           # subset: no other elements
                        continue
                elif not (want <= els):             # must contain all requested
                    continue
            if getattr(args, "organic", False) and not ({"C", "H"} <= els):
                continue
            if excl and (els & excl):               # drop if any excluded element present
                continue
            filtered.append(r)
        rows = filtered

    # Optional de-duplication: keep at most N per (reduced formula, space group).
    if args.max_per_formula:
        seen = {}
        deduped = []
        for r in rows:
            key = (r.get("calcformula") or r.get("formula"), r.get("sgNumber"))
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= args.max_per_formula:
                deduped.append(r)
        rows = deduped

    if args.limit:
        rows = rows[: args.limit]
    return rows


def rows_from_ids(id_list):
    """Build minimal candidate rows from an explicit COD id list (no db3)."""
    return [{"file": int(i)} for i in id_list]


# ============================================================================
# Fetch: download CIFs from COD (with a local cache)
# ============================================================================

def fetch_cif(cod_id, cache_dir, timeout=30, retries=3, pause=0.34, ua=DEFAULT_UA):
    """Return CIF text for a COD id, using/refreshing a local cache."""
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{cod_id}.cif")
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            with open(cache_path, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read()

    url = COD_CIF_URL.format(cid=cod_id)
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", errors="ignore")
            if cache_path:
                with open(cache_path, "w", encoding="utf-8") as fh:
                    fh.write(text)
            time.sleep(pause)              # be polite to the COD servers
            return text
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            time.sleep(pause * (attempt + 1) * 2)
    raise RuntimeError(f"could not fetch COD {cod_id}: {last_err}")


def mirror_cif_path(mirror_root, cod_id):
    """Resolve a COD id to a CIF path inside a local rsync mirror.
    The COD archive stores 7-digit ids hierarchically, e.g.
    1000017 -> <root>/cif/1/00/00/1000017.cif. Several common layouts
    (with/without the 'cif' prefix, and flat) are tried."""
    s = str(cod_id)
    sub = os.path.join(s[0], s[1:3], s[3:5]) if len(s) == 7 else ""
    candidates = [
        os.path.join(mirror_root, "cif", sub, f"{s}.cif"),
        os.path.join(mirror_root, sub, f"{s}.cif"),
        os.path.join(mirror_root, f"{s}.cif"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def get_cif(cod_id, cfg):
    """Return CIF text for a COD id from a local mirror if configured, else HTTP."""
    if cfg.get("mirror"):
        path = mirror_cif_path(cfg["mirror"], cod_id)
        if not path:
            raise FileNotFoundError(f"CIF {cod_id} not found in mirror {cfg['mirror']}")
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    return fetch_cif(cod_id, cfg.get("cache"), ua=cfg.get("user_agent", DEFAULT_UA),
                     pause=cfg.get("pause", 0.34))


# ============================================================================
# Compute: CIF -> powder pattern via pymatgen
# ============================================================================

def compute_pattern(cif_text, wavelength, tt_range):
    """Return (peak_2theta[], peak_intensity[]) from CIF text.

    Parses tolerantly (a generous occupancy tolerance salvages the many
    disordered / partially-occupied COD entries the default parser rejects)
    and quietly (pymatgen's CIF warnings are suppressed)."""
    import warnings
    from pymatgen.io.cif import CifParser
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parser = CifParser.from_str(cif_text, occupancy_tolerance=100.0, site_tolerance=1e-4)
        except AttributeError:  # older pymatgen
            parser = CifParser.from_string(cif_text, occupancy_tolerance=100.0)
        structures = parser.parse_structures(primitive=False)
        if not structures:
            raise ValueError("no parseable structure in CIF")
        calc = XRDCalculator(wavelength=wavelength)
        pat = calc.get_pattern(structures[0], two_theta_range=tuple(tt_range))
    return np.asarray(pat.x, dtype=float), np.asarray(pat.y, dtype=float)


def broaden(peaks_x, peaks_y, xmin, xmax, step, sigma):
    """Gaussian-broaden a stick pattern onto a grid, normalised to 100.
    Matches the broadening the app uses for reference overlays."""
    x = np.arange(xmin, xmax + step, step)
    y = np.zeros_like(x)
    for c, h in zip(peaks_x, peaks_y):
        y += h * np.exp(-((x - c) / sigma) ** 2)
    if y.max() > 0:
        y = y / y.max() * 100.0
    return x, y


# ============================================================================
# Bake: write the .h5 library
# ============================================================================

def display_name(row, formula):
    for k in ("mineral", "commonname", "chemname"):
        v = row.get(k)
        if v and str(v) not in ("\\N", "None"):
            return str(v)
    return formula or f"COD {row['file']}"


def reduced_formula(row):
    f = row.get("calcformula") or row.get("formula")
    if not f:
        return ""
    return str(f).replace("-", " ").strip()


def _build_one(task):
    """Worker: fetch/read a CIF and compute its reflection list.
    Returns (cod_id, meta, peaks_x, peaks_y, error). Kept small (peaks only)
    so it pickles cheaply across processes; broadening happens in the parent."""
    cid, meta, cfg = task
    try:
        cif = get_cif(cid, cfg)
        px, py = compute_pattern(cif, cfg["wavelength"], cfg["tt"])
        if px.size == 0:
            raise ValueError("no reflections in range")
        mx = cfg.get("max_peaks")
        if mx and px.size > mx:                       # keep the strongest N reflections
            order = np.argsort(py)[::-1][:mx]
            order = order[np.argsort(px[order])]
            px, py = px[order], py[order]
        return (cid, meta, px.astype("float32"), py.astype("float32"), None)
    except Exception as e:
        return (cid, meta, None, None, str(e))


def build_library(rows, args):
    xmin, xmax = args.two_theta
    cfg = {"wavelength": args.wavelength, "tt": (xmin, xmax),
           "peaks_only": args.peaks_only, "mirror": args.mirror,
           "cache": args.cache, "pause": args.pause, "user_agent": args.user_agent,
           "max_peaks": args.max_peaks or None}
    tasks = []
    for r in rows:
        formula = reduced_formula(r)
        tasks.append((int(r["file"]),
                      {"name": display_name(r, formula), "formula": formula,
                       "sg": str(r.get("sg") or "")}, cfg))
    total = len(tasks)
    jobs = max(1, args.jobs)
    if jobs > 1 and not args.mirror:
        print(f"Note: --jobs {jobs} opens {jobs} parallel connections to COD. "
              f"For large builds prefer a local --mirror (rsync) to stay polite and fast.")

    def results_iter():
        if jobs > 1:
            from multiprocessing import Pool
            with Pool(jobs) as pool:
                for res in pool.imap_unordered(_build_one, tasks, chunksize=16):
                    yield res
        else:
            for t in tasks:
                yield _build_one(t)

    kept = failed = 0
    with h5py.File(args.out, "w") as h5:
        h5.attrs["source"] = "COD"
        h5.attrs["wavelength"] = args.wavelength
        h5.attrs["two_theta_min"] = xmin
        h5.attrs["two_theta_max"] = xmax
        h5.attrs["storage"] = "peaks" if args.peaks_only else "curve"
        spectra = h5.create_group("spectra")

        for n, (cid, meta, px, py, err) in enumerate(results_iter(), 1):
            if err or px is None:
                failed += 1
                if err:
                    print(f"  [skip] COD {cid}: {err}")
            else:
                g = spectra.create_group(str(cid))
                if not args.peaks_only:               # store the ready-to-plot curve too
                    gx, gy = broaden(px, py, xmin, xmax, args.step, args.sigma)
                    g.create_dataset("x", data=gx.astype("float32"), compression="gzip")
                    g.create_dataset("y", data=gy.astype("float32"), compression="gzip")
                g.attrs["name"] = meta["name"]
                g.attrs["cod_id"] = str(cid)
                g.attrs["rruff_id"] = ""               # loader compatibility
                g.attrs["url"] = COD_PAGE_URL.format(cid=cid)
                # peaks/intensities as DATASETS (attributes are capped at 64 KB by
                # HDF5; low-symmetry cells can exceed that with many reflections).
                _ds(g, "peaks", px)
                _ds(g, "intensities", py)
                g.attrs["formula"] = meta["formula"]
                g.attrs["sg"] = meta["sg"]
                g.attrs["wavelength"] = args.wavelength
                g.attrs["source"] = "COD"
                kept += 1
            if n % 100 == 0 or n == total:
                print(f"  ... {n}/{total} processed ({kept} kept, {failed} skipped)")

    mode = "peaks-only" if args.peaks_only else "curve+peaks"
    print(f"\nDone: {kept} pattern(s) -> {args.out}  ({failed} skipped, storage={mode})")
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
            flat_convert(args.out)
            after = os.path.getsize(args.out)
            print("  --flat: %.1f MB -> %.1f MB (%.1fx smaller)"
                  % (before / 1e6, after / 1e6, before / max(1, after)))

    if kept == 0:
        print("WARNING: no patterns were written. Check your selection / mirror / network.")


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build an ad-hoc COD powder-XRD .h5 library for the XRD toolkit.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_argument_group("selection (choose entries)")
    src.add_argument("--db3", help="Profex COD SQLite index (cod-*.db3) for offline search")
    src.add_argument("--ids", help="explicit comma-separated COD ids (bypasses db3 search)")
    src.add_argument("--ids-file", help="text file with one COD id per line")
    src.add_argument("--mineral", help="mineral-name LIKE filter (e.g. Quartz)")
    src.add_argument("--name", help="LIKE filter across mineral/chemical/common names")
    src.add_argument("--formula-contains", dest="formula_contains",
                     help="substring match on the COD formula string")
    src.add_argument("--elements", help="comma/space list, e.g. 'Ti,O'")
    src.add_argument("--only-elements", action="store_true",
                     help="with --elements: exclude phases containing any other element")
    src.add_argument("--exclude-elements", dest="exclude_elements",
                     help="drop phases containing any of these elements (comma/space list)")
    src.add_argument("--inorganic", action="store_true",
                     help="keep only inorganic phases (no carbon and no hydrogen)")
    src.add_argument("--organic", action="store_true",
                     help="keep only organic-like phases (contain both C and H)")
    src.add_argument("--minerals-only", dest="minerals_only", action="store_true",
                     help="keep only entries that have a mineral name")
    src.add_argument("--sg", type=int, help="space-group number filter")
    src.add_argument("--max-per-formula", type=int, default=0,
                     help="keep at most N entries per (formula, space group)")
    src.add_argument("--limit", type=int, default=0, help="cap number of entries")

    comp = p.add_argument_group("pattern computation")
    comp.add_argument("--wavelength", default="CuKa",
                      help="X-ray wavelength keyword or Angstrom value (default CuKa)")
    comp.add_argument("--two-theta", nargs=2, type=float, default=[5.0, 90.0],
                      metavar=("MIN", "MAX"), help="2-theta range (default 5 90)")
    comp.add_argument("--step", type=float, default=0.02, help="grid step (default 0.02)")
    comp.add_argument("--sigma", type=float, default=0.10,
                      help="Gaussian broadening sigma (default 0.10)")

    store = p.add_argument_group("storage / performance")
    p.add_argument("--flat", action="store_true",
                   help="write the consolidated layout (one concatenated peaks "
                        "array plus an offset index): ~16x smaller than one "
                        "group per entry, and read by the apps from "
                        "2026.07.28.1 onward")
    store.add_argument("--peaks-only", dest="peaks_only", action="store_true",
                       help="store only reflection peaks (no broadened curve); ~5-10x smaller, "
                            "browser-friendly. Apps rebuild the curve on load. Ideal for huge sets.")
    store.add_argument("--max-peaks", dest="max_peaks", type=int, default=0,
                       help="keep only the N strongest reflections per phase (0 = all). "
                            "Recommended for organic/large-cell phases, e.g. 300.")
    store.add_argument("--jobs", type=int, default=1,
                       help="parallel worker processes for CIF->pattern (use with --mirror)")

    net = p.add_argument_group("fetching")
    net.add_argument("--mirror",
                     help="read CIFs from a local COD rsync mirror directory instead of HTTP "
                          "(rsync://www.crystallography.net/cif/). Enables fast offline builds.")
    net.add_argument("--cache", default="cod_cif_cache",
                     help="directory for cached CIFs when fetching over HTTP (default cod_cif_cache)")
    net.add_argument("--pause", type=float, default=0.34,
                     help="seconds between HTTP downloads (default 0.34)")
    net.add_argument("--user-agent", default=DEFAULT_UA, help="HTTP User-Agent")

    p.add_argument("--out", default="cod_library.h5", help="output .h5 path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    ids = []
    if args.ids:
        ids += [int(x) for x in re.split(r"[,\s]+", args.ids) if x.strip()]
    if args.ids_file:
        with open(args.ids_file) as fh:
            ids += [int(x) for x in re.split(r"[,\s]+", fh.read()) if x.strip()]

    if ids:
        rows = rows_from_ids(ids)
        # enrich with db3 metadata if available (nicer names/formulae)
        if args.db3 and os.path.exists(args.db3):
            con = sqlite3.connect(args.db3)
            con.row_factory = sqlite3.Row
            q = ",".join("?" * len(ids))
            meta = {r["file"]: dict(r) for r in con.execute(
                f"SELECT file,sg,sgNumber,formula,calcformula,cellformula,"
                f"mineral,chemname,commonname,Z FROM data WHERE file IN ({q})",
                ids).fetchall()}
            con.close()
            for r in rows:
                r.update(meta.get(r["file"], {}))
    elif args.db3:
        rows = select_from_db3(args.db3, args)
    else:
        sys.exit("Provide --ids / --ids-file, or --db3 plus a search filter.")

    if not rows:
        sys.exit("No entries matched your selection.")

    print(f"Selected {len(rows)} COD entr(y/ies). Fetching CIFs and computing patterns...")
    build_library(rows, args)


if __name__ == "__main__":
    main()
