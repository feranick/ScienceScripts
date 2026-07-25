# COD (Crystallography Open Database) integration

This adds Crystallography Open Database reference patterns to the XRD Peak
Analysis Toolkit, alongside the existing Materials Project and RRUFF sources.

## Why a "builder" instead of using COD directly

COD distributes crystal **structures** (CIF files), not ready-made diffraction
patterns, and the Profex `cod-*.db3` is a **metadata-only search index** (unit
cell, space group, formula, names — no atomic coordinates, no patterns). So a
diffraction pattern always has to be *computed* from a CIF. Two consequences:

- The **Python** app can do this live (fetch a CIF from COD, simulate with
  pymatgen) because it has network access and no CORS restriction.
- The **browser** app cannot fetch CIFs at run time (CORS blocks
  crystallography.net), so it works purely from a pre-built `.h5` library.

`build_cod_powder_library.py` bridges the two: it selects entries, fetches the
CIFs once, computes the patterns, and bakes a compact `.h5` that **both** apps
load offline — using the same schema as the RRUFF powder library, so overlays
carry real reflection intensities and peak-matching works unchanged.

## 1. Build a library — `build_cod_powder_library.py`

Requirements: `pip install pymatgen h5py numpy`

**Getting the `cod-*.db3` index.** It ships with Profex (the open-source XRD /
Rietveld package). Download it from the Profex download page —
<https://www.profex-xrd.org/download/> — under the COD database section (the
file is named like `cod-YYMMDD.zip`; unzip it to get `cod-YYMMDD.db3`, ~1 GB).
It's a read-only search index; this toolkit never modifies it.

Selection uses that Profex `cod-*.db3` as an **offline search index over all
~530k COD entries** (any chemistry, not just minerals); only the CIFs you
actually select are downloaded.

```bash
# Everything named "quartz", one representative per (formula, space group)
python build_cod_powder_library.py --db3 cod-260101.db3 \
       --name quartz --max-per-formula 1 --out quartz.h5

# All TiO2 polymorphs (anatase/rutile/brookite), Ti and O only
python build_cod_powder_library.py --db3 cod-260101.db3 \
       --elements Ti,O --only-elements --out tio2.h5

# A specific set of COD ids (no db3 needed for selection)
python build_cod_powder_library.py --ids 1011097,9008213 --out picks.h5
```

### Supplying CIFs: online vs. a local rsync mirror

Small builds can fetch CIFs over HTTP on demand (the default; cached in
`--cache`). For large builds (tens of thousands of entries) that is slow and
impolite to the COD servers — mirror the archive once and build offline:

```bash
# one-time: mirror all COD CIFs (~a few GB) — then rebuild any library offline
rsync -av --delete rsync://www.crystallography.net/cif/ ./cod-cif-mirror/cif/
```

Then pass `--mirror ./cod-cif-mirror` and parallelise with `--jobs N`.

### The three general-purpose libraries

With the mirror in place, these are the recommended defaults (adjust `--jobs`
to your CPU):

```bash
# 1) Inorganic — no C/H (~62k). Full curves; browser-friendly.
python build_cod_powder_library.py --db3 cod-260101.db3 --inorganic \
       --mirror ./cod-cif-mirror --jobs 8 --out cod_inorganic.h5

# 2) Minerals — entries with a mineral name (~16k). Full curves.
python build_cod_powder_library.py --db3 cod-260101.db3 --minerals-only \
       --mirror ./cod-cif-mirror --jobs 8 --out cod_minerals.h5

# 3) Organic — C+H (~448k). PEAKS-ONLY + capped reflections (large set).
python build_cod_powder_library.py --db3 cod-260101.db3 --organic \
       --mirror ./cod-cif-mirror --jobs 8 --peaks-only --max-peaks 300 \
       --out cod_organic.h5
```

(No mirror? Drop `--mirror` and use `--jobs 1` — it fetches over HTTP, fine for
the minerals set, slow for the larger two.)

Key options — selection: `--inorganic` / `--organic` / `--minerals-only`,
`--elements` (+`--only-elements`), `--exclude-elements`, `--name` / `--mineral`
/ `--formula-contains` / `--sg`, `--ids` / `--ids-file`, `--max-per-formula`,
`--limit`. Pattern: `--wavelength` (default `CuKa`), `--two-theta MIN MAX`,
`--step`, `--sigma`. Storage / speed: `--peaks-only` (no curve; ~5–10× smaller,
rebuilt on load), `--max-peaks N` (keep N strongest reflections — recommended
for organic/large-cell phases), `--jobs N` (parallel), `--mirror DIR`,
`--cache DIR`.

Output `.h5` schema (per phase): attributes `name`, `cod_id`, `url`, `peaks`
(reflection 2θ), `intensities`, `formula`, `sg`, `wavelength`; plus datasets
`x`,`y` (broadened profile 0–100) **unless** `--peaks-only`, in which case the
apps rebuild the curve from `peaks`+`intensities` on load.

## 2. Python app (`xrd_plotter.py`)

New **🌐 COD** panel (above RRUFF). One button — **🗂️ Open COD Source…** — asks
how you want to work:

- **🌐 db3 index + Online** — pick a Profex `cod-*.db3` to search all of COD;
  *Overlay Selected* fetches that entry's CIF and simulates it on the fly, and
  *Match by Selected Peaks* fetches + simulates the current search hits and ranks
  them (needs `pymatgen` + network). No `.h5` required in this mode.
- **📚 Local .h5 libraries** — add one or more baked `.h5` files and work fully
  offline: search, overlay (real intensities, existing dot-dash reference
  style), and instant peak-matching against the union of active libraries.

**Multi-library manager.** You can load several libraries (e.g. inorganic +
minerals + organic) and tick which ones are **active**; search/overlay/match use
only the active set. Each row has a checkbox and an ❌ to remove it. The set is
remembered across launches in `cod_libraries.json` (next to the script) and
re-loaded automatically — so you set it up once and just toggle thereafter.

Peaks-only libraries (e.g. the organic one) are supported: the overlay curve is
rebuilt from the stored reflections on the fly. Overlay labels link to the COD
page, and the whole sidebar is scrollable. `mp_api` is no longer required for COD
or for pattern simulation — only for the Materials Project search.

## 3. Browser app (`xrd_plotter.html`)

New **🌐 COD** panel (above RRUFF), offline only. **📚 Add COD .h5 Library**
reads baked files in-browser via h5wasm (the same mechanism the RRUFF panel
uses). Like the Python app it's a **multi-library manager**: add several
libraries, tick which are active (checkbox), remove with ❌; search / overlay /
match use the union of active libraries, and peaks-only libraries are rebuilt on
the fly. The set (and the files) are cached in IndexedDB and restored on the next
visit. A `cod_powder_library.h5` placed next to the page (or `?codlib=<url>`)
auto-loads on first run.

## Notes / limitations

- Calculated patterns assume an ideal, randomly-oriented powder with default
  thermal parameters, so **peak positions are reliable but relative intensities
  can differ** from a measured pattern (preferred orientation, etc.). This is
  the same caveat as the Materials Project overlays and is fine for phase ID.
- COD is community-contributed; a few CIFs are partial and will be skipped by
  the builder (reported, never fatal).
- Verified: quartz (COD 1011097) computes its strongest reflection at 26.66°
  2θ (Cu-Kα, the (101) line), and marked quartz peaks rank the quartz entry at
  100% vs. 25% for anatase in the match tool.
