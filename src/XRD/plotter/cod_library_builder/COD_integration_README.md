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

# Space group 152 silicates, capped at 50 entries
python build_cod_powder_library.py --db3 cod-260101.db3 \
       --elements Si,O --sg 152 --limit 50 --out sg152.h5

# A specific set of COD ids (no db3 needed for selection)
python build_cod_powder_library.py --ids 1011097,9008213 --out picks.h5
```

Key options: `--name` / `--mineral` / `--formula-contains` / `--elements`
(+ `--only-elements`) / `--sg` / `--ids` / `--ids-file` for selection;
`--max-per-formula` and `--limit` to trim duplicates; `--wavelength` (default
`CuKa`), `--two-theta MIN MAX`, `--step`, `--sigma` for the pattern; `--cache`
keeps downloaded CIFs so re-builds are instant and offline.

Output `.h5` schema (per phase): datasets `x`,`y` (broadened profile, 0–100)
and attributes `name`, `cod_id`, `url`, `peaks` (reflection 2θ),
`intensities`, `formula`, `sg`, `wavelength`.

## 2. Python app (`xrd_plotter.py`)

New **🌐 COD** panel (above RRUFF). One button — **🗂️ Open COD Source…** — asks
how you want to work:

- **🌐 db3 index + Online** — pick a Profex `cod-*.db3` to search all of COD;
  *Overlay Selected* fetches that entry's CIF and simulates it on the fly, and
  *Match by Selected Peaks* fetches + simulates the current search hits and ranks
  them (needs `pymatgen` + network). No `.h5` required in this mode.
- **📚 Local .h5 library** — pick a baked `.h5` and work fully offline: search,
  overlay (real intensities, existing dot-dash reference style), and instant
  peak-matching against the whole library.

Overlay labels link to the entry's COD page. The whole sidebar is now scrollable.
`mp_api` is no longer required for COD or for pattern simulation — only for the
Materials Project search.

## 3. Browser app (`xrd_plotter.html`)

New **🌐 COD** panel (above RRUFF), offline only. **📚 Open COD .h5 Library**
reads the baked file in-browser via h5wasm (the same mechanism the RRUFF panel
uses), then Search / Overlay / Match by Selected Peaks work identically. The
library is cached in IndexedDB, and a `cod_powder_library.h5` placed next to the
page (or `?codlib=<url>`) auto-loads on startup.

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
