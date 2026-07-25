# ROD (Raman Open Database) integration

This adds Raman Open Database reference spectra to the Raman Peak Analysis
Toolkit, alongside the existing RRUFF source.

ROD is the Raman sibling of the COD, run by the same group in Vilnius, with all
data placed in the public domain under **CC0**. It is small — ~1100 entries —
and cross-linked to COD crystal structures, so a matched spectrum can be opened
straight through to its structure.

## Why a builder here, when ROD serves spectra directly

Unlike COD, ROD ships **measured spectra**, so nothing has to be simulated. The
builder exists for a different reason: **CORS**.

```
$ curl -sI -H "Origin: https://example.com" \
    "https://solsa.crystallography.net/rod/result?text=quartz&format=json" \
    | grep -i access-control-allow-origin
Access-Control-Allow-Origin: https://thermott.ibt.lt
```

CORS *is* configured on ROD's server — but pinned to one third-party origin, so
a browser fetch from any other page (including `file://`, which sends
`Origin: null`) is blocked. Consequences:

- The **Python** app can query ROD live; there is no CORS restriction outside a
  browser.
- The **browser** app cannot, and works purely from a pre-built `.h5` library.

`build_rod_library.py` bridges the two. Because ROD is small, the recommended
build is simply **the whole database** — roughly 24 MB on a 1 cm⁻¹ grid, which
both apps hold comfortably in memory. There is no need to subset the way COD
requires.

The output uses the **same `.h5` schema as the RRUFF and COD libraries**, so it
opens through the existing library buttons even in an un-patched copy of either
app, and peak-matching works unchanged.

> **If you would rather have live browser access:** ROD's config already has the
> CORS block, so widening it is a one-line change for their admins. The correct
> ask is `Access-Control-Allow-Origin: *` **together with** removing
> `Access-Control-Allow-Credentials: true` — the two are illegal in combination
> and browsers reject the pair. Contact address is on the ROD homepage.

## 1. Build a library — `build_rod_library.py`

Requirements: `pip install h5py numpy scipy`
(scipy is optional — a fallback peak finder is used without it, but band
positions may differ slightly from the app's own detector.)

No index file to download, no API key: entry discovery goes through ROD's
documented REST search endpoint.

```bash
# The whole database — the recommended default
python build_rod_library.py --all --step 1 --out rod_raman_library.h5

# Free-text metadata search (names, mineral, bibliography)
python build_rod_library.py --text calcite --out calcite.h5

# Everything containing Ti and O
python build_rod_library.py --elements Ti,O --out tio2.h5

# A specific set of ROD ids
python build_rod_library.py --ids 1000076,1000506 --out picks.h5
```

Every download is cached in `--cache` (default `rod_cache/`), so re-running is
free and `--offline` re-bakes with no network at all — handy when tuning
`--step` or `--prominence`:

```bash
python build_rod_library.py --all --offline --step 2 --out rod_small.h5
```

### Sizing

Two requests per entry (`.jdx` spectrum + `.rod` CIF metadata) at the default
0.34 s pause means a full build takes roughly 13 minutes. Add `--no-metadata`
to halve that, at the cost of falling back to the JCAMP `TITLE`/`MOLFORM` for
names.

Measured, extrapolated to 1133 entries:

| Grid | Size | Notes |
| --- | --- | --- |
| native (`--step 0`) | ~38 MB | as deposited, typically ~0.5 cm⁻¹ |
| `--step 1` | ~24 MB | **recommended** — ROD deposits are mostly 4 cm⁻¹ resolution, so this is lossless in practice |
| `--step 2` | ~21 MB | diminishing returns; HDF5 per-group overhead dominates |

Run `--limit 20` first for a true figure on your selection — the numbers above
extrapolate from smooth spectra and real noisy deposits compress worse.

Key options — selection: `--all`, `--text`, `--formula`, `--elements` /
`--exclude-elements`, `--journal` / `--year` / `--doi`, `--ids` / `--ids-file`,
`--limit`. Processing: `--xmin` / `--xmax` (crop), `--step` (resample),
`--prominence` (band detection, default 0.04 — matches the apps),
`--max-peaks N` (default 40), `--require-peaks`. Fetching: `--cache DIR`,
`--offline`, `--no-metadata`, `--pause`, `--retries`, `--timeout`.

Output `.h5` schema (per spectrum, group `/spectra/<rod_id>`): datasets `x`,`y`
(Raman shift cm⁻¹, intensity normalized 0–100), `peaks`, `intensities`;
attributes `name`, `rod_id`, `rruff_id` (= `rod_id`, for back-compat with the
existing loaders), `cod_id`, `url`, `cod_url`, `peaks`, `mineral`, `formula`,
`laser_nm`, `resolution_cm1`, `instrument`, `publication`, `year`, `source`,
`license`. File-level attributes record the build date and `license=CC0-1.0`.

## 2. Python app (`raman_plotter.py`)

New **🔬 ROD** panel (below RRUFF), with a **Source** radio:

- **Offline .h5** — add one or more baked libraries and work fully offline:
  search, overlay, and instant peak-matching against the union of active
  libraries.
- **Online** — search queries `solsa.crystallography.net` directly and *Overlay
  Selected* downloads that spectrum (cached in `~/.raman_plotter_rod/`). No
  library file needed. Peak matching also works here, against a narrowed subset
  — see below.

**Multi-library manager**, same as the XRD tool's COD panel. Load several
libraries, tick which are **active**; search/overlay/match use only the active
set. Each row has a checkbox and an ❌ to remove it. The set is remembered
across launches in `~/.raman_plotter_rod_libraries.json` and re-loaded
automatically.

The match window ranks candidates by band alignment and offers **🔗 Open ROD
Page(s)**, **⬡ Open Linked COD Structure(s)** (for entries with a COD
cross-reference), and **➕ Overlay Selected Match(es)**. Overlay labels in the
Layers panel link to the ROD page.

### Online peak matching — narrow, then scan

Ranking *all* of ROD live would mean downloading the whole database. Instead,
online matching is **bounded**: you narrow the candidate set with a query, and
only that subset is fetched, band-detected, and ranked.

The **as** dropdown next to the search box chooses how the query is sent to
ROD:

| `as` | Query example | ROD search key |
| --- | --- | --- |
| `auto` | `calcite` / `1000076` | text, or id if numeric |
| `text` | `calcite` | free text over names + bibliography |
| `elements` | `Ti,O` | `el1..el8` — all must be present |
| `formula` | `C8 H10 N4 O2` | empirical formula, Hill notation |
| `id` | `1000076,1000506` or `10000%` | entry ids, `%` wildcard allowed |

Workflow: mark your peaks → type a narrowing query → **🎯 Match by Selected
Peaks (ROD)**. If the query matches more entries than the **max online** cap
(default 100, next to the dropdown), you are asked whether to scan the first N
or go back and narrow further. During the scan the button becomes **⏹ Stop
scanning** and progress is reported in the status line.

Downloads land in the shared cache (`~/.raman_plotter_rod/`), so re-running the
same scan — with a different tolerance, say — costs nothing. Running a match
with an empty query is refused, with a prompt explaining the options rather
than silently doing nothing.

Practically: `elements Ti,O` or a mineral-family name gets you a few dozen
candidates and a scan that finishes in well under a minute. If you find
yourself repeatedly scanning the same corner of ROD, bake that subset into an
`.h5` instead — matching is then instant and works offline.

## 3. Browser app (`raman_plotter.html`)

New **🔬 ROD** panel (below RRUFF), offline only, for the CORS reason above.
**📚 Add ROD .h5 Library** reads baked files in-browser via h5wasm — the same
mechanism the RRUFF panel uses. Like the Python app it is a multi-library
manager: add several, tick which are active, remove with ❌; search / overlay /
match use the union of active libraries.

The set *and the files* are cached in IndexedDB and restored on the next visit.
A `rod_raman_library.h5` placed next to the page (or `?rodlib=<url>`) auto-loads
on first run.

## Notes / limitations

- ROD is community-deposited and modest in size (~1100 entries), weighted
  toward minerals and the SOLSA project's materials. It complements RRUFF
  rather than replacing it.
- Spectra are **measured**, so unlike the COD/Materials Project XRD overlays
  there is no simulation caveat — but excitation wavelength varies by deposit
  (recorded in the `laser_nm` attribute and shown in search results), and
  relative band intensities are wavelength- and orientation-dependent.
- The builder's band detector mirrors `detect_reference_peaks()` in the app
  exactly (`prominence=0.04` on min-max normalized intensity, `distance=3`,
  40 bands max), so match scores are consistent between a baked library and raw
  spectra scanned from a folder.
- Metadata is scraped from the CIF by substring-matching tag names rather than
  hardcoded ones, because ROD's `cif_raman`/`cif_rod` dictionaries have changed
  over the project's life. Missing fields degrade to empty strings, never
  errors.
- Verified: JCAMP-DX parsing agrees between builder and app on a real ROD
  deposit (2422 points); synthetic bands at 206/464/1085 cm⁻¹ are recovered to
  within 0.2 cm⁻¹; a library built from them scores 100% against marked quartz
  peaks. Test harnesses are `_test_rod_builder.py` and `_test_rod_plotter.py`.

## Citation

ROD data are CC0 — no permission needed — but the original depositors should be
acknowledged. Each entry's `publication` attribute carries the citation where
ROD provides one.

El Mendili, Y. *et al.* (2019). "Raman Open Database: first interconnected
Raman–X-ray diffraction open-access resource for material identification."
*J. Appl. Cryst.* **52**, 618–625.
