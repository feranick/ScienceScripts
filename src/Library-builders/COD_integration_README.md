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
# one-time: mirror all COD CIFs (several GB, ~500k files) — then build offline.
# Create the destination first (rsync won't make nested parents on its own):
mkdir -p ./cod-cif-mirror/cif
rsync -av --delete rsync://www.crystallography.net/cif/ ./cod-cif-mirror/cif/
# (rsync >= 3.2.3 alternative: add --mkpath instead of the mkdir line.)
```

rsync is resumable — if it times out mid-run (the tree is ~500k files), just
re-run; it skips what's already there. For reliability, sync one leading-digit
bucket at a time inside a retry loop:

```bash
for d in 1 2 3 4 5 6 7 8 9; do
  until rsync -a --partial --timeout=600 --contimeout=60 --delete \
        rsync://www.crystallography.net/cif/$d/ ./cod-cif-mirror/cif/$d/; do
    echo "  $d stalled — retrying in 30s…"; sleep 30
  done
done
```

Then pass `--mirror ./cod-cif-mirror` and parallelise with `--jobs N`.

### Sizing: build for the desktop app, or for the browser?

**This is the decision that matters most, and getting it wrong produces a file
the browser cannot open.** Two things drive size: whether you store curves, and
which *layout* you store the reflections in.

Figures below are measured on real builds, not estimated. An earlier version of
this table quoted 0.5 KB per phase for a `--max-peaks 60` build and predicted
30 MB for 62k phases. That was the **data payload** — 60 reflections × 2 arrays ×
4 bytes = 480 bytes. What HDF5 actually writes is ~14× more, because a group per
entry costs roughly 2 KB in object headers and B-tree nodes regardless of what
you put in it. Measured: `cod_inorganic_web.h5` is 366 MB for 51,225 phases, not
25 MB. If you built an organic set expecting ~100 MB and got 1.4 GB, that table
is why.

| Build | Per phase | 51,225 phases | 195,135 phases |
| --- | --- | --- | --- |
| full curves (default, 4251 pts) | ~33 KB | ~1.7 GB | ~6.4 GB |
| `--peaks-only --max-peaks 60` | 7.1–7.3 KB | **366 MB** | **1423 MB** |
| the same, `repack_library.py --no-compress` | ~3.1 KB | ~160 MB | **597 MB** |
| `--peaks-only --max-peaks 60 --flat` | 0.4–0.5 KB | **~23 MB** | **~90 MB** |

Sizes are decimal (1 MB = 10⁶ bytes). **Bold figures are measured files**; the
rest follow from the per-phase cost. Per-phase varies a little with how long the
formula and name strings are, hence the ranges — do not expect the rows to
multiply out to the last MB.

Two results in that table are counter-intuitive and both are measured:

- **Turning gzip OFF makes the file smaller** — 7.3 → 3.1 KB per phase. A
  60-value float32 dataset does not compress, and the chunk index plus filter
  pipeline HDF5 stores per dataset costs more than the compression saves. The
  builders now decide by array size, so curves (thousands of points, with a
  regular `x` grid) are still compressed, where gzip genuinely saves ~39%.
- **`--flat` is 16× smaller again** — 3.1 → 0.46 KB per phase. It stores every
  phase's reflections in one concatenated array with an offset index instead of
  one HDF5 group each, which removes the per-group overhead entirely. Read by
  the apps from `2026.07.28.1` onward; older app versions cannot open it.

- **`xrd_plotter.py` (desktop)** reads `x`/`y` lazily through h5py, one phase at
  a time. Size is irrelevant; a multi-GB curve library is fine.
- **`xrd_plotter.html` (browser)** must hold the library in the tab. Full-curve
  libraries above a few hundred MB **cannot work** — see the next section.

So for anything large, build twice: full curves for the desktop, `--peaks-only
--max-peaks 60 --flat` for the browser. Peak positions and relative intensities
are identical either way; only the stored profile differs, and the browser
rebuilds it on overlay.

### Shrinking a library you already built

`repack_library.py` reads an existing `.h5` and writes a new one. It never
re-downloads or recomputes anything, so this is minutes rather than hours:

```bash
# 1423 MB -> 88 MB, in place
python3 repack_library.py cod_organic.h5 --flat -o cod_organic.h5

# same schema, no app changes needed at all: 1423 -> 597 MB
python3 repack_library.py cod_organic.h5 --no-compress -o cod_organic_web.h5

# audit a library for damage; exits 1 if any entry is unreadable
python3 repack_library.py cod_organic.h5 --check
```

Both conversions verify a sample of entries against the source before replacing
anything, and write to a temporary file first, so a failure leaves the original
intact. `--check` exists because corrupt gzip chunks are silent: a library can
sit on a server looking fine until something tries to read the damaged entry.

### Why the browser cannot take a multi-GB library

Loading a library in the tab costs three copies, and the third dominates:

| | 2.1 GB library |
| --- | --- |
| the downloaded `ArrayBuffer` (JS heap) | 2.1 GB |
| the Emscripten MEMFS copy inside h5wasm | 2.1 GB |
| per-phase JS arrays for `x` and `y` | **~4.2 GB** |

h5wasm is compiled for wasm32, so its entire address space caps at 4 GB
regardless of how much RAM the machine has. A renderer asked to do this is
killed part-way through loading — Chrome shows **"Aw, Snap!", error code 5**,
with nothing in the console because the process is gone.

The apps now defend against this rather than crashing:

- above **1.2 GB** the library is refused with a message telling you to rebuild
  peaks-only;
- above **150 MB** stored curves are dropped at load and profiles are rebuilt
  from `peaks`+`intensities` on overlay (reported in the status line);
- above **300 MB** the file is not cached to IndexedDB;
- spectral data is held as `Float32Array` rather than JS `Array` — a quarter of
  the memory.

Those guards make an oversized file fail cleanly. They are not a substitute for
building peaks-only.

A `--flat` library sidesteps most of this arithmetic. The 195k-phase organic set
is 88 MB on disk, and its entries are zero-copy views into two typed arrays
rather than two JS arrays per phase, so the third row above — historically the
one that killed the tab — becomes a single 94 MB allocation.

### The recommended library set

With the mirror in place, these are sensible defaults (adjust `--jobs` to your
CPU). Note the inorganic and minerals sets are built twice:

```bash
# 1) Inorganic — no C/H (~62k)
#    desktop: full curves (~1.7 GB)
python build_cod_powder_library.py --db3 cod-260101.db3 --inorganic \
       --mirror ./cod-cif-mirror --jobs 8 --out cod_inorganic.h5
#    browser: peaks-only, flat (~23 MB)
python build_cod_powder_library.py --db3 cod-260101.db3 --inorganic \
       --mirror ./cod-cif-mirror --jobs 8 --peaks-only --max-peaks 60 --flat \
       --out cod_inorganic_web.h5

# 2) Minerals — entries with a mineral name (~16k)
#    desktop: full curves (~540 MB); browser: peaks-only + --flat (~7 MB)
python build_cod_powder_library.py --db3 cod-260101.db3 --minerals-only \
       --mirror ./cod-cif-mirror --jobs 8 --out cod_minerals.h5
python build_cod_powder_library.py --db3 cod-260101.db3 --minerals-only \
       --mirror ./cod-cif-mirror --jobs 8 --peaks-only --max-peaks 60 \
       --out cod_minerals_web.h5

# 3) Organic — C+H (~448k). Peaks-only for both; full curves would be ~15 GB.
python build_cod_powder_library.py --db3 cod-260101.db3 --organic \
       --mirror ./cod-cif-mirror --jobs 8 --peaks-only --max-peaks 60 \
       --out cod_organic.h5
```

### Generating `libraries.json` — `make_libraries_manifest.py`

The apps read a `libraries.json` to learn what is on the server (see
Troubleshooting for why they cannot just list the folder). Rather than couple
building to deployment, a separate scanner reads the finished `.h5` files:

```bash
python make_libraries_manifest.py /var/www/html/tools/xrd-plotter
```

```
Scanning /var/www/html/tools/xrd-plotter
  ok    cod_inorganic_web.h5      COD inorganic — 62,431 patterns · peaks-only · CuKa · 31 MB
  ok    cod_minerals_web.h5       COD minerals — 16,102 patterns · peaks-only · CuKa · 8 MB
  ok    rruff_powder_library.h5   RRUFF powder — 3,782 patterns · 79 MB  [rruff panel]
  skip  index.html                not a reference library

Wrote /var/www/html/tools/xrd-plotter/libraries.json  (3 libraries)
```

Labels and notes come from each file's own attributes (`source`, `technique`,
`license`, `storage`, spectrum count) plus its filename, so nothing is guessed
and **no rebuild is needed** — run it on libraries you built months ago. Only
the attribute block and the group count are read, never the spectra, so a 2 GB
library is inspected in milliseconds.

A file whose name says `rruff` is tagged `"panel": "rruff"`, which routes it to
the apps' single-library RRUFF panel instead of offering it twice.

Anything that is not one of our libraries is skipped: a LabSpec data file, an
HTML error page saved with an `.h5` extension, `index.html`. Options:
`--dry-run` to preview, `--keep-labels` so hand-edited labels survive a
re-scan, `--include`/`--exclude` to filter, `--out` for a different path.

`--max-peaks 60` keeps the 60 strongest reflections`--max-peaks 60` keeps the 60 strongest reflections, which is far more than
peak matching needs and still resolves crowded low-symmetry patterns. Drop to
30 if you want the organic set smaller; raise it if you overlay peaks-only
references and want a denser rebuilt profile.

Check a build before committing to a long run:

```bash
python build_cod_powder_library.py --db3 cod-260101.db3 --inorganic \
       --limit 200 --peaks-only --max-peaks 60 --out probe.h5
# multiply the resulting size by (total entries / 200)
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

## 2. Both apps: one **🔬 Reference Databases** panel

RRUFF, COD and (in the desktop app) the Materials Project used to have a panel
each. They are now a single panel with one search box:

- **Sources** are listed as checkboxes with live counts — the RRUFF folder or
  `.h5`, plus every loaded COD library. Tick which ones are live; ❌ removes a
  library.
- **One search** queries every enabled source and merges the results into one
  list, each row tagged with its origin (`[RRUFF]`, `[COD]`).
- **One Match** ranks across every enabled source in a single pass, in a window
  with a Source column. Previously this meant matching once per panel and
  comparing the rankings by eye.
- **Manage sources** (collapsed by default) holds the setup controls: adding
  `.h5` libraries, the RRUFF folder, the COD db3 index, and the Materials
  Project API key.

**Multi-library manager.** Load several libraries (inorganic + minerals +
organic) and toggle which are active. The set is remembered across launches —
`cod_libraries.json` next to the script in the desktop app, IndexedDB in the
browser — so you set it up once and just toggle thereafter.

Peaks-only libraries are supported everywhere: the overlay curve is rebuilt from
the stored reflections on the fly. Overlay labels link to the COD page.
`mp_api` is required only for the Materials Project search, not for COD or for
pattern simulation.

### Desktop-only: the db3 index

**🗂️ Open COD Source…** under Manage sources also accepts a Profex `cod-*.db3`,
which searches all of COD without any `.h5`: *Overlay* fetches that entry's CIF
and simulates it on the fly, and matching fetches + simulates the current search
hits (needs `pymatgen` + network).

### Browser-only: libraries are offered, not auto-loaded

On load the page works out which `.h5` files the server has and lists them with
their sizes and a **Load** button — nothing is downloaded until you ask, since a
library can be tens or hundreds of MB. Libraries you loaded before are restored
from IndexedDB automatically.

Discovery tries, in order: `?codlib=a.h5,b.h5` → a `libraries.json` manifest you
provide → the server's directory index → a list of conventional filenames
(`cod_powder_library.h5`, `cod_inorganic.h5`, `cod_minerals.h5`,
`cod_organic.h5`, …). A manifest is the reliable choice for a fixed deployment:

```json
["cod_inorganic_web.h5", "cod_minerals_web.h5", "cod_organic.h5"]
```

Under `file://` none of this works — no directory listing, and `fetch` is
blocked — so add libraries by hand there.

## Notes / limitations

- Calculated patterns assume an ideal, randomly-oriented powder with default
  thermal parameters, so **peak positions are reliable but relative intensities
  can differ** from a measured pattern (preferred orientation, etc.). This is
  the same caveat as the Materials Project overlays and is fine for phase ID.
- COD is community-contributed; a few CIFs are partial and will be skipped by
  the builder (reported, never fatal).
- A peaks-only overlay is a Gaussian rebuild from the stored reflections, not
  the profile the builder computed. Positions and relative intensities match;
  the peak shape is nominal. For publication figures, overlay from a full-curve
  library in the desktop app.
- Verified: quartz (COD 1011097) computes its strongest reflection at 26.66°
  2θ (Cu-Kα, the (101) line), and marked quartz peaks rank the quartz entry at
  100% vs. 25% for anatase in the match tool.

## Troubleshooting

**Chrome shows "Aw, Snap!" with error code 5 while a library initializes.**
The renderer ran out of memory — the tab was killed, which is why the console is
empty. The library is a full-curve build too large for a browser. Rebuild it
with `--peaks-only --max-peaks 60` and keep the full one for the desktop app.
Current builds refuse politely above 1.2 GB instead of crashing, but a file
between roughly 300 MB and 1.2 GB will still load slowly and drop its stored
curves; peaks-only is the right answer at that size.

**Why not just list whatever `.h5` is in the folder?** A browser cannot: there
is no filesystem API over HTTP, only "fetch this exact URL and see if it 404s".
The server has to volunteer a listing, and Apache does not by default (and an
`index.html` in the directory suppresses it even with `Options +Indexes`). Hence
the manifest, which is also the only way to give a library a readable name.

**The browser only offers some of the `.h5` files, or none.** Discovery looks
**relative to the page**, so a library must live in the same directory the HTML
is served from — `https://host/tools/xrd-plotter/cod_inorganic.h5` for a page at
`https://host/tools/xrd-plotter/`. A library uploaded elsewhere on the server is
invisible. Open the network tab: `404 (Not Found)` on the candidate names means
exactly this.

Three ways to fix it, in increasing order of robustness:

1. Put the `.h5` files next to the page.
2. Add a `libraries.json` next to the page — the recommended option. Entries
   may be relative paths or absolute URLs, so the files can live anywhere, and
   an object form lets you give each library a readable name and a one-line
   note that the panel displays instead of the filename:
   ```json
   [
     { "file": "cod_inorganic_web.h5",
       "label": "COD inorganic",
       "note": "~62k phases, no C/H · peaks-only" },
     { "file": "../shared/cod_minerals_web.h5",
       "label": "COD minerals",
       "note": "~16k named minerals · peaks-only" }
   ]
   ```
   A bare string still works (`["cod_inorganic_web.h5"]`) — the label and note
   are optional. `title`/`description` are accepted as aliases for
   `label`/`note`, and `url`/`path` for `file`.
3. Pass them per-visit: `?codlib=../libraries/cod_inorganic_web.h5`

With no manifest and no directory index, discovery falls back to a list of
conventional filenames (`cod_powder_library.h5`, `cod_inorganic.h5`,
`cod_minerals.h5`, `cod_organic.h5`, …), which is why matching those names
happens to work. Run `dbDiagnoseDiscovery()` in the console to see what each
step actually found.

**A library loads but Match finds nothing.** Check the tolerance — the default
is 0.2° 2θ, which is tight for a poorly-calibrated diffractometer. Also confirm
the library is actually ticked in the source list.
