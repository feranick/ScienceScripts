# Science Scripts
Collection of Scripts for Handling Scientific Data

## Spectroscopy plotters

`xrd_plotter`, `raman_plotter` and `ftir_plotter` are three sibling tools with
the same design. Each exists as a local Python script (Tkinter GUI) and as a
stand-alone HTML page that runs from a file or a web server with no build step.

They share a **🔬 Reference Databases** panel: one search box over every enabled
source, results merged into one list tagged by origin, and a single *Match by
Selected Peaks* that ranks across all enabled sources in one pass. Sources are
ticked on and off individually; setup controls (adding libraries, online modes,
link-outs) live in a collapsible *Manage sources*.

Reference libraries are `.h5` files built by the scripts in
[`src/Library-builders/`](src/Library-builders) — see
[Reference database builders](#reference-database-builders) below. All three apps
read one schema, so a library is usable by whichever tool matches its technique.

### Identify peaks on hover

Tick **🔎 Identify on hover** and move the cursor along a spectrum. Each peak
reports how many references are still consistent with *every* peak you have
marked, and what happens to that number when this one is added — say
`add this reflection: 2,292 become 711 (−69%)`. Names appear once the set is
small enough to mean something; above 200 candidates you get the count and a
prompt to mark more peaks.

That threshold is the point of the feature. A single peak is not an
identification: in a 67k-phase XRD set roughly a third of all phases have *some*
reflection within ±0.2° of any given position, and picking five names out of
twenty thousand would look authoritative while being arbitrary. Watching the
count collapse as peaks accumulate teaches what peak matching actually depends
on. Each card also reports how common that position is across the library, so a
distinctive band is visibly worth more than a crowded one.

Costs one index build when first enabled (~4M band positions, well under a
second) and about 0.2 ms per hover.

    src/XRD/     xrd_plotter.py    xrd_plotter.html    converter/
    src/Raman/   raman_plotter.py  raman_plotter.html
    src/FTIR/    ftir_plotter.py   ftir_plotter.html
    src/Library-builders/

### Installing

The Python side installs as a wheel. Nothing needs rearranging — the build
remaps the technique folders onto one import package, `sciencescripts`.

```bash
python -m pip install build
python -m build --wheel
python -m pip install dist/sciencescripts-*.whl
```

That puts ten commands on PATH: `xrd-plotter`, `raman-plotter`, `ftir-plotter`,
the six `build-*-library` builders, and `make-libraries-manifest`. Every script
still runs directly from a checkout exactly as before.

Optional extras cover the dependencies that are imported lazily, so the base
install stays small: `[cod]` for computing powder patterns from COD structures,
`[mp]` for the Materials Project lookup, `[openspecy]` for reading Open Specy's
`.rds` release, or `[all]`.

The three plotters need **tkinter**, which ships with Python but is a separate OS
package on most Linux distributions and cannot be installed with pip — `apt
install python3-tk`, `dnf install python3-tkinter`, or `brew install python-tk`.
The builders and the manifest tool do not import it and run headless.

## XRD
1. **xrd_converter (GUI and CLI)** (`src/XRD/converter/`): Convert xrd data in complex csv or xrdml formats into simple csv format for plotting, or archiving. Both GUI and command line versions available.
2. **xrd_plotter**: Plot one or more XRD data files (csv or xrdml format). Available both as a local python script (with GUI) or as a stand-alone web script (running either locally or in remote server).

    ### Functionality of the plotter:
    - Curve fitting
    - Curve normalization to peak
    - Cropping to view
    - Smoothing curves
    - Background subtraction (via regularization or through reference diffractogram)
    - Correct angular offsets with reference data
    - Reference search and peak matching against:
        - [Crystallography Open Database](https://www.crystallography.net/cod/) — offline `.h5` libraries, or live search of the whole of COD via a Profex `cod-*.db3` index (python version only)
        - [RRUFF](https://www.rruff.net/zipped_data_files/powder/) measured mineral patterns
        - Materials Project, by formula or peak selection (python version only, needs an API key)

    See [COD_integration_README.md](src/Library-builders/COD_integration_README.md) for building COD
    libraries, sizing them, and serving them.

## Raman
1. **raman_plotter**: Plot one or more Raman data files (H5, xml, or txt formats from Horiba LabSpec). Available both as a local python script (with GUI) or as a stand-alone web script (running either locally or in remote server).

    ### Functionality of the plotter:
    - Curve fitting
    - Curve normalization to peak
    - Cropping to view
    - Smoothing curves
    - Background subtraction (via regularization or through reference spectra)
    - Reference search and peak matching against:
        - [RRUFF](https://www.rruff.net/zipped_data_files/raman/) measured mineral spectra
        - [Raman Open Database](https://solsa.crystallography.net/rod/) (ROD, CC0) — offline `.h5`, or live query with a bounded scan (python version only)
        - [Open Specy](https://openspecy.org) (CC-BY) — polymers, microplastics, pigments and other organics
        - [SDBS](https://sdbs.db.aist.go.jp/) — link-out only; SDBS prohibits automated retrieval, so you download a spectrum yourself and import the `.jdx`

    See [ROD_integration_README.md](src/Library-builders/ROD_integration_README.md) for the ROD and
    Open Specy integration, and why ChemSpider and SDBS cannot be pulled
    automatically.

## FTIR
1. **ftir_plotter**: Plot one or more FTIR data files (jdx or csv formats from Thermo Nicolet). Available both as a local python script (with GUI) or as a stand-alone web script (running either locally or in remote server).

    ### Functionality of the plotter:
    - Curve fitting
    - Curve normalization to peak
    - Cropping to view
    - Smoothing curves
    - Background subtraction (via regularization or through reference spectra)
    - Reference search and peak matching against:
        - [RRUFF](https://www.rruff.net/zipped_data_files/infrared/) measured mineral spectra
        - [Open Specy](https://openspecy.org) FT-IR (CC-BY) — the infrared holdings are larger than the Raman ones
        - [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/) — ~16k IR spectra fetched on demand and cached locally (python version only)
        - [SDBS](https://sdbs.db.aist.go.jp/) — link-out plus `.jdx` import, as above

    JCAMP-DX import handles both `(X++(Y..Y))` and `(XY..XY)` forms including
    ASDF compression (SQZ/DIF/DUP), so files from Thermo, Bruker, NIST, SDBS and
    ROD all read correctly.

## Reference database builders

In [`src/Library-builders/`](src/Library-builders). Each writes an `.h5` in one shared schema (`/spectra/<id>` with `x`, `y`,
`peaks`, `intensities` and metadata attributes). Band detection matches the apps
exactly, so match scores are consistent whichever source a reference came from.

| Script | Source | Licence |
| --- | --- | --- |
| `build_cod_powder_library.py` | COD structures → computed powder patterns | CC0 |
| `build_rod_library.py` | Raman Open Database | CC0 |
| `build_openspecy_library.py` | Open Specy, `--only raman\|ftir\|both` | CC-BY |
| `build_rruff_library.py` | RRUFF Raman | — |
| `build_rruff_ir_library.py` | RRUFF infrared | — |
| `build_rruff_powder_library.py` | RRUFF powder XRD | — |

**`make_libraries_manifest.py`** scans a folder of built `.h5` files and writes
the `libraries.json` the HTML apps read, deriving a label and note from each
file's own attributes. No rebuilding needed, and only the attribute block is
read, so a multi-GB library is inspected in milliseconds.

```bash
python src/Library-builders/make_libraries_manifest.py /var/www/html/tools/xrd-plotter
```

### Serving the HTML apps

Drop the `.html` file and your `.h5` libraries in one directory. On load the page
lists what it finds with sizes and a **Load** button — nothing is downloaded
until asked, since a library can be tens or hundreds of MB.

A browser cannot list a directory (HTTP has no such call), so discovery tries, in
order: `?codlib=`/`?rodlib=`/`?speclib=` on the URL → a `libraries.json` manifest
→ the server's directory index → a set of conventional filenames. **The manifest
is the reliable option**, and the only one that can give a library a readable
name. Under `file://` none of it works and libraries must be added by hand.

**Size matters for the browser only.** The Python apps read spectra lazily and
are indifferent to library size; the HTML apps must hold a library in the tab,
where h5wasm caps out at 4 GB of address space. Build large libraries twice —
full curves for the desktop, `--peaks-only` for the browser, which is 50–70×
smaller with identical peak positions and intensities. The apps refuse anything
too large with an explanation rather than crashing the tab.

## SEM/EDS
1. **Summarizer**: Create a summary given the spectra and the images.

