#!/usr/bin/env python3
"""
make_libraries_manifest.py
==========================

Scan a folder of already-built `.h5` reference libraries and write the
`libraries.json` manifest the browser apps read. No rebuilding required.

WHY THIS EXISTS
---------------
A browser cannot list a directory -- HTTP has no such call -- so the apps either
guess conventional filenames or read a manifest. The builders write their own
entry as they go, but if the libraries already exist (or were built before that
existed, or renamed since) this reads them off disk instead.

Everything it needs is already inside each file: the builders stamp `source`,
`technique`, `license`, `storage` and a spectrum count, so labels and notes are
derived rather than guessed. Nothing is recomputed and no spectral data is read
-- only the small attribute block and the group count, so a 2 GB library is
inspected in milliseconds.

USAGE
-----
  # scan the current folder, write ./libraries.json
  python make_libraries_manifest.py

  # a specific folder, previewing without writing
  python make_libraries_manifest.py /var/www/html/tools/xrd-plotter --dry-run

  # keep hand-edited labels for files already in the manifest
  python make_libraries_manifest.py --keep-labels

REQUIREMENTS
------------
  pip install h5py
"""

import argparse
import json
import os
import sys

try:
    import h5py
except ImportError:
    sys.exit("This tool needs h5py:  pip install h5py")


HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

# A library whose filename says "rruff" belongs to the apps' single-library
# RRUFF panel; anything else goes to the multi-library panel.
RRUFF_HINT = "rruff"


def _decode(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace").strip()
    if isinstance(v, str):
        return v.strip()
    if v is None:
        return ""
    return str(v).strip()


def is_hdf5(path):
    try:
        with open(path, "rb") as fh:
            return fh.read(8) == HDF5_MAGIC
    except Exception:                                # noqa: BLE001
        return False


def inspect_library(path):
    """Reads a library's own attributes. Returns a dict, or None if it is not one.

    Only attributes and the group count are touched -- never x/y -- so this is
    fast regardless of file size.
    """
    if not is_hdf5(path):
        return None
    try:
        with h5py.File(path, "r") as f:
            if "spectra" not in f:
                # Flat-schema library (repack_library.py --flat): no /spectra
                # group, so the entry count comes from the offset index. Only
                # the attribute block is read either way, so this stays fast.
                if "peaks_all" in f and "offsets" in f:
                    a = f.attrs
                    n = int(a.get("count", 0)) or max(0, int(f["offsets"].shape[0]) - 1)
                    return {
                        "count": n,
                        "source": _decode(a.get("source")),
                        "database": _decode(a.get("database")),
                        "technique": _decode(a.get("technique")),
                        "license": _decode(a.get("license")),
                        "storage": _decode(a.get("storage")) or "peaks",
                        "built": _decode(a.get("built")),
                        "library_type": _decode(a.get("library_type")),
                        "wavelength": _decode(a.get("wavelength")),
                        "units_x": _decode(a.get("units_x")),
                        "schema": "flat",
                    }
                return None                          # not one of our libraries
            a = f.attrs
            info = {
                "count": len(f["spectra"]),
                "source": _decode(a.get("source")),
                "database": _decode(a.get("database")),
                "technique": _decode(a.get("technique")),
                "license": _decode(a.get("license")),
                "storage": _decode(a.get("storage")),
                "built": _decode(a.get("built")),
                "library_type": _decode(a.get("library_type")),
                "wavelength": _decode(a.get("wavelength")),
                "units_x": _decode(a.get("units_x")),
            }
            # Older RRUFF builds record the count and set names differently.
            if not info["count"]:
                info["count"] = int(a.get("Count", 0) or 0)
            if not info["source"]:
                sets = a.get("Datasets")
                if sets is not None:
                    info["source"] = "RRUFF"
                    info["sets"] = [_decode(x) for x in list(sets)][:4]
            # A peaks-only build has no stored curves; detect it if unstamped.
            if not info["storage"]:
                first = next(iter(f["spectra"].values()), None)
                if first is not None:
                    info["storage"] = "curve+peaks" if "x" in first else "peaks"
            return info
    except Exception as e:                           # noqa: BLE001
        print(f"  [warn] {os.path.basename(path)}: {e}")
        return None


def describe(filename, info, size_bytes):
    """A readable label and one-line note, from the file's own attributes."""
    base = os.path.splitext(os.path.basename(filename))[0]
    src = info.get("source") or ""
    tech = (info.get("technique") or "").lower()

    if src.upper() == "COD":
        label = "COD"
    elif src.upper() == "ROD":
        label = "ROD (Raman Open Database)"
    elif src == "OpenSpecy":
        label = "Open Specy"
    elif src.upper() == "RRUFF" or RRUFF_HINT in base.lower():
        label = "RRUFF"
    else:
        label = src or base.replace("_", " ")

    # The filename usually carries the distinction the attributes cannot
    # (inorganic vs minerals vs organic), so fold it in.
    for hint, word in (("inorganic", "inorganic"), ("mineral", "minerals"),
                       ("organic", "organic"), ("powder", "powder"),
                       ("ftir", "FT-IR"), ("_ir", "infrared"),
                       ("raman", "Raman")):
        if hint in base.lower() and word.lower() not in label.lower():
            label = f"{label} {word}"
            break
    if tech in ("raman", "ftir", "both") and tech not in label.lower():
        pretty = {"raman": "Raman", "ftir": "FT-IR", "both": "Raman + FT-IR"}[tech]
        if pretty.lower() not in label.lower():
            label = f"{label} {pretty}"

    bits = []
    if info.get("count"):
        unit = "patterns" if "powder" in base.lower() or src.upper() == "COD" else "spectra"
        bits.append(f"{info['count']:,} {unit}")
    storage = info.get("storage") or ""
    if storage.startswith("peaks") and "curve" not in storage:
        bits.append("peaks-only")
    if info.get("license"):
        bits.append(info["license"].replace("-1.0", "").replace("-4.0", ""))
    if info.get("library_type"):
        bits.append(info["library_type"])
    if info.get("wavelength"):
        bits.append(str(info["wavelength"]))
    if size_bytes:
        bits.append(f"{size_bytes / 1e6:.0f} MB" if size_bytes >= 1e6
                    else f"{size_bytes / 1e3:.0f} KB")
    return label, " · ".join(bits)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Scan a folder of .h5 reference libraries and write "
                    "libraries.json for the browser apps.")
    p.add_argument("folder", nargs="?", default=".",
                   help="folder holding the .h5 files (default: current)")
    p.add_argument("--out", default=None,
                   help="manifest path (default: libraries.json inside FOLDER)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="print what would be written, change nothing")
    p.add_argument("--keep-labels", dest="keep_labels", action="store_true",
                   help="preserve labels/notes already in the manifest for files "
                        "it already lists (so hand edits survive a re-scan)")
    p.add_argument("--include", default=None,
                   help="only files whose name contains this substring")
    p.add_argument("--exclude", default=None,
                   help="skip files whose name contains this substring")
    args = p.parse_args(argv)

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        sys.exit(f"Not a folder: {folder}")
    target = args.out or os.path.join(folder, "libraries.json")

    previous = {}
    if args.keep_labels and os.path.exists(target):
        try:
            with open(target, encoding="utf-8") as fh:
                old = json.load(fh)
            if isinstance(old, dict):
                old = old.get("libraries", [])
            for item in old or []:
                if isinstance(item, dict):
                    key = os.path.basename(str(item.get("file") or item.get("url") or ""))
                    if key:
                        previous[key] = item
        except Exception as e:                       # noqa: BLE001
            print(f"[warn] could not read existing {target}: {e}")

    names = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".h5", ".hdf5")))
    if args.include:
        names = [n for n in names if args.include.lower() in n.lower()]
    if args.exclude:
        names = [n for n in names if args.exclude.lower() not in n.lower()]
    if not names:
        sys.exit(f"No .h5 files in {folder}")

    print(f"Scanning {folder}")
    entries, skipped = [], []
    for name in names:
        path = os.path.join(folder, name)
        size = os.path.getsize(path)
        info = inspect_library(path)
        if info is None:
            skipped.append(name)
            print(f"  skip  {name:34} not a reference library")
            continue
        label, note = describe(name, info, size)
        entry = {"file": name, "label": label}
        if note:
            entry["note"] = note
        if RRUFF_HINT in name.lower() or (info.get("source") or "").upper() == "RRUFF":
            entry["panel"] = "rruff"
        if name in previous:                         # honour hand edits
            kept = previous[name]
            if args.keep_labels:
                if kept.get("label"):
                    entry["label"] = kept["label"]
                if kept.get("note"):
                    entry["note"] = kept["note"]
                if kept.get("panel"):
                    entry["panel"] = kept["panel"]
        entries.append(entry)
        tag = "  [rruff panel]" if entry.get("panel") == "rruff" else ""
        print(f"  ok    {name:34} {label} — {note}{tag}")

    if not entries:
        sys.exit("Found .h5 files, but none is a reference library "
                 "(no top-level 'spectra' group).")

    payload = json.dumps(entries, indent=2, ensure_ascii=False) + "\n"
    if args.dry_run:
        print(f"\n--dry-run, would write {target}:\n")
        print(payload)
        return

    with open(target, "w", encoding="utf-8") as fh:
        fh.write(payload)
    print(f"\nWrote {target}  ({len(entries)} librar"
          f"{'y' if len(entries) == 1 else 'ies'}"
          f"{f', {len(skipped)} skipped' if skipped else ''})")
    print("Copy it next to the app's .html and the panel will name each library.")


if __name__ == "__main__":
    main()
