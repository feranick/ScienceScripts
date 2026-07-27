"""ScienceScripts: spectroscopy plotters and reference-library builders.

This file exists only so the wheel has a real package rather than a namespace
package. The modules themselves are remapped here at build time from the
technique folders (src/XRD, src/Raman, src/FTIR, src/Library-builders) -- see the
`[tool.hatch.build.targets.wheel.sources]` table in pyproject.toml. Goes in the
repo at src/sciencescripts/__init__.py.

Nothing is imported eagerly: importing this package must not pull in tkinter,
matplotlib or h5py, so that `make-libraries-manifest` stays usable on a headless
machine and `python -c "import sciencescripts"` is cheap.
"""

__all__ = [
    "xrd_plotter",
    "raman_plotter",
    "ftir_plotter",
    "build_cod_powder_library",
    "build_rod_library",
    "build_openspecy_library",
    "build_rruff_library",
    "build_rruff_ir_library",
    "build_rruff_powder_library",
    "make_libraries_manifest",
]

try:                                    # installed
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("sciencescripts")
    except PackageNotFoundError:        # running from a checkout
        __version__ = "0.dev0"
except ImportError:                     # pragma: no cover  (Python < 3.8)
    __version__ = "0.dev0"
