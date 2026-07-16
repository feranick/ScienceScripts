#!/usr/bin/env python3
"""
Plot a 'necklace' of spherical particles from a PACKMOL-style PDB file.

Each ATOM record is one sphere:
  - columns 31-54 hold the x, y, z coordinates (Angstrom)
  - the B-factor column (61-66) holds radius * 10, so radius = Bfactor / 10
    (this captures the per-sphere size distribution)

Spheres are drawn as true 3D surfaces and connected, in file order,
by a line to show the necklace chain.

Usage:
    python plot_necklace.py structure.pdb
    python plot_necklace.py structure.pdb --save                 # -> structure.png
    python plot_necklace.py structure.pdb --save pdf             # -> structure.pdf
    python plot_necklace.py structure.pdb --save necklace.png    # explicit filename
    python plot_necklace.py structure.pdb --no-chain             # hide connecting line
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def read_pdb(path):
    """Return (coords Nx3 array, radii length-N array)."""
    coords, radii = [], []
    with open(path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            # Fixed-column PDB parsing
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            bfac = line[60:66].strip()
            r = float(bfac) / 10.0 if bfac else 1.0
            coords.append((x, y, z))
            radii.append(r)
    return np.array(coords), np.array(radii)


def sphere_mesh(center, radius, n=16):
    """Return x, y, z mesh grids for a sphere surface."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def main():
    ap = argparse.ArgumentParser(description="Plot a necklace of spheres from a PDB file.")
    ap.add_argument("pdb", help="input PDB file")
    ap.add_argument("--save", nargs="?", const="", metavar="FILE_OR_EXT",
                    help="save figure instead of showing. With no value, uses the input "
                         "filename with a .png extension. A bare extension (e.g. 'pdf') "
                         "keeps the input name but changes the extension. Or give a full filename.")
    ap.add_argument("--no-chain", action="store_true", help="do not draw the connecting chain line")
    ap.add_argument("--res", type=int, default=16, help="sphere mesh resolution (default 16)")
    ap.add_argument("--alpha", type=float, default=0.9, help="sphere opacity (default 0.9)")
    args = ap.parse_args()

    coords, radii = read_pdb(args.pdb)
    if len(coords) == 0:
        sys.exit("No ATOM/HETATM records found.")

    n = len(coords)
    print(f"{n} spheres | radius: mean={radii.mean():.3f}, "
          f"min={radii.min():.3f}, max={radii.max():.3f}")

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Color spheres by their position along the chain
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, n))

    # Connecting chain line (drawn first, behind spheres)
    if not args.no_chain:
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                color="0.4", linewidth=1.0, alpha=0.7, zorder=1)

    # Spheres
    for c, r, col in zip(coords, radii, colors):
        x, y, z = sphere_mesh(c, r, n=args.res)
        ax.plot_surface(x, y, z, color=col, alpha=args.alpha,
                        linewidth=0, antialiased=True, shade=True)

    # Equal aspect ratio so spheres are not distorted
    ax.set_box_aspect((1, 1, 1))
    max_range = (coords.max(axis=0) - coords.min(axis=0)).max() / 2.0
    mid = coords.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Necklace of {n} spheres")

    plt.tight_layout()
    if args.save is not None:
        base, _ = os.path.splitext(args.pdb)
        if args.save == "":                       # --save with no value
            out = base + ".png"
        elif "." not in os.path.basename(args.save):   # bare extension, e.g. "pdf"
            out = base + "." + args.save.lstrip(".")
        else:                                     # full filename given
            out = args.save
        plt.savefig(out, dpi=150)
        print(f"Saved to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
