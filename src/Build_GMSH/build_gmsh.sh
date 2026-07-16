#!/usr/bin/env bash
#
# build_gmsh.sh — Build and install gmsh 4.15.2 on an old Linux system
# (e.g. RHEL/CentOS 7, glibc 2.17) using a self-contained conda GCC 11
# toolchain. No root required.
#
# The build environment lives INSIDE the directory the script is run from:
#   ./miniconda3   conda + GCC 11 toolchain
#   ./build        downloaded source + build tree
#
# The install prefix is a separate, OPTIONAL argument. If omitted, gmsh is
# installed to ./gmsh inside the current folder.
#
# Usage:
#   ./build_gmsh.sh [install-prefix] [num-jobs]
#
# Example:
#   cd /home/nicola/gmsh_local
#   /path/to/build_gmsh.sh                        # install to ./gmsh, all cores
#   /path/to/build_gmsh.sh /home/nicola/exec      # install to /home/nicola/exec
#   /path/to/build_gmsh.sh /home/nicola/exec 8    # ... with 8 parallel jobs
#
# The resulting gmsh binary is linked with -static-libstdc++ -static-libgcc,
# so it runs on the system without needing the conda environment active.

set -euo pipefail

# ---------------------------------------------------------------------------
# The build environment is anchored to the current working directory
# ---------------------------------------------------------------------------
BASE_DIR="$(pwd)"
MINICONDA_PREFIX="${BASE_DIR}/miniconda3"
WORKDIR="${BASE_DIR}/build"

# ---------------------------------------------------------------------------
# Arguments
#   $1 (optional): install prefix   (default: ./gmsh in the current folder)
#   $2 (optional): number of jobs   (default: all cores)
# ---------------------------------------------------------------------------
INSTALL_PREFIX="${1:-${BASE_DIR}/gmsh}"
JOBS="${2:-$(nproc 2>/dev/null || echo 4)}"

# Resolve a relative install prefix against the launch directory (not ./build),
# since CMake/make install run from inside the build tree.
case "${INSTALL_PREFIX}" in
    /*) : ;;                                   # already absolute
    *)  INSTALL_PREFIX="${BASE_DIR}/${INSTALL_PREFIX}" ;;
esac

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GMSH_VERSION="4.15.2"
GMSH_TARBALL="gmsh-${GMSH_VERSION}-source.tgz"
GMSH_URL="https://gmsh.info/src/${GMSH_TARBALL}"
GMSH_SRCDIR="gmsh-${GMSH_VERSION}-source"

# This installer still runs on glibc 2.17 (newer ones require >= 2.28):
MINICONDA_INSTALLER="Miniconda3-py39_4.12.0-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}"

CONDA_ENV="gcc11"
GCC_VERSION="11"

echo "=============================================================="
echo " gmsh ${GMSH_VERSION} build (self-contained)"
echo "   base directory : ${BASE_DIR}"
echo "   conda prefix   : ${MINICONDA_PREFIX}"
echo "   install prefix : ${INSTALL_PREFIX}"
echo "   build/work dir : ${WORKDIR}"
echo "   parallel jobs  : ${JOBS}"
echo "=============================================================="

mkdir -p "${WORKDIR}"

# ---------------------------------------------------------------------------
# Helper: download a URL with wget or curl (whichever exists)
# ---------------------------------------------------------------------------
download() {
    local url="$1" out="$2"
    if [[ -f "${out}" ]]; then
        echo ">> ${out} already present, skipping download."
        return 0
    fi
    echo ">> Downloading ${url}"
    if command -v wget >/dev/null 2>&1; then
        wget -O "${out}" "${url}"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o "${out}" "${url}"
    else
        echo "ERROR: neither wget nor curl is available." >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 1. Install Miniconda into ./miniconda3 (only if not already present)
# ---------------------------------------------------------------------------
if [[ ! -x "${MINICONDA_PREFIX}/bin/conda" ]]; then
    echo ">> Installing Miniconda to ${MINICONDA_PREFIX}"
    download "${MINICONDA_URL}" "${WORKDIR}/${MINICONDA_INSTALLER}"
    bash "${WORKDIR}/${MINICONDA_INSTALLER}" -b -p "${MINICONDA_PREFIX}"
else
    echo ">> Miniconda already installed at ${MINICONDA_PREFIX}"
fi

# Load conda into this shell. Conda's own activation scripts are not written
# to be safe under 'set -u' (they reference variables like ADDR2LINE before
# defining them), so relax nounset while sourcing / activating conda.
set +u
# shellcheck disable=SC1091
source "${MINICONDA_PREFIX}/etc/profile.d/conda.sh"
set -u

# ---------------------------------------------------------------------------
# 2. Create the GCC 11 environment (only if not already present)
# ---------------------------------------------------------------------------
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo ">> Creating conda env '${CONDA_ENV}' with GCC ${GCC_VERSION}"
    conda create -n "${CONDA_ENV}" -c conda-forge \
        "gxx_linux-64=${GCC_VERSION}" cmake make gmp -y
else
    echo ">> Conda env '${CONDA_ENV}' already exists"
    echo ">> Ensuring GMP is present in '${CONDA_ENV}'"
    conda install -n "${CONDA_ENV}" -c conda-forge gmp -y
fi

set +u
conda activate "${CONDA_ENV}"
set -u

# $CC and $CXX are set by the conda compiler activation scripts.
echo ">> Using CC=${CC}"
echo ">> Using CXX=${CXX}"
"${CXX}" --version | head -n1

# ---------------------------------------------------------------------------
# 3. Download and extract gmsh source into ./build
# ---------------------------------------------------------------------------
cd "${WORKDIR}"
download "${GMSH_URL}" "${GMSH_TARBALL}"

if [[ ! -d "${GMSH_SRCDIR}" ]]; then
    echo ">> Extracting ${GMSH_TARBALL}"
    tar xf "${GMSH_TARBALL}"
fi

# ---------------------------------------------------------------------------
# 4. Configure, build, install into ./gmsh
# ---------------------------------------------------------------------------
cd "${GMSH_SRCDIR}"
rm -rf build
mkdir build
cd build

echo ">> Configuring with CMake"
cmake \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
    -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
    ..

echo ">> Building (make -j${JOBS})"
make -j"${JOBS}"

echo ">> Installing to ${INSTALL_PREFIX}"
make install

echo "=============================================================="
echo " Done."
echo " gmsh installed to: ${INSTALL_PREFIX}/bin/gmsh"
echo
echo " Add it to your PATH:"
echo "   export PATH=${INSTALL_PREFIX}/bin:\$PATH"
echo
echo " (Built with -static-libstdc++/-static-libgcc, so the conda"
echo "  environment does NOT need to be active to run gmsh.)"
echo "=============================================================="
