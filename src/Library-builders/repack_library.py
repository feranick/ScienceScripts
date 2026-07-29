#!/usr/bin/env python3
"""Shrink or split an existing reference .h5 without rebuilding it.

Nothing here re-downloads or recomputes anything: it reads the library you
already have and writes a new file. Three independent levers, usable together:

  --no-compress   drop gzip. Counter-intuitively this SHRINKS the file: each
                  entry stores ~60 float32 values, and a gzip chunk that small
                  costs more in filter/chunk bookkeeping than the compression
                  saves. Measured on cod_organic.h5: 1423 MB -> 597 MB.
  --drop-attrs    remove per-entry attributes that are redundant or derivable.
                  `url` is 48 bytes/entry and is just the COD id in a template;
                  `source` and `wavelength` are already file-level attributes.
  --split-mb N    write several shards instead of one file. The apps already
                  accept multiple libraries with a selector, so shards appear as
                  separate sources you can enable individually.

What this cannot fix: HDF5 spends roughly 2 KB per group on object headers and
B-tree nodes, so a group-per-entry library has a floor near 2.5 KB/entry however
much you trim. For 195k entries that is ~490 MB. The only way below that is to
stop using one group per entry -- see the note at the end of --help.

Examples
--------
  # the usual case: same schema, no app changes needed, 2.4x smaller
  python3 repack_library.py cod_organic.h5 --no-compress -o cod_organic_web.h5

  # trim as well: 1423 -> ~512 MB
  python3 repack_library.py cod_organic.h5 --no-compress \\
      --drop-attrs url,rruff_id,source,wavelength -o cod_organic_web.h5

  # four shards of ~150 MB each
  python3 repack_library.py cod_organic.h5 --no-compress --split-mb 150
"""
import argparse
import os
import sys
import time

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("This tool needs h5py:  pip install h5py")


# Attributes that are safe to drop: derivable from cod_id, duplicated at file
# level, or always empty in practice. Listed so --help can show them.
SUGGESTED_DROP = ('url', 'rruff_id', 'source', 'wavelength')


def human(n):
    return '%.1f MB' % (n / 1e6) if n < 1e9 else '%.2f GB' % (n / 1e9)


def entry_names(src):
    """Group names under /spectra, in file order."""
    return list(src['spectra'].keys())


def entry_peaks(g):
    """(peaks, intensities) for one group, whichever way they are stored.

    The RRUFF builders keep peaks in a group ATTRIBUTE, not a dataset -- and
    intensities not at all. The loaders have always accepted both forms; this
    tool did not, so --flat crashed with a bare KeyError on any RRUFF library.
    Returns (None, None) when an entry carries no peaks at all.
    """
    px = None
    if 'peaks' in g:
        px = np.asarray(g['peaks'][:], dtype='float32')
    elif 'peaks' in g.attrs:
        px = np.atleast_1d(np.asarray(g.attrs['peaks'], dtype='float32'))
    if px is None or px.size == 0:
        return None, None
    py = None
    if 'intensities' in g:
        py = np.asarray(g['intensities'][:], dtype='float32')
    elif 'intensities' in g.attrs:
        py = np.atleast_1d(np.asarray(g.attrs['intensities'], dtype='float32'))
    if py is None or py.size != px.size:
        # No stored heights: treat every reflection as equally strong rather
        # than inventing a ranking. peak_index_rank already handles this.
        py = np.ones(px.size, dtype='float32')
    return px, py


def copy_entry(dst_spectra, name, g, drop, keep_gzip, max_peaks):
    px, py = entry_peaks(g)
    if px is None:
        return 0
    if max_peaks and px.size > max_peaks:
        # Keep the strongest, then restore ascending position order so the
        # loaders and the match code see the same convention as before.
        if py is not None:
            idx = np.argsort(py)[::-1][:max_peaks]
            idx = np.sort(idx)
            px, py = px[idx], py[idx]
        else:
            px = px[:max_peaks]
    out = dst_spectra.create_group(name)
    kw = dict(compression='gzip') if keep_gzip else {}
    out.create_dataset('peaks', data=px.astype('float32'), **kw)
    if py is not None:
        out.create_dataset('intensities', data=py.astype('float32'), **kw)
    # the curve, when the source library carries one
    for extra in ('x', 'y'):
        if extra in g:
            out.create_dataset(extra, data=g[extra][:].astype('float32'), **kw)
    for k, v in g.attrs.items():
        if k not in drop:
            out.attrs[k] = v
    return px.size


# 'group' carries the source group name. Positional identity is not enough: an
# entry with no peaks is dropped, after which flat slot j no longer corresponds
# to source entry j -- which is exactly how the verifier first mis-reported a
# correct file as corrupt.
STR_FIELDS = (('name', 'name'), ('formula', 'formula'), ('sg', 'sg'),
              ('url', 'url'), ('id', None), ('group', None))


def write_flat(src, names, dest, file_attrs, src_name, drop, max_peaks, t0):
    """Consolidated layout: one peaks array plus an offset index.

    HDF5 charges ~2 KB per group, so a group-per-entry library is ~93% metadata
    for peaks-only data. Concatenating every entry's reflections into two arrays
    and recording where each entry starts drops that to ~0.44 KB/entry, which is
    a 16x reduction on cod_organic.h5. The parallel string arrays hold the
    metadata that used to live in per-group attributes.
    """
    n = len(names)
    offs = np.zeros(n + 1, dtype='uint32')
    peaks_parts, inten_parts = [], []
    # Entries without peaks are dropped, so the offset array is filled by KEPT
    # position, not by source position -- otherwise a skipped entry leaves a
    # zero-length hole and every later entry reads the wrong slice.
    kept = []
    cols = {field: [] for field, _ in STR_FIELDS}
    for i, nm in enumerate(names):
        g = src['spectra'][nm]
        px, py = entry_peaks(g)
        if px is None:
            continue                      # nothing to carry for this entry
        if max_peaks and px.size > max_peaks:
            keep = np.sort(np.argsort(py)[::-1][:max_peaks])
            px, py = px[keep], py[keep]
        peaks_parts.append(px)
        inten_parts.append(py)
        kept.append(i)
        offs[len(kept)] = offs[len(kept) - 1] + px.size
        a = g.attrs
        for field, key in STR_FIELDS:
            if field == 'group':
                val = nm
            elif field == 'id':
                val = a.get('cod_id') or a.get('rod_id') or a.get('rruff_id') or nm
            else:
                val = a.get(key, '')
            cols[field].append('' if field in drop else str(val))
        if (i + 1) % 20000 == 0:
            print('    ... %s/%s entries (%.0fs)'
                  % (format(i + 1, ','), format(n, ','), time.time() - t0))

    n_kept = len(kept)
    offs = offs[:n_kept + 1]
    with h5py.File(dest, 'w') as h:
        for k, v in file_attrs.items():
            h.attrs[k] = v
        h.attrs['schema'] = 'flat'
        h.attrs['count'] = n_kept
        if n_kept != n:
            h.attrs['skipped_no_peaks'] = n - n_kept
        h.attrs['repacked_from'] = os.path.basename(src_name)
        pk = np.concatenate(peaks_parts) if peaks_parts else np.zeros(0, 'float32')
        it = np.concatenate(inten_parts) if inten_parts else np.zeros(0, 'float32')
        # A chunk may not exceed the dataset, which a fixed 64 K breaks on any
        # library smaller than ~1000 entries.
        chunk = (max(1, min(1 << 16, pk.size)),)
        h.create_dataset('peaks_all', data=pk, compression='gzip', chunks=chunk)
        h.create_dataset('inten_all', data=it, compression='gzip', chunks=chunk)
        h.attrs['n_peaks'] = int(offs[-1])
        # uint32 rather than int64: h5wasm hands int64 back as BigInt64Array,
        # which every consumer would then have to convert. 4 G reflections is
        # far beyond any library we build.
        h.create_dataset('offsets', data=offs, compression='gzip')
        for field, _ in STR_FIELDS:
            vals = cols[field]
            if not any(vals):
                continue                      # nothing to store, skip the dataset
            w = max(1, max(len(v.encode('utf-8')) for v in vals))
            h.create_dataset(field, dtype='S%d' % w, compression='gzip',
                             data=np.array([v.encode('utf-8') for v in vals],
                                           dtype='S%d' % w))
    return int(offs[-1])


def flat_convert(src_path, dest_path=None, drop=(), max_peaks=0, quiet=False):
    """Convert a group-per-entry library to the flat layout, in place if asked.

    Importable so the builders can offer --flat without duplicating any of this:
    they write the ordinary schema, then call here. Writing to a temp file and
    replacing means a failure leaves the original intact.
    """
    in_place = dest_path is None or os.path.abspath(dest_path) == os.path.abspath(src_path)
    tmp = (src_path + '.flat.tmp') if in_place else dest_path
    t0 = time.time()
    with h5py.File(src_path, 'r') as src:
        if 'spectra' not in src:
            raise ValueError('%s has no /spectra group' % src_path)
        names = entry_names(src)
        npk = write_flat(src, names, tmp, dict(src.attrs), src_path,
                         set(drop), max_peaks, t0)
    if in_place:
        os.replace(tmp, src_path)
        final = src_path
    else:
        final = dest_path
    if not quiet:
        sz = os.path.getsize(final)
        print('  flat layout: %s entries, %s reflections, %s  (%.2f KB/entry)'
              % (format(len(names), ','), format(npk, ','), human(sz),
                 sz / max(1, len(names)) / 1024))
    return final


def open_shard(path, file_attrs, src_name, shard, n_shards):
    h = h5py.File(path, 'w')
    for k, v in file_attrs.items():
        h.attrs[k] = v
    h.attrs['repacked_from'] = os.path.basename(src_name)
    if n_shards > 1:
        h.attrs['shard'] = shard
        h.attrs['shard_of'] = n_shards
    return h, h.create_group('spectra')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Below ~2.5 KB/entry the group-per-entry schema itself is the "
               "limit. A consolidated layout (one concatenated peaks array plus "
               "an offset index) measures 0.44 KB/entry -- 89 MB for a 195k-entry "
               "library -- but the apps need code to read it, so it is not one of "
               "the options here.")
    ap.add_argument('src', help='existing .h5 library')
    ap.add_argument('-o', '--out', help='output path (default: <stem>_web.h5)')
    ap.add_argument('--no-compress', dest='no_compress', action='store_true',
                    help='drop gzip (usually makes the file much SMALLER)')
    ap.add_argument('--keep-compression', dest='no_compress', action='store_false',
                    help='keep gzip as in the source')
    ap.set_defaults(no_compress=True)
    ap.add_argument('--drop-attrs', default='',
                    help='comma-separated per-entry attributes to omit. Safe: '
                         + ','.join(SUGGESTED_DROP))
    ap.add_argument('--max-peaks', type=int, default=0,
                    help='keep only the N strongest reflections per entry')
    ap.add_argument('--split-mb', type=float, default=0,
                    help='write shards of about this size instead of one file')
    ap.add_argument('--limit', type=int, default=0,
                    help='only the first N entries (for a quick trial run)')
    ap.add_argument('--verify', type=int, default=200,
                    help='re-read this many random entries and compare against '
                         'the source (0 to skip). Default 200')
    ap.add_argument('--flat', action='store_true',
                    help='consolidated layout: one concatenated peaks array plus '
                         'an offset index. ~16x smaller than the group-per-entry '
                         'schema. Needs app support (2026.07.28.1 or later)')
    args = ap.parse_args()

    if not os.path.exists(args.src):
        sys.exit('no such file: %s' % args.src)
    drop = {s.strip() for s in args.drop_attrs.split(',') if s.strip()}
    stem = os.path.splitext(os.path.basename(args.src))[0]
    src_size = os.path.getsize(args.src)

    t0 = time.time()
    with h5py.File(args.src, 'r') as src:
        if 'spectra' not in src:
            sys.exit('%s has no /spectra group -- is it one of our libraries?'
                     % args.src)
        file_attrs = dict(src.attrs)
        names = entry_names(src)
        if args.limit:
            names = names[:args.limit]
        n = len(names)
        print('  source: %s   %s entries   %s' % (args.src, format(n, ','), human(src_size)))
        print('  gzip: %s   drop attrs: %s   max-peaks: %s'
              % ('off' if args.no_compress else 'kept',
                 ','.join(sorted(drop)) or 'none', args.max_peaks or 'all'))

        if args.flat:
            if args.split_mb:
                sys.exit('--flat and --split-mb together are not supported: the '
                         'flat layout is small enough that sharding is pointless.')
            dest = args.out or os.path.join(
                os.path.dirname(os.path.abspath(args.src)), stem + '_flat.h5')
            npk = write_flat(src, names, dest, file_attrs, args.src, drop,
                             args.max_peaks, t0)
            sz = os.path.getsize(dest)
            print()
            print('  wrote %-46s %s  (%s entries, %.2f KB/entry)'
                  % (os.path.basename(dest), human(sz), format(n, ','), sz / n / 1024))
            print('  total %s -> %s   (%.1fx smaller)'
                  % (human(src_size), human(sz), src_size / sz))
            print('  %s reflections carried over, in %.0fs'
                  % (format(npk, ','), time.time() - t0))
            if args.verify:
                verify_flat(args.src, dest, args.verify, args.max_peaks)
            print('\n  Next: re-run make_libraries_manifest.py on the serving '
                  'directory so libraries.json picks up the new file.')
            return

        # One shard unless asked otherwise. Sizes are only known as we go, so
        # shards are cut on a running byte estimate and the count is not known
        # up front -- hence the two-pass rename at the end.
        budget = args.split_mb * 1e6 if args.split_mb else float('inf')
        shard_paths, written = [], []
        idx = 1
        cur_path = '/tmp/_repack_shard_%d.h5' % idx
        h, sp = open_shard(cur_path, file_attrs, args.src, idx, 1)
        count_here = 0
        total_peaks = 0
        for i, nm in enumerate(names):
            total_peaks += copy_entry(sp, nm, src['spectra'][nm], drop,
                                      not args.no_compress, args.max_peaks)
            count_here += 1
            # ~2.5 KB/entry is the observed floor for this schema; good enough
            # to cut shards on without flushing and stat-ing every iteration.
            if count_here * 2560 >= budget:
                h.attrs['count'] = count_here
                h.close()
                shard_paths.append(cur_path); written.append(count_here)
                idx += 1
                cur_path = '/tmp/_repack_shard_%d.h5' % idx
                h, sp = open_shard(cur_path, file_attrs, args.src, idx, 1)
                count_here = 0
            if (i + 1) % 20000 == 0:
                print('    ... %s/%s entries (%.0fs)'
                      % (format(i + 1, ','), format(n, ','), time.time() - t0))
        h.attrs['count'] = count_here
        h.close()
        shard_paths.append(cur_path); written.append(count_here)

    n_shards = len(shard_paths)
    outdir = os.path.dirname(os.path.abspath(args.out or args.src))
    finals = []
    for i, (p, cnt) in enumerate(zip(shard_paths, written), start=1):
        if n_shards == 1:
            dest = args.out or os.path.join(outdir, stem + '_web.h5')
        else:
            dest = os.path.join(outdir, '%s_%02dof%02d.h5' % (stem, i, n_shards))
            with h5py.File(p, 'a') as h:
                h.attrs['shard'] = i
                h.attrs['shard_of'] = n_shards
        os.replace(p, dest)
        finals.append((dest, cnt))

    print()
    out_total = sum(os.path.getsize(d) for d, _ in finals)
    for d, cnt in finals:
        sz = os.path.getsize(d)
        print('  wrote %-46s %s  (%s entries, %.2f KB/entry)'
              % (os.path.basename(d), human(sz), format(cnt, ','), sz / cnt / 1024))
    print('  total %s -> %s   (%.1fx smaller)'
          % (human(src_size), human(out_total), src_size / out_total))
    print('  %s reflections carried over, in %.0fs' % (format(total_peaks, ','), time.time() - t0))

    if args.verify:
        verify(args.src, finals, args.verify, drop, args.max_peaks)

    print('\n  Next: re-run make_libraries_manifest.py on the serving directory '
          'so libraries.json picks up the new file(s).')


def verify_flat(src_path, dest, k, max_peaks):
    """Reconstruct random entries from the flat file and compare to the source.

    This is the check that matters: the offset arithmetic is the whole design, so
    an off-by-one would silently shift every entry's reflections by one slot.
    """
    print('\n  verifying %d random entries against the source ...' % k)
    rng = np.random.default_rng(0)
    bad = []
    with h5py.File(src_path, 'r') as src, h5py.File(dest, 'r') as out:
        names = list(src['spectra'].keys())
        offs = out['offsets'][:]
        pa_all, ia_all = out['peaks_all'], out['inten_all']
        ids = out['id'][:] if 'id' in out else None
        groups = out['group'][:] if 'group' in out else None
        n = int(out.attrs['count'])
        if groups is None and len(names) != n:
            sys.exit('  cannot verify: file predates the group column and entries '
                     'were dropped, so slots cannot be matched to source entries')
        for j in rng.choice(n, size=min(k, n), replace=False):
            # Look the source entry up by its recorded group name. Comparing by
            # position silently drifts as soon as one entry is dropped.
            nm = groups[j].decode() if groups is not None else names[j]
            g = src['spectra'][nm]
            want, _ = entry_peaks(g)
            if want is None:
                continue           # dropped for having no peaks; nothing to compare
            got = pa_all[offs[j]:offs[j + 1]]
            if max_peaks and want.size > max_peaks:
                if got.size != max_peaks or not np.isin(got, want).all():
                    bad.append((nm, 'peaks (trimmed)'))
                    continue
            elif want.size != got.size or not np.allclose(want, got, atol=1e-6):
                bad.append((nm, 'peaks %d vs %d' % (want.size, got.size)))
                continue
            if ids is not None:
                src_id = str(g.attrs.get('cod_id') or g.attrs.get('rod_id')
                             or g.attrs.get('rruff_id') or nm)
                if ids[j].decode() != src_id:
                    bad.append((nm, 'id %r vs %r' % (ids[j].decode(), src_id)))
        # the last offset must equal the total length, or entries were dropped
        if int(offs[-1]) != pa_all.shape[0]:
            bad.append(('<offsets>', 'last offset %d != peaks_all length %d'
                        % (int(offs[-1]), pa_all.shape[0])))
        if 'inten_all' in out and ia_all.shape[0] != pa_all.shape[0]:
            bad.append(('<arrays>', 'peaks and intensities differ in length'))
    if bad:
        print('  %d problems -- do not use this output:' % len(bad))
        for nm, why in bad[:10]:
            print('    %s: %s' % (nm, why))
        sys.exit(1)
    print('  all %d sampled entries reconstruct exactly, offsets consistent' % min(k, n))


def verify(src_path, finals, k, drop, max_peaks):
    """Compare a random sample of entries against the source."""
    print('\n  verifying %d random entries against the source ...' % k)
    rng = np.random.default_rng(0)
    bad = 0
    checked = 0
    with h5py.File(src_path, 'r') as src:
        for dest, _ in finals:
            with h5py.File(dest, 'r') as out:
                names = list(out['spectra'].keys())
                pick = rng.choice(len(names), size=min(k, len(names)), replace=False)
                for j in pick:
                    nm = names[j]
                    a, b = src['spectra'][nm], out['spectra'][nm]
                    pa, _ = entry_peaks(a)
                    pb, _ = entry_peaks(b)
                    if pa is None or pb is None:
                        continue
                    if max_peaks:
                        if pb.size > max_peaks or not np.isin(pb, pa.astype('float32')).all():
                            bad += 1
                    elif pa.size != pb.size or not np.allclose(pa, pb, atol=1e-6):
                        bad += 1
                    # metadata that was not dropped must survive intact
                    for key, v in a.attrs.items():
                        if key in drop:
                            continue
                        if key not in b.attrs or str(b.attrs[key]) != str(v):
                            bad += 1
                            break
                    checked += 1
    if bad:
        print('  %d of %d sampled entries DIFFER -- do not use this output' % (bad, checked))
        sys.exit(1)
    print('  all %d sampled entries match (peaks and retained metadata)' % checked)


if __name__ == '__main__':
    main()
