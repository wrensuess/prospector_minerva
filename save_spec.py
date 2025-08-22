#!/usr/bin/env python3
"""collect chains and observed quantities from individual NPZ files into a single HDF5 file with multiprocessing"""
import os
import numpy as np
from astropy.table import Table
import utils as ut_cwd
import argparse
import h5py
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import time

# ----------------------
# Helpers (pickleable)
# ----------------------

def get_file_info(file_path):
    """Get (objid, filename) from basename without loading data."""
    filename = os.path.basename(file_path)
    return int(filename.split('_')[1]), filename

def chi2(modmags, obsmags, obsunc):
    _obsunc = np.clip(obsunc, a_min=obsmags*0.05, a_max=None)
    return (((modmags-obsmags)/_obsunc)**2).sum()

def process_single_file(indexed_info, dir_indiv, cat, filts):
    """
    Load one NPZ and return all relevant arrays for this object.
    indexed_info = (idx, objid, filename)
    """
    idx, objid, filename = indexed_info
    full_path = os.path.join(dir_indiv, filename)
    result = {}

    with np.load(full_path, allow_pickle=True) as dat:
        result['modspec_map'] = dat['modspec_map'].astype(np.float32)
        result['modmag_map'] = dat['modmags_map'].astype(np.float32)
        result['zred'] = dat['zred']  # maximum likelihood redshift

    _idx = np.where(cat['id'] == objid)[0][0]
    obs_fnu = ut_cwd.get_fnu_maggies(idx=_idx, catalog=cat, filts=filts)
    obs_enu = ut_cwd.get_enu_maggies(idx=_idx, catalog=cat, filts=filts)
    result['obsmag'] = obs_fnu.astype(np.float32)
    result['obsmag_unc'] = obs_enu.astype(np.float32)
    result['objid'] = objid
    result['nbands'] = np.sum((obs_enu > 0) & np.isfinite(obs_fnu))
    
    
    # Photometry mask
    phot_mask = (obs_enu > 0) & np.isfinite(obs_fnu)
    _mask = np.ones_like(obs_fnu, dtype=bool)
    for k in range(len(obs_fnu)):
        if obs_enu[k] > 0:
            if obs_fnu[k] < 0 and obs_fnu[k] + 5*obs_enu[k] < 0:
                _mask[k] = False
    phot_mask &= _mask
    mask = phot_mask

    obsmags_masked = obs_fnu[mask]
    obsunc_masked = obs_enu[mask]
    fsps_mags_masked = result['modmag_map'][mask]
    result['chi2_fsps'] = chi2(fsps_mags_masked, obsmags_masked, obsunc_masked)

    return idx, result

def process_chunk(indexed_chunk, dir_indiv, cat, filts):
    results, errors = [], []
    for entry in indexed_chunk:
        try:
            results.append(process_single_file(entry, dir_indiv, cat, filts))
        except Exception as e:
            idx, objid, filename = entry
            errors.append(f"{filename}: {e}")
    return results, errors

def write_results(results, datasets):
    for idx, res in results:
        for key, ds in datasets.items():
            ds[idx] = res[key]
    return len(results)

# ----------------------
# Main
# ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=str, default='phisfh', help='phisfh, phisfhzfixed')
    parser.add_argument('--catalog_path', type=str, default="UNCOVER_v5.0.1_LW_SUPER_CATALOG.fits")
    parser.add_argument('--dir_indiv', type=str, default='chains_parrot')
    parser.add_argument('--dir_collected', type=str, default='results')
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--chunk_size', type=int, default=25)
    parser.add_argument('--io_buffer', type=int, default=10)
    args = parser.parse_args()

    n_workers = args.n_workers or min(cpu_count(), 64)
    os.makedirs(args.dir_collected, exist_ok=True)

    sname = os.path.join(args.dir_collected, f'spec_{args.prior}.h5')
    print(f"Output file: {sname}")
    print(f"Using {n_workers} workers with chunk size {args.chunk_size}")

    # Load catalog
    # mdir = ut_cwd.photdir
    cat = Table.read(args.catalog_path)
    all_filternames = np.array([f[2:] for f in cat.dtype.names if f.startswith('f_f')])
    filter_dict = ut_cwd.filter_dictionary(all_filternames)
    filts = list(filter_dict.keys())
    print("Fitting filters:", filts)

    # Discover files
    all_files = sorted([f for f in os.listdir(args.dir_indiv) if f.endswith(f'_spec_{args.prior}.npz')])
    if not all_files:
        raise RuntimeError(f"No files found in {args.dir_indiv} matching '*_spec_{args.prior}.npz'")
    n_obj = len(all_files)

    # Build indexed info
    file_infos = [get_file_info(f) for f in all_files]  # (objid, filename)
    indexed_infos = [(i, objid, fname) for i, (objid, fname) in enumerate(file_infos)]

    # Inspect first file for shapes
    sample_file = os.path.join(args.dir_indiv, all_files[0])
    with np.load(sample_file, allow_pickle=True) as dat:
        modspec_map_shape = dat['modspec_map'].shape
        modmag_map_shape = dat['modmags_map'].shape
        weff = dat['weff'] # photometric effective wavelengths
        wavspec = dat['wavspec'] # wavelengths for fsps model
    obs_fnu_shape = ut_cwd.get_fnu_maggies(idx=0, catalog=cat, filts=filts).shape
    obs_enu_shape = ut_cwd.get_enu_maggies(idx=0, catalog=cat, filts=filts).shape
    print('loaded weff, wavspec from sample file')

    # Create HDF5 datasets
    with h5py.File(sname, 'w') as h5f:
        datasets = {}
        datasets['objid'] = h5f.create_dataset('objid', shape=(n_obj,), dtype=np.int32, compression='gzip', chunks=True)
        datasets['chi2_fsps'] = h5f.create_dataset('chi2_fsps', shape=(n_obj,), dtype=np.float32, compression='gzip', chunks=True)
        datasets['nbands'] = h5f.create_dataset('nbands', shape=(n_obj,), dtype=np.int32, compression='gzip', chunks=True)
        datasets['zred'] = h5f.create_dataset('zred', shape=(n_obj,), dtype=np.float32, compression='gzip', chunks=True)

        datasets['obsmag'] = h5f.create_dataset('obsmag', shape=(n_obj,) + obs_fnu_shape, dtype=np.float32,
                                                compression='gzip', chunks=(1,) + obs_fnu_shape)
        datasets['obsmag_unc'] = h5f.create_dataset('obsmag_unc', shape=(n_obj,) + obs_enu_shape, dtype=np.float32,
                                                    compression='gzip', chunks=(1,) + obs_enu_shape)
        datasets['modspec_map'] = h5f.create_dataset('modspec_map', shape=(n_obj,) + modspec_map_shape, dtype=np.float32,
                                                     compression='gzip', chunks=(1,) + modspec_map_shape)
        datasets['modmag_map'] = h5f.create_dataset('modmag_map', shape=(n_obj,) + modmag_map_shape, dtype=np.float32,
                                                    compression='gzip', chunks=(1,) + modmag_map_shape)
        datasets['weff'] = h5f.create_dataset('weff', data=weff, dtype=np.float32)
        datasets['wavspec'] = h5f.create_dataset('wavspec', data=wavspec, dtype=np.float32)
        print('created datasets in HDF5')

        # Chunking
        chunks = [indexed_infos[i:i + args.chunk_size] for i in range(0, n_obj, args.chunk_size)]
        total_chunks = len(chunks)

        start_time = time.time()
        files_processed = 0
        processed_chunks = 0

        # Multiprocessing executor
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            inflight = {}
            chunk_iter = iter(enumerate(chunks))

            # Prime pipeline
            try:
                while len(inflight) < args.io_buffer:
                    idx, chunk = next(chunk_iter)
                    fut = executor.submit(process_chunk, chunk, args.dir_indiv, cat, filts)
                    inflight[fut] = idx
            except StopIteration:
                pass

            # Consume chunks as completed
            while inflight:
                done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    chunk_idx = inflight.pop(fut)
                    try:
                        results, errors = fut.result()
                        for msg in errors:
                            print("Error processing", msg)
                        written = write_results(results, datasets)
                        files_processed += written
                        processed_chunks += 1
                        if processed_chunks % 10 == 0 or processed_chunks == total_chunks:
                            elapsed = time.time() - start_time
                            rate = files_processed / elapsed if elapsed > 0 else 0.0
                            print(f"Completed chunk {processed_chunks}/{total_chunks}: "
                                  f"{files_processed} files ({rate:.1f} files/sec)")
                    except Exception as e:
                        print(f"Error consuming chunk {chunk_idx}: {e}")

                    # Refill inflight queue
                    try:
                        idx, chunk = next(chunk_iter)
                        fut = executor.submit(process_chunk, chunk, args.dir_indiv, cat, filts)
                        inflight[fut] = idx
                    except StopIteration:
                        pass

    total_time = time.time() - start_time
    print("\nCompleted!")
    print(f"Total objects: {files_processed}/{n_obj}")
    print(f"Total time: {total_time:.2f} sec")
    rate = files_processed / total_time if total_time > 0 else 0.0
    print(f"Average rate: {rate:.1f} files/sec")
    print(f"Saved to: {sname}")

if __name__ == '__main__':
    main()
