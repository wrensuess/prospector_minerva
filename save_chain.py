#!/usr/bin/env python3
"""collect chains from individual NPZ files into a single HDF5 file"""
import os
import numpy as np
import argparse
import h5py
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import time

# ----------------------
# Helpers (top-level so they're pickleable)
# ----------------------

def get_file_info(file_path):
    """Get (objid, filename) from basename without loading data."""
    filename = os.path.basename(file_path)
    return int(filename.split('_')[1]), file_path  # (objid, filename)

def process_single_file(indexed_info, dir_indiv):
    """
    Worker: load one NPZ and return (index, objid, chain_data[nsamp,26]).
    indexed_info = (idx, objid, filename)
    """
    idx, objid, filename = indexed_info
    full_path = os.path.join(dir_indiv, filename)

    with np.load(full_path, allow_pickle=True) as dat:
        chains = dat['chains'][()]  # dict-like

    n_samples = chains['zred'].shape[0]
    chain_data = np.empty((n_samples, 26), dtype=np.float32)

    # Fill columns 
    chain_data[:, 0]  = chains['zred']
    chain_data[:, 1]  = chains['total_mass']
    chain_data[:, 2]  = chains['stellar_mass']
    chain_data[:, 3]  = chains['logzsol']
    chain_data[:, 4]  = chains['mwa']
    chain_data[:, 5:8] = chains['sfr']
    chain_data[:, 8:11] = chains['ssfr'] 
    chain_data[:, 11] = chains['dust2']
    chain_data[:, 12] = chains['dust_index']
    chain_data[:, 13] = chains['dust1_fraction']
    chain_data[:, 14] = chains['log_fagn']
    chain_data[:, 15] = chains['log_agn_tau']
    chain_data[:, 16] = chains['gas_logz']
    chain_data[:, 17] = chains['duste_qpah']
    chain_data[:, 18] = chains['duste_umin']
    chain_data[:, 19] = chains['log_duste_gamma']
    chain_data[:, 20] = chains['logsfr_ratios_1']
    chain_data[:, 21] = chains['logsfr_ratios_2']
    chain_data[:, 22] = chains['logsfr_ratios_3']
    chain_data[:, 23] = chains['logsfr_ratios_4']
    chain_data[:, 24] = chains['logsfr_ratios_5']
    chain_data[:, 25] = chains['logsfr_ratios_6']

    return idx, objid, chain_data

def process_single_file_h5(indexed_info, dir_indiv):
    """
    Worker: load one HDF5 and return (index, objid, chain_data[nsamp,26]).
    indexed_info = (idx, objid, filename)
    """
    idx, objid, filename = indexed_info
    full_path = os.path.join(dir_indiv, filename)

    with h5py.File(full_path, "r") as f:
        chains = f["chains"]

    n_samples = chains["zred"].shape[0]
    chain_data = np.empty((n_samples, 26), dtype=np.float32)

    # Fill columns
    chain_data[:, 0]  = chains["zred"][:]
    chain_data[:, 1]  = chains["total_mass"][:]
    chain_data[:, 2]  = chains["stellar_mass"][:]
    chain_data[:, 3]  = chains["logzsol"][:]
    chain_data[:, 4]  = chains["mwa"][:]
    chain_data[:, 5:8] = chains["sfr"][:]
    chain_data[:, 8:11] = chains["ssfr"][:]
    chain_data[:, 11] = chains["dust2"][:]
    chain_data[:, 12] = chains["dust_index"][:]
    chain_data[:, 13] = chains["dust1_fraction"][:]
    chain_data[:, 14] = chains["log_fagn"][:]
    chain_data[:, 15] = chains["log_agn_tau"][:]
    chain_data[:, 16] = chains["gas_logz"][:]
    chain_data[:, 17] = chains["duste_qpah"][:]
    chain_data[:, 18] = chains["duste_umin"][:]
    chain_data[:, 19] = chains["log_duste_gamma"][:]
    chain_data[:, 20] = chains["logsfr_ratios_1"][:]
    chain_data[:, 21] = chains["logsfr_ratios_2"][:]
    chain_data[:, 22] = chains["logsfr_ratios_3"][:]
    chain_data[:, 23] = chains["logsfr_ratios_4"][:]
    chain_data[:, 24] = chains["logsfr_ratios_5"][:]
    chain_data[:, 25] = chains["logsfr_ratios_6"][:]

    return idx, objid, chain_data


def process_chunk(indexed_chunk, dir_indiv, n_samples, n_params):
    """Worker: process a whole chunk; returns (results, errors)."""
    results, errors = [], []
    for entry in indexed_chunk:
        try:
            #results.append(process_single_file(entry, dir_indiv))
            results.append(process_single_file_h5(entry, dir_indiv))
        except Exception as e:
            idx, objid, filename = entry
            # Fill with NaN for failed files
            results.append((idx, objid, np.full((n_samples, n_params), np.nan, dtype=np.float32)))
            errors.append(f"{filename}: {e}")
    return results, errors

def write_results(results, chains_ds, objids_ds):
    """Single-writer: place each result at its final index (preserve input order)."""
    for idx, objid, chain_data in results:
        chains_ds[idx] = chain_data
        objids_ds[idx] = objid
    return len(results)

# ----------------------
# Main
# ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=str, default='phisfh')
    parser.add_argument('--dir_indiv', type=str, default='chains_parrot')
    parser.add_argument('--dir_collected', type=str, default='results')
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--chunk_size', type=int, default=25,
                        help='files per chunk for parallel processing')
    parser.add_argument('--io_buffer', type=int, default=10,
                        help='number of chunks to keep in-flight')
    args = parser.parse_args()

    n_workers = args.n_workers or min(cpu_count(), 64)

    keys = ['zred', 'total_mass', 'stellar_mass', 'logzsol', 'mwa',
            'sfr10', 'sfr30', 'sfr100', 'ssfr10', 'ssfr30', 'ssfr100',
            'dust2', 'dust_index', 'dust1_fraction', 'log_fagn', 'log_agn_tau',
            'gas_logz', 'duste_qpah', 'duste_umin', 'log_duste_gamma',
            'logsfr_ratios_1', 'logsfr_ratios_2', 'logsfr_ratios_3',
            'logsfr_ratios_4', 'logsfr_ratios_5', 'logsfr_ratios_6']

    os.makedirs(args.dir_collected, exist_ok=True)
    sname = os.path.join(args.dir_collected, f'chains_{args.prior}.h5')

    print(f"Output file: {sname}")
    print(f"Using {n_workers} workers with chunk size {args.chunk_size}")

    # Discover files (keep your original pattern, preserving sorted order)
    '''
    all_files = sorted([f for f in os.listdir(args.dir_indiv)
                        if f.endswith(f'unw_{args.prior}.npz')])
    if not all_files:
        raise RuntimeError(f"No files found in {args.dir_indiv} matching '*unw_{args.prior}.npz'")
    '''
    all_files = sorted([f for f in os.listdir(args.dir_indiv)
                        if f.endswith(f'unw_{args.prior}.h5')])
    if not all_files:
        raise RuntimeError(f"No files found in {args.dir_indiv} matching '*unw_{args.prior}.h5'")

    # Build list of (objid, filename) and an indexed version for exact ordering
    file_infos = [get_file_info(f) for f in all_files]  # (objid, filename)
    file_infos.sort(key=lambda x: x[0])
    indexed_infos = [(i, objid, fname) for i, (objid, fname) in enumerate(file_infos)]
    n_obj = len(indexed_infos)

    # Inspect shape from the first file
    '''
    sample_file = os.path.join(args.dir_indiv, all_files[0])
    with np.load(sample_file, allow_pickle=True) as sample_data:
        sample_chain = sample_data['chains'][()]
        n_samples = sample_chain['zred'].shape[0]
    '''
    sample_file = os.path.join(args.dir_indiv, all_files[0])
    with h5py.File(sample_file, "r") as f:
        n_samples = f["chains"]["zred"].shape[0]
    n_params = 26
    print(f"Processing {n_obj} files: {n_samples} samples × {n_params} parameters each")

    # Chunk the work for the executor
    chunks = [indexed_infos[i:i + args.chunk_size]
              for i in range(0, n_obj, args.chunk_size)]
    total_chunks = len(chunks)

    start_time = time.time()

    # Create HDF5 (single-writer, preallocated, input order rows)
    with h5py.File(sname, 'w', libver='latest') as h5f:
        h5f.create_dataset('theta_labels', data=np.array(keys, dtype='S'))
        chains_ds = h5f.create_dataset(
            'chains',
            shape=(n_obj, n_samples, n_params),
            dtype=np.float32,
            compression='gzip',
            chunks=(100, n_samples, n_params),  # or tune as needed
            shuffle=True,
            track_times=False
        )
        
        objid_chunk = max(1, min(1000, n_obj))
        
        objids_ds = h5f.create_dataset('objid', shape=(n_obj,), dtype=np.int64,
            compression='gzip', chunks=(1000,))

        files_processed = 0
        processed_chunks = 0

        # Rolling submission: keep up to io_buffer chunks in flight; refill as they complete
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            inflight = {}
            chunk_iter = iter(enumerate(chunks))

            # Prime the pipeline
            try:
                while len(inflight) < args.io_buffer:
                    idx, chunk = next(chunk_iter)
                    fut = executor.submit(process_chunk, chunk, args.dir_indiv, n_samples, n_params)
                    inflight[fut] = idx
            except StopIteration:
                pass

            # Consume results as soon as any chunk finishes
            while inflight:
                done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    chunk_idx = inflight.pop(fut)
                    try:
                        results, errors = fut.result()

                        # Log per-file read errors (if any)
                        for msg in errors:
                            print(f"Error processing {msg}")

                        written = write_results(results, chains_ds, objids_ds)
                        files_processed += written
                        processed_chunks += 1

                        if processed_chunks % 10 == 0 or processed_chunks == total_chunks:
                            elapsed = time.time() - start_time
                            rate = files_processed / elapsed if elapsed > 0 else 0.0
                            print(f"Completed chunk {processed_chunks}/{total_chunks}: "
                                  f"{files_processed} files ({rate:.1f} files/sec)")

                    except Exception as e:
                        print(f"Error consuming chunk {chunk_idx}: {e}")

                    # Top up the inflight queue
                    try:
                        idx, chunk = next(chunk_iter)
                        fut = executor.submit(process_chunk, chunk, args.dir_indiv, n_samples, n_params)
                        inflight[fut] = idx
                    except StopIteration:
                        pass

    total_time = time.time() - start_time
    print("\nCompleted!")
    print(f"Total objects: {files_processed}/{n_obj}")
    print(f"Total time: {total_time:.2f} seconds")
    rate = files_processed / total_time if total_time > 0 else 0.0
    print(f"Average rate: {rate:.1f} files/sec")
    print(f"Saved to: {sname}")

if __name__ == '__main__':
    main()
