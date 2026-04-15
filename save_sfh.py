#!/usr/bin/env python3
"""collect SFH data from individual NPZ files into a single HDF5 file"""
import os
import numpy as np
import argparse
import h5py
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import time

# ----------------------
# Helpers
# ----------------------

def get_file_info(file_path):
    """Get (objid, filename) from basename without loading data."""
    filename = os.path.basename(file_path)
    return int(filename.split('_')[1]), file_path  # (objid, filename)

def process_single_file(indexed_info, dir_indiv, perc):
    """
    Worker: load one NPZ and return (index, objid, agebins_max, sfh_q).
    indexed_info = (idx, objid, filename)
    """
    idx, objid, filename = indexed_info
    full_path = os.path.join(dir_indiv, filename)

    with np.load(full_path, allow_pickle=True) as dat:
        chains = dat['chains'][()]

    agebins_max = chains['agebins_max']  
    sfh_q = np.quantile(chains['sfh'], perc, axis=0)  # shape (n_percentiles, n_bins)

    return idx, objid, agebins_max, sfh_q

def process_single_file_h5(indexed_info, dir_indiv, perc):
    """
    Worker: load one HDF5 and return (index, objid, agebins_max, sfh_q).
    indexed_info = (idx, objid, filename)
    """
    idx, objid, filename = indexed_info
    full_path = os.path.join(dir_indiv, filename)

    with h5py.File(full_path, "r") as f:
        chains = f["chains"]

        # scalar/array
        def read_any(name):
            ds = chains[name]
            return ds[()] if ds.shape == () else ds[:]

        agebins_max = read_any("agebins_max")

        sfh = read_any("sfh")   # shape (nsamp, nbins)
        sfh_q = np.quantile(sfh, perc, axis=0)

    return idx, objid, agebins_max, sfh_q

def process_chunk(indexed_chunk, dir_indiv, perc, n_percentiles, n_bins):
    results, errors = [], []
    for entry in indexed_chunk:
        try:
            results.append(process_single_file(entry, dir_indiv, perc))
        except Exception as e:
            idx, objid, filename = entry
            results.append((idx, objid, np.nan, np.full((n_percentiles, n_bins), np.nan, dtype=np.float32)))
            errors.append(f"{filename}: {e}")
    return results, errors

def write_results(results, objids_ds, tmax_ds, sfh_ds):
    """Single-writer: place each result at its final index (preserve input order)."""
    for idx, objid, agebins_max, sfh_q in results:
        objids_ds[idx] = objid
        tmax_ds[idx] = agebins_max
        sfh_ds[idx] = sfh_q
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

    # Percentiles for SFH
    perc = np.array([0.1, 2.3, 15.9, 50, 84.1, 97.7, 99.9]) * 0.01
    n_percentiles = len(perc)

    os.makedirs(args.dir_collected, exist_ok=True)
    sname = os.path.join(args.dir_collected, f'sfh_{args.prior}.h5')

    print(f"Output file: {sname}")
    print(f"Using {n_workers} workers with chunk size {args.chunk_size}")

    # Discover files
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
    file_infos.sort(key=lambda x: x[0])  # Sort by objid
    indexed_infos = [(i, objid, fname) for i, (objid, fname) in enumerate(file_infos)]
    n_obj = len(indexed_infos)

    # Inspect shape from the first file
    '''
    sample_file = os.path.join(args.dir_indiv, all_files[0])
    with np.load(sample_file, allow_pickle=True) as sample_data:
        sample_chain = sample_data['chains'][()]
        n_bins = sample_chain['sfh'].shape[1]
    '''
    sample_file = os.path.join(args.dir_indiv, all_files[0])
    with h5py.File(sample_file, "r") as f:
        n_bins = f["chains"]["sfh"].shape[1]

    print(f"Processing {n_obj} files with {n_bins} SFH bins, {n_percentiles} percentiles")

    # Chunk the work for the executor
    chunks = [indexed_infos[i:i + args.chunk_size]
              for i in range(0, n_obj, args.chunk_size)]
    total_chunks = len(chunks)

    start_time = time.time()

    # Create HDF5 (single-writer, preallocated, input order rows)
    with h5py.File(sname, 'w', libver='latest') as h5f:
        h5f.create_dataset('percentiles', data=perc)
        objids_ds = h5f.create_dataset('objid', shape=(n_obj,), dtype=np.int64)
        '''
        tmax_ds = h5f.create_dataset(
            'agebins_max',
            shape=(n_obj),
            dtype=np.float32,
            compression='gzip',
            chunks=(1000,),
            shuffle=True,
            track_times=False
        )
        '''
        objid_chunk = max(1, min(1000, n_obj))
        tmax_ds = h5f.create_dataset(
            'agebins_max',
            shape=(n_obj),
            dtype=np.float32,
            compression='gzip',
            chunks=(objid_chunk,),
            shuffle=True,
            track_times=False
        )
        
        sfh_ds = h5f.create_dataset(
            'sfh',
            shape=(n_obj, n_percentiles, n_bins),
            dtype=np.float32,
            compression='gzip',
            chunks=(1000, n_percentiles, n_bins),
            shuffle=True,
            track_times=False
        )

        files_processed = 0
        processed_chunks = 0

        # Rolling submission
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            inflight = {}
            chunk_iter = iter(enumerate(chunks))

            # Prime the pipeline
            try:
                while len(inflight) < args.io_buffer:
                    idx, chunk = next(chunk_iter)
                    fut = executor.submit(process_chunk, chunk, args.dir_indiv, perc, n_percentiles, n_bins)
                    inflight[fut] = idx
            except StopIteration:
                pass

            # Consume results
            while inflight:
                done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    chunk_idx = inflight.pop(fut)
                    try:
                        results, errors = fut.result()

                        # Log per-file read errors (if any)
                        for msg in errors:
                            print(f"Error processing {msg}")

                        written = write_results(results, objids_ds, tmax_ds, sfh_ds)
                        files_processed += written
                        processed_chunks += 1

                        if processed_chunks % 10 == 0 or processed_chunks == total_chunks:
                            elapsed = time.time() - start_time
                            rate = files_processed / elapsed if elapsed > 0 else 0.0
                            print(f"Completed chunk {processed_chunks}/{total_chunks}: "
                                  f"{files_processed} files ({rate:.1f} files/sec)")

                    except Exception as e:
                        print(f"Error consuming chunk {chunk_idx}: {e}")

                    # Top up inflight
                    try:
                        idx, chunk = next(chunk_iter)
                        fut = executor.submit(process_chunk, chunk, args.dir_indiv, perc, n_percentiles, n_bins)
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
