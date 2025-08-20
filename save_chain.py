"""Ultra-fast version for supercomputers using memory mapping and parallel HDF5"""
import os
import numpy as np
import argparse
import h5py
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import time

# ----------------------
# Worker-safe functions
# ----------------------

def get_file_info(file_path):
    """Quick function to get objid from filename without loading data"""
    filename = os.path.basename(file_path)
    return int(filename.split('_')[1]), file_path  # (objid, filename only)

def process_single_file(file_info, dir_indiv):
    """Process a single file and return minimal data structure"""
    objid, filename = file_info
    full_path = os.path.join(dir_indiv, filename)

    try:
        with np.load(full_path, allow_pickle=True) as dat:
            chains = dat['chains'][()]

        n_samples = chains['zred'].shape[0]
        chain_data = np.empty((n_samples, 26), dtype=np.float32)

        # Fill columns
        chain_data[:, 0] = chains['zred']
        chain_data[:, 1] = chains['total_mass']
        chain_data[:, 2] = chains['stellar_mass']
        chain_data[:, 3] = chains['logzsol']
        chain_data[:, 4] = chains['mwa']
        chain_data[:, 5:8] = chains['sfr']       # or [:,5]=chains['sfr10']; etc.
        chain_data[:, 8:11] = chains['ssfr']     # or per-key if stored separately
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

        return objid, chain_data

    except Exception as e:
        return None, f"{filename}: {e}"

def process_chunk(chunk, dir_indiv):
    """Top-level function submitted to the pool (pickleable)."""
    out = []
    errors = []
    for fi in chunk:
        objid, data_or_err = process_single_file(fi, dir_indiv)
        if objid is None:
            errors.append(data_or_err)  # string with context
        else:
            out.append((objid, data_or_err))
    return out, errors

def write_chunk_to_hdf5(results, chunk, chains_ds, objids_array, index_map):
    """Write results back into their original input order."""
    n_written = 0
    for (objid, chain_data), fi in zip(results, chunk):
        idx = index_map[fi]
        chains_ds[idx] = chain_data
        objids_array[idx] = objid
        n_written += 1
    return n_written

# ----------------------
# Driver
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
                        help='number of chunks to buffer in memory')
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

    # Discover files
    all_files = sorted([f for f in os.listdir(args.dir_indiv)
                        if f.endswith(f'unw_{args.prior}.npz')])
    if not all_files:
        raise RuntimeError(f"No files found in {args.dir_indiv} for prior '{args.prior}'")

    file_infos = [get_file_info(f) for f in all_files]
    n_obj = len(file_infos)

    # Make index map for exact ordering
    index_map = {fi: i for i, fi in enumerate(file_infos)}

    # Determine sample sizes
    sample_file = os.path.join(args.dir_indiv, all_files[0])
    with np.load(sample_file, allow_pickle=True) as sample_data:
        sample_chain = sample_data['chains'][()]
        n_samples = sample_chain['zred'].shape[0]

    n_params = len(keys)
    print(f"Processing {n_obj} files: {n_samples} samples × {n_params} parameters each")

    # Chunk the work
    chunks = [file_infos[i:i + args.chunk_size]
              for i in range(0, len(file_infos), args.chunk_size)]

    start_time = time.time()

    # HDF5 creation (preallocated)
    with h5py.File(sname, 'w') as h5f:
        h5f.create_dataset('theta_labels', data=np.array(keys, dtype='S'))

        chunk_shape = (min(args.chunk_size, n_obj), n_samples, n_params)
        chains_ds = h5f.create_dataset(
            'chains',
            shape=(n_obj, n_samples, n_params),
            dtype=np.float32,
            compression='lzf',
            chunks=chunk_shape,
            shuffle=True,
            track_times=False
        )
        objids_array = h5f.create_dataset(
            'objid',
            shape=(n_obj,),
            dtype=np.int32
        )

        files_processed = 0
        total_chunks = len(chunks)

        # Rolling submission/consumption
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            inflight = {}
            chunk_iter = iter(enumerate(chunks))

            # Prime the pump
            try:
                while len(inflight) < args.io_buffer:
                    idx, chunk = next(chunk_iter)
                    fut = executor.submit(process_chunk, chunk, args.dir_indiv)
                    inflight[fut] = (idx, chunk)
            except StopIteration:
                pass

            processed_chunks = 0
            while inflight:
                done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    chunk_idx, chunk = inflight.pop(fut)
                    try:
                        (results, errors) = fut.result()

                        for msg in errors:
                            print(f"Error processing {msg}")

                        written = write_chunk_to_hdf5(
                            results, chunk, chains_ds, objids_array, index_map
                        )
                        files_processed += written
                        processed_chunks += 1

                        if processed_chunks % 10 == 0 or processed_chunks == total_chunks:
                            elapsed = time.time() - start_time
                            rate = files_processed / elapsed if elapsed > 0 else 0.0
                            print(f'Completed chunk {processed_chunks}/{total_chunks}: '
                                  f'{files_processed} files ({rate:.1f} files/sec)')

                    except Exception as e:
                        print(f"Error consuming chunk {chunk_idx}: {e}")

                    # Top up
                    try:
                        idx, chunk = next(chunk_iter)
                        fut = executor.submit(process_chunk, chunk, args.dir_indiv)
                        inflight[fut] = (idx, chunk)
                    except StopIteration:
                        pass

    total_time = time.time() - start_time
    print('\nCompleted!')
    print(f'Total objects: {files_processed}/{n_obj}')
    print(f'Total time: {total_time:.2f} seconds')
    rate = files_processed / total_time if total_time > 0 else 0.0
    print(f'Average rate: {rate:.1f} files/sec')
    print(f'Saved to: {sname}')

if __name__ == '__main__':
    main()
