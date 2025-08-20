"""Ultra-fast version for supercomputers using memory mapping and parallel HDF5"""
import os
import numpy as np
import argparse
import h5py 
from multiprocessing import Pool, cpu_count, Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from functools import partial

def get_file_info(file_path):
    """Quick function to get objid from filename without loading data"""
    filename = os.path.basename(file_path)
    return int(filename.split('_')[1]), file_path

def process_single_file(file_info, dir_indiv):
    """Process a single file and return minimal data structure"""
    objid, filename = file_info
    full_path = os.path.join(dir_indiv, filename)
    
    try:
        # Use context manager for automatic cleanup
        with np.load(full_path, allow_pickle=True) as dat:
            chains = dat['chains'][()]
        
        # Pre-allocate and fill array more efficiently
        n_samples = chains['zred'].shape[0]
        chain_data = np.empty((n_samples, 26), dtype=np.float32)
        
        # Direct assignment is faster than stacking
        chain_data[:, 0] = chains['zred']
        chain_data[:, 1] = chains['total_mass'] 
        chain_data[:, 2] = chains['stellar_mass']
        chain_data[:, 3] = chains['logzsol']
        chain_data[:, 4] = chains['mwa']
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
        
        return objid, chain_data
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None, None

def write_chunk_to_hdf5(results, h5_file, chains_ds, objids_array, start_idx):
    """Write a chunk of results to HDF5 file"""
    valid_results = [(objid, data) for objid, data in results if data is not None]
    
    if not valid_results:
        return 0
    
    # Sort by objid for better data locality
    valid_results.sort(key=lambda x: x[0])
    
    end_idx = start_idx + len(valid_results)
    
    for i, (objid, chain_data) in enumerate(valid_results):
        chains_ds[start_idx + i] = chain_data
        objids_array[start_idx + i] = objid
    
    return len(valid_results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=str, default='phisfh')
    parser.add_argument('--dir_indiv', type=str, default='chains_parrot')
    parser.add_argument('--dir_collected', type=str, default='results')
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--chunk_size', type=int, default=500, 
                        help='files per chunk for parallel processing')
    parser.add_argument('--io_buffer', type=int, default=10,
                        help='number of chunks to buffer in memory')
    args = parser.parse_args()
    
    n_workers = args.n_workers or min(cpu_count(), 64)  # Cap at 64 for most systems
    
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

    # Get all file info
    all_files = sorted([f for f in os.listdir(args.dir_indiv) 
                       if f.endswith(f'unw_{args.prior}.npz')])
    
    file_infos = [get_file_info(f) for f in all_files]
    n_obj = len(file_infos)
    
    # Get dimensions from sample file
    sample_file = os.path.join(args.dir_indiv, all_files[0])
    with np.load(sample_file, allow_pickle=True) as sample_data:
        sample_chain = sample_data['chains'][()]
        n_samples = sample_chain['zred'].shape[0]
    
    n_params = len(keys)
    print(f"Processing {n_obj} files: {n_samples} samples × {n_params} parameters each")

    # Create chunks
    chunks = [file_infos[i:i + args.chunk_size] 
              for i in range(0, len(file_infos), args.chunk_size)]
    
    # Initialize HDF5 file with optimal settings for parallel access
    with h5py.File(sname, 'w') as h5f:
        h5f.create_dataset('theta_labels', data=np.array(keys, dtype='S'))
        
        # Optimized chunking for supercomputer storage
        chunk_shape = (min(args.chunk_size, n_obj), n_samples, n_params)
        chains_ds = h5f.create_dataset(
            'chains',
            shape=(n_obj, n_samples, n_params),
            dtype=np.float32,
            compression='lzf',  # Faster than gzip
            chunks=chunk_shape,
            shuffle=True,
            track_times=False  # Disable timestamp tracking for speed
        )
        
        objids_array = np.empty(n_obj, dtype=np.int32)
        
        start_time = time.time()
        files_processed = 0
        
        # Process chunks with controlled concurrency
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit initial batch of jobs
            future_to_chunk = {}
            
            for chunk_idx, chunk in enumerate(chunks):
                if len(future_to_chunk) < args.io_buffer:
                    future = executor.submit(
                        lambda c: [process_single_file(fi, args.dir_indiv) for fi in c], 
                        chunk
                    )
                    future_to_chunk[future] = chunk_idx
                else:
                    break
            
            remaining_chunks = chunks[len(future_to_chunk):]
            remaining_iter = iter(remaining_chunks)
            
            # Process results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                
                try:
                    results = future.get()
                    
                    # Write results to HDF5
                    chunk_start = chunk_idx * args.chunk_size
                    written = write_chunk_to_hdf5(
                        results, h5f, chains_ds, objids_array, chunk_start
                    )
                    
                    files_processed += written
                    
                    if chunk_idx % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = files_processed / elapsed if elapsed > 0 else 0
                        print(f'Completed chunk {chunk_idx + 1}/{len(chunks)}: '
                              f'{files_processed} files ({rate:.1f} files/sec)')
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx}: {e}")
                
                # Submit next job if available
                try:
                    next_chunk = next(remaining_iter)
                    next_chunk_idx = len(chunks) - len(remaining_chunks) + \
                                   (chunks.index(next_chunk) if next_chunk in chunks else -1)
                    
                    future = executor.submit(
                        lambda c: [process_single_file(fi, args.dir_indiv) for fi in c],
                        next_chunk
                    )
                    future_to_chunk[future] = len(chunks) - len(remaining_chunks) - 1
                    
                except StopIteration:
                    pass
        
        # Clean up objids array
        objids_array = objids_array[:files_processed]
        h5f.create_dataset('objid', data=objids_array)

    total_time = time.time() - start_time
    print(f'\nCompleted!')
    print(f'Total objects: {files_processed}')
    print(f'Total time: {total_time:.2f} seconds')
    print(f'Average rate: {files_processed/total_time:.1f} files/sec')
    print(f'Saved to: {sname}')

if __name__ == '__main__':
    main()