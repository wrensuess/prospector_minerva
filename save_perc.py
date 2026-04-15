#!/usr/bin/env python3
"""collect percentiles from individual NPZ files into a single HDF5 file with multiprocessing"""
import os
import numpy as np
import argparse
import h5py
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import time
from astropy.table import Table

# ----------------------
# Helpers (top-level so they're pickleable)
# ----------------------

def get_file_info(file_path):
    """Get (objid, filename) from basename without loading data."""
    filename = os.path.basename(file_path)
    return int(filename.split('_')[1]), file_path  # (objid, filename)

def process_single_file(indexed_info, dir_indiv, catalog_ids, catalog_zspec):
    """
    Worker: load one NPZ and return all percentile fields.
    indexed_info = (idx, objid, filename)
    """
    idx, objid, filename = indexed_info
    full_path = os.path.join(dir_indiv, filename)

    with np.load(full_path, allow_pickle=True) as dat:
        perc = dat['percentiles'][()]
        zred_ml = dat['chain_ml'][0]

    # Lookup spec-z from catalog
    z_spec = catalog_zspec[catalog_ids == objid][0]

    # Return all relevant fields
    result = dict(
        objid=objid,
        zred=perc['zred'],
        total_mass=perc['total_mass'],
        stellar_mass=perc['stellar_mass'],
        met=perc['logzsol'],
        mwa=perc['mwa'],
        sfr10=perc['sfr'][0,:],
        sfr30=perc['sfr'][1,:],
        sfr100=perc['sfr'][2,:],
        ssfr10=perc['ssfr'][0,:],
        ssfr30=perc['ssfr'][1,:],
        ssfr100=perc['ssfr'][2,:],
        dust2=perc['dust2'],
        dust_index=perc['dust_index'],
        dust1_fraction=perc['dust1_fraction'],
        log_fagn=perc['log_fagn'],
        log_agn_tau=perc['log_agn_tau'],
        gas_logz=perc['gas_logz'],
        duste_qpah=perc['duste_qpah'],
        duste_umin=perc['duste_umin'],
        log_duste_gamma=perc['log_duste_gamma'],
        rest_UVJugi=perc['rest_UVJugi'],
        rest_UVJugi_map=perc['rest_UVJugi_map'],
        rest_UVJugi_colors=perc['rest_UVJugi_colors'],
        rest_UVJugi_colors_map=perc['rest_UVJugi_colors_map'],
        rest_gz=perc['rest_gz'],
        rest_gz_map=perc['rest_gz_map'],
        rest_gz_colors=perc['rest_gz_colors'],
        rest_gz_colors_map=perc['rest_gz_colors_map'],
        rest_NUVrJ=perc['rest_NUVrJ'],
        rest_NUVrJ_map=perc['rest_NUVrJ_map'],
        rest_NUVrJ_colors=perc['rest_NUVrJ_colors'],
        rest_NUVrJ_colors_map=perc['rest_NUVrJ_colors_map'],
        zred_ml=zred_ml,
        zred_spec=z_spec
    )
    return idx, result

def process_single_file_h5(indexed_info, dir_indiv, catalog_ids, catalog_zspec):
    idx, objid, filename = indexed_info
    full_path = os.path.join(dir_indiv, filename)

    with h5py.File(full_path, "r") as f:
        perc = f["percentiles"]

        def read_any(name):
            ds = perc[name]
            return ds[()] if ds.shape == () else ds[:]

        # chain_ml
        zred_ml = f["chain_ml"][0]

        match = (catalog_ids == objid)
        z_spec = catalog_zspec[match][0] if np.any(match) else -99.0

        sfr = read_any('sfr')
        ssfr = read_any('ssfr')

        result = dict(
            objid=objid,

            zred=read_any('zred'),
            total_mass=read_any('total_mass'),
            stellar_mass=read_any('stellar_mass'),
            met=read_any('logzsol'),
            mwa=read_any('mwa'),

            sfr10=sfr[0, :],
            sfr30=sfr[1, :],
            sfr100=sfr[2, :],

            ssfr10=ssfr[0, :],
            ssfr30=ssfr[1, :],
            ssfr100=ssfr[2, :],

            dust2=read_any('dust2'),
            dust_index=read_any('dust_index'),
            dust1_fraction=read_any('dust1_fraction'),
            log_fagn=read_any('log_fagn'),
            log_agn_tau=read_any('log_agn_tau'),
            gas_logz=read_any('gas_logz'),
            duste_qpah=read_any('duste_qpah'),
            duste_umin=read_any('duste_umin'),
            log_duste_gamma=read_any('log_duste_gamma'),

            rest_UVJugi=read_any('rest_UVJugi'),
            rest_UVJugi_map=read_any('rest_UVJugi_map'),
            rest_UVJugi_colors=read_any('rest_UVJugi_colors'),
            rest_UVJugi_colors_map=read_any('rest_UVJugi_colors_map'),

            rest_gz=read_any('rest_gz'),
            rest_gz_map=read_any('rest_gz_map'),
            rest_gz_colors=read_any('rest_gz_colors'),
            rest_gz_colors_map=read_any('rest_gz_colors_map'),

            rest_NUVrJ=read_any('rest_NUVrJ'),
            rest_NUVrJ_map=read_any('rest_NUVrJ_map'),
            rest_NUVrJ_colors=read_any('rest_NUVrJ_colors'),
            rest_NUVrJ_colors_map=read_any('rest_NUVrJ_colors_map'),

            zred_ml=zred_ml,
            zred_spec=z_spec
        )

    return idx, result

def process_chunk(indexed_chunk, dir_indiv, catalog_ids, catalog_zspec):
    """Worker: process a whole chunk; returns (results, errors)."""
    results, errors = [], []
    for entry in indexed_chunk:
        try:
            #results.append(process_single_file(entry, dir_indiv, catalog_ids, catalog_zspec))
            results.append(process_single_file_h5(entry, dir_indiv, catalog_ids, catalog_zspec))
        except Exception as e:
            idx, objid, filename = entry
            errors.append(f"{filename}: {e}")
    return results, errors

def write_results(results, datasets):
    """Single-writer: place each result at its final index (preserve input order)."""
    for idx, res in results:
        for key, ds in datasets.items():
            ds[idx] = res[key]
    return len(results)

# ----------------------
# Main
# ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=str, default='phisfh')
    parser.add_argument('--dir_indiv', type=str, default='chains_parrot')
    parser.add_argument('--dir_collected', type=str, default='results')
    parser.add_argument('--catalog_path', type=str, default="UNCOVER_v5.0.1_LW_SUPER_CATALOG.fits")
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--chunk_size', type=int, default=25)
    parser.add_argument('--io_buffer', type=int, default=10)
    args = parser.parse_args()

    n_workers = args.n_workers or min(cpu_count(), 64)
    os.makedirs(args.dir_collected, exist_ok=True)
    sname = os.path.join(args.dir_collected, f'quant_{args.prior}.h5')
    print(f"Output file: {sname}")
    print(f"Using {n_workers} workers with chunk size {args.chunk_size}")

    # Load catalog
    cat = Table.read(args.catalog_path)
    catalog_ids = cat['id'].data
    catalog_zspec = cat['z_spec'].data

    # Discover files
    '''
    all_files = sorted([f for f in os.listdir(args.dir_indiv) if f.endswith(f'perc_{args.prior}.npz')])
    if not all_files:
        raise RuntimeError(f"No files found in {args.dir_indiv} matching '*perc_{args.prior}.npz'")
    '''
    all_files = sorted([f for f in os.listdir(args.dir_indiv) if f.endswith(f'perc_{args.prior}.h5')])
    if not all_files:
        raise RuntimeError(f"No files found in {args.dir_indiv} matching '*perc_{args.prior}.h5'")
    
    n_obj = len(all_files)

    # Inspect first file to get shapes
    '''
    sample_file = os.path.join(args.dir_indiv, all_files[0])
    with np.load(sample_file, allow_pickle=True) as dat:
        perc = dat['percentiles'][()]
        rest_shapes = {k: v.shape for k, v in perc.items() if k.startswith('rest_') and 
                       not k.endswith('map')}
        map_shapes = {k: v.shape for k, v in perc.items() if k.endswith('map')}
    n_perc = perc['zred'].shape[0]
    '''
    sample_file = os.path.join(args.dir_indiv, all_files[0])
    with h5py.File(sample_file, "r") as f:
        perc = f["percentiles"]

        def shape_of(name):
            ds = perc[name]
            return ds.shape

        n_perc = perc["zred"].shape[0]

        rest_shapes = {k: shape_of(k) for k in perc.keys() if k.startswith("rest_") and not k.endswith("map")}
        map_shapes = {k: shape_of(k) for k in perc.keys() if k.endswith("map")}
    
    print(rest_shapes)

    # print('perc_sfr shape:', perc['sfr'].shape)
    # print('perc_ssfr shape:', perc['ssfr'].shape)
    # print('zphot shape:', perc['zred'].shape)    

    # Build indexed info
    file_infos = [get_file_info(f) for f in all_files]
    indexed_infos = [(i, objid, fname) for i, (objid, fname) in enumerate(file_infos)]

    # Pre-create HDF5 datasets
    with h5py.File(sname, 'w') as h5f:
        datasets = {}
        scalar_keys = ['objid','zred','total_mass','stellar_mass','met','mwa',
                       'sfr10','sfr30','sfr100','ssfr10','ssfr30','ssfr100',
                       'dust2','dust_index','dust1_fraction','log_fagn','log_agn_tau',
                       'gas_logz','duste_qpah','duste_umin','log_duste_gamma','zred_ml','zred_spec']
        for key in scalar_keys:
            if key == 'objid': 
                datasets[key] = h5f.create_dataset('objid', shape=(n_obj,), dtype=np.int64)
            elif key == 'zred_spec':
                datasets[key] = h5f.create_dataset('zred_spec', data=np.zeros(n_obj)-99., dtype=np.float32, 
                                                   compression='gzip', chunks=True)    # default -99 for no spec-z
            elif key == 'zred_ml':
                datasets[key] = h5f.create_dataset('zred_ml', data=np.zeros(n_obj)-99., dtype=np.float32, 
                                                   compression='gzip', chunks=True)
            else:    
                datasets[key] = h5f.create_dataset(key, shape=(n_obj,n_perc), dtype=np.float32,
                                                   compression='gzip', chunks=True)
        
        for key, shape in rest_shapes.items():
            datasets[key] = h5f.create_dataset(key, shape=(n_obj,shape[0],n_perc), dtype=np.float32,
                                              compression='gzip', chunks=(1,shape[0],n_perc))
        '''
        for key, shape in map_shapes.items():
            datasets[key] = h5f.create_dataset(key, shape=(n_obj,shape[0]), dtype=np.float32,
                                              compression='gzip', chunks=(1,shape[0]))
        '''
        for key, shape in map_shapes.items():
            if len(shape) == 0:   # scalar
                datasets[key] = h5f.create_dataset(key, shape=(n_obj,), dtype=np.float32, compression='gzip', chunks=True)
            else:
                datasets[key] = h5f.create_dataset(key, shape=(n_obj, *shape), dtype=np.float32, compression='gzip', chunks=(1, *shape))
            
        # Chunking
        chunks = [indexed_infos[i:i+args.chunk_size] for i in range(0, n_obj, args.chunk_size)]
        total_chunks = len(chunks)

        start_time = time.time()
        files_processed = 0
        processed_chunks = 0

        # Multiprocessing
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            inflight = {}
            chunk_iter = iter(enumerate(chunks))
            try:
                while len(inflight) < args.io_buffer:
                    idx, chunk = next(chunk_iter)
                    fut = executor.submit(process_chunk, chunk, args.dir_indiv, catalog_ids, catalog_zspec)
                    inflight[fut] = idx
            except StopIteration:
                pass

            while inflight:
                done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    chunk_idx = inflight.pop(fut)
                    try:
                        results, errors = fut.result()
                        for msg in errors:
                            print(f"Error processing {msg}")
                        written = write_results(results, datasets)
                        files_processed += written
                        processed_chunks += 1
                        if processed_chunks % 10 == 0 or processed_chunks == total_chunks:
                            elapsed = time.time() - start_time
                            rate = files_processed / elapsed if elapsed>0 else 0
                            print(f"Completed chunk {processed_chunks}/{total_chunks}: {files_processed} files ({rate:.1f} files/sec)")
                    except Exception as e:
                        print(f"Error consuming chunk {chunk_idx}: {e}")
                    try:
                        idx, chunk = next(chunk_iter)
                        fut = executor.submit(process_chunk, chunk, args.dir_indiv, catalog_ids, catalog_zspec)
                        inflight[fut] = idx
                    except StopIteration:
                        pass

    total_time = time.time() - start_time
    print("\nCompleted!")
    print(f"Total objects: {files_processed}/{n_obj}")
    print(f"Total time: {total_time:.2f} sec")
    print(f"Saved to: {sname}")

if __name__ == '__main__':
    main()
