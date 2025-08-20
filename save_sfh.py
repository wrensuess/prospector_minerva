import os
import numpy as np
import argparse
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='phisfh', help='phisfh, phisfhzfixed')
parser.add_argument('--dir_indiv', type=str, default='chains_parrot', help='input folder storing chains')
parser.add_argument('--dir_collected', type=str, default='results', help='output folder storing unweighted chains and quantiles')
parser.add_argument('--nproc', type=int, default=16, help='number of parallel processes')
parser.add_argument('--chunk_size', type=int, default=100, help='files per chunk')
args = parser.parse_args()
print(args)

which_prior = args.prior

sname = os.path.join(args.dir_collected, f'sfh_{args.prior}.h5')
print('sfhs will be saved to', sname)

all_files = sorted([f for f in os.listdir(args.dir_indiv) if f.endswith(f'unw_{which_prior}.npz')])
n_obj = len(all_files)
print(f"Found {n_obj} files")

# percentiles for SFH
perc = np.array([0.1, 2.3, 15.9, 50, 84.1, 97.7, 99.9]) * 0.01
n_percentiles = len(perc)

# Check shape from first file
sample_file = os.path.join(args.dir_indiv, all_files[0])
sample_chain = np.load(sample_file, allow_pickle=True)['chains'][()]
n_bins = sample_chain['sfh'].shape[1]

# ---------- Worker function ----------
def process_chunk(chunk_files, dir_indiv, perc):
    objid_chunk = []
    agebins_chunk = []
    sfh_chunk = []

    for this_file in chunk_files:
        mid = int(this_file.split('_')[1])
        dat = np.load(os.path.join(dir_indiv, this_file), allow_pickle=True)
        chains = dat['chains'][()]
        objid_chunk.append(mid)
        agebins_chunk.append(chains['agebins_max'])
        sfh_chunk.append(np.quantile(chains['sfh'], perc, axis=0))

    return (
        np.array(objid_chunk, dtype=np.int32),
        np.array(agebins_chunk, dtype=np.float32),
        np.array(sfh_chunk, dtype=np.float32)
    )

# ---------- Main parallel loop ----------
with h5py.File(sname, 'w') as h5f:
    # Preallocate datasets
    dset_objid = h5f.create_dataset('objid', shape=(n_obj,), dtype=np.int32,
                                    compression='gzip', chunks=True)
    dset_agebins = h5f.create_dataset('agebins_max', shape=(n_obj,), dtype=np.float32,
                                      compression='gzip', chunks=True)
    dset_sfh = h5f.create_dataset('sfh', shape=(n_obj, n_percentiles, n_bins), dtype=np.float32,
                                  compression='gzip', chunks=(1, n_percentiles, n_bins))
    h5f.create_dataset('percentiles', data=perc)

    # Split work into chunks
    chunks = [all_files[i:i+args.chunk_size] for i in range(0, n_obj, args.chunk_size)]

    with ProcessPoolExecutor(max_workers=args.nproc) as executor:
        futures = {executor.submit(process_chunk, chunk, args.dir_indiv, perc): idx
                   for idx, chunk in enumerate(chunks)}

        offset = 0
        for fut in as_completed(futures):
            objid_chunk, agebins_chunk, sfh_chunk = fut.result()
            size = len(objid_chunk)

            # Write back in the correct place
            dset_objid[offset:offset+size] = objid_chunk
            dset_agebins[offset:offset+size] = agebins_chunk
            dset_sfh[offset:offset+size, :, :] = sfh_chunk

            offset += size
            if offset % 10000 == 0:
                print(f'Processed {offset} files')

print('length:', n_obj)
print('saved to', sname)
