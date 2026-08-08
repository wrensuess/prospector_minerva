#!/usr/bin/env python3
import os
import re
import glob
import argparse
import numpy as np
import h5py

### what we need to execute
#python combine_chain.py --dir_split /scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2/chain_split --output /scratch/alpine/ikmi3774/slurm/postprocess_UDS_n3.0_v1.2/chains_phisfh.h5

#python combine_chain.py --dir_split /scratch/alpine/ikmi3774/slurm/postprocess_COSMOS_n3.0_v1.0/chain_split --output /scratch/alpine/ikmi3774/slurm/postprocess_COSMOS_n3.0_v1.0/chains_phisfh.h5

def get_start_end(fname):
    """
    chains_phisfh_0_500.h5 -> (0, 500)
    """
    base = os.path.basename(fname)
    m = re.search(r"_(\d+)_(\d+)\.h5$", base)
    if m is None:
        raise ValueError(f"Cannot parse start/end from filename: {base}")
    return int(m.group(1)), int(m.group(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=str, default="phisfh")
    parser.add_argument("--dir_split", type=str, default="chain_split")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--compression", type=str, default="gzip")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.dir_split, f"chains_{args.prior}.h5")

    pattern = os.path.join(args.dir_split, f"chains_{args.prior}_*_*.h5")
    files = sorted(glob.glob(pattern), key=lambda f: get_start_end(f)[0])

    if len(files) == 0:
        raise RuntimeError(f"No split files found: {pattern}")

    print(f"Found {len(files)} split files")

    # check shapes and total size
    total_nobj = 0
    n_samples = None
    n_params = None
    theta_labels = None

    file_info = []

    for ff in files:
        start, end = get_start_end(ff)

        with h5py.File(ff, "r") as f:
            n_obj_i = f["chains"].shape[0]
            n_samples_i = f["chains"].shape[1]
            n_params_i = f["chains"].shape[2]
            print(ff.split("/")[-1],n_obj_i,list(f.keys()))

            if n_samples is None:
                n_samples = n_samples_i
                n_params = n_params_i
                theta_labels = f["theta_labels"][:]
            else:
                if n_samples_i != n_samples or n_params_i != n_params:
                    raise ValueError(f"Shape mismatch in {ff}")

        file_info.append((start, end, ff, n_obj_i))
        total_nobj += n_obj_i

    print(f"Total objects: {total_nobj}")
    print(f"Shape: ({total_nobj}, {n_samples}, {n_params})")
    print(f"Output: {args.output}")

    if os.path.exists(args.output):
        raise RuntimeError(f"Output already exists: {args.output}")

    objid_chunk = max(1, min(1000, total_nobj))

    with h5py.File(args.output, "w", libver="latest") as out:
        out.create_dataset("theta_labels", data=theta_labels)

        chains_out = out.create_dataset(
            "chains",
            shape=(total_nobj, n_samples, n_params),
            dtype=np.float32,
            compression=args.compression,
            shuffle=True,
            chunks=(objid_chunk, n_samples, n_params),
            track_times=False,
        )

        objid_out = out.create_dataset(
            "objid",
            shape=(total_nobj,),
            dtype=np.int64,
            compression=args.compression,
            chunks=(objid_chunk,),
        )

        pos = 0

        for i, (start, end, ff, n_obj_i) in enumerate(file_info):
            with h5py.File(ff, "r") as f:
                chains = f["chains"][:]
                objid = f["objid"][:]

            chains_out[pos:pos+n_obj_i] = chains
            objid_out[pos:pos+n_obj_i] = objid

            print(
                f"{i+1}/{len(file_info)}: "
                f"{os.path.basename(ff)} -> rows {pos}:{pos+n_obj_i}"
            )

            pos += n_obj_i

    print("Done.")


if __name__ == "__main__":
    main()
