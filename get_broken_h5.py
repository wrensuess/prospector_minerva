import h5py, glob, pdb
from tqdm import tqdm
import numpy as np

check_dir1 = "/scratch/alpine/ikmi3774/slurm/chains_parrot_COSMOS_n3.0_v1.0_ACS+WEBB_Kf444w_SUPER_spsbeta/COSMOS4/"
check_dir2 = "/scratch/alpine/ikmi3774/slurm/chains_parrot_COSMOS_n3.0_v1.0_ACS+WEBB_Kf444w_SUPER_spsbeta/COSMOS2/"
check_dir3 = "/scratch/alpine/ikmi3774/slurm/chains_parrot_COSMOS_n3.0_v1.0_ACS+WEBB_Kf444w_SUPER_spsbeta/COSMOS3/"
check_dir4 = "/scratch/alpine/ikmi3774/slurm/chains_parrot_COSMOS_n3.0_v1.0_ACS+WEBB_Kf444w_SUPER_spsbeta/COSMOS5/"
output_name = "chains_parrot_COSMOS_n3.0_v1.0_ACS+WEBB_Kf444w_SUPER_spsbeta/"

idlist1 = glob.glob(check_dir1+output_name+"id_*_mcmc_phisfh.h5")
idlist2 = glob.glob(check_dir2+output_name+"id_*_mcmc_phisfh.h5")
idlist3 = glob.glob(check_dir3+output_name+"id_*_mcmc_phisfh.h5")
idlist4 = glob.glob(check_dir4+output_name+"id_*_mcmc_phisfh.h5")
fittedid1 = np.array([int(a.split("/")[-1].split("_")[1]) for a in idlist1])
fittedid2 = np.array([int(a.split("/")[-1].split("_")[1]) for a in idlist2])
fittedid3 = np.array([int(a.split("/")[-1].split("_")[1]) for a in idlist3])
fittedid4 = np.array([int(a.split("/")[-1].split("_")[1]) for a in idlist4])
fittedid_all = np.concatenate([fittedid1,fittedid2,fittedid3,fittedid4])
fittedpath_all = idlist1+idlist2+idlist3+idlist4
fittedpathid_all = np.array([int(a.split("/")[-1].split("_")[1]) for a in fittedpath_all])

processed_dir = "/scratch/alpine/ikmi3774/slurm/postprocess_COSMOS_n3.0_v1.0/npz/"
post1 = glob.glob(processed_dir+"id_*_spec_phisfh.h5")
post2 = glob.glob(processed_dir+"id_*_perc_phisfh.h5")
post3 = glob.glob(processed_dir+"id_*_unw_phisfh.h5")
postid1 = np.array([int(a.split("/")[-1].split("_")[1]) for a in post1])
postid2 = np.array([int(a.split("/")[-1].split("_")[1]) for a in post2])
postid3 = np.array([int(a.split("/")[-1].split("_")[1]) for a in post3])

common_post = np.intersect1d(np.intersect1d(postid1, postid2), postid3)
all_unique = np.unique(np.concatenate([postid1, postid2, postid3]))
not_in_all = np.setdiff1d(all_unique, common_post)

print("[Info] all fitting ids in TOTAL:",len(fittedid_all))
print("[Info] post-process runned:",len(postid1))
print("[Info] post-process completed:",len(all_unique))
print("[Info] post-process interrupted?:",len(not_in_all))
print("[Info] Interrupted IDs:")
print(not_in_all)

pdb.set_trace()

'''
##########################################################
################### for fitting file #####################
##########################################################
common = np.intersect1d(fittedid_all, all_unique)
only_fitted = np.setdiff1d(fittedid_all, all_unique)

check_path = []
id_list = [int(a) for a in list(only_fitted)]
for k in range(0,len(id_list)):
    cid = np.where(id_list[k]==fittedpathid_all)[0][0]
    check_path.append(fittedpath_all[cid])

failed_ids = []

#for obj_id in id_list:
    #h5file = f"id_{obj_id}_spec_phisfh.h5"
for obj_id in tqdm(check_path):
    h5file = obj_id
    
    try:
        with h5py.File(h5file, "r") as f:
            list(f.keys())
        #print(f"{obj_id}: OK")
        
    except OSError as e:
        msg = str(e).lower()
        if "truncated file" in msg:
            print(f"{obj_id}: FAILED (truncated file)")
        elif "bad object header version number" in msg:
            print(f"{obj_id}: FAILED (bad object header)")
        else:
            print(f"{obj_id}: FAILED (other OSError)")
            print(f"    {e}")
        failed_ids.append(obj_id)

    except Exception as e:
        print(f"{obj_id}: FAILED (other Exception)")
        print(f"    {e}")

        failed_ids.append(obj_id)

print("[Info] Number of failed files:",len(failed_ids))
#print("[Info] Failed IDs:")
#print(failed_ids)
print("[Info] TOTAL-processed =",len(fittedid_all)-len(all_unique))

#np.save("./../phot_catalog/brokenid_MINERVA-COSMOS_n3.0_v1.0_ACS+WEBB_Kf444w_SUPER_CATALOG.txt",np.array(failed_ids))
'''

#################################################################
################### for post-processed file #####################
#################################################################
failed_ids1 = []
failed_ids2 = []
failed_ids3 = []

for j in tqdm(range(0,len(post1))):
    h5file1 = post1[j]
    h5file2 = post2[j]
    h5file3 = post3[j]
    
    try:
        with h5py.File(h5file1, "r") as f:
            list(f.keys())
    except OSError as e:
        msg = str(e).lower()
        if "truncated file" in msg:
            print(f"{obj_id}: FAILED (truncated file)")
        elif "bad object header version number" in msg:
            print(f"{obj_id}: FAILED (bad object header)")
        else:
            print(f"{obj_id}: FAILED (other OSError)")
            print(f"    {e}")
        failed_ids1.append(obj_id)
    except Exception as e:
        print(f"{obj_id}: FAILED (other Exception)")
        print(f"    {e}")

        failed_ids1.append(obj_id)

    try:
        with h5py.File(h5file2, "r") as f:
            list(f.keys())
    except OSError as e:
        msg = str(e).lower()
        if "truncated file" in msg:
            print(f"{obj_id}: FAILED (truncated file)")
        elif "bad object header version number" in msg:
            print(f"{obj_id}: FAILED (bad object header)")
        else:
            print(f"{obj_id}: FAILED (other OSError)")
            print(f"    {e}")
        failed_ids2.append(obj_id)
    except Exception as e:
        print(f"{obj_id}: FAILED (other Exception)")
        print(f"    {e}")

        failed_ids2.append(obj_id)

    try:
        with h5py.File(h5file3, "r") as f:
            list(f.keys())
    except OSError as e:
        msg = str(e).lower()
        if "truncated file" in msg:
            print(f"{obj_id}: FAILED (truncated file)")
        elif "bad object header version number" in msg:
            print(f"{obj_id}: FAILED (bad object header)")
        else:
            print(f"{obj_id}: FAILED (other OSError)")
            print(f"    {e}")
        failed_ids3.append(obj_id)
    except Exception as e:
        print(f"{obj_id}: FAILED (other Exception)")
        print(f"    {e}")

        failed_ids3.append(obj_id)

print("[Info] Number of failed spec files:",len(failed_ids1))
print("[Info] Number of failed perc files:",len(failed_ids2))
print("[Info] Number of failed unw files:",len(failed_ids3))

np.save("./../phot_catalog/brokenspec_MINERVA-COSMOS_n3.0_v1.0_ACS+WEBB_Kf444w_SUPER_CATALOG.txt",np.array(failed_ids1))
np.save("./../phot_catalog/brokenperc_MINERVA-COSMOS_n3.0_v1.0_ACS+WEBB_Kf444w_SUPER_CATALOG.txt",np.array(failed_ids2))
np.save("./../phot_catalog/brokenunw_MINERVA-COSMOS_n3.0_v1.0_ACS+WEBB_Kf444w_SUPER_CATALOG.txt",np.array(failed_ids3))
