import numpy as np
import glob
from astropy.table import Table
import utils as ut_cwd

ver = "COSMOS_n3.0_v1.0"
ver = "EGS_n2.0_v1.3"

### first large calculation
cat_dir = "/projects/ikmi3774/minerva_sps_git/stellar_pop_catalog_bb/phot_catalog/"
cat_ori1 = "fitid1_MINERVA-"+ver+"_ACS+WEBB_Kf444w_SUPER_CATALOG.txt"
cat_ori2 = "fitid2_MINERVA-"+ver+"_ACS+WEBB_Kf444w_SUPER_CATALOG.txt"
cat_ori3 = "fitid3_MINERVA-"+ver+"_ACS+WEBB_Kf444w_SUPER_CATALOG.txt"

check_dir1 = "/scratch/alpine/ikmi3774/slurm/chains_parrot_"+ver+"_ACS+WEBB_Kf444w_SUPER_spsbeta/EGS1/"
check_dir2 = "/scratch/alpine/ikmi3774/slurm/chains_parrot_"+ver+"_ACS+WEBB_Kf444w_SUPER_spsbeta/EGS2/"
check_dir3 = "/scratch/alpine/ikmi3774/slurm/chains_parrot_"+ver+"_ACS+WEBB_Kf444w_SUPER_spsbeta/EGS3/"
output_name = "chains_parrot_"+ver+"_ACS+WEBB_Kf444w_SUPER_spsbeta/"

fitid1 = np.loadtxt(cat_dir+cat_ori1)
fitid2 = np.loadtxt(cat_dir+cat_ori2)
fitid3 = np.loadtxt(cat_dir+cat_ori3)
fitid_all = np.concatenate([fitid1,fitid2,fitid3])
fitid_unique = np.unique(fitid_all)
print("[Info] overlap check:",len(fitid_unique),"/",len(fitid_all),"= 1?")

idlist1 = glob.glob(check_dir1+output_name+"id_*_mcmc_phisfh.h5")
idlist2 = glob.glob(check_dir2+output_name+"id_*_mcmc_phisfh.h5")
idlist3 = glob.glob(check_dir3+output_name+"id_*_mcmc_phisfh.h5")
fittedid1 = np.array([int(a.split("/")[-1].split("_")[1]) for a in idlist1])
fittedid2 = np.array([int(a.split("/")[-1].split("_")[1]) for a in idlist2])
fittedid3 = np.array([int(a.split("/")[-1].split("_")[1]) for a in idlist3])
fittedid_all = np.concatenate([fittedid1,fittedid2,fittedid3])

common1 = np.intersect1d(fittedid1, fitid1)
only_fitted1 = np.setdiff1d(fittedid1, fitid1)
only_fit1 = np.setdiff1d(fitid1, fittedid1)

common2 = np.intersect1d(fittedid2, fitid2)
only_fitted2 = np.setdiff1d(fittedid2, fitid2)
only_fit2 = np.setdiff1d(fitid2, fittedid2)

common3 = np.intersect1d(fittedid3, fitid3)
only_fitted3 = np.setdiff1d(fittedid3, fitid3)
only_fit3 = np.setdiff1d(fitid3, fittedid3)

common_all = np.intersect1d(fittedid_all, fitid_all)
only_fitted_all = np.setdiff1d(fittedid_all, fitid_all)
only_fit_all = np.setdiff1d(fitid_all, fittedid_all)

print("[Info] all fitting ids in "+check_dir1)
print("=====>",len(fitid1))
print("[Info] N fitting completed")
print("=====>",len(fittedid1))
print("overlap check:",len(common1))
print("only fitting (=remaining):",len(only_fit1))
print("only fitted:",len(only_fitted1))

print("[Info] all fitting ids in "+check_dir2)
print("=====>",len(fitid2))
print("[Info] N fitting completed")
print("=====>",len(fittedid2))
print("overlap check:",len(common2))
print("only fitting (=remaining):",len(only_fit2))
print("only fitted:",len(only_fitted2))

print("[Info] all fitting ids in "+check_dir3)
print("=====>",len(fitid3))
print("[Info] N fitting completed")
print("=====>",len(fittedid3))
print("overlap check:",len(common3))
print("only fitting (=remaining):",len(only_fit3))
print("only fitted:",len(only_fitted3))

print("[Info] all fitting ids in TOTAL")
print("=====>",len(fitid_all))
print("[Info] N fitting completed")
print("=====>",len(fittedid_all))
print("overlap check:",len(common_all))
print("only fitting (=remaining):",len(only_fit_all))
print("only fitted:",len(only_fitted_all))

#print(only_fit_all[:100])
#print(fitid1[:100])
#np.savetxt(cat_dir+"refitid_MINERVA-"+ver+"_ACS+WEBB_Kf444w_SUPER_CATALOG.txt",only_fit_all)
pdb.set_trace()

### no phot object?
cat = Table.read(cat_dir+"MINERVA-"+ver+"_ACS+WEBB_Kf444w_SUPER_CATALOG.fits")

id_list = only_fit_all

all_filts = [f[2:] for f in cat.colnames if f.startswith("f_f")]
filter_dict = ut_cwd.filter_dictionary(all_filts)
filts = list(filter_dict.keys())

n_nophot = []
for objid in id_list:
    match = np.where(cat["id"] == objid)[0]
    if len(match) == 0:
        print(f"{objid}: ID not found")
        continue
    idx = match[0]
    flux = np.array([np.ma.filled(cat[f"f_{f}"][idx], np.nan) for f in filts], dtype=float)
    err = np.array([np.ma.filled(cat[f"e_{f}"][idx], np.nan) for f in filts], dtype=float)
    n_valid = np.sum(np.isfinite(flux) & np.isfinite(err) & (err > 0))
    print(f"{objid}: {'NO PHOT' if n_valid == 0 else 'OK'} ({n_valid} bands)")
    
    if n_valid == 0:
        n_nophot.append(objid)
        print(f"{objid}: NO PHOT")
print(f"\nNO PHOT: {len(n_nophot)} / {len(id_list)} ({len(n_nophot) / len(id_list) * 100:.1f}%)")

print(n_nophot)

'''
### remaining sources
cat_ori4 = "refitid_MINERVA-COSMOS_n3.0_v1.0_ACS+WEBB_Kf444w_SUPER_CATALOG.txt"
check_dir4 = "/scratch/alpine/ikmi3774/slurm/chains_parrot_COSMOS_n3.0_v1.0_ACS+WEBB_Kf444w_SUPER_spsbeta/COSMOS5/"
fitid4 = np.loadtxt(cat_dir+cat_ori4)
idlist4 = glob.glob(check_dir4+output_name+"id_*_mcmc_phisfh.h5")
fittedid4 = np.array([int(a.split("/")[-1].split("_")[1]) for a in idlist4])
fittedid_all = np.concatenate([fittedid1,fittedid2,fittedid3,fittedid4])
fittedid_unique = np.unique(fittedid_all)
print("[Info] overlap check:",len(fittedid_unique),"/",len(fittedid_all),"= 1?")

common4 = np.intersect1d(fittedid4, fitid4)
only_fitted4 = np.setdiff1d(fittedid4, fitid4)
only_fit4 = np.setdiff1d(fitid4, fittedid4)

common_all = np.intersect1d(fittedid_all, fitid_all)
only_fitted_all = np.setdiff1d(fittedid_all, fitid_all)
only_fit_all = np.setdiff1d(fitid_all, fittedid_all)

print("[Info] all fitting ids in "+check_dir4)
print("=====>",len(fitid4))
print("[Info] N fitting completed")
print("=====>",len(fittedid4))
print("overlap check:",len(common4))
print("only fitting (=remaining):",len(only_fit4))
print("only fitted:",len(only_fitted4))

print("[Info] all fitting ids in TOTAL")
print("=====>",len(fitid_all))
print("[Info] N fitting completed")
print("=====>",len(fittedid_all))
print("overlap check:",len(common_all))
print("only fitting (=remaining):",len(only_fit_all))
print("only fitted:",len(only_fitted_all))
'''
