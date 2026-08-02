import numpy as np
import glob

cat_dir = "/projects/ikmi3774/minerva_sps_git/stellar_pop_catalog_bb/phot_catalog/"
cat_ori1 = "fitid1_MINERVA-COSMOS_n3.0_v1.0_ACS+WEBB_Kf444w_SUPER_CATALOG.txt"
cat_ori2 = "fitid2_MINERVA-COSMOS_n3.0_v1.0_ACS+WEBB_Kf444w_SUPER_CATALOG.txt"
cat_ori3 = "fitid3_MINERVA-COSMOS_n3.0_v1.0_ACS+WEBB_Kf444w_SUPER_CATALOG.txt"

check_dir1 = "/scratch/alpine/ikmi3774/slurm/COSMOS4/"
check_dir2 = "/scratch/alpine/ikmi3774/slurm/COSMOS2/"
check_dir3 = "/scratch/alpine/ikmi3774/slurm/COSMOS3/"
output_name = "chains_parrot_COSMOS_n3.0_v1.0_ACS+WEBB_Kf444w_SUPER_spsbeta/"

fitid1 = np.loadtxt(cat_dir+cat_ori1)
fitid2 = np.loadtxt(cat_dir+cat_ori2)
fitid3 = np.loadtxt(cat_dir+cat_ori3)
fitid_all = np.concatenate([fitid1,fitid2,fitid3])
fitid_unique = np.unique(fitid_all)
print("[Info] overlap check:",len(fitid_unique),"/",len(fitid_all),"= 1?")

idlist1 = glob.glob(check_dir1+output_name+"id_*_mcmc_phisfh.h5")
idlist2 = glob.glob(check_dir1+output_name+"id_*_mcmc_phisfh.h5")
idlist3 = glob.glob(check_dir1+output_name+"id_*_mcmc_phisfh.h5")
fittedid1 = np.array([int(a.split("/")[-1].split("_")[1]) for a in idlist1])
fittedid2 = np.array([int(a.split("/")[-1].split("_")[1]) for a in idlist2])
fittedid3 = np.array([int(a.split("/")[-1].split("_")[1]) for a in idlist3])

print(type(fitid1[0]))
print(type(fittedid1[0]))

common1 = np.intersect1d(fittedid1, fitid1)
only_fitted1 = np.setdiff1d(fittedid1, fitid1)
only_fit1 = np.setdiff1d(fitid1, fittedid1)

common2 = np.intersect1d(fittedid2, fitid2)
only_fitted2 = np.setdiff1d(fittedid2, fitid2)
only_fit2 = np.setdiff1d(fitid2, fittedid2)

common3 = np.intersect1d(fittedid3, fitid3)
only_fitted3 = np.setdiff1d(fittedid3, fitid3)
only_fit3 = np.setdiff1d(fitid3, fittedid3)

print("[Info] all fitting ids in "+check_dir1)
print("=====>",len(fitid1))
print("overlap check:",len(common1))
print("only fitting (=remaining):",len(only_fit1))
print("only fitted:",len(only_fitted1))

print("[Info] all fitting ids in "+check_dir2)
print("=====>",len(fitid2))
print("overlap check:",len(common2))
print("only fitting (=remaining):",len(only_fit2))
print("only fitted:",len(only_fitted2))

print("[Info] all fitting ids in "+check_dir3)
print("=====>",len(fitid3))
print("overlap check:",len(common3))
print("only fitting (=remaining):",len(only_fit3))
print("only fitted:",len(only_fitted3))
