#!/usr/bin/env python3
import subprocess
import glob,time,os,pdb
import numpy as np
from datetime import datetime

CHECK_INTERVAL_HOURS = 6 #6 for first run, 1-2 for second run, 0.5 for later?
CHECK_INTERVAL_SEC = 3600*CHECK_INTERVAL_HOURS


def wait_job_finish():
    """
    squeue -j JOBID in CHECK_INTERVAL_HOURS, and check job is completed
    """
    user = os.environ.get("USER")
    if not user:
        raise RuntimeError("no USER in environmental variable")

    print(f"[INFO] User {user}' job check in {CHECK_INTERVAL_HOURS} hours")
    print("[INFO] Finish if no job. Ctrl+C for force cancel.")

    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            # execute squeue -u <user>
            res = subprocess.run(
                ["squeue", "-u", user],
                text=True,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"[{now}] squeue error: {e}")
            print(e.stdout)
            print(e.stderr)
            time.sleep(CHECK_INTERVAL_SEC)
            continue

        lines = res.stdout.strip().splitlines()

        # Nline = 1 :only header, no job, Nline = 2: header+sbatch chain_job.sh
        if len(lines) <= 2:
            print(f"[{now}] no job. finished.")
            # a few columns
            for line in lines:
                print("  ", line)
            break
        else:
            num_jobs = len(lines) - 1
            print(f"[{now}] {num_jobs} jobs now")
            # a few columns
            for line in lines[: min(5, len(lines))]:
                print("  ", line)
            if num_jobs > 4:
                print("  ...")

        print(f"[INFO] next check will be {CHECK_INTERVAL_HOURS} hours later\n")
        time.sleep(CHECK_INTERVAL_SEC)

def get_unfit_number(chaindir,catpath_check):
    ### check path is consistent with submit_loop.py
    fitted = glob.glob(chaindir+"id_*.h5")
    fitted_id = [a.split("/")[-1].split("_")[1] for a in fitted]

    whole_id_ = np.loadtxt(catpath_check)
    whole_id = [str(int(a)) for a in list(whole_id_)]

    not_fitted =  list(set(fitted_id)^set(whole_id))
    print("fitted:",len(fitted_id)/len(whole_id),"unfitted:",len(not_fitted)/len(whole_id))

    not_fitted_out = [int(a) for a in not_fitted]
    not_fitted_out.sort()

    return int(len(not_fitted)), np.array(not_fitted_out)

    

field = "EGS"
ver = "n2.0_v1.3"
cathead = "fitid1" #name of catalog for the first attempt, fitid or refitid 
cathead_ori = "fitid1" #name of reference catalog to check fitting is completed

spsver = 'spsbeta' #not set as a variable, need to edit submit_loop.py
outdir = '/scratch/alpine/ikmi3774/slurm/EGS1/'
catdir = '/projects/ikmi3774/minerva_sps_git/stellar_pop_catalog_bb/phot_catalog/'
catpath = catdir+cathead+"_MINERVA-"+field+"_"+ver+"_ACS+WEBB_Kf444w_SUPER_CATALOG.txt"
catpath_check = catdir+cathead_ori+"_MINERVA-"+field+"_"+ver+"_ACS+WEBB_Kf444w_SUPER_CATALOG.txt"
chaindir = outdir+'chains_parrot_{}_{}_ACS+WEBB_Kf444w_SUPER_{}/'.format(field, ver, spsver)

if "refitid" in cathead:
    refitids = np.loadtxt(catpath)
    print(refitids)
    #pdb.set_trace()
    for j in range(0,len(refitids)):
        print("rm "+chaindir+"id_"+str(int(refitids[j]))+"_mcmc_phisfh.h5")
        os.system("rm "+chaindir+"id_"+str(int(refitids[j]))+"_mcmc_phisfh.h5")

#subprocess.run(["python", "submit_loop.py", cathead, field, ver, outdir, catdir], check=True)
subprocess.Popen(["python", "submit_loop.py", cathead, field, ver, outdir, catdir])

time.sleep(CHECK_INTERVAL_SEC)
print(f"[INFO] start checking since run spent {CHECK_INTERVAL_HOURS} hours")

k=0
while True:
    print(f"[INFO] wait_job_finish")
    wait_job_finish()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] waited")
    Nresi, unfit_id_array = get_unfit_number(chaindir,catpath_check)
    if Nresi==0:
        break
    else:
        k=k+1
        cathead = cathead+str(int(k)) #name of catalog for k-th attempt
        catpath = catdir+cathead+"_MINERVA-"+field+"_"+ver+"_ACS+WEBB_Kf444w_SUPER_CATALOG.txt"
        np.savetxt(catpath,np.array(unfit_id_array))
        subprocess.run(["python", "submit_loop.py", cathead, field, ver, outdir, catdir], check=True)
        print(f"[INFO] job with "+cathead+" catalog is running...")
        time.sleep(CHECK_INTERVAL_SEC)

print(f"[INFO] all fitting is completed")

