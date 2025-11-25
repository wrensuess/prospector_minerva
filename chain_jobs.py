#!/usr/bin/env python3
import subprocess
import glob,time,os
import numpy as np
from datetime import datetime

CHECK_INTERVAL_HOURS = 0.1
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
            # squeue -u <user> を実行
            res = subprocess.run(
                ["squeue", "-u", user],
                text=True,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"[{now}] squeue 実行エラー: {e}")
            print(e.stdout)
            print(e.stderr)
            time.sleep(CEHCK_INTERVAL_SEC)
            continue

        lines = res.stdout.strip().splitlines()

        # Nline = 1 :only header, no job
        if len(lines) <= 1:
            print(f"[{now}] no job. finished.")
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

def get_unfit_number(catdir,cathead,field,ver):
    fitted = glob.glob("/scratch/alpine/ikmi3774/slurm/chains_parrot_"+ver+"_LW_Kf444w_SUPER_spsbeta/id_*.h5")
    fitted_id = [a.split("/")[-1].split("_")[1] for a in fitted]

    whole_id_ = np.loadtxt(catdir+"fitid_MINERVA-"+field+"_"+ver+"_LW_Kf444w_SUPER_CATALOG.txt")
    whole_id = [str(int(a)) for a in list(whole_id_)]

    not_fitted =  list(set(fitted_id)^set(whole_id))
    print("fitted:",len(fitted_id)/len(whole_id),"unfitted:",len(not_fitted)/len(whole_id))

    not_fitted_out = [int(a) for a in not_fitted]
    not_fitted_out.sort()

    return int(len(not_fitted)), np.array(not_fitted_out)

    

field = "UDS"
ver = "n2.3_v1.1"
catdir = '../phot_catalog/'
cathead = "fitid"
catpath = catdir+cathead+"_MINERVA-"+field+"_"+ver+"_LW_Kf444w_SUPER_CATALOG.txt"
subprocess.run(["python", "submit_loop.py", cathead, field, ver], check=True)
time.sleep(CHECK_INTERVAL_SEC)

k=0
while True:
    wait_job_finish()
    print("waited")
    Nresi, unfit_id_array = get_unfit_number(catdir,cathead,field,ver)
    if Nresi==0:
        break
    else:
        k=k+1
        cathead = "unfitid"+str(int(k))
        catpath = catdir+cathead+"_MINERVA-"+field+"_"+ver+"_LW_Kf444w_SUPER_CATALOG.txt"
        np.savetxt(catpath,np.array(unfit_id_array))
        subprocess.run(["python", "submit_loop.py", cathead, field, ver], check=True)
        print(f"[INFO] job with "+cathead+" catalog is running...")
        time.sleep(CHECK_INTERVAL_SEC)

print(f"[INFO] all fitting is completed")
