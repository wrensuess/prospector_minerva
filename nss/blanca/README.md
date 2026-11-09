# Running the NSS profile and timing on Blanca

Production fits get exactly one core each, so all timing here is one physical core per fit.
Mac timings are multi-threaded (XLA used 1.75–4 cores) and can't stand in for that.

Layout on Blanca (next to the prospector session's `/projects/$USER/speedtest/`):
```
/projects/$USER/nss_blanca/nss_test/        code, inputs, logs/, results/ (outputs are small)
/projects/$USER/nss_blanca/efsps-photfit/   the efsps-photfit package + emulator
/projects/$USER/software/anaconda/envs/nss  separate env; prosp is never touched
```

## On the Mac

1. Build the bundle (~27 MB). It prints an md5 to compare on Blanca:
   `bash ~/Projects/MINERVA/nss_test/blanca/make_bundle.sh`
2. Copy it (expect a Duo push):
   `scp ~/nss_blanca_bundle.tar.gz <curc-user>@login.rc.colorado.edu:/projects/$USER/`

## On Blanca, once

3. `ssh <curc-user>@login.rc.colorado.edu`
4. Unpack and check the md5:
   `mkdir -p /projects/$USER/nss_blanca && cd /projects/$USER/nss_blanca && tar -xzf ../nss_blanca_bundle.tar.gz && md5sum ../nss_blanca_bundle.tar.gz`
5. Build the env. Use a compile node if CURC wants installs there (`acompile`); this has
   not been checked:
   `module load anaconda && conda create -y -p /projects/$USER/software/anaconda/envs/nss python=3.11`
   `source activate /projects/$USER/software/anaconda/envs/nss && pip install -r /projects/$USER/nss_blanca/nss_test/blanca/requirements_nss.txt`
   `python -c "import jax, blackjax; print(jax.__version__, blackjax.__version__)"` should print `0.10.2 1.7.1`.

## Jobs (on the login node, from `/projects/$USER/nss_blanca/nss_test/blanca`)

First, in each new login shell: `module load slurm/blanca` (so `sbatch` goes to Blanca,
not Alpine). The job scripts don't load it; inside a job it only gives an Lmod error.

6. **One-core profile on CASA's own nodes** (≤4 h): `sbatch profile_nss.sbatch`
   The numerics tests run first (reduced size) and the job stops if they fail. Then the
   full profile: lane counts, the cost of each component inside the NSS step, and the two
   candidate fixes.
7. **One-core profile on preemptable nodes**, the short variant set. Submit 2–3 times to
   sample node types:
   `sbatch --partition=blanca --account=blanca-casa --qos=preemptable --requeue --time=02:00:00 --export=ALL,PROFILE_ARGS="--skip-micro --variants baseline fix_twopoint fix_tables baseline" profile_nss.sbatch`
8. **20 full fits in the production layout** (preemptable, 21 tasks, `mpirun lb`,
   `--hint=nomultithread`, seeds 0–19): `sbatch fullfit_layout.sbatch`
9. Optional: confirm the binding in the NSS env (the prospector session already measured it):
   `sbatch --hint=nomultithread check_binding.sbatch`

Check jobs with `squeue -u $USER`. Slurm's `.out`/`.err` files land in the submit
directory.

## Bring back to the Mac

10. `mkdir -p ~/Projects/MINERVA/nss_test/blanca_runs`
    `scp -r <curc-user>@login.rc.colorado.edu:/projects/$USER/nss_blanca/nss_test/logs ~/Projects/MINERVA/nss_test/blanca_runs/`
    `scp -r <curc-user>@login.rc.colorado.edu:/projects/$USER/nss_blanca/nss_test/results ~/Projects/MINERVA/nss_test/blanca_runs/`
    `scp '<curc-user>@login.rc.colorado.edu:/projects/$USER/nss_blanca/nss_test/blanca/*.out' ~/Projects/MINERVA/nss_test/blanca_runs/`

## What to look at

- Profile header: ONE physical core, i.e. 1 CPU, or 2 hyperthreads of one core on a
  hyperthreaded node. Otherwise there's a WARNING and the timings are not one-core.
- Baseline wall time for 60 iterations. Times ~306/60, that's the one-core sampling time
  of a full fit. The Mac (multi-threaded) took 139 s for 60 iterations.
- `fix_twopoint` and `fix_tables`: the speed-up on one core. `logZ` must equal the
  baseline's, which shows the trajectory stayed identical.
- CPU/wall ≈ 1.00 on nodes without hyperthreading; up to ~2 on hyperthreaded ones.
- Full-fit job: the first line of each `results/blanca_layout_<job>/seed*.log` must say
  `1 physical core(s)`. The `.out` ends with a table of logZ, z, logM, wall, RSS and CPU%
  per fit, plus the logZ scatter across the 20 seeds.

## Why Slurm, not XLA_FLAGS

XLA sizes its CPU thread pool from the process's CPU affinity. The flags usually used to
cap it, `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`, are ignored by
jax 0.6.1 and 0.10.2 (tested 2026-10-08). The prospector session measured on Blanca
(OpenMPI 4.1.1):
- nodes without hyperthreading: `mpirun lb` binds each fit to one core;
- hyperthreaded nodes (Xeon Gold 6130): 21 tasks get 11 physical cores and fits aren't
  bound (~2× slower) unless the job sets `#SBATCH --hint=nomultithread`. All jobs here do.
