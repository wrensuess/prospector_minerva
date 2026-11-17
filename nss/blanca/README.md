# Running NSS on Blanca

Production fits get exactly one core each, so all timing here is one physical core per fit.
Mac timings are multi-threaded (XLA used 1.75–4 cores) and can't stand in for that.

Commands below run either on the Mac or on Blanca, as each step says. On the Mac, write
your CURC user name for `<curc-user>` (in the remote paths too: `$USER` there would be the
Mac's user). On Blanca, `$USER` is your CURC user.

Layout on Blanca (next to the prospector session's `/projects/$USER/speedtest/`):
```
/projects/$USER/nss_blanca/nss_test/         code, inputs, logs/, results/ (from the bundle)
/projects/$USER/nss_blanca/efsps-photfit/    the efsps-photfit package + emulator (from the bundle)
/projects/$USER/nss_blanca/E-FSPS/           E-FSPS source, for spectra (with .git)
/projects/$USER/nss_blanca/efsps_lib/        the weights-only stellar library, for spectra
/projects/$USER/software/anaconda/envs/nss   the fits' env; prosp is never touched
/projects/$USER/software/anaconda/envs/efsps the spectra env (a clone of nss + E-FSPS's imports)
```

## Code: the bundle (Mac → Blanca, every time the code changes)

1. On the Mac, build the bundle. It prints an md5 to compare on Blanca, and records the
   commits it was built from in `nss_test/BUNDLE_INFO.txt`:
   `EFSPS_PARENT=/Users/wren/Projects/minerva/efsps-photfit bash /Users/wren/Projects/minerva/prospector_minerva_nss/nss/blanca/make_bundle.sh`
2. On the Mac, copy it (Duo push):
   `scp ~/nss_blanca_bundle.tar.gz <curc-user>@login.rc.colorado.edu:/projects/<curc-user>/`
3. On Blanca (`ssh <curc-user>@login.rc.colorado.edu`), unpack and check. Unpacking replaces
   the code and keeps `results/`:
   `mkdir -p /projects/$USER/nss_blanca && cd /projects/$USER/nss_blanca && tar -xzf ../nss_blanca_bundle.tar.gz && md5sum ../nss_blanca_bundle.tar.gz && cat nss_test/BUNDLE_INFO.txt`

## Environments (once; done: nss 2026-10-08, efsps 2026-11-16)

Installs must run on a compile node, `acompile`, which belongs to Alpine: load
`slurm/alpine` first, and `slurm/blanca` again after leaving it (before any `sbatch` or
`squeue` for Blanca). On the compile node `module load anaconda` works; the login nodes have
no anaconda module.

4. On Blanca, go to a compile node:
   `module load slurm/alpine && acompile`
5. The fits' env (`requirements_nss.txt`: jax/jaxlib 0.10.2, blackjax 1.7.1, numpy 2.4.6, ...):
   `module load anaconda && conda create -y -p /projects/$USER/software/anaconda/envs/nss python=3.11`
   `/projects/$USER/software/anaconda/envs/nss/bin/pip install -r /projects/$USER/nss_blanca/nss_test/blanca/requirements_nss.txt`
   Check: `/projects/$USER/software/anaconda/envs/nss/bin/python -c "import jax, blackjax; print(jax.__version__, blackjax.__version__)"` prints `0.10.2 1.7.1`.
6. The spectra env: a clone of the nss env plus E-FSPS's imports, with the jax/numpy pins kept
   (pip leaves them as they are):
   `module load anaconda && conda create -y -p /projects/$USER/software/anaconda/envs/efsps --clone /projects/$USER/software/anaconda/envs/nss`
   `/projects/$USER/software/anaconda/envs/efsps/bin/pip install arviz numpyro optax dill matplotlib corner dynesty nautilus-sampler jax==0.10.2 jaxlib==0.10.2 numpy==2.4.6`
   Check (after step 8 below): `/projects/$USER/software/anaconda/envs/efsps/bin/python -c "import sys; sys.path.insert(0, '/projects/$USER/nss_blanca/E-FSPS'); import efsps, jax; print('efsps ok, jax', jax.__version__)"` prints `efsps ok, jax 0.10.2` (an ArviZ FutureWarning before it is harmless).
7. Leave the compile node and switch back to Blanca:
   `exit`
   `module load slurm/blanca`

## Spectra inputs (once; done 2026-11-16)

After its fits, each `tasklist_layout.sbatch` element runs `spectra_nss.py` as `NSPEC`
(default 10) parallel processes over its galaxies, one core and ~2.5 GB each (spectrum
percentiles, best-fit spectrum, rest-frame colours, added to each `_post.h5`). One Blanca
core: 7.6 s per galaxy on EPYC 9354, 16.4 s on Gold 6230, plus ~26 s to start each process
(2026-11-16). It needs the efsps env (step 6) and the two inputs below; without them the job
runs the fits only, says so at the top of its `.out` and exits 2. Override the paths with
`EFSPS_ENV`, `EFSPS_SRC`, `EFSPS_LIB` in `--export`; `SPECTRA=0` turns the step off.

8. E-FSPS source (Emily's private repo; read only, never commit to it). Keep `.git`: every
   spectrum records the E-FSPS commit (`git rev-parse` on this folder), which is `unknown`
   without it. Either on Blanca, clone it pinned to the commit the tests used (needs GitHub
   access from Blanca: an SSH key, or a read-only token at the prompt):
   `git clone git@github.com:efburnham/E-FSPS.git /projects/$USER/nss_blanca/E-FSPS && git -C /projects/$USER/nss_blanca/E-FSPS checkout fbfe4fa`
   or on the Mac, copy the Mac's checkout, `.git` included:
   `rsync -a --partial --exclude __pycache__ /Users/wren/Projects/minerva/E-FSPS/ <curc-user>@login.rc.colorado.edu:/projects/<curc-user>/nss_blanca/E-FSPS/`
   Check on Blanca: `git -C /projects/$USER/nss_blanca/E-FSPS rev-parse --short HEAD`.
9. On the Mac, the weights-only library (241 MB; made with `spectra_nss.py --make-slim`). rsync
   prints nothing when it succeeds; `-v` lists the 5 files:
   `rsync -av --partial /Users/wren/Software/efsps/c3k_hr-mist-afe_cc1b132_slim <curc-user>@login.rc.colorado.edu:/projects/<curc-user>/nss_blanca/efsps_lib/`
10. Check on Blanca that all three are there:
    `ls -d /projects/$USER/software/anaconda/envs/efsps/bin/python /projects/$USER/nss_blanca/E-FSPS/efsps /projects/$USER/nss_blanca/efsps_lib/c3k_hr-mist-afe_cc1b132_slim/meta_data.pkl`

## Jobs (on the Blanca login node, from `/projects/$USER/nss_blanca/nss_test/blanca`)

First, in each new login shell (and after `acompile`): `module load slurm/blanca` (so
`sbatch` goes to Blanca, not Alpine). The job scripts don't load it; inside a job it only
gives an Lmod error.

11. **Production-style runs: `tasklist_layout.sbatch`**, a job array over a task list
    (`CATALOG ID DUST SEED` per line), 20 fits at a time per node, then spectra; each element
    retries and requeues until its fits are done (see "Restarts and retries"). Settings and
    defaults are at the top of the script. E.g. 1052 fits, 100 per element:
    `sbatch --array=0-10 --export=ALL,TASKLIST=blanca/pp_specz1k_tasks.txt,RUN=specz1k_pp,CHUNK=100 tasklist_layout.sbatch`
12. **Catalogues after the arrays: `catalog_runs.sbatch`**, with a dependency on them (it
    counts finished and failed fits, then builds the collected HDF5 files and FITS catalogues
    per catalogue and seed):
    `sbatch --dependency=afterany:<arrayjob> --export=ALL,RUNS=specz1k_pp:blanca/pp_specz1k_tasks.txt catalog_runs.sbatch`
13. Profiling and older tests:
    - one-core profile on CASA's own nodes (≤4 h): `sbatch profile_nss.sbatch`. The numerics
      tests run first (reduced size) and the job stops if they fail; then lane counts, the
      cost of each component inside the NSS step, and the two candidate fixes;
    - one-core profile on preemptable nodes, the short variant set (submit 2–3 times to
      sample node types):
      `sbatch --partition=blanca --account=blanca-casa --qos=preemptable --requeue --time=02:00:00 --export=ALL,PROFILE_ARGS="--skip-micro --variants baseline fix_twopoint fix_tables baseline" profile_nss.sbatch`
    - 20 seeds of one galaxy in the production layout: `sbatch fullfit_layout.sbatch`;
    - the binding check: `sbatch --hint=nomultithread check_binding.sbatch`.

Check jobs with `squeue -u $USER -o "%.12i %.20j %.9T %.10M %R"`. Slurm's `.out`/`.err` files
land in the submit directory (`nss_test/blanca/`); the fits' logs and outputs are under
`nss_test/results/<RUN>/`.

## Bring results back to the Mac

From the Mac, with rsync (one Duo push; copies only what the filters name; resumes with
`--partial`). Never `tar` on Blanca, and never copy the full per-fit `*_seed<S>.npz` files
(~14 MB each). One run's jsons, logs, `_post.h5`, `_draws.npz` and catalogues:
`rsync -av --partial --prune-empty-dirs --include='*/' --include='*.json' --include='*.log' --include='*_post.h5' --include='*_draws.npz' --include='catalog/***' --exclude='*' <curc-user>@login.rc.colorado.edu:/projects/<curc-user>/nss_blanca/nss_test/results/<RUN> /Users/wren/Projects/minerva/nss_private_handoff/results/`
The job `.out` files:
`rsync -av --partial '<curc-user>@login.rc.colorado.edu:/projects/<curc-user>/nss_blanca/nss_test/blanca/nss_tasklist_<jobid>_*.out' /Users/wren/Projects/minerva/nss_private_handoff/results/blanca/`
Results stay in the private folder (MINERVA results never go in this public repo).

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
- Task-list job: the first line of each fit's log must say `1 physical core(s)`; the `.out`
  ends with the element's final state (`pipeline_status: DONE n, ...` and `===== done`).

## Why Slurm, not XLA_FLAGS

XLA sizes its CPU thread pool from the process's CPU affinity. The flags usually used to
cap it, `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`, are ignored by
jax 0.6.1 and 0.10.2 (tested 2026-10-08). The prospector session measured on Blanca
(OpenMPI 4.1.1):
- nodes without hyperthreading: `mpirun lb` binds each fit to one core;
- hyperthreaded nodes (Xeon Gold 6130): 21 tasks get 11 physical cores and fits aren't
  bound (~2× slower) unless the job sets `#SBATCH --hint=nomultithread`. All jobs here do.

## Restarts and retries (tasklist_layout.sbatch)

An element keeps working until every fit of its lines is fitted, postprocessed and has
spectra. Its state is read from the files only (`pipeline_status.py`): a valid json (written
last and atomically by `nss_jax.py`), a valid `_post.h5` of the current format, spectra in it.
- Preempted: Slurm requeues it (`--requeue`); the next run continues from the files.
- Near the 2 h limit: Slurm signals USR1 10 min before and the element requeues itself
  (`MAX_REQUEUE`, default 20 restarts in all). A requeued job restarts after ~2 min.
- Killed or crashed mid-write: every write is atomic, so a file is either complete or absent;
  temporary files are removed at the next status.
- A fit's own postprocessing failed: `nss_jax.py` kept the full `.npz`, and the element reruns
  `postprocess_nss.py` on it (which then records the rerun in the json).
- A stage that runs to its end without its output (a real failure, not a kill) is counted in
  `results/<RUN>/.state/<stem>.json`; after `MAX_FAILS` (3) the fit is given up, listed at the
  end of the `.out`, and the element exits 1. Fix the cause, delete that `.state` file and
  resubmit the element.
- The `.out` is appended across restarts (`--open-mode=append`); each run starts with
  `===== <date> on <node>, job <id>_<i>, restart <n>` and ends with its final state.

Catch-up by hand (normally unnecessary): `make_catalog_nss.py` writes `missing_spectra.txt`
for galaxies without spectra; resubmitting their elements does them, or run, in a job:
`EFSPS_SRC=/projects/$USER/nss_blanca/E-FSPS EFSPS_PARENT=/projects/$USER/nss_blanca/efsps-photfit /projects/$USER/software/anaconda/envs/efsps/bin/python spectra_nss.py --libdir /projects/$USER/nss_blanca/efsps_lib/c3k_hr-mist-afe_cc1b132_slim --list missing_spectra.txt`

## Checking a run

On the login node (no module needed; the env's python is called directly). Every fit of the
task list, or `--first/--last` for one element's lines; `--list` prints each fit's state:
`cd /projects/$USER/nss_blanca/nss_test && /projects/$USER/software/anaconda/envs/nss/bin/python pipeline_status.py --root . --run <RUN> --tasklist <TASKLIST>`
It prints `DONE n, FIT n, POST n, SPEC n, GAVE_UP n`. Run it only when no job of that run is
running: it removes temporary files it takes for stale.

## Restart test (CHAOS=1)

Proves that an element ends with everything done although it is interrupted every way it can
be: `blanca/chaos_monkey.sh` (only with `CHAOS=1`) requeues it mid-fit (as a preemption),
then truncates a json, deletes a `_post.h5` (with and without its full `.npz`), leaves stale
temporary files, SIGKILLs a running fit and a running spectra process, and sends the
time-limit signal. 9 quick fits (`blanca/restart_test_tasks.txt`), one element, full `.npz`
kept (`EXTRA=`) so the postprocessing rerun is exercised; ~35 min on CASA's own node (never
preempted, so the only requeues are the test's; it needs 21 free cores there, i.e. 42 idle
CPUs in `sinfo -p blanca-casa -N -o "%N %C"`). From `nss_test/blanca`, after
`module load slurm/blanca`:
`rm -rf ../results/restart_test && sbatch --array=0 --partition=blanca-casa --qos=blanca-casa --export=ALL,TASKLIST=blanca/restart_test_tasks.txt,RUN=restart_test,CHUNK=9,NSPEC=3,EXTRA=,CHAOS=1 tasklist_layout.sbatch`
When `squeue` no longer lists it (it goes through restarts 0, 1, 2):
`cd .. && /projects/$USER/software/anaconda/envs/nss/bin/python blanca/check_restart_test.py --root . --run restart_test --tasklist blanca/restart_test_tasks.txt --out blanca/nss_tasklist_<jobid>_0.out`
It must end `RESULT: PASS`. Passed on Blanca 2026-11-16 (job 29641695, bgpu-casa1, 18/18;
~35 min with two ~2 min requeue gaps). A Mac rehearsal of the same job script (fake
`mpirun lb` and `scontrol`, 5 fits, `CHAOS_N_DONE=4`) passed all of these checks on
2026-11-16. The pure-logic cases (every damaged-file state, failure counting, give-up, exit
codes) are in `test_pipeline_status.py`, which runs anywhere.
