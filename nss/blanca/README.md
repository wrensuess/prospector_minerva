# Running NSS on Blanca

Production fits get exactly one core each, so all timing here is one physical core per fit.
Mac timings are multi-threaded (XLA used 1.75–4 cores) and can't stand in for that.

Commands below run either on the Mac or on Blanca, as each step says. On the Mac, write
your CURC user name for `<curc-user>` (in the remote paths too: `$USER` there would be the
Mac's user). On Blanca, `$USER` is your CURC user.

Layout on Blanca (next to the prospector session's `/projects/$USER/speedtest/`):
```
/projects/$USER/nss_blanca/nss_test/         code, inputs, logs/, results/ (from the bundle)
/scratch/alpine/$USER/nss_results/<RUN>/     production results (RESULTS_ROOT; /projects is nearly full)
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
runs the fits only, says so at the top of its `.out` and exits 4. Override the paths with
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
    defaults are at the top of the script. Results go to `RESULTS_ROOT/<RUN>/`; the default
    `RESULTS_ROOT` is `nss_test/results` (so older runs are found where they are), but
    production runs must set it to scratch, since `/projects` has ~27 GB free. Task-list and
    catalogue paths stay relative to `nss_test/`. E.g. 1052 fits, 100 per element:
    `mkdir -p /scratch/alpine/$USER/nss_results && sbatch --array=0-10 --export=ALL,TASKLIST=blanca/pp_specz1k_tasks.txt,RUN=specz1k_pp,CHUNK=100,RESULTS_ROOT=/scratch/alpine/$USER/nss_results tasklist_layout.sbatch`
    Submit each element once: a second job for the same lines while the first is still in
    `squeue` (running, pending or requeued) stops at once with exit 7 (see "Exit status").
12. **Catalogues after the arrays: `catalog_runs.sbatch`**, with a dependency on them (it
    counts finished and failed fits, then builds the collected HDF5 files and FITS catalogues
    per catalogue and seed):
    `sbatch --dependency=afterany:<arrayjob> --export=ALL,RUNS=specz1k_pp:blanca/pp_specz1k_tasks.txt catalog_runs.sbatch`
    It still reads `nss_test/results/<RUN>` only (not `RESULTS_ROOT`; production catalogue
    building is a later task).
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
`RESULTS_ROOT/<RUN>/` (default `nss_test/results/<RUN>/`).

## Bring results back to the Mac

From the Mac, with rsync (one Duo push; copies only what the filters name; resumes with
`--partial`). Never `tar` on Blanca, and never copy the full per-fit `*_seed<S>.npz` files
(~14 MB each). One run's jsons, logs, `_post.h5`, `_draws.npz` and catalogues (a run with
`RESULTS_ROOT` set: replace `nss_blanca/nss_test/results` by that folder):
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
  ends with the element's final state (`pipeline_status: DONE n, ...`, `===== done` and
  `===== <date> exit 0`).

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
spectra. Its state is read from the files only (`pipeline_status.py`): a fitted json (written
last and atomically by `nss_jax.py`; `"status": "ok"`, or a `logZ` and no `timed_out` in files
from before the status key), a valid `_post.h5` of the current format whose logZ matches the
json (every dataset's last element read, shapes checked against the fit summary inside it, so a
cut-short or zero-filled file that still opens is caught), spectra in it. A file that exists but
cannot be read is read again a few times, then treated as damaged and listed as `DAMAGED`; so
is one that makes the HDF5 library hang (each look inside a `_post.h5` runs in a child process
killed after 60 s; a zero-filled tail hung h5py in a test) or crash.
- One job per element: the element holds `RESULTS_ROOT/<RUN>/.state/lock_<first>-<last>` (its
  job id). A second job for the same lines exits 7 while the holder is still in `squeue`
  (pending, running or requeued); a lock left by a job that is gone is taken over.
- Before each pass the status removes stale temporary files and the old json, `_post.h5`,
  `.npz` and `_draws.npz` of every fit it schedules for a refit (the old log is kept as
  `.log.prev<k>`), so a fit killed half way can never leave an old json next to a new `_post.h5`.
- Every task line writes its own exit code (`<stem>.rc` in the pass's work folder) after its
  process ends. A fit counts as failed only if it ended without its output (exit code there and
  still undone, or a json with status `timed_out`, `capped` or `nonfinite`); a fit with no exit
  code was killed (preemption, requeue, `mpirun` dying), which never counts. The `mpirun lb`
  exit status is printed, and processes of a pass still alive after it are killed.
- Spectra: `spectra_nss.py --progress` records each galaxy as it starts; when a spectra process
  ends badly, only the galaxy it was working on and any it finished without spectra are
  charged; galaxies it never reached are not.
- Circuit breaker: if at least `BREAKER_MIN` (5) and at least `BREAKER_FRAC` (50%) of the fits
  of a pass that ended (counting only fits that had not failed before) failed, or every spectra
  process failed before its first galaxy, the node or setup is at fault: no failures are
  recorded, the `.out` shows `!! CIRCUIT BREAKER` with the first error lines of the logs, and the
  element requeues itself once (it may land on another node). A second trip in a row stops it
  (exit 6): fix the setup (missing catalogue, broken env, full disk) and resubmit.
- `"status": "bad_input"` (the id is not in its catalogue, no usable band): `BAD_INPUT`, final,
  never retried or counted, listed with its reason; it does not make the element fail. Delete
  that json to retry it (e.g. after fixing the catalogue).
- Preempted: Slurm requeues it (`--requeue`); the next run continues from the files.
- Near the 2 h limit: Slurm signals USR1 10 min before and the element requeues itself. The
  element's own requeues (time limit, work left after `MAX_PASSES`, circuit breaker) are counted
  in `.state/job_<arrayjob>_<i>.self_requeues` and bounded by `MAX_REQUEUE` (20); preemptions
  are not counted (`SLURM_RESTART_COUNT` counts both). A resubmission starts a new count. After
  `scontrol requeue` of itself the script waits for Slurm to stop it; a requeued job restarts
  after ~2 min.
- Killed or crashed mid-write: every write is atomic, so a file is either complete or absent.
- A fit's own postprocessing failed: `nss_jax.py` kept the full `.npz`, and the element reruns
  `postprocess_nss.py` on it (which then records the rerun in the json).
- A real failure is counted in `RESULTS_ROOT/<RUN>/.state/<stem>.json`; after `MAX_FAILS` (3)
  the fit is given up, listed at the end of the `.out`, and the element exits 3. Fix the cause,
  delete that `.state` file and resubmit the element.
- A task-list line without exactly `CATALOG ID DUST SEED` stops the element with exit 9 and
  the line number in the `.out`.
- The `.out` is appended across restarts (`--open-mode=append`); each run starts with
  `===== <date> on <node>, job <id>_<i> (id <jobid>), restart <n>` and an element that ends
  for good ends with `===== <date> exit <code>`.

### Exit status of an element

| Exit | Meaning | What to do |
|---|---|---|
| 0 | done: every fit fitted, postprocessed, with spectra (`BAD_INPUT` fits are listed, final) | nothing |
| 1 | setup missing before any work (env, loadbalance module, task list, results folder) | fix, resubmit |
| 3 | done except fits given up after `MAX_FAILS` failures (listed) | fix, delete their `.state` files, resubmit |
| 4 | fits done, no spectra: spectra setup missing | see "Spectra inputs", resubmit |
| 5 | work left, not requeued (`MAX_REQUEUE` self-requeues used, no `scontrol`, or the requeue failed) | resubmit the element |
| 6 | circuit breaker tripped twice in a row (or its requeue failed); nothing was charged | read the diagnosis, fix node/setup, resubmit |
| 7 | another live job holds this element's lock (duplicate submission); nothing touched | none, or cancel the other job |
| 8 | `pipeline_status.py` itself failed (Python error or bad arguments) | read the `.out`; a bug |
| 9 | malformed task-list line (line number in the `.out`) | fix the task list, resubmit |

A requeued run has no exit status of its own: `scontrol requeue` of a running job puts it back
to PENDING under the same job id and Slurm stops the running script (Slurm docs: `scontrol
requeue` "Requeue a running, suspended or finished Slurm batch job into pending state"; the
Blanca restart test of 2026-11-16 showed it restarting under the same id). The script never
exits on its own after a successful requeue; it waits up to 5 min to be stopped and only then
exits 5. `sacct -j <jobid> --duplicates -o JobID,State,ExitCode` lists each run; requeued runs
show `REQUEUED`. (A bash error before the first status could give 1, 2 or 127.)
`pipeline_status.py`'s own codes (0 done, 10 work left, 11 given up, 12 circuit breaker, 13
malformed task list) avoid Python's 1 (uncaught exception) and 2 (usage error).

Catch-up by hand (normally unnecessary): `make_catalog_nss.py` writes `missing_spectra.txt`
for galaxies without spectra; resubmitting their elements does them, or run, in a job:
`EFSPS_SRC=/projects/$USER/nss_blanca/E-FSPS EFSPS_PARENT=/projects/$USER/nss_blanca/efsps-photfit /projects/$USER/software/anaconda/envs/efsps/bin/python spectra_nss.py --libdir /projects/$USER/nss_blanca/efsps_lib/c3k_hr-mist-afe_cc1b132_slim --list missing_spectra.txt`

## Checking a run

On the login node (no module needed; the env's python is called directly). Every fit of the
task list, or `--first/--last` for one element's lines; `--list` prints each fit's state and
why; `--results-root` if the run was submitted with `RESULTS_ROOT`:
`cd /projects/$USER/nss_blanca/nss_test && /projects/$USER/software/anaconda/envs/nss/bin/python pipeline_status.py --root . --run <RUN> --tasklist <TASKLIST> --results-root /scratch/alpine/$USER/nss_results`
It prints `DONE n, FIT n, POST n, SPEC n, BAD_INPUT n, GAVE_UP n`, then any `DAMAGED`,
`BAD_INPUT` and `GAVE UP` fits with their reasons. It is read-only and safe to run while jobs
of the run are running: it removes, renames and records nothing. Only the job script cleans
(it passes `--clean` between passes, while it holds the element's lock); never pass `--clean`
by hand while a job of that run is running. Each results folder is listed once, so a whole-run
check is linear in the number of files (one folder of 200,000 files, 2026-11-17 on the Mac:
an element's 200 fits 0.9 s, was 280 s; all 50,000 fits 7.3 s, was ~1 s per fit; these fits had
no `_post.h5`; each `_post.h5` checked adds ~30 ms on the Mac, so an element's 200 take ~6 s).

## Restart test (CHAOS=1)

Proves that an element ends with everything done, and that only real failures are charged,
although it is interrupted every way it can be: `blanca/chaos_monkey.sh` (only with `CHAOS=1`)
first makes every fit of a pass fail at once (expect the circuit breaker: no failures, one
requeue), then requeues it mid-fit (as a preemption: the fits it kills must have no failures),
then truncates a json, deletes a `_post.h5` (with and without its full `.npz`), leaves stale
temporary files, cuts a `_post.h5` to 70% and pads it with zeros so it still opens, SIGKILLs a
running fit (one counted failure; a fit the preemption did not kill, where there is one) and 30 s later `mpirun lb` itself (the fits in flight: killed,
no failures), SIGKILLs a running spectra process and sends the time-limit signal. The task
list `blanca/restart_test_tasks.txt` has 9 quick fits and one id that is not in its catalogue
(expect `BAD_INPUT`, not retried; needs `nss_jax.py` with the `status` key, 2026-11-17 or
later). One element, full `.npz` kept (`EXTRA=`) so the postprocessing rerun is exercised;
restarts 0, 1, 2, 3; expected ~40-55 min on CASA's own node (never preempted, so the only
requeues are the test's; it needs 21 free cores there, i.e. 42 idle CPUs in
`sinfo -p blanca-casa -N -o "%N %C"`). From `nss_test/blanca`, after
`module load slurm/blanca`:
`rm -rf ../results/restart_test && sbatch --array=0 --partition=blanca-casa --qos=blanca-casa --export=ALL,TASKLIST=blanca/restart_test_tasks.txt,RUN=restart_test,CHUNK=10,NSPEC=3,EXTRA=,CHAOS=1 tasklist_layout.sbatch`
When `squeue` no longer lists it:
`cd .. && /projects/$USER/software/anaconda/envs/nss/bin/python blanca/check_restart_test.py --root . --run restart_test --tasklist blanca/restart_test_tasks.txt --out blanca/nss_tasklist_<jobid>_0.out`
It must end `RESULT: PASS`. The previous version (no breaker, mpirun kill, zero-padded file or
bad input; restarts 0-2) passed on Blanca 2026-11-16 (job 29641695, bgpu-casa1, 18/18, ~35
min). This version passed a Mac rehearsal (fake `mpirun lb`, `scontrol`, `squeue`; 6 fits and
the bad id, all 7 at once) on 2026-11-17; Blanca not yet run. The pure-logic cases (every
damaged-file state, json statuses, failure attribution, circuit breaker, exit codes) are in
`test_pipeline_status.py`, which runs anywhere.
