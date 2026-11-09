# NSS + E-FSPS for MINERVA: startup, status, next steps

Handoff guide (2026-11-09) for whoever picks this up next, human or Claude. Read this first,
then [`STATUS.md`](STATUS.md) for the whole-project picture: both fitting paths, costs,
robustness, decisions and ranked next steps.
The public parts live here, on branch `nss-efsps` of `wrensuess/prospector_minerva`, in
`nss/`. Unreleased MINERVA photometry and MINERVA-specific results are NOT in this public
repo; they are in the private handoff folder (see "Data and private files").

## 1. What this is

A pure-JAX photometric SED fitter: the efsps-photfit model (Joel Leja; E-FSPS emulator,
Prospector-beta priors, 26 free parameters) sampled with blackjax **nested slice sampling**
(NSS; Yallup, Kroupa & Handley 2026, arXiv:2601.23252). The goal is a fast, *robust* fitter
for ~500k MINERVA galaxies on Blanca, one core per fit. It is a different model from the
production pipeline (dynesty + Parrot in `prospector_minerva`), so its output would be a new
catalogue, not a faster copy of the old one.

## 2. Repositories and versions

| What | Where | Version used |
|---|---|---|
| this code | `wrensuess/prospector_minerva`, branch `nss-efsps`, folder `nss/` | |
| efsps-photfit (model, priors, emulator files) | `jrleja/efsps-photfit` (Joel's) | commit `6d5ab6d` (v0.2.1), **unmodified** |
| E-FSPS (reference only: its `PhotModel` branch has the emulator forward pass we ported) | `efburnham/E-FSPS` (Emily's) | read-only |

Rule from Wren: **never commit to Joel's or Emily's repos; fork them under `wrensuess` first**
(needed when this code is ported into efsps-photfit).

## 3. Setup on a new machine

```bash
git clone -b nss-efsps https://github.com/wrensuess/prospector_minerva.git
git clone https://github.com/jrleja/efsps-photfit.git && git -C efsps-photfit checkout 6d5ab6d
python3.11 -m venv ~/envs/nss            # a PERMANENT location: the last venv lived in /private/tmp and macOS deleted it
~/envs/nss/bin/pip install -r prospector_minerva/nss/blanca/requirements_nss.txt
export EFSPS_PARENT=$PWD/efsps-photfit   # every script finds efsps-photfit (and the emulator) through this
```
`requirements_nss.txt` pins jax/jaxlib 0.10.2, **blackjax 1.7.1** (1.2.x has no `blackjax.ns`),
numpy 2.4.6, scipy 1.17.1, astropy 8.0.1, h5py 3.16.0. Python 3.11.

**Intel Mac:** the pip recipe above fails, because PyPI has no Intel-macOS jaxlib after 0.4.38
and blackjax >= 1.4 needs jaxlib >= 0.9 (blackjax 1.3, the last to install, has no
`blackjax.ns`). conda-forge still builds them, so use a conda env with the same pins
(validated 2026-11-09: all numerics tests pass, matching the Apple-silicon logs):
```bash
conda create -y -p ~/envs/nss -c conda-forge --override-channels python=3.11 jax=0.10.2 jaxlib=0.10.2 blackjax=1.7.1 numpy=2.4.6 scipy=1.17.1 astropy=8.0.1 h5py=3.16.0
```

Quick check (minutes, from `nss/`): `python test_tabulated_mass_prior.py`,
`python test_jax_loglike.py --n 2000 --n-fixed 200`, `python test_jax_prior.py --n 5000 --n-fixed 500`.
Each must end `RESULT: PASS`. Then a 1-second NSS smoke run:
`python nss_jax.py --catalog uncover_test.csv --id 58858 --tag smoke --mass-prior tabulated --n-live 200 --num-delete 20 --max-seconds 1 --out /tmp`.

For MINERVA galaxies, copy `minerva_uds_test9.fits` from the private folder into `nss/`
(it is git-ignored).

## 4. Code map (`nss/`)

| File | Role |
|---|---|
| `jax_prior.py` | JAX port of the efsps-photfit prior transform (u -> theta). Built from the numpy `FitModel`'s own tables. |
| `tabulated_mass_prior.py` | Tabulated Prospector-beta mass prior (no mass-function evaluation per call). **Use this** (`--mass-prior tabulated`). |
| `jax_model.py` | `JaxLogLike`: prior transform + w256 emulator + efsps Gaussian likelihood, pure JAX. |
| `photometry.py` | One reader for UNCOVER and MINERVA SUPER catalogues (FITS) and efsps CSV. mu = 1 if no `mu` column; drops F098M (no emulator band); masks fluxes < -5 sigma (as prospector does). |
| `nss_efsps.py` | Original numpy-callback NSS prototype; also `build()` (FitModel from photometry, emulator path). |
| `nss_jax.py` | **The runner**: NSS with the pure-JAX likelihood. Summary JSON records settings, photometry handling, emulator md5. Starts from the best n_live of 1e6 prior draws by default (`--n-prior-start`; 0 = standard start); see §5. |
| `test_prior_start.py` | Known-answer toys (10-d, logZ = 0, two-mode bottlenecks) for the prior start's evidence bookkeeping and mode loss. |
| `prior_start_check.py` | Offline diagnostic: share of the best n_live of N prior draws near a target redshift, for N = 1e3..1e7 (finds a galaxy's volume bottleneck without sampling). |
| `analyze_prior_start.py` | Prior-start vs standard runs on the agreed criteria (found rate, logZ, median shifts, iterations/time). |
| `test_jax_prior.py`, `test_jax_loglike.py`, `test_tabulated_mass_prior.py` | Numerics tests vs numpy, by redshift bin, `--catalog/--id/--mass-prior`. Run before trusting any change. |
| `profile_nss.py` | Where the time goes inside the real NSS step (component doubling, lane counts, variants). |
| `analyze_inner_steps.py` | Convergence analysis of 20-seed runs (logZ scatter vs quoted error, parameter-median scatter, shift vs a reference setting). |
| `compare_runs.py`, `combine_windows.py`, `audit_prior_draws.py` | Older analysis helpers (two-run comparison; z-window evidence combination; brute-force best logL per z bin). |
| `make_minerva_subset.py`, `make_csv.py`, `make_mocks.py`, `mass_ppf_candidates.py` | Helpers / superseded experiments. |
| `blanca/` | Everything for Blanca: `README.md` (run order), `profile_nss.sbatch`, `fullfit_layout.sbatch` (20 seeds of one galaxy in the production layout), `check_binding.*`, `posterior_draws.*`, `high_z_dead_points.py`, `make_bundle.sh`, `requirements_nss.txt`. |
| `results/`, `logs/` | Reference runs for UNCOVER 58858 (used by the tests), Blanca profiles/runs for 58858, all test/profile logs. |

## 5. Status (2026-11-09)

**Done and validated**
- JAX model == numpy to <= 1e-11 relative (theta and log L) at z = 0.01-15, every z bin,
  fixed z at all prior switch points, unit-cube edges, on UNCOVER 58858 and a MINERVA galaxy;
  on Mac (jax 0.5.3 and 0.10.2) and on 5 Blanca CPU types. The full NSS fit of 58858
  reproduces the numpy-callback run point for point.
- Tabulated mass prior: identical to the exact prior above z = 3 and below 0.2 (rounding),
  <= 4e-6 dex (KS <= 2e-7) for 0.2 < z <= 3 with nodes every 0.01. 5-10x faster fits on one core.
- MINERVA photometry reader; emulator is always w256 (`v4_n22000000_w256_d5_s0_parrot.npy`,
  md5 c9567f26..., the +-2.5% net; never use w64 for real data).

**Speed (one core, production layout = 20 fits per node, tabulated prior)**: cost is linear
in inner steps; median 37 ms per (iteration x inner step) (10-90%: 26-48; EPYC 9475F 15,
EPYC 7713P 45). ~230-365 iterations per fit at 1000 live points. At the median CPU:
52 steps (2d) 8 min, 78 (3d) 13 min, 130 (5d) 21 min, 260 (10d) 42 min per fit.

**NOT done: convergence (the gate for production)**
- 20 seeds x 9 MINERVA galaxies x inner steps 2d/3d/5d/10d: reproducibility improves
  steadily with inner steps but does not plateau by 10d (parameter medians still move
  ~0.14 posterior widths between seeds; logZ scatter 1.3-3.8x its quoted error).
- **Redshift mode-finding is the main failure**: for one high-z galaxy ~30% of runs miss the
  correct redshift solution at every inner-step setting (a discovery race lost in the first
  ~20% of the run; dead points show the failed runs never got near the good fit). Another
  galaxy has two genuine redshift solutions that runs weight inconsistently.
- Per-galaxy numbers are in the private folder (`PRIVATE_STATUS.md`).

**Update 2026-11-09: prior start (fixes the redshift discovery failures; now the default)**
- Cause, found with `prior_start_check.py`: the failing galaxy's correct solution holds only
  ~0.4% of the constrained prior volume over a stretch early in the run (~4 of 1000 live
  points), so runs lose it by chance. The share recovers later (7% by 1e7 prior draws). The
  other test galaxies never fall below ~1% and never failed. The starting points do not
  decide the outcome: the same initial set succeeds at some inner-step settings and fails at
  others.
- Fix (`nss_jax.py --n-prior-start N`, default 1e6): draw N prior points, delete the worst
  N - n_live without replacement (no MCMC; this compresses ln(N/n_live) nats almost for free),
  run NSS from the best n_live. This is nested sampling with a live set that shrinks from N to
  n_live, so it is valid. The handover volume is X_h ~ Beta(n_live + 1, N - n_live), sampled
  per Monte Carlo draw; the NSS phase uses blackjax's weights scaled by X_h.
- Toys (`test_prior_start.py`, `logs/test_prior_start_toys.log`): logZ unbiased at every
  start. Lost mode 15/30 -> 1/30 and 28/30 -> 3/30 at 1e7. 30-50% fewer iterations.
- Blanca, 9 MINERVA galaxies x 20 seeds, 130 inner steps, 1e6 and 1e7:
  - the failing galaxy found its redshift in 20/20 (standard 11/20);
  - no harm elsewhere: wrong-z runs unchanged or fewer; logZ within 2 combined SE at 1e6;
    median-shift tests at the chance rate, none surviving Bonferroni;
  - 20-30% fewer iterations at the same cost per iteration. The 1e6 draws take ~30 s on one
    core; 1e7 takes 3-6 min, so it saves less overall. Hence 1e6.
  - Details: `PRIVATE_STATUS.md`.
- **Not fixed:** in single-solution galaxies, logZ scatter between seeds is still 1-5x the
  quoted error and parameter medians still move between seeds. Also not fixed: the
  inconsistent weights of two genuine solutions. Both are now the main open problems.

## 6. Next steps, in order

0. **(New, 2026-11-09) Single-solution convergence**, now the main robustness gap. Diagnose
   before buying more inner steps (which is what costs time). Candidates: curved degeneracies
   (age-dust-metallicity) that one linear covariance shape handles badly; deleting 100 of 1000
   per iteration; inner chains that do not decorrelate from their start points. Also the speed
   lever: blackjax's lockstep waste (only ~35% of likelihood lanes are used).
1. **Clustering kernel: SET ASIDE 2026-11-09.** It doesn't address the failures above: the
   redshift losses happen before the live points separate into clusters (now fixed by the
   prior start), and single-solution fits have one cluster. NSS is essentially PolyChord's
   algorithm, so running PolyChord with clustering on/off is the cheap way to test clustering
   if it's ever wanted. Its one likely benefit is PolyChord-style allocation by cluster volume,
   for the inconsistent weights of two genuine solutions. Original plan, for reference:
   PolyChord-style (Handley+2015, arXiv:1506.00171 Sec. 5.3): cluster live points with
   mutual k-nearest neighbours (k increased from 2 until the partition stops changing);
   each new point's slice chain uses its *starting* point's cluster covariance for the
   whole chain (fixed per chain keeps the MCMC valid); shrink small-cluster covariances
   toward the global one. Plug into blackjax via `nss(..., inner_kernel_params=...,
   update_strategy=...)` (copy `blackjax/ns/from_mcmc.update_with_mcmc_take_last`, give each
   chain its own `covariance_factor`); no blackjax changes needed. Validate first on toy
   problems with known answers (single Gaussian; separated multi-mode Gaussians in 26-d with
   unequal weights) over many seeds: evidence and mode weights must be right; measure overhead.
2. **Test matrix on Blanca** (superseded by the prior-start test above; keep the 1000 vs 2000
   live-point comparison in mind for the convergence work): {global, clustered} x {1000, 2000 live points} at 130 inner
   steps, 20 seeds, the 9 MINERVA test galaxies (`fullfit_layout.sbatch`, extend it with the
   new option). Agree acceptance criteria with Wren *before* looking: e.g. zero catastrophic
   redshift failures in 20 seeds; logZ scatter <= ~1.5x quoted; parameter-median scatter
   <= 0.1 posterior width. Pick the cheapest passing setting (Wren: robustness over speed).
3. Broader validation: ~50-100 galaxies across z and type (EELGs, dusty, quiescent, faint,
   spec-z as truth), 3-5 seeds each; failure rates; comparison with spec-z and the
   production dynesty catalogue.
4. **Model configuration with Wren/Ikki** - AGN first (production pins it because free AGN
   sent dusty high-z galaxies to low z; efsps currently has `fagn` log-uniform 1e-5..3).
   Wren's idea (2026-11-09): allow AGN but penalise fagn above ~0.1-0.3, where real galaxies
   saturate. Recommended form: flat in log10(fagn) from -5 to -1, then a half-Gaussian tail
   in log10(fagn) (sigma ~0.3 dex) up to the emulator limit; closed-form inverse CDF (flat
   part + `ndtri` for the tail), easy in numpy and JAX. NOT a half-Gaussian in *linear*
   fagn: sampled in log space it peaks at fagn ~ sigma, i.e. it would *favour* AGN.
   Prior mass by fagn range (computed 2026-11-09, sampled in log10 fagn over 1e-5..3):

   | prior | <1e-3 | 1e-3-0.1 | 0.1-0.3 | 0.3-1 | 1-3 | density at 0.3 / 1 / 3 rel. to 0.01 |
   |---|---|---|---|---|---|---|
   | current log-uniform | 36.5% | 36.5% | 8.7% | 9.5% | 8.7% | 1 / 1 / 1 |
   | half-Gaussian in linear fagn, sigma 0.1 | 0.8% | 67.5% | 31.5% | 0.3% | 0% | 0.33 / 2e-20 / ~0 |
   | flat to log -1, tail sigma 0.3 dex | 45.7% | 45.7% | 7.6% | 1.0% | 0% | 0.28 / 4e-3 / 5e-6 |
   | flat to log -1, tail sigma 0.2 dex | 47.1% | 47.1% | 5.8% | 0.1% | 0% | 0.06 / 4e-6 / 1e-12 |

   Needs Wren's sign-off on break and width, then a new prior kind + tests; validate on the
   galaxies where free AGN caused catastrophic low-z solutions in production.
5. Engineering for production: port JAX prior/tabulated prior/reader/tests into a **fork** of
   efsps-photfit; production driver (many galaxies per process, checkpointing for
   preemption, outputs Ikki's postprocessing can read, redshift-mode flags, provenance);
   end-to-end scale test (~1000 galaxies) for throughput, failure rate and cost
   (~21-45 min/fit -> roughly 0.2-0.4M core-hours for 500k galaxies).

## 7. Wren's standing preferences (how to work on this)

- Production fits get **exactly one core**. Judge speed single-threaded; Mac timings are
  multi-threaded and do not count. Never propose multi-core or concurrent-fit speedups.
- **Robustness over speed**: sampler settings = the minimum at which repeated seeds give the
  same answers; slightly longer fits are fine. Never cut settings just for speed.
- Every test covers a range of redshifts (low z < 0.15 through high z, including the prior's
  switch points) and reports by z bin. Use **real galaxies, not mocks**.
- Test before committing; commit/push only when Wren says; commit messages describe the
  change in words.
- Report before each step of multi-step work; Wren decides.
- For Blanca: always give **exact, copy-paste commands** (full paths, one per block) for every
  transfer, job submission and progress check.
- No Blanca node type should be much slower for a *preventable* reason (binding, ISA, ...).
- Give justification from documentation/papers for sampler settings.

## 8. Blanca essentials (details in `blanca/README.md`)

- Login `ssh <curc-user>@login.rc.colorado.edu` (Duo push per ssh/scp). Code in
  `/projects/$USER/...`; `/scratch/alpine/$USER` is mounted on Blanca nodes.
- On the LOGIN node, `module load slurm/blanca` before `sbatch`/`squeue` (inside jobs it
  only gives an Lmod error, so the scripts don't load it).
- Preemptable: `--account=blanca-casa --partition=blanca --qos=preemptable --requeue`;
  CASA's own nodes: `--partition=blanca-casa`.
- **Dedicated node bgpu-casa1** (`--partition=blanca-casa --qos=blanca-casa`): 40 physical
  cores (Gold 6230), ~385 GB, V100. It's never preempted and starts at once when free, and
  it's shared with the group. Use it for matched-CPU timing comparisons and for jobs that are
  costly to lose to preemption. Check it's free with `sinfo -p blanca-casa -N -o "%N %C %T"`.
- **Always `#SBATCH --hint=nomultithread`.** On hyperthreaded nodes (Xeon Gold 6130 etc.) a
  21-task job otherwise gets 11 physical cores, mpirun doesn't bind, fits run ~2x slower.
- XLA thread flags (`--xla_cpu_multi_thread_eigen=false`, `intra_op_parallelism_threads=1`)
  are **ignored** by jax 0.6.1 and 0.10.2; XLA sizes its thread pool from CPU affinity, so
  one core must come from Slurm/mpirun binding.
- `--constraint=skylake` can land on Cascade Lake (the bhpc-c7 nodes are labelled skylake).
- Build the bundle with `EFSPS_PARENT=... bash nss/blanca/make_bundle.sh`; it unpacks to the
  `nss_test/` + `efsps-photfit/` layout the job scripts expect.

## 9. Pitfalls already hit (don't rediscover)

- `jnp.cumsum` sums in a different order from `np.cumsum`; the exact mass CDF uses a
  sequential `lax.scan` (moot with the tabulated prior).
- XLA fuses `u*cum[-1] - c0` into an FMA; masses at u within ~1e-9 of 1 (flat CDF tail) then
  differ visibly while the CDF agrees to 1 ulp. Tests judge those rows in probability space.
- On x86 the mass-function pdf (~4000 pow/exp per evaluation) was ~80% of the likelihood
  time - hence the tabulated prior. "Fixes" that kept the mass function (two-point gather,
  tables + two-point) were *slower* on one x86 core.
- blackjax vmaps its slice while-loops over `num_delete` chains in lockstep: only ~35% of
  likelihood lanes are needed by any chain.
- NSS's quoted logZ error covers shrinkage only, not imperfect mixing - use seed repeats.
- Running full NSS in every z window costs ~1 full fit per window (rejected approach).
- macOS deletes old files in `/private/tmp`: never keep environments or results there.
