# MINERVA SED fitting: project status report

**As of 2026-11-09.** This is the whole-project picture: both fitting paths, what they cost,
how robust they are, and what to do next. For setup, the code map and pitfalls, read
[`HANDOFF.md`](HANDOFF.md). Update this file when the picture changes, and keep the date
current.

This repo is public. MINERVA catalogue ids are fine here (Wren, 2026-11-09). MINERVA
photometry and per-fit result files stay in the private handoff folder, which also has the
detailed per-galaxy numbers (`PRIVATE_STATUS.md`).

**Test set:** 9 MINERVA UDS galaxies (catalogue n3.0_v1.2):
- 1174326 (z≈0.44)
- 1113001 (z≈1.5)
- 1169924 (z≈1.5, no spec-z)
- 1212906 (z_spec 2.09)
- 1192179 (z≈2.3, dusty)
- 1198917 (z≈2.7)
- 1241601 (z≈2.9, second genuine low-z solution)
- 1197855 (z≈3.9)
- 1152437 (z≈6.1 EELG)

## 1. Goal and constraints

- **Goal:** fit ~500k MINERVA galaxies on CU Blanca, accurately and fast. Production gives each
  fit **exactly one core**, so every timing here is single-core.
- **Robustness comes before speed** (Wren): sampler settings are the smallest at which repeated
  seeds give the same answer. Slightly longer fits are fine; untrustworthy fits are not.
- **What "fast" means:** at 500k fits, every minute per fit is ~8.3k core-hours.

## 2. Two paths

| | A. dynesty + Parrot | B. NSS + E-FSPS |
|---|---|---|
| Model | production Prospector-beta model, Parrot emulator, 18 free parameters, AGN pinned off | efsps-photfit model (Joel Leja), E-FSPS emulator (w256), Prospector-beta priors, 26 free parameters, AGN free |
| Sampler | dynesty (rwalk, multi-ellipsoid) | blackjax nested slice sampling (NSS), pure JAX |
| Where | `prospector_minerva`, branch `speedup-dynesty` (pushed, 6c08ce9); production is `dev_curc` | `prospector_minerva`, branch `nss-efsps`, folder `nss/` |
| Status | exact speed-ups done; robustness at production settings is poor | main effort; redshift failures fixed on the test set; convergence and production engineering open |

**These are different models.** Path B would produce a new catalogue, not a faster copy of the
production one. So the cost comparison below is between pipelines, not a like-for-like speed-up.

## 3. Cost per fit and for 500k fits (one Blanca core)

| Pipeline | Per fit | 500k fits | Robustness on the 9 test galaxies |
|---|---|---|---|
| A, production code (`dev_curc`) | ~60–90 min (Wren's figure: 1–1.5 h) | ~600k core-h | runs lock onto different solutions; 1192179 (dusty, z≈2.3) is at the wrong z in most runs |
| A, speed-up branch, nlive 1600 | 4–6× cheaper per call (identical outputs); ~20–55 min | ~250k | same problems |
| A, speed-up branch, nlive 3200 | ~60–90 min | ~600k | better; 1192179 is still wrong in ~35% |
| B, exact mass prior, 52 inner steps | ~64 min | — | — |
| B, tabulated prior, 52 steps | 8 min | ~70k | not converged |
| B, tabulated, 130 steps | 24 min | ~200k | 1152437 (z≈6) is at the wrong z in 45% of runs |
| **B, tabulated, 130 steps, 10⁶ prior start (current default)** | **~19 min (estimated)** | **~160k** | all 9 at the right z in every run; medians move ~0.2 posterior half-widths between seeds |

- Path B timings are medians over Blanca CPU types: 26–48 ms per (iteration × inner step);
  EPYC 9354 is fastest at ~24, Gold 6130 and E5-2680 v4 slowest at ~51–58.
- **The prior-start row is an estimate.** Its raw median (23.4 min) equals the standard start's
  (23.9 min), because its jobs happened to land on slower CPUs. The ~0.8× comes from 20–30%
  fewer iterations at an unchanged cost per step on the one shared CPU type (EPYC 9354, +5%).
  A matched run on the dedicated node bgpu-casa1 would measure it directly.

## 4. Robustness: what's fixed, what isn't

**Fixed (on the 9-galaxy test set):**
- **Redshift discovery.** 1152437, the z≈6 EELG, lost its correct solution early in ~45% of
  runs. The
  cause was a volume bottleneck: the solution holds ~0.4% of the constrained prior volume for a
  while (≈4 of 1000 live points) and dies out by chance.
  - The 10⁶ prior start (draw 10⁶ prior points, delete the worst without replacement, start
    NSS from the best 1000) passes that point with ~10⁵ effective live points.
  - Result: 20/20 seeds right, against 11/20 before. No harm to the other 8 galaxies.
  - Toy tests confirm the evidence stays unbiased.

**Not fixed:**
1. **Single-solution repeatability.** At 130 inner steps, logZ scatter between seeds is
   1–5× its quoted error, and parameter medians move ~0.2 posterior half-widths (worst
   parameter ~0.35).
   - It improves with inner steps but hasn't plateaued by 260.
   - Acceptance criteria are now set (§5, 2026-11-09). Against them, k=130 sits at R̂ ≈ 1.024
     (median parameter); dynesty at nlive 3200 managed ~0.1–0.3 half-widths (R̂ ≈ 1.005–1.04).
2. **Two genuine redshift solutions get inconsistent weights.** 1241601 (z≈2.9) has a second,
   real low-z solution (z≈0.3), and runs weight the two anywhere from 10% to 90%. After two solutions
   separate, each new point is grown from a point in its own solution, so the split of live
   points between them drifts at random.
   - PolyChord-style allocation by cluster volume is the known fix.
   - More live points also help.
3. **Only 9 galaxies tested.** 1152437 needed a start of 10⁵–10⁶; some of the 500k
   will need more. The broader validation (§6, item 4) measures how often.

## 5. Decisions made

| Date | Decision |
|---|---|
| 2026-10-08 | Tabulated mass prior adopted (≤4e-6 dex from exact; 5–10× faster fits). |
| 2026-10-09 | The emulator is always w256 (±2.5%); w64 is never used for real data. |
| 2026-10-09 | Full NSS runs in redshift windows: rejected (each window costs ~1 full fit). |
| 2026-11-09 | Clustering kernel set aside. It doesn't address the failures we have: the redshift losses happen before the live points form clusters, and single-solution fits have one cluster. |
| 2026-11-09 | 10⁶ prior start adopted as the default (`nss_jax.py --n-prior-start`; 0 = standard start). |
| 2026-11-09 | Convergence acceptance (single-solution galaxies, 20 seeds, judged on the median over galaxies; the worst galaxy is a warning, since 20 seeds measure scatter to ~16%). **Parameters:** Gelman–Rubin R̂ < 1.01 for every parameter (Vehtari+2021), i.e. median scatter between seeds ≤ 0.14 posterior half-widths, since R̂ ≈ √(1+S²). A parameter above 2× that threshold (S > 0.28) is a warning. Use rank-normalized R̂ where draws exist, so broad, skewed or edge-piled posteriors (dust2, dust_index) are judged fairly. **Width:** one run's 16–84 width within 10% of a reference posterior (long run or NUTS, not the pooled seeds). **logZ:** seed scatter ≤ 1.5× quoted to start, ideally lower (implementation error well below statistical); otherwise inflate the quoted error by the measured factor. **Plus:** no catastrophic failures (wrong-z or wrong-solution runs). Goal: generally quite good with no big obvious failure modes, not perfect. |
| standing | One core per fit; robustness before speed; every test spans redshift, uses real galaxies and reports by z bin; never commit to Joel's or Emilie's repos (fork first). |

## 6. Next steps, ranked

Robustness first; the speed gains are estimates until measured on one core.

1. **Single-solution convergence.** This is the main robustness gap and the largest speed
   lever: cost is linear in inner steps, and 130 instead of 52 is 2.5×. Diagnose before buying
   more steps. Candidates:
   - curved degeneracies between age, dust and metallicity, which one linear covariance shape
     handles badly;
   - deleting 100 of 1000 live points per iteration;
   - inner chains that don't decorrelate from their start points.

   Acceptance criteria: §5 (2026-11-09). Plan: (0) criteria, set; (1) mine the existing
   20-seed summaries; (2) a toy that reproduces the scatter; (3) instrumented diagnosis on two
   MINERVA galaxies (insertion-index test, chain decorrelation, slice statistics, reference
   posterior); (4) fixes chosen by the diagnosis; (5) Blanca test matrix and adoption.
   Brute force is expensive: the scatter falls roughly as k^−0.64, so S ≤ 0.14 needs
   k ≈ 260 (~42 min per fit) for the median parameter, and more for the worst.
2. **blackjax lockstep waste: maybe 1.5–2×, medium effort.** The 100 chains run in lockstep,
   and only ~35% of the likelihood evaluations computed are used by any chain. One idea: draw
   several candidate points per slice at once and take the first valid one (plain rejection
   within the bracket, a valid sampler). It must pass the toy and numerics tests.
3. **Emulator in float32: maybe 1.3–1.8× on the network's share, low effort.** Everything runs
   in float64. Rounding at float32 (~1e-7) is far below the emulator's ±2.5% accuracy, but it
   needs the by-redshift numerics tests.
4. **Broader validation: 50–100 galaxies** across redshift and type (EELGs, dusty, quiescent,
   faint, spec-z as truth), 3–5 seeds each. Measure failure rates; compare with spec-z and the
   production catalogue.
5. **Gold 6130 slowness: cheap; serves the "no slow node types" goal.** The 6130 runs at 51 ms
   against 31 ms for its sibling the Gold 6150, and clock speed (2.1 vs 2.7 GHz) explains only
   1.3× of that 1.6×. Test an AVX2 cap (`XLA_FLAGS=--xla_cpu_max_isa=AVX2`) and the
   hyperthreading setup on one job.
6. **Model configuration with Wren and Ikki, AGN first. IN PROGRESS: Wren is testing a new fagn
   prior (2026-11-09); its form and results aren't recorded here yet.** The earlier
   recommendation was flat in log10 fagn from −5 to −1, then a half-Gaussian tail with
   σ ≈ 0.3 dex; never a half-Gaussian in linear fagn, which favours AGN (table in HANDOFF §6.4).
   Validate on the galaxies where free AGN sent dusty high-z objects to low z in production.
7. **Production engineering:**
   - port the code into a fork of efsps-photfit;
   - a driver that runs many galaxies per process (saves ~10 s of startup per fit);
   - checkpointing, since a preempted job currently restarts all its fits;
   - outputs Ikki's postprocessing can read, redshift-mode flags, provenance;
   - an end-to-end test on ~1000 galaxies.

**Not worth optimizing now:** prior draws (~30 s, ~3% of a fit) and per-fit startup (~10 s,
~1%).

**Rough ceiling:** at ~160k core-hours, path B is already well under production's ~600k.
Items 2, 3 and 5 might bring it to ~70–100k; item 1 could halve that again. Robustness, not
speed, is what gates production.

## 7. Open items for other people

- **Ikki (production, `dev_curc`):**
  - add `#SBATCH --hint=nomultithread`: on hyperthreaded nodes 20 fits otherwise share ~11
    cores and run ~2× slower;
  - the python-fsps stale-redshift bug: the first `predict` after a redshift change returns
    stale output, and postprocessing calls it once per posterior sample;
  - why `target_n_effective` is 20000;
  - which config made the production SPS catalogue (AGN free, maybe MIRI).
- **Joel:** before porting the NSS code into efsps-photfit, fork it under `wrensuess`.
  We depend on commit `6d5ab6d` (v0.2.1), unmodified.
- **Wren:**
  - the AGN prior: being tested by Wren (2026-11-09);
  - whether to push `nss-efsps`.

## 8. Where things live

- **Code:** `wrensuess/prospector_minerva`, branches `nss-efsps` (this folder) and
  `speedup-dynesty`. efsps-photfit at `6d5ab6d`.
- **Private data and results:** `nss_private_handoff/` (MINERVA test subset, per-galaxy
  results, `PRIVATE_STATUS.md`). On Blanca:
  `/projects/kasu8993/nss_blanca/nss_test/{results,logs,blanca/*.out}`.
- **Environments:** Linux and Apple silicon use the pip recipe in HANDOFF §3; an Intel Mac uses
  the conda-forge recipe there. Blanca: `/projects/kasu8993/software/anaconda/envs/nss`.
- **Old-Mac only, not on the current machine:** the dynesty speed-up worktree and its
  `speedup_runs/` test data, and the per-fit NSS sample files (also on Blanca).
