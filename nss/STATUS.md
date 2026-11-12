# MINERVA SED fitting: project status report

**As of 2026-11-11.** This is the whole-project picture: both fitting paths, what they cost,
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
| B, tabulated, 130 steps, 10⁶ prior start | ~19 min (estimated) | ~160k | all 9 at the right z in every run; medians move ~0.2 posterior half-widths between seeds |
| **B, tabulated, 10⁶ prior start, mass integrated out, 32 steps (current default, 2026-11-11)** | **4.7 min on EPYC 9354, 6.9 on Gold 5418Y (fast types only)** | **~40–60k** | 21 galaxies (12 with several redshift solutions): every seed finds every solution; median R̂ 1.004 (worst parameter 1.008); weight std ≤ 0.10 in 10 of 11 multi-solution galaxies; per-seed posteriors and SFHs agree by eye (§4) |

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
   - Acceptance criteria are now set (§5, 2026-11-09). Dynesty at nlive 3200 managed ~0.1–0.3
     half-widths (R̂ ≈ 1.005–1.04).
   - **Phase 1 (2026-11-09, `analyze_convergence.py` on the 1080 existing summaries,
     `sim_ideal_ns.py`; numbers on right-solution runs only, bimodal 1241601 left out, so lower
     than the 0.22 quoted above, which pooled everything):**
     - At the default (k=130, 1e6 start), the median parameter's S is 0.17 (R̂ 1.014). 43% of
       (galaxy, parameter) pairs pass R̂ < 1.01, and 13% are above the warning level. At
       k=260: 0.11, with 75% passing and no warnings.
     - **It depends on the galaxy, not the parameter.** In 1192179 (dust2 2.6), 1198917
       (fagn ≈ 1.3) and 1212906, nearly every parameter scatters (S ≈ 0.23–0.25; each run is
       worth ~25–30 independent draws, against Kish ESS ~7000). 1152437, 1169924 and 1197855
       pass easily (0.07–0.10).
     - **Broad parameters are not the problem:** S is unrelated to posterior/prior width
       (Spearman +0.01). But tails move more than medians: fagn has p50 0.08 and p84 0.48, and
       the SFR ratios have p84 ~0.35.
     - **Noise, not bias:** at k ≥ 130, single-run widths are within 2–3% of k=260, and
       significant mean shifts are at the chance rate. At k=52, runs are 9% too narrow.
     - **Floor:** a perfect sampler with these settings gives S 0.01–0.04 and R ≈ 1,
       including num_delete=100. All the excess comes from imperfect inner chains.
     - **logZ is a separate problem:** R is 2.4–6 even where parameters pass (1197855: S 0.10,
       R 6.0). R improves with a bigger prior start (4.0 → 3.0 at 1e7), which points to the
       early, multimodal-redshift phase. Phase 3's staged insertion-index test checks this.
     - **Redshift-solution weights, all runs kept:** only 1241601 has a second solution in this
       test set. There, the weight off the main solution is 0.10 ± 0.20 between seeds (range
       0–0.74) at the default, and doesn't settle with k. **Criterion (Wren, 2026-11-09): the
       std of a solution's weight across seeds ≤ 0.10** (0.05 is too strict).
     - **How common second solutions are** (`select_bimodal.py`, from the 7 redshift quantiles
       of the production catalogues; private outputs `bimodal_{cosmos,uds}.csv`):
       - COSMOS (212k galaxies): 39% one solution; a second solution of 0.1–2.3% / 2.3–16% /
         16–50% in 13% / 15% / 13%; 21% broad. UDS (216k): 33%, 11% / 16% / 16%, 25% broad.
       - It depends steeply on brightness: ~90–95% have one solution at F444W < 24, but only
         9–14% do above 28.
       - With a strong second solution and a known spec-z, the spec-z lies in the minority
         solution about half the time. So the weights matter.
       - **These undercount.** Production dynesty locks onto one solution: it puts 1241601
         at z ≈ 1.56 with no second solution, where NSS finds z 2.8 (spec-z 2.91) plus a real
         z ≈ 0.3 solution. Production and NSS are also different models.
     - **New test set: 12 COSMOS galaxies with a strong (16–50%) second solution in
       production** (`minerva_cosmos_bimodal12.fits`, private; ids in
       `bimodal_cosmos_test12.csv`). Rules (Wren): F444W S/N ≥ 3, use_phot = 1, ≥ 4 MINERVA
       medium bands. All have 9–11 medium bands and 20–24 usable bands, span z ≈ 0.3–6.4,
       and 5 have a spec-z. Numerics tests pass on two of them, and the runner works on
       COSMOS.
     - **COSMOS 20-seed results at the default** (jobs 29426142–53, fast nodes only;
       `analyze_rhat.py` with the new per-solution report, log `analyze_rhat_cosmos12.log`,
       private):
       - **Several solutions under NSS too:** 10 of 12. The other 2 (2099068, 2088585) have
         one.
       - **Solution-weight criterion (std ≤ 0.10):**
         - passes in 5: 2273446 0.015, 2355389 0.007, 2200517 0.039, 2248619 0.061,
           2243320 0.088;
         - fails in 5: 2057694 0.11, 2276122 0.13, 2076167 0.13, 2137258 0.17,
           2329830 0.28 (0–100% between seeds).
       - **R̂ within the main solution:** below 1.01 for the median parameter in 4 (2355389,
         2248619, 2076167, 2088585), 1.013–1.03 in 6, and failing in 2329830 (1.048) and
         2099068 (1.06). The minority solutions are often worse (R̂ 1.03–1.13).
       - **Against spec-z:**
         - 2099068: NSS gets 2.20 against spec-z 2.10, with one solution (production had
           1.14/3.07).
         - 2276122: the spec-z solution (2.77) holds every run's best point but gets only
           34% of the weight. That's plausibly a volume effect.
         - **2076167: a discovery failure at high z.** The z ≈ 5.8 solution (spec-z 5.84)
           has the best log L (526.5 against 524.0), but 18 of 20 fits give it ≤ 1–2%.
           The 1e6 prior start isn't enough here.
         - 2137258 (spec-z 5.86) and 2355389 (spec-z 4.44): no fit finds a solution near
           the spec-z.
         - **z-window check** (one Mac run per galaxy, window around the spec-z; weight =
           P_window × Z_window against the free runs' Z, as in `combine_windows.py`; logs
           `zwindow_*.log`, private):
           - 2076167: best log L 526.5 at z 5.83 (the best anywhere); expected weight ≈ 5%.
           - 2137258: 595.7 at z 5.88, as good as the best the free runs found (595.1–595.8);
             expected weight ≈ 3%. It is never found, so this is a discovery miss, but a small
             one.
           - 2355389: 470.7 at z 4.41, worse than the free runs' 472.9; expected weight ≈ 0.1%.
             Our model disfavours the spec-z.
           - **So the high-z spec-z solutions fit as well or better, but under our priors they
             hold only a few % of the weight.** The prior mass at z 5.6–6.1 is only 4%, and
             more of parameter space fits well at low z. That's a model and prior question,
             not a convergence one. Production puts ≥16% (2137258) and >50% (2076167) at
             high z. The sampler problem is the per-run noise: 2076167 gets 0% in most runs
             but 57% in one, against an expected ~5%.
       - logZ R is again 2–7× in most of them.
     - **Phase 3 diagnostics (2026-11-09, `diagnose_nss.py`: an instrumented copy of
       blackjax's update strategy, validated on 26-d Gaussian and banana toys; MINERVA outputs
       in `results/diag/`, private):**
       - **The inner chains hardly move on real galaxies.** After 130 steps:
         - each new point keeps a correlation of 0.8–0.9 with its start point in *every*
           parameter (Gaussian toy: 0.01);
         - the distance travelled is 0.02–0.27 of an independent draw (toy: 0.99);
         - each slice step shrinks 4–5.5 times (toy: 1.6) and rarely expands.
         - This holds even for 1197855, which passes R̂. New points are near-clones of their
           start points, so k=130 buys far less than it costs.
       - **The cause is the logmass coordinate.** 96–100% of live points have u_logmass
         within 0.02 of 1, because galaxies sit in the high-mass tail of the mass-function
         prior. There, logmass grows like log(1−u): 0.3–0.5 dex per decade. Late in
         1197855's run, the whole mass posterior is a sliver 1−u ≈ 3e-5 to 1.2e-4. Mass
         correlates with nearly every parameter, so degeneracies that are roughly linear in
         θ become strongly curved in u, and every direction is blocked. The cube walls are
         not the cause: the likelihood stops the step first 79–85% of the time.
       - **Tested and rejected: a linear logmass coordinate** (`linear_mass.py`, with the
         mass-function prior as a log-density; the same prior, validated by
         `test_linear_mass.py`, all PASS). Chain memory at k=130 only went from 0.79–0.85 to
         0.75–0.80 (1197855), and not at all for 1192179. The squeezed u coordinate is not
         the main cause.
       - **The real thin direction is mass along the brightness ridge** (geometry tests on
         live points, log `geometry_1197855.log`, private):
         - The likelihood surface is smooth: 0–2 local maxima along fine line scans.
         - At fixed other parameters, the allowed logmass range is only 0.025–0.03 dex,
           against a live-point spread of 0.10–0.29 dex. Brightness = mass × L/M(age, dust,
           Z, z), so mass is pinned along a curved ridge.
         - Room to the likelihood contour along random covariance directions, bounds not
           counted (near / far; an ellipsoid gives ~0.4 / ~1.4):
           - mass kept: 0.16 / 0.56 (1197855) and 0.08 / 0.30 (1192179);
           - **mass optimized out: 0.85 / 1.97 and 0.45 / 1.60**, as roomy as an ellipsoid.
           - The remaining limit is mostly the parameter bounds (first in 82–88% of steps),
             which slice sampling handles.
         - A local 100-neighbour covariance only gives about 1.7× more room. Physical-θ
           coordinates without removing mass give none.
       - **Candidate fix: integrate mass out exactly, inside the likelihood.**
         L'(rest) = ∫ L(m, rest) p(m|z) p(logzsol|m) p(ratios|z,m) dm, then draw m from the
         same integrand for every posterior sample. This gives the same joint posterior and
         evidence, and every mass-dependent prior stays inside the integral.
         - logmass is not an emulator input (the net predicts flux per 1e10 Msun), so it
           costs one network call plus cheap arithmetic on a mass grid.
         - logzsol and the SFH ratios are then sampled in physical units, their conditional
           priors as densities.
         - The ratio prior clips at ±4: 0.2–0.5% of each ratio's prior sits on the bound.
           That stays exact if the sampler moves the *unclipped* ratio
           y = expected(z,m) + t-offset, with its truncated-t density inside the integral and
           clip(y, −4, 4) passed to the emulator.
         - **Built (2026-11-10): `mass_marginal.py`, tests in `test_mass_marginal.py` (PASS
           on 1197855 and 1192179).**
           - Same theta from the same prior draw to 4e-16. The SFH ratios differ by 3.6e-6,
             which is the expected-ratio table: E_j(s) on the shifted age
             s = log10 age(z) + delta_t_dex(m), ≤ 3.5e-6 dex from the current chain;
             approved by Wren.
           - The metallicity and ratio-offset densities match their inverse CDFs.
           - Mass integral: analytic peak (the flux is linear in 10^m), 41 points over
             ±8σ, Newton-shifted for the prior's slope. Accurate to ≤ 4e-4 nats at real live
             points and ≤ 6e-3 at prior draws within 1000 nats of the best.
           - **Open:** where the mass prior at the likelihood peak is < e^-100 of its
             maximum (deep mass-function tail), errors reach several nats. The cause is not
             found yet; two fixes tried, neither helped.
           - Cost ≈ 3× the standard likelihood per call (Mac, multi-threaded; one core still
             to measure).
         - **Mixing with mass integrated out** (diagnose_nss.py --mass-coord marginal; k=130):
           - **Memory of the start after 130 steps: 0.27–0.50 (1197855) and 0.26–0.54
             (1192179), against 0.79–0.91 in the standard 26-d runs.**
           - Distance travelled 0.40–0.50 of an independent draw (was 0.06–0.27).
           - 2.4–2.6 shrinks per slice (was 3.4–5.5).
           - Insertion index uniform; no stuck chains.
           - **Memory after only 32 steps (0.57–0.83) ≈ the standard sampler after 130.**
           - **30–45% fewer iterations** (116 vs 201; 148 vs 212), because the mass
             information no longer has to be compressed through.
           - Rough cost per fit ∝ k × iterations × cost per call: at about the standard
             sampler's mixing, k ≈ 32–40 → (40 × 0.6 × 3) / 130 ≈ 0.55 of today's cost. At
             equal cost (k ≈ 70), much better mixing. To be measured on one core.
           - What's left: the prior-dominated parameters (dust emission, gas, AGN) keep ~0.5
             memory at 130 and decay slower than the SFH ratios. Steps are now mostly
             limited by the parameter bounds.
           - **In the runner (2026-11-10):** `nss_jax.py --mass-coord marginal`, and
             `MASSCOORD=marginal` in `fullfit_layout.sbatch` (output directories end `_mm`).
             Mass is drawn back for every posterior sample. A full Mac fit of 1197855 at k=52
             agrees with the 20 standard seeds:
             - logZ 537.61, against 536.92 ± 0.78 (k=130) and 536.87 ± 0.55 (k=260);
             - medians and 16/84% within about one seed-scatter for z, mass, Z, dust2 and SFH;
             - 114 iterations, against 212 standard.
           - **Blanca test (submitted 2026-11-10, bundle md5 39d3ad8e…):** 9 UDS + 12 COSMOS
             galaxies × k = 32, 52, 78 × 20 seeds, 1e6 prior start, fast nodes only. Judge on
             R̂ (all fits and per solution), solution weights, logZ against the standard runs,
             and one-core wall time per fit. The job script now calls the env's python
             directly instead of `module load anaconda`.
           - **Blanca results (2026-11-10; `compare_massmarg.py`, log
             `compare_massmarg.log`, private): 21 galaxies × k = 32/52/78 × 20 seeds,
             against standard k=130 (1e6 start).** 1152437 k=52 has 18 seeds: two were
             killed when the job ran out of memory; a rerun is pending.

             | median over galaxies [worst] | standard k=130 | marg k=32 | marg k=52 | marg k=78 |
             |---|---|---|---|---|
             | R̂, median parameter | 1.013 [1.179] | 1.004 [1.013] | 1.002 [1.009] | 1.001 [1.007] |
             | R̂, worst parameter | 1.031 [1.338] | 1.008 [1.029] | 1.004 [1.017] | 1.003 [1.011] |
             | solution-weight std (multi-solution galaxies) | 0.113 [0.28] | 0.037 [0.12] | 0.021 [0.10] | 0.018 [0.08] |
             | share with weight std ≤ 0.10 | 45% | 91% | 92% | 100% |
             | logZ R (scatter / quoted) | 3.2 [10.7] | 1.8 [3.4] | 1.3 [3.0] | 1.5 [2.7] |
             | iterations | 164 | 116 | 116 | 116 |
             | cost per fit vs standard (same galaxy and CPU) | 1 | 0.52 | 0.73 | 1.03 |

           - Hardest galaxies: 1241601's weight std goes 0.22 → 0.02 at k=52, R̂ 1.34 → 1.008;
             2329830's goes 0.28 → 0.07. The worst at k=52 is 2137258 (weight std 0.10, R̂
             1.017).
           - **k=32 vs k=52, per parameter** (all 26 parameters, R̂ ≥ 1.01 counted; "shift" =
             std of the 20 run means / pooled posterior std; the ideal-sampler floor is
             ~0.03):

             | | k=32 | k=52 | k=78 |
             |---|---|---|---|
             | galaxies with any R̂ ≥ 1.01 / ≥ 1.02 | 6 / 3 | 3 / 0 | 2 / 0 |
             | parameter–galaxy pairs ≥ 1.01 | 42 | 11 | 3 |
             | worst shift, median galaxy [worst] | 0.14 [0.27] | 0.11 [0.21] | 0.09 [0.19] |

             - k=32: 1198917, 1212906, 1241601, 2099068, 2137258 and 2273446. Mostly
               logmass and logsfr_ratios_1/2 (also dust2, logzsol), so not only the broad dust
               parameters. Worst: 2273446 logmass 1.029 (shift 0.27), 1198917
               logsfr_ratios_1 1.027.
             - k=52: 1212906 (logsfr_ratios_1 1.011), 2273446 (logmass 1.012), 2137258 (9
               parameters, worst logzsol 1.017).
             - Paired wall time on EPYC 9354: k=32 costs 0.66–0.70 of k=52 in all 16
               galaxies with both on that CPU type, so a third less, not the 32/52 = 0.62 of
               the step count.
             - **Per-seed posteriors** (`plot_k_compare.py`; PDF in the private
               `logs/k_compare/`): p(z), the SFH of each redshift solution, and histograms
               of every other parameter, per seed and setting. Every seed finds every
               solution at all three k. The R̂ exceedances at k=32 don't show up as visibly
               different posteriors: e.g. 1198917 (logsfr_ratios_1 R̂ 1.027 at k=32) has
               per-seed median SFHs inside the pooled 16–84% at every k. Weights in 2137258
               (std over seeds): z≈0.81 0.69±0.12 / 0.71±0.10 / 0.69±0.08 at k=32/52/78.
             - **Scaling with k** (scratch fit over the 21 galaxies, k = 32/52/78):
               - Seed scatter of run medians (in posterior half-widths) falls as ~k^−0.5, so
                 its variance goes as 1/k. Median galaxy, median parameter: 0.080 / 0.063 /
                 0.052.
               - Cost per fit ∝ k + 8.6. A fixed overhead worth ~9 steps fits the paired
                 ratios (32→52: 0.67 predicted and measured; 52→78: 1.43 predicted, 1.41
                 measured). Iterations are constant (median 116) across k.
               - Weight std falls only weakly (median slope −0.12): mostly the floor set by
                 the finite number of draws. Worst galaxy (2137258) ~k^−0.5.
             - **A principled k** (proposed, untested below 32):
               - Sampler scatter s inflates the effective error bar by √(1+s²): s = 0.14
                 adds 1%, s = 0.2 adds 2%. Pick the tolerance, then
                 k_min = k_ref (s_ref / s_tol)².
               - PolyChord's guidance for the same kind of sampler (slice steps along random
                 directions): ~d repeats for posteriors, ~5d for evidences. d = 25 once mass
                 is integrated out. Below ~d, a chain can't have moved along every
                 direction, so the k^−0.5 law shouldn't be trusted there.
               - Both point to k ≈ 25: the extrapolated scatter is ~0.09 for the median
                 galaxy and ~0.2 for the worst. Cost ≈ 0.83 of k=32. Needs a Blanca check
                 before adoption.
             - **Blanca check at k = 16 and 25 (2026-11-11; logs
               `compare_massmarg_k16_78.log`, `k_scaling_k16_78.log`; plots
               `k_compare/k16_k25_k32_k52_params.pdf`, all private).** Same 21 galaxies × 20
               seeds. The seed-17/18 rerun is merged, so 1152437 k=52 now has 20 seeds.

               | | k=16 | k=25 | k=32 | k=52 | k=78 |
               |---|---|---|---|---|---|
               | galaxies with any R̂ ≥ 1.01 / ≥ 1.02 / ≥ 1.05 | 12 / 7 / 2 | 8 / 5 / 0 | 6 / 3 / 0 | 3 / 0 / 0 | 2 / 0 / 0 |
               | worst R̂ [galaxy] | 1.075 [1152437] | 1.037 [2273446] | 1.029 | 1.017 | 1.011 |
               | scatter of run means / posterior std, median galaxy (median param) | 0.116 | 0.097 | 0.089 | 0.066 | 0.056 |
               | same, worst param: median galaxy [worst galaxy] | 0.18 [0.38] | 0.15 [0.39] | 0.14 [0.27] | 0.11 [0.22] | 0.09 [0.19] |
               | weight std, median [worst] | 0.042 [0.14] | 0.038 [0.11] | 0.037 [0.12] | 0.032 [0.10] | 0.028 [0.08] |
               | logZ R, median | 2.1 | 1.8 | 1.8 | 1.3 | 1.5 |
               | paired cost vs k=32 (EPYC 9354) | 0.63 | 0.83 | 1 | 1.51 | 2.14 |
               | model (k+8.6)/40.6 | 0.61 | 0.83 | 1 | 1.49 | 2.13 |

               - The cost model holds. The 1/√k law holds for the median galaxy down to k=16:
                 0.097 measured at k=25, 0.098 predicted from k ≥ 32.
               - The tail breaks below ~d, as expected. At k=25 the worst galaxy's scatter is
                 0.39, against ~0.30 extrapolated from k=32. At k=16, 7 galaxies reach R̂ ≥ 1.02.
               - Visible in the plots:
                 - k=16: 1198917's dust2, logmass and nebular parameters scatter between
                   seeds.
                 - k=25: one seed of 2273446 puts extra weight in a low-mass tail (logmass
                   7.8–8.5); a milder version appears at k=32.
                 - SFHs agree at every k, including k=16.
               - Peak RSS per fit is median 1.7 GB, max 2.3 GB, at every k. The update_info fix
                 didn't change the peak; something else sets it, probably the prior start.
                 `fullfit_layout.sbatch` now asks for `--mem=48G` (20 fits; 30G ran out).
           - **logZ:** within the standard runs' own scatter for most galaxies. In 1192179,
             1197855, 1241601 and 2276122 the mass-marginal value is 0.15–0.6 nats higher
             (2.2–2.7 SE). For 1192179 and 1241601, standard k=260 moves toward it, so the
             poorly mixed standard runs probably sit low. Not proven; a high-precision
             reference would settle it.
           - Absolute one-core time on EPYC 9354 (20-fit layout): standard k=130 ~7 min,
             marg k=52 ~7 min, marg k=32 ~4.7 min. These medians mix galaxies; use the paired
             ratios above. Slow CPU types were not tested.
           - The memory fix in nss_jax.py (no longer storing the slice brackets) was not yet
             in these runs. Results are bit-identical with it.
           - A bug found and fixed on the way: the diagnostic runner drew prior points in 25
             instead of 26 dimensions, JAX clamped the index, and two parameters were
             identical (singular covariance). `x_from_u` now refuses wrong-length input.
       - **What sets the weights between high- and low-z solutions** (`evidence_budget.py` on
         windowed fits of 2076167, 2137258 and 2248619; log `evidence_budget.log`, private).
         log(weight_high / weight_low) splits into a z-prior term, a fit term <log L>, a
         prior-volume cost per parameter, and a correlation term:
         - **The z ≈ 6 solution always fits better:** <log L> +1.7 to +3.6.
         - **It always pays more prior volume:** −3.7 to −5.4 nats, and loses overall
           (weights 2–17%).
         - **Mass is the largest single term: +1.4 to +2.9 nats.** The z ≈ 6 solutions have
           logM 9.0–9.7, at or above the 97.5th percentile of the z ≈ 6 mass-function prior
           (median 7.44). The mean log prior density is −2.7 to −4.6, against −1.0 to −1.5 for
           the low-z solutions, which also have broader mass posteriors.
         - Next: dust2 (+1.1 to +1.6), because the high-z solution needs dust tightly near
           zero while the dusty low-z interloper fits over a broad dust range. Then the p(z)
           prior (−0.8 to −0.9 nats; 4% of the prior mass at z 5.6–6.1), then
           delta_gas_logqion (+0.6), logsfr_ratios_1 and dust_index.
         - 2076167's low-z solution uses an AGN (fagn ≈ 0.85, costing 1.45 nats): the free-AGN
           low-z interloper again.
         - **So the place to look is the high-z mass-function prior** (Tacchella+18 at
           z ≥ 4: fixed Schechter parameters, no uncertainties, pre-JWST). Even a modest
           10⁹ M☉ galaxy at z ≈ 6 (2137258; EAZY ≈ 9.2) sits at its 97.5th percentile. This
           is *not* a detection problem: selection depends only on the observed fluxes, which
           every solution reproduces, so it cancels in a single galaxy's posterior. The
           question is whether the prior's relative abundance of z ≈ 6 galaxies and low-z
           interlopers is right. Production uses the same Prospector-beta prior, so the
           production/NSS difference in weights is more likely dynesty locking onto one
           solution. Proposal: §6, item 6.
       - **Solution weights (1241601):** chains cross between redshift solutions 4–13% of the
         time early on, then 0–1% after about iteration 60. From then until the losing
         solutions die out (by about iteration 90), each solution's live-point count drifts
         at random. With near-clone new points it drifts in clumps. This confirms the
         mechanism behind the weight scatter. **Wren (2026-11-09): most real
       galaxies probably have more than one solution, so they can't be left out.** So the test set
       (1 bimodal galaxy in 9) is unrepresentative. Solution-weight scatter is a main
       criterion, not a side issue. This reopens the clustering decision of 2026-11-09:
       PolyChord-style allocation of new points by cluster volume is the known fix.
     - **Rank-normalized R̂** (`analyze_rhat.py` on all 54 runs' `draws.npz`, every fit kept;
       log `analyze_rhat.log`, private). It agrees with the median-scatter numbers
       (R̂ ≈ √(1+S²)). Per galaxy, median / worst parameter:
       - At the default (k=130, 1e6 start):
         - at or just over the threshold: 1152437 1.002/1.010, 1169924 1.004/1.011,
           1197855 1.004/1.011;
         - about half the parameters pass: 1174326 1.010/1.032, 1113001 1.011/1.031;
         - fail: 1212906 1.025/1.042, 1198917 1.026/1.154, 1192179 1.034/1.106 (46% of its
           parameters above the 1.04 warning level);
         - 1241601 1.18/1.34, with solution weights 0.87 ± 0.22 (criterion ≤ 0.10).
       - At k=260 (standard start), every single-solution galaxy has median ≤ 1.009 and worst
         ≤ 1.034, with no warnings. 1241601 is still at 1.12 (weights ± 0.21).
       - **1241601 has three solutions:** z ≈ 2.8 (spec-z 2.91), 0.31, and 1.58 (where
         production puts it). Inside the z ≈ 2.8 solution, R̂ is still 1.05–1.43, because
         fits split weight differently between z ≈ 2.7 and 2.88. So multimodality also
         happens on scales smaller than the ±0.3(1+z) window, and R̂ within the main solution
         catches it.
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
| 2026-11-11 | **Mass integrated out of the likelihood** (`mass_marginal.py`, `--mass-coord marginal`) adopted as the default. Mathematically the same prior, implemented differently; this is allowed when it speeds sampling (Wren). The only approximation, the tabulated expected SFH ratios, was approved ("you should tabulate it"). Mass is still in the model: it sets the SFH-ratio and metallicity priors and enters the redshift prior, and is drawn back for every posterior sample. |
| 2026-11-11 | **Soft-ceiling dust2 prior, mass only, adopted as the NSS default** (Wren): today's flat dust2 prior below a ceiling of A_V = 1 mag at log M < 8, rising to 4 (no change) at log M 9; σ 0.2 penalty above. Only removes the low-z dusty-dwarf trap; no measurable effect on 35 non-trap test galaxies (§6 item 5b). |
| 2026-11-11 | **32 inner steps** as the default for mass-marginal NSS (Wren). 20-seed Blanca tests at 16/25/32/52/78 on 21 galaxies: 32 is the smallest setting where the hard galaxies still follow the 1/√k scatter law; 25 is 17% cheaper with a worse tail, 16 breaks (R̂ up to 1.075). The strict "R̂ < 1.01 for every parameter" is missed in 6 of 21 galaxies (worst 1.029), but per-seed posteriors and SFHs agree by eye, the standard Wren applies. `nss_jax.py` and `fullfit_layout.sbatch` defaults are now n_live 1000, num_delete 100, 32 steps, tabulated prior, mass integrated out. |
| 2026-11-11 | Results come back from Blanca by `rsync` pulled from the Mac, never `tar czf` on the login node. |
| 2026-11-09 | Convergence acceptance (single-solution galaxies, 20 seeds, judged on the median over galaxies; the worst galaxy is a warning, since 20 seeds measure scatter to ~16%). **Parameters:** Gelman–Rubin R̂ < 1.01 for every parameter (Vehtari+2021), i.e. median scatter between seeds ≤ 0.14 posterior half-widths, since R̂ ≈ √(1+S²). A parameter above 2× that threshold (S > 0.28) is a warning. Use rank-normalized R̂ where draws exist, so broad, skewed or edge-piled posteriors (dust2, dust_index) are judged fairly. **Width:** one run's 16–84 width within 10% of a reference posterior (long run or NUTS, not the pooled seeds). **logZ:** seed scatter ≤ 1.5× quoted to start, ideally lower (implementation error well below statistical); otherwise inflate the quoted error by the measured factor. **Plus:** no catastrophic failures (wrong-z or wrong-solution runs). Goal: generally quite good with no big obvious failure modes, not perfect. |
| standing | One core per fit; robustness before speed; every test spans redshift, uses real galaxies and reports by z bin; never commit to Joel's or Emilie's repos (fork first). |

## 6. Outstanding items to get NSS into production

Convergence (the old item 1) is solved for now: with mass integrated out and 32 steps, the
21-galaxy tests meet the criteria in spirit (§4, §5). What's left, grouped; within each group,
most important first.

**A. Correctness checks still open (before production)**

1. **Deep-tail integral error in `mass_marginal.py`.** The mass integral is off by up to
   several nats at points where the mass prior at the likelihood peak is below e^−100 of its
   maximum. No test-galaxy posterior is affected, but find the cause before production.
2. **A high-precision reference for a few galaxies** (e.g. mass-marginal, 260 steps, more live
   points). It settles two things at once:
   - the logZ offsets: 1192179, 1197855, 1241601 and 2276122 come out 0.15–0.6 nats higher
     than the standard 130-step runs. Standard 260-step runs move toward the new values, so
     the old runs probably sat low, but that isn't proven;
   - the width criterion (one run's 16–84% within 10% of a reference posterior), never checked
     for mass-marginal runs.
3. **logZ error bars.** At 32 steps the seed scatter of logZ is ~1.8× the quoted error.
   Either inflate the quoted logZ error by that factor, or ignore it if logZ isn't used
   downstream.
4. **Broader validation: 50–100 galaxies** across redshift and type (EELGs, dusty, quiescent,
   faint, spec-z as truth), 3–5 seeds each, at the production setting. Measure failure rates;
   compare with spec-z and the production catalogue. The 21 test galaxies were chosen to be
   hard, so they say little about typical failure rates.

**B. Model decisions (with Wren, Ikki and Joel)**

5. **AGN prior. DECIDED 2026-11-11; IMPLEMENTED here 2026-11-11 (uncommitted).**
   `nss_priors.py`; `nss_efsps.build()` applies it by default (`nss_jax.py --fagn-prior`,
   default `flat1_s025`; summaries record `fagn_prior`). Tests: `test_nss_priors.py` PASS (ppf
   vs numerical inverse ≤ 4e-9 dex, vs the prospector `FastFlatGaussTail` ≤ 5e-12 dex for
   u < 1−1e-6, NumPy vs JAX 9e-16 dex, P(fagn > 0.1) = 0.0726); all numerics suites PASS with
   it (prior and loglike, exact and tabulated, on 58858 and 1212906; mass-marginal on
   1197855). The tail is written in complement form, so it doesn't lose precision at u → 1.
   Still to do: rerun the AGN test set in `photometry.py`. Wren adopted "flat1_s025" as the
   production prior:
   - x = log10 fagn on [−5, log10 3] (same range as efsps now);
   - flat for x ≤ −1, then a half-Gaussian tail with σ = 0.25 dex;
   - P(fagn > 0.1) = 7.3%.

   Inverse CDF, with m_flat = 4, m_tail = σ√(π/2)·erf((log3+1)/(σ√2)), norm = m_flat + m_tail:
   - u ≤ m_flat/norm: x = −5 + u·norm;
   - otherwise: x = −1 + σ√2·erfinv((u·norm − m_flat)/(σ√(π/2))).

   Reference code is in the prospector worktree `prospector_minerva_fagn` (branch fagn-prior,
   uncommitted): `FastFlatGaussTail` in `fagn_priors.py`, used by Ikki's production
   `params_prosp_parrot_AGNfree.py`.

   Validation was a free-z refit of 49 UDS galaxies that production put at low z with high
   fagn, plus 39 bright right-z controls, 2 repeats each:
   - It fixed 22/40 bright ones in both repeats and broke 1/39 controls.
   - Harsher tails (σ 0.1) fixed no more and broke a real AGN (UDS 1241214).

5b. **Dust prior against the low-z dwarf trap. ADOPTED 2026-11-11 and the NSS default (uncommitted).** Production
   puts z_spec > 5.5 galaxies at z ≈ 0.2 as dusty (dust2 1.2–3.4), old, log M 6–7.7 dwarfs.
   Candidate fix: a dust2 prior conditional on mass and SFR₁₀₀, after Nagaraj+22 (ApJ 932, 54)
   Fig. 7, calibrated on secure-z production fits, applied inside the mass integral
   (`mass_marginal.py`). A 48-galaxy test set (dwarf traps, dusty quiescent, SMG-like, real low-z dwarfs,
   controls); plan, sample and figures are private (`../dust_prior/`). Note: efsps's dust2 prior
   is uniform on [0, 4] and its mass floor is 10⁷; production uses TruncNormal(0.3, 1) and 10⁶.
   Most production trap solutions sit below 10⁷, so NSS behaviour on them must be measured first.
   Variants to test (Wren, 2026-11-11): dust2 given mass only, SFR only, and mass + SFR; no
   redshift dependence for now. Wren wants the prior as close to flat as possible, so the form is
   a soft ceiling: today's flat prior × a half-Gaussian penalty only above d_max(M, SFR), with
   d_max 0.8 at low mass / low SFR, ramping to 4 (no prior) by log M 10 / log SFR 0.
   **Implemented 2026-11-11 (uncommitted):** `dust_prior.py`; `MassMarginal(dust_prior=...)` adds
   ln w inside the mass integral (SFR₁₀₀ = 10^m · f(ratios, z), ported from efsps
   `sfh_quantities`); `nss_jax.py --dust-prior {none,mass,sfr,mass_sfr}` (default none),
   `--save-draws N`. `test_dust_prior.py` PASS: SFR₁₀₀ JAX vs efsps ≤ 7e-13 dex, ln w JAX vs
   NumPy ≤ 6e-14, no-penalty ceiling bitwise identical to no prior, error the penalty adds to the
   mass integral ≤ 3e-7 nats. ln C = −1.08 / −0.88 / −0.83 (mass / sfr / mass_sfr): the ceiling
   removes 53–72% of the prior volume (most prior mass is at low mass), a constant offset in logZ.
   Blanca test: 48 galaxies × {none, mass, sfr, mass_sfr} × 3 seeds (`blanca/dust_tasks.txt`,
   `blanca/tasklist_layout.sbatch`, array 0–28, fast nodes, results/dust48/).
   **Results (2026-11-11, 576/576 fits OK; private: `dust_prior/analyze_dust48.py`, `dust48_*.csv`,
   `dust48_results.png`):**
   - Without a dust prior NSS already escapes much of the trap: 6 of the 11 z > 5.5 galaxies that
     production put at z < 0.6 get ≥ 0.9 weight at z_spec (mean 0.69); production had 0.
   - Mean weight at z_spec for the 11, none / mass / sfr / mass_sfr: 0.69 / **0.88** / 0.83 / 0.82.
     Mass only is best: 1184059 0.19 → 0.81 (0.44 with an SFR term), 1201575 0.21 → 1.00
     (0.77–0.82), 1187083 (z 3.2) 0.73 → 0.93 (0.74–0.77). The SFR ramp lets low-mass dusty
     solutions with log SFR −1 to 0 through (1184059's z ≈ 1.1, log M 7.6, dust2 1.4 solution).
   - No collateral in any variant: the 35 non-trap galaxies (real low-z dwarfs incl. the dustiest,
     dusty quiescent, SMG-like, controls) keep their z weights, and their posterior medians move
     ≤ 0.2 half-widths (median 0.01–0.05), inside the seed scatter (0.05–0.08). Reported logZ is
     unchanged for them (w = 1 where their posterior is).
   - Not fixed, and not the dust prior's job: 2132888 leaves z 0.24 but lands at z ≈ 5.3, not 7.12
     (Hα ↔ [OIII]); 1210695 goes to a massive dusty z ≈ 2 solution; SMG-like 1088165 (z 4.09) and
     2243143 (3.16) split weight with z ≈ 2.6–2.7 solutions in every setting.
   - Wall time unchanged (median 302–341 s per fit across settings).
   **Adopted setting (Wren, 2026-11-11):** mass only; ceiling A_V = 1 mag (dust2 0.92) below
   log M 8, rising linearly to 4 (no change) at log M 9; tail σ 0.2; today's flat prior everywhere
   below the ceiling. Wren narrowed the ramp from 8–10 to 8–9 because real dusty galaxies exist at
   10^8–10 and the goal is only the trap. Chosen by reweighting the no-prior posteriors (checked
   against the 8–10 refits to ±0.05): mean trap weight at z_spec 0.79 (0.65 without; 0.82 for
   the tested 8–10, 0.80 ceiling 0.8); the cost of 1 mag vs 0.8 is 1184059 (0.75 → 0.58). No
   non-trap galaxy affected (≤ 0.1% of any right-z posterior pays > 0.5 nat). `dust_prior.CEILING`;
   `nss_jax.py --dust-prior` now defaults to `mass` (`none` to turn it off; the cdf mass coordinate
   needs `none`); `fullfit_layout.sbatch` has `DUST` (default mass). ln C = −0.89. Tests PASS
   (`test_dust_prior.py`, `test_mass_marginal.py`; private logs `test_dust_prior_1197855_adopted.log`).
   Not yet run on Blanca at the adopted setting. **Validation prepared 2026-11-11** (bundle md5
   20e6087c…): (1) the 48 test galaxies at the adopted setting, 3 seeds (144 fits,
   `blanca/dust48_adopted_tasks.txt`, RUN dust48_adopted, CHUNK 72, array 0–1); (2) every secure
   spec-z galaxy the prior could affect, UDS + COSMOS, with and without the prior, seed 0: 195 at
   z_spec < 1 with production log M < 9.5, 417 at z_spec > 5.5, 61 at z_spec 3–5.5 that production
   put at z < 1 and log M < 8 (673 galaxies, 1346 fits, `minerva_{uds,cosmos}_specval.fits`
   (private), `blanca/specval_tasks.txt`, RUN specval, CHUNK 100, array 0–13). `nss_jax.py
   --no-full-npz` skips the 14 MB sample file (tasklist_layout default). Pass criteria: z_spec > 5.5
   at z < 1 essentially gone with the prior (any left explained by another failure);
   ≥ 99% of the low-z galaxies change weight at z_spec by < 0.05, none by > 0.1; the z 3–5.5 group
   reported, not judged. With one seed, check galaxies that move against the no-prior draws
   reweighted by w (no seed noise), and rerun movers with more seeds if needed.
   **Validation results (2026-11-11; 1490/1490 fits OK; private `dust_prior/analyze_validation.py`,
   `.log`, `validation_specval.csv`, `validation_specval.png`):**
   - The 48 at the adopted setting (3 seeds): mean trap weight at z_spec 0.65 → 0.81 (reweighting
     predicted 0.79; 1184059 refit 0.75 vs predicted 0.58, the rest within 0.03). The 35 non-traps:
     z_spec weight changes ≤ 0.01; log M and dust2 medians move ≤ 0.2 half-widths (seed floor 0.06–0.08).
   - **PASS, z_spec > 5.5 (417):** majority weight at z < 1 goes 33 → 11 (production: 33). The 11 left
     have z ≈ 0.8, log M 7.1–7.7 solutions with dust2 0.4–1.0 (one 1.4 at 10^8.7): not the dusty dwarf.
     Five galaxies lose > 0.1 at z_spec, but to neighbouring high-z solutions, with none of their
     no-prior draws penalised and no change under reweighting: one-seed sampler noise (2171172:
     0.95 → 0.24, weight moved above z_spec), not the prior.
   - **PASS, real z < 1, log M < 9.5 (195):** every galaxy changes its weight at z_spec by < 0.05
     (max 0.041); the prior's own effect (reweighting) is ≤ 0.017.
   - z 3–5.5 trapped by production (61): majority at z_spec 16 → 18; mostly not a dust problem.
   - **New, separate issue:** on z_spec > 5.5 NSS trails production and eazy at the faint end
     (majority weight at the right z, adopted: 0.88 / 0.74 / 0.59 / 0.32 for F444W < 26 / 26–27 /
     27–28 / > 28; production 0.85 / 0.85 / 0.81 / 0.62; eazy 0.94 / 0.90 / 0.87 / 0.82). The lost
     weight is mostly at z 1–3 (18.5% of all weight; 73 galaxies with > 50% there): solutions at median
     z 1.4, log M 7.8, dust2 0.5, log SFR −1.1 — Lyman/Balmer break confusion, not dust. Related to
     item 6 (high-z priors), not yet diagnosed.
     - **The priors explain only part of it** (`dust_prior/reweight_zmass_prior.py`, private). NSS and
       production do NOT share redshift and mass priors: production `PhiSFH` has z uniform on
       [0.001, 20] and Leja+20 frozen above z = 3 (`const_phi=True`) over 10⁶–10^12.5; NSS (efsps
       NZSFH) has the Prospector-β number-density p(z) and Tacchella+18 above z ≈ 4 over 10⁷–10^12.5.
       NSS's priors favour a z 1.4, log M 7.8 solution over a z 6–8, log M 9 one by 1.7–4 nats more.
       Reweighting the NSS posteriors to production's z, mass and dust2 priors raises the z > 5.5
       majority-right fraction 0.65 → 0.73 (F444W < 26 / 26–27 / 27–28 / > 28: 0.93 / 0.82 / 0.68 /
       0.40, against production 0.85 / 0.85 / 0.81 / 0.62). The rest of the faint-end gap is
       unexplained (candidates, untested: the model itself — E-FSPS with free nebular parameters vs
       Parrot — and dynesty locking onto one solution in production).
   - Wall time per fit median 5.8 min (90% 8.5) on the fast types.
   Side note for item A1: at 1197855 prior draws the existing mass-integral error reached
   2.1e-2 nats in this test (60 draws), above the 6e-3 recorded earlier. **Later:** a q_PAH prior tied to
   metallicity or mass (the trap puts PAH 3.3 µm on [OIII]/Hα), tested separately.
6. **High-z mass prior.** High-z spec-z solutions lose weight mainly to the mass-function
   prior at z ≈ 6 (then dust2 and p(z)); production dynesty hides this by locking onto single
   solutions.
   - **FUTURE DIRECTION (Wren, 2026-11-09; she'll come back to it): a high-z mass prior that
     doesn't kill known-right solutions.**
     - **Goal (Wren):** the prior must not fully suppress solutions we know are right, such as
       the spec-z ≈ 5.85 solutions of COSMOS 2076167 and 2137258 (2–7% weight now). "The universe
       looks different than we thought it did in 2018". It must still strongly downweight things
       like z = 11, logM = 11. There are no good JWST-era mass functions yet.
     - **Why the obvious tweaks don't work:** Gaussian smearing (σ = 0.3 dex) or raising M*
       by 0.5 dex gains logM ≈ 9 at z ≈ 6 only about 0.2 nats (log density −4.3 → −4.1). The
       penalty comes from the steep faint end (α ≈ −1.6 puts most prior mass near the lower
       bound) and from the extra mass the high-z solution needs, not from the exponential
       cutoff. A "detection-aware" prior is the wrong fix (§4).
     - **Proposal:**
       - **A mixture:** p(logM | z) = (1 − ε) T18 + ε F, with F flat in logM from the lower
         bound up to a cap M_max(z). A solution then pays at most about log(ε / width) instead
         of an unbounded tail penalty.
       - **The cap keeps the guardrail.** Make M_max(z) physical, e.g. the baryon budget of
         the rarest halo expected in the survey volume (halo mass function × f_b), or simply a
         hand-set line (≈ 11.5 at z 6 falling to ≈ 10 at z 11). Above the cap only T18 remains.
       - **Where ε applies:** ε > 0 only at z ≳ 4 (or ≳ 3). Below that, Leja+20 is
         data-based, so leaving ε = 0 there keeps the prior identical to production.
       - **Revise p(z) together with the mass prior, not as an extra.** p(z) is NOT flat. It is
         Prospector-beta's `pdf_of_z_l20t18.txt` ("probability of observing a galaxy at z"),
         which is presumably built from the same Leja+20 / T18 abundances. That's unverified: ask
         Joel how it was made and for what depth. As a normalized density it is 0.22 at z 1,
         0.12 at z 3, 0.17 at z 4, 0.075 at z 6, 0.016 at z 8 and 0.001 at z 10, so z ≈ 6 sits at
         about 1/3 of z ≈ 1. It cost the z ≈ 6 solutions about 0.8 nats. It rests on the same 2018
         abundances, so a mass-prior floor without a consistent p(z) change would be
         inconsistent. Also check the bump at z = 4 (0.12 at z 3 → 0.17 at z 4), probably the
         L20 → T18 seam.
     - **How to choose ε:** on a secure spec-z sample across z (e.g. DJA graded spectra in
       ~/Projects/minerva, matched to MINERVA), counting both failure types: true high-z
       suppressed, and true low-z given spurious high-z weight. Not on these two galaxies alone.
     - **Cheap first step (no refitting):** reweight existing posterior samples by
       π_new / π_old. This is valid because the floor only adds probability where the old prior
       already had some. Do it for the windowed fits (`results/zwindow/`, `evidence_budget.py`)
       and get the solution weights of 2076167 and 2137258 against ε in minutes. Only the final
       choice needs reruns, plus the prior and loglike numerics tests, since the tabulated
       table rows change.
     - **Caveats:** check the spec-z grades of 2076167 and 2137258 first. 2248619 (no spec-z)
       is not evidence; Wren doubts it's high-z. This departs from Prospector-beta and from
       production, so agree it with Joel and Ikki with numbers in hand. Never change Joel's
       repo; the change goes in our tabulated prior or a fork.

**C. Production engineering**

7. **Bring memory down. PROFILED 2026-11-11** (Blanca job 29475943, Gold 5418Y, 16 fits of
   1212906 and 1152437 at the defaults; kit `blanca/memprof/`; outputs private,
   `nss_private_handoff/results/memprof/`). Peak RSS ~1.6–2.0 GB per fit, but the data the
   fit actually holds is only 0.15–0.35 GB (0.6 GB at the end). The rest:
   - **The JAX/XLA compile of the NSS step** (~28 s, also ~6–8% of fit time): ~0.85 GB in use
     at its peak, which sets a per-fit floor of ~1.25 GB private memory. Compiling the
     prior-start batch function adds ~0.35 GB transiently.
   - **glibc keeps what the compiles free:** ~0.8 GB held free for the rest of the fit.
   - **End of the run** (`log_weights`, mass draws, theta): the large arrays grow the heap
     on top of that, +0.3–0.5 GB. This is the per-fit peak in every base run, and it scales
     with iterations.
   - Not the cause: the prior start's 20,000-row batches (0.93 GB transient, below the compile
     peak; 5,000-row batches save nothing on Linux), the kept discards (2 MB), the 10⁶ log L
     values (8 MB), `update_info` at k=32 (0.1 MB per iteration). Shared libraries: 0.16 GB,
     counted once per node.
   - glibc settings (`MALLOC_ARENA_MAX=2`, fixed 1 MiB mmap threshold, 32 MiB trim threshold)
     remove the end spike: per-fit peak 1.85 → 1.54 GB (`time -v`), outputs bitwise identical,
     but fits are ~5% slower (+1 to +12%, paired).
   - Job level: the memory cgroup peaked at 20.4 GB for 16 fits (1.27 GB per fit), set by the
     simultaneous compiles at job start. 20 seeds of one galaxy also finish together, so their
     end spikes stack (this is what ran past 30G).
   - **FIXED 2026-11-11 (uncommitted): `malloc_tools.py`**, called from
     `nss_jax.run_nss_core`: malloc_trim after the prior start and after the step compile;
     large allocations mmapped after the NSS loop. glibc only; `NSS_MALLOC_TUNING=0` turns
     it off; summaries record `malloc_tuning` and `xla_flags`.
     - **Blanca job 29478805** (Gold 5418Y; 1212906 and 1152437 × seeds 0, 1; same node and
       seeds as the reference without the fix; dust prior `mass`, the new default):
       - **outputs bitwise identical in 4/4 fits;**
       - time per iteration ×1.000 (0.993–1.006);
       - peak private memory 1.69 → 1.31 GB, `time -v` 1.81 → 1.43 GB;
       - during the NSS loop 1.32 → 0.72 GB; end of run 1.69 → 1.13 GB.
     - The per-fit peak is now the step compile (1.31 GB), at job start, when all fits
       compile at once. So a 20-fit job needs ~27 GB: `fullfit_layout.sbatch` and
       `tasklist_layout.sbatch` now ask for `--mem=32G` (was 48G).
     - **Every CPU type checked** (jobs 29479115–29479120, 2026-11-11; fix vs no fix, same
       galaxies and seeds, 8 fits per node). On all six types: outputs bitwise identical
       (4/4 each), time per iteration unchanged (medians ×0.98–1.007), and the peak is the
       step compile. Median peak private memory with the fix (without):
       Gold 6130 1.30 (1.80), Gold 6230 1.20 (1.58), Gold 6342 1.30 (1.83),
       E5-2680 v4 1.32 (1.86), EPYC 7452 1.23 (1.81), EPYC 7413 1.37 (1.73) GB.
       The highest single fit: 1.37 GB (EPYC 7413; `time -v` 1.49). The AVX2-only chips
       compile to no more memory than the AVX-512 ones. 20 × 1.37 = 27.4 GB, so 32G holds on
       every type (~4.5 GB headroom).
   - **XLA flags (same job, each on top of the fix):**
     - codegen split count 1, 256-bit vectors, no expensive LLVM passes: bitwise identical,
       no change in memory or speed (the Mac's 30% slowdown for no-expensive-passes doesn't
       happen on Linux).
     - AVX2 cap: 1e-13-level differences, no memory change.
     - **Legacy fusion emitters (`--xla_cpu_use_fusion_emitters=false`):** halves the step
       compile's memory. Peak 1.31 → 0.87 GB private (`time -v` 0.99 GB); compile 19% faster.
       Results differ at the 1e-13 level (same iterations and logZ; percentiles move ~1e-13
       half-widths). But each iteration is ~6% slower (+1 to +10%; on the placements where
       other fits ran ~10% fast, it is still no faster than the reference), so likely ~9%.
       Not adopted: at 500k fits that costs more than the memory is worth, unless memory
       turns out to be the binding constraint.
     - Timing noise: fits on some cores of the same node ran ~10% faster (prior start 55 vs
       95 s), so paired ratios are good to ~±5%.
   - Under 1 GB without a slowdown: a persistent compilation cache with the photometry passed
     as arguments rather than constants (with item 10). It would also save the ~28 s compile
     per fit (~6–8% of fit time).
   - **In production use (dust-prior validation, 2026-11-11, 1490 fits on the fast node types,
     malloc tuning on):** no failures; peak RSS per fit median 1.42 GB, max 1.55, against median
     1.71 / max 2.07 in the earlier dust48 run without it (different galaxies, not paired).
8. **Update postprocessing** (Wren will take this on in a separate thread). Turn NSS fit
   outputs into the catalogue products `postprocess_parrot.py` makes now (percentiles, SFHs,
   stellar mass, sSFR, UVJ):
     - **What a fit writes now:** `<id>_<tag>.npz` holds the full weighted posterior: `theta`
       (all 26 parameters; with mass integrated out, logmass is drawn back once per sample),
       `w`, `logl`, `u`, `logz_draws`. `<id>_<tag>.json` has the summary.
     - `blanca/posterior_draws.py` only resamples 2,000 equal-weight draws per fit, for
       small transfers in the seed tests. It isn't a step production needs.
     - **Mapping:** production `postprocess_parrot.run_all` reads the dynesty h5 and calls
       `resample_equal(chain, weights)`, so `theta` / `w` take the place of `chain` /
       `weights`. Either write the h5 layout `prospect.io.reader.results_from` expects
       (fields unchecked), or have the fit write equal-weight draws directly.
     - **Storage:** 13.6 MB per fit (33,900 float64 samples, ESS ~7,900) is ~7 TB for 500k
       fits. 4,000 float32 equal-weight draws is ~0.4 MB per fit (~200 GB), still above the
       1,000 the postprocessing uses.
     - **Best-fit trap with mass integrated out:** `run_all` appends `chain[-1]` as the
       maximum-likelihood point. For NSS output with mass integrated out, `logl` is L′ (mass
       integrated out), so the top sample's logmass is a random draw, not the best fit. A
       real best fit means maximising over mass at that point; mass_marginal.py already
       finds the mass peak analytically.
   - Fix the python-fsps stale-redshift bug on the way (§7, Ikki): the first `predict`
     after a redshift change returns stale output, and postprocessing calls it once per
     posterior sample.
9. **Port into a fork of efsps-photfit** (under `wrensuess`; never commit to Joel's repo).
10. **A driver that runs many galaxies per process** (saves ~10 s of startup and the ~16 s
    JAX compile per fit).
11. **Checkpointing:** a preempted job currently restarts all 20 of its fits.
12. **Redshift-mode flags and provenance** in the outputs: solutions found and their weights,
    code and emulator versions, settings.
13. **CPU types.** **First timings, 2026-11-11** (the memprof jobs above; production
    defaults, one core per fit, but only 8 fits on each node, so a full node may be slower).
    Median ms per NSS iteration, and the ratio to Gold 5418Y (job 29478805):
    Gold 5418Y 2211 (1.00), EPYC 7413 2273 (1.03), Gold 6342 2437 (1.10),
    EPYC 7452 2952 (1.34), Gold 6230 3003 (1.36), Gold 6130 3790 (1.71),
    E5-2680 v4 4085 (1.85). The prior start scales about the same way (53–104 s).
    EPYC 9354 not in this set. **Allowed now (2026-11-12, Wren):** EPYC 9354, Gold 5418Y,
    plus EPYC 7413 and Gold 6342 (within ~10% of the fast types). **EPYC 7452 and Gold 6230
    (~1.35×) are also decent: for large runs they can and should be included** (more
    nodes beats per-fit speed); not in the default constraint, add `|rome|cascadelake`.
    Blanca tags: `milan` is only the two EPYC 7413 nodes (EPYC 7713 nodes are tagged
    `epyc7713`/`epyc7713p`); `icelake` is three Gold 6342 nodes and one untimed 32-core node.
    `tasklist_layout.sbatch` now uses `epyc9354|gold5418y|5418y|milan|icelake`. Gold 6130 and
    E5-2680 v4 (1.7–1.85×) stay out. Still to do:
    - time a full node (21 tasks) per type;
    - the Gold 6130 slowness: 1.26× the Gold 6230 here at the same clock (earlier, 51 ms
      against 31 ms per call for the Gold 6150, where clock speed explains only 1.3× of the
      1.6×). Test an AVX2 cap (`XLA_FLAGS=--xla_cpu_max_isa=AVX2`) and the hyperthreading
      setup. This serves the "no slow node types" goal.
14. **A production spot check:** refit a random ~1% of galaxies with a second seed at 52
    steps and compare. It's the only way to catch, in production, galaxies harder than the 21
    tested.
15. **An end-to-end test on ~1000 galaxies** through fit, postprocessing and catalogue.
16. ~~Commit and push.~~ Done 2026-11-11: the convergence work and the mass-marginal default
    are committed and pushed to `origin/nss-efsps` on wrensuess/prospector_minerva (public),
    up to `d94c518`.

**D. Optional speed-ups (not blocking)**

17. **blackjax lockstep waste: maybe 1.5–2×, medium effort.** The 100 chains run in
    lockstep, and only ~35% of the likelihood evaluations computed are used by any chain.
    Draw several candidate points per slice at once and take the first valid one (plain
    rejection within the bracket, a valid sampler). Must pass the toy and numerics tests.
18. **Emulator in float32: maybe 1.3–1.8× on the network's share, low effort.** Rounding at
    float32 (~1e-7) is far below the emulator's ±2.5% accuracy, but it needs the by-redshift
    numerics tests. It would also cut memory.
19. **25 inner steps** would save another 17%, with a somewhat worse tail (§4). Revisit after
    the broader validation.

**Not worth optimizing now:** prior draws (~20 s, ~5% of a fit) and per-fit startup (~10 s
plus the ~16 s compile; item 10 removes both).

**Rough ceiling:** at 32 steps, ~40–60k core-hours for 500k fits on the fast node types,
against production's ~600k. Items 17 and 18 might take it to ~25–40k.

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
  - the AGN prior: adopted 2026-11-11 (flat1_s025, §6 item 5); implemented in NSS and committed
    2026-11-12, together with the adopted dust prior (§6 item 5b);
  - **a later discussion of model + priors (Wren, 2026-11-12):** why NSS trails production on faint
    z > 5.5 galaxies (§6 item 5b, "New, separate issue"). NSS and production differ in redshift,
    mass and dust2 priors and in the model (E-FSPS with free nebular physics vs Parrot), not only the
    sampler; the priors explain about a third of the faint-end gap. A clean best-χ² comparison of
    the two solutions per galaxy was proposed but not run.

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
