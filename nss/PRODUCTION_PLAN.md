# NSS production-readiness plan

**Status (2026-11-16):** Wren has made decisions D1–D7 (§3); work proceeds in the order of §6.
The restart work of another session is committed (`4f7d337`) and passed its Blanca restart
test, so no file is off-limits any more. The only code change under this plan so far is the
opt-in float32 switch in `jax_model.py` (task S4, uncommitted, off by default). `STATUS.md`
was brought into line with this plan on 2026-11-16 (uncommitted).

**Rule for editing this file (Wren, 2026-11-16):** it must be consistent the whole way through.
A change in one place is propagated to every number, status, cross-reference, task row, step
and risk it touches, so the file is accurate when read top to bottom. No appended corrections
that contradict earlier text. The same holds for `STATUS.md`, `HANDOFF.md` and the READMEs.

Read order for a new agent or person: [`HANDOFF.md`](HANDOFF.md) (setup, code map, working
rules) → [`STATUS.md`](STATUS.md) (what has been measured and decided) → this file (what is left
to do before and during production, in what order, and how each piece is judged done).

## 1. What production has to deliver (Wren, 2026-11-16)

1. Fits that are **reliable**: robust to several redshift solutions, and consistent if the
   catalogue is re-run.
2. Fits that are **fast**: hundreds of thousands of galaxies per run on Blanca, one core each.
3. **Photo-z statistics similar to eazy's** (σ_NMAD and outlier fraction). Individual redshifts
   are not expected to match eazy: it is a template code, and SPS codes historically do worse on
   photo-z with better physics (D1).
4. **Real galaxies out to z ≈ 11 are very important.** Every validation reports z 7–9, 9–11 and
   > 11 separately, and nothing is adopted that lowers the recovery of real z ≥ 9 galaxies.
5. Post-processing products that are **made efficiently, stored compactly and easy for the team
   to use**; each band-set catalogue is a single complete product (D4).
6. **Prior re-weighting from the stored products** (e.g. other n(z) priors) without refitting;
   where re-weighting is not enough, the tools must say so and name the galaxies to refit.
7. **Easy for a postdoc to run**: "no looping through tons of scripts trying to deal with fits
   that cut off or were corrupted partway through".

### Scale (measured 2026-11-16 from the catalogues on `/Volumes/DarkPhoenix/Surveys/MINERVA`)

| Field | Catalogue | Rows | `use_phot = 1` | with ≥ 1 MB band (D3 definition) | with any MIRI mophongo flux |
|---|---|---|---|---|---|
| UDS | n3.0_m3.1_v1.2.1 | 345,792 | 229,769 | 216,107 | 107,925 |
| COSMOS | n3.0_m3.0_v1.0.1 | 294,126 | 219,770 | 148,816 | 181,238 |
| EGS | n2.0_m2.1_v1.3.1 | 520,875 | 270,269 | 156,938 | 96,639 |
| **Total** | | 1,160,793 | **719,808** | 521,861 | 385,802 |

- **Selection:** everything with `use_phot = 1` (Wren). GOODS-N later, perhaps GOODS-S.
- **Three runs:** BB only, BB+MB, BB+MB+MIRI (band sets defined in D3).
- **Fits needed:** 2.16M if every run fits every galaxy; **1.63M** because a galaxy whose band
  set does not change between two runs is not refit (D4, task S7): 719,808 + 521,861 + 385,802.
  At the weekend run's median of 8.2 min per fit (all allowed CPU types) that is ~220k
  core-hours; ~135k at the fast types' ~5 min. STATUS.md's "~500k fits" and "40–60k core-hours"
  predate these counts.
- **Throughput:** the preemptable QoS allows 2,000 cores per user (D6), i.e. ~95 jobs of 20
  fits at once, ~14,000 fits per hour at 8.2 min per fit: ~5 days of continuous full-cap
  running for 1.63M fits, more with preemptions and queue waits.
- **Faint galaxies:** with all bands, < 2% of `use_phot = 1` galaxies have total S/N < 5
  (median 14–19; no error floor). BB-only fits will sit lower (task R7).
- **Storage:** today ~1.08 MB and 4 files per fit, all in one directory per run: ~1.8 TB and
  6.5M files for 1.63M fits. The permanent home has 4.2 TB free and `/projects` 27 GB (D6), so
  the compact format (P2, target ~0.3 MB per fit, ~0.5 TB in all) and packing (O3) must land
  before production.

## 2. Where things stand (review of 2026-11-16)

Four read-only reviews (core fit code, post-processing, Blanca job scripts, photo-z results).
Line numbers refer to the working copy of 2026-11-16, which had uncommitted edits.

**Sound, keep:**
- The sampler setup (STATUS §4–5): 10⁶ prior start, mass integrated out, 32 inner steps. On the
  21 hard galaxies × 20 seeds every seed finds every redshift solution, median R̂ 1.004.
- Determinism: every random number derives from `--seed`. Same seed gives identical iterations
  and logZ on 7 CPU models; percentiles differ by ≤ 4e-12.
- The file-based restart logic (`pipeline_status.py`, `tasklist_layout.sbatch`; committed
  2026-11-16 as `4f7d337`; Blanca restart test with injected failures PASS, 18/18): state
  judged from files only, atomic writes, requeue on preemption and before the time limit,
  bounded retries. Its gaps at production scale are task O0.
- Per-fit postprocessing at the end of each fit (3.5 s), spectra as a separate restartable pass,
  streaming catalogue builder.

**Gaps, by goal:**

| Goal | Main gaps found |
|---|---|
| Reliable | Convergence was only tested with the full non-MIRI band set and on hand-picked hard galaxies; BB-only (more multi-modal), BB+MB+MIRI and a representative sample are untested. No per-fit convergence or multi-modality diagnostic is saved. No iteration or time cap by default; a loop condition that never ends if every live point has log L = −inf (`nss_jax.py:200`); `timed_out` fits count as done; a JSON can hold NaN. |
| Photo-z | On the 1041-galaxy spec-z test (specz1k) NSS has 16.6% outliers against eazy's 8.4%, but that sample over-represents z > 5.5 and contains 61 galaxies selected because production failed. Weighted back to the parent spec-z pool: **NSS 8.1 ± 1.2%, eazy 5.0%**; NSS matches eazy at z < 3, and the excess sits at z 3–5.5 and 5.5–9. **163 of 173 NSS outliers have ≥ 1% of their posterior at the spec-z (108 have > 10%)**, and the best likelihood at the truth is within ~1 nat of the best elsewhere: a prior/model problem, not a sampler one. The truth is less clean than first thought: re-checked by Z1, only 90 of the 173 outliers have a grade-3 DJA v3 spectrum within 0.3″ at the same z (98 with v4.2), not 172 as first reported. At z ≥ 9 specz1k has only 14 spec-z (the Z1 benchmark has 34); NSS is biased +0.04 there (eazy −0.005). The emulator itself is off by 0.048 mag on realistic SFHs (see J1). |
| Fast | Fixed overhead per fit is ~58–70 s of ~390–520 s (imports, build, compile, prior start), because the photometry is compiled in as constants and every galaxy recompiles. Gold 6230 nodes ran 1.98× slower than EPYC 9354 at full load (39% of weekend fits), against 1.35× in one-core timings; unexplained. Job-level idle cores: lb master 5%, list tail ~5%, spectra stage 2–4%, partial fits lost at the 2 h limit on slow nodes. blackjax lockstep waste (only ~35% of likelihood evaluations used) is untouched. |
| Products | ~1.08 MB per fit where ~0.3 MB would do (duplicate 4000-draw file, derivable columns, uncompressed draws, a wavelength grid in every file). No p(z) product, no redshift-solution table, no quality or multi-modality flags in the catalogue. Code version is "unknown" in every Blanca product (the bundle has no `.git`; `BUNDLE_INFO.txt` is never read). Collected file names collide across runs. No column/units documentation, no reader. |
| Re-weighting | Works today from 4000 equal-weight draws (validated: re-weighting the new-prior draws to the old prior reproduces the old refits), but that file is a third of the storage cost, equal-weight draws cover the tails poorly, and nothing reports when a galaxy needs a refit. A `FORMAT_VERSION` bump currently forces refits, because production keeps no samples to redo postprocessing from. |
| Easy to run | From a new catalogue to a final catalogue is ~8 manual steps, with array ranges computed by hand. No band-set switch exists; MIRI is reachable only through `EXTRA`, which also turns the 14 MB sample file back on, the postprocessing rerun drops the MIRI flag, and the reader finds no MIRI columns at all in the current `_mophongo` catalogues (they are named `_ap` and `_mophongo`). `catalog_runs.sbatch` knows only UDS and COSMOS (case-sensitive match fails on real catalogue names), counts files with `ls` (fails above ~50k files) and deletes and rebuilds everything in a 2 h preemptable job. Status scans are quadratic in files per directory. Running the status check by hand during a run deletes live temporary files. No guard against submitting the same run twice. Results were written under `/projects`, which is nearly full. |

## 3. Decisions (all made by Wren, 2026-11-16)

| # | Decision | Outcome |
|---|---|---|
| D1 | What "as good as eazy" means | **Aggregate statistics, not agreement:** σ_NMAD and outlier fraction similar to eazy's, weighted to the parent spec-z pool; Wren judges "similar" from the benchmark report (Z1). eazy is a benchmark only: no eazy-agreement flag as a quality criterion and no hybrid redshift. |
| D2 | Speed work that changes fit numerics | **Yes:** multi-galaxy driver (S3), float32 network (S4, on only if it shows a real one-core gain), time-boxed lockstep prototype (S5). |
| D3 | Band sets | **"Pre- and post-MINERVA" (arXiv:2507.19706).** BB = every HST band + JWST wide bands + F410M. MB = every other NIRCam medium band (F140M, F162M, F182M, F210M, F250M, F300M, F335M, F360M, F430M, F460M, F480M). MIRI = the **mophongo** fluxes; a further MIRI error term is Wren's to handle separately; for now the error is never below the aperture catalogue's, read as e = max(e_mophongo, e_aperture) (the mophongo error is the smaller one for 22–70% of galaxies, by band). The lists are defined in code, not taken from the catalogues' `n_bands_mb`, which counts F410M in COSMOS but not in UDS. Bands without an emulator band (F098M, F470N) stay dropped: not needed. Open, for GOODS-N only: its pre-MINERVA F182M, F210M (FRESCO) and F335M (JADES). |
| D4 | Skip refits for unchanged band sets | **Yes,** and every band-set catalogue is complete on its own: a skipped galaxy's results are copied into it (every product, not only the catalogue row), with a column naming the run the fit came from. |
| D5 | The other session's restart work | **It finishes first; the review's fixes (O0) go on top.** Done 2026-11-16: committed as `4f7d337`, Blanca restart test PASS, so O0 and every task touching `nss_jax.py` can start. |
| D6 | Where things live on Blanca | **Work files and results on `/scratch/alpine/$USER` during a run; packed, validated shards and collected catalogues copied to `/pl/active/suess-minerva/` (the permanent home) before scratch's 90-day purge; only code and environments under `/projects`.** Facts behind this, from the CURC docs and Wren's commands on 2026-11-16: `/projects/kasu8993` 224 of 250 GB used (27 GB free); `/scratch/alpine` 9.5 TB free, 20M-file limit, files removed 90 days after creation, no backup; `/pl/active/suess-minerva` 9.1 TB, 4.2 TB free, shared with Ikki; preemptable QoS 24 h wall limit, 2,000 cores and 1,000 running jobs per user, no submit limit, 120 s grace on preemption, requeued by default; `MaxArraySize` 4,000,001, `MaxJobCount` 50,000 (array size is no constraint). |
| D7 | Emulator | **Joel retrains the emulator before production,** once Emilie's E-FSPS is final (expected the week of 2026-11-16), with mixture training draws instead of flat ones (his estimate: ~0.02 mag accuracy). He asks for 3–5 days' notice. We need it at Gate A (§6), i.e. about 2026-12-04; an earlier draft would reduce re-tuning. Our requests: task J1. |

Standing decisions (STATUS §5, HANDOFF §7) still hold: one core per fit; robustness before
speed; tests span redshift on real galaxies; MIRI only in the MIRI run; never commit to Joel's
or Emily's repos; report and wait for go-ahead before every commit; exact copy-paste commands for
every Blanca step; results come back by rsync.

## 4. Design choices this plan rests on

1. **One run manifest per (field, band set).** A small JSON written by `nss_run.py init`:
   catalogue path, sha256 and row count; selection; band set and the exact filter list; sampler
   and prior settings; seed rule; commits of nss, efsps-photfit and E-FSPS; emulator md5;
   library versions; output format version; results root. Every fit JSON and every collected
   product carries the manifest's hash, and the status check refuses results made under a
   different manifest. This is what makes a re-run reproducible and a mixed-up run detectable.
2. **Keep a compact weighted posterior sample per fit, and derive everything else from it.**
   The nested-sampling dead points with their weights (parameters, the drawn mass, log L),
   not only equal-weight draws. Dead points cover low-posterior regions densely, which is what
   re-weighting to a different prior needs. Postprocessing products can then be regenerated
   from this store, so a format change never forces a refit, and likelihood changes (e.g. a
   different emulator error term) can be re-weighted by re-running the emulator on stored
   points (~40 µs each). Size and thinning are chosen by test (P1), not by guess.
3. **Pack finished work.** When a job element's galaxies are all done and validated, their
   per-galaxy files go into one shard file per element; per-galaxy files are then removed.
   A field-run is then a few thousand files, not a million; status reads shard markers, and
   copying to PetaLibrary and back to the Mac is cheap.
4. **Everything that changes fit numerics lands before the validation gate** (band sets, prior
   changes from the photo-z work, speed changes, Joel's retrained emulator). The gate is run
   once, on the frozen configuration, and production starts only from a passed gate.
5. **One driver, idempotent.** `nss_run.py submit` works out what is missing and submits only
   that; running it again after any interruption is always safe and always the right next step.

## 5. Tasks

Size: S ≤ 1 day, M 2–4 days, L ≥ 1 week of agent work, excluding Blanca turnaround (Wren or
the postdoc runs every Blanca command). "Numerics" = the fit's numbers change, so the task must
be finished before Gate A (§6). Status: todo / doing / done / dropped.

### R. Reliability of each fit

| ID | Task | Done when | Size | Numerics | Status |
|---|---|---|---|---|---|
| R1 | Guards: iteration cap, default `--max-seconds`, NaN-safe termination test, JSON never holds NaN. A `status` field (`ok`, `timed_out`, `capped`, `nonfinite`, `bad_input`) that `pipeline_status.py` and `make_catalog_nss.py` honour. Deterministic input failures (id absent, no usable band) are recorded once, not retried three times | toy with all log L = −inf ends; unit tests for each status; a `timed_out` fit is not DONE | S | no | **done 2026-11-17, committed 2026-11-19**: defaults `--max-iterations 5000`, `--max-seconds 5400` (sampling loop only); non-`ok` fits get no `_post.h5`, `nonfinite` no `.npz`; only per-object refusals are `bad_input` (exit 0), a broken catalogue still exits non-zero. `test_fit_guards.py` PASS (toys + 6 galaxies z 0.12–8.49); old vs new code identical on 6 small fits and a production-setting fit. Blanca restart test (job 29666545) PASS: the `bad_input` task recorded once, not retried |
| R2 | Per-fit diagnostics saved in the JSON: insertion-rank test (needs birth log L kept), slice shrink/acceptance summary, and a redshift-solution table (solutions found by gaps in the weighted z samples: z range, weight, best log L, median mass) | reproduces `analyze_rhat.py`'s solution weights on the 21-galaxy set; insertion-rank p-values uniform on the Gaussian toy | M | no | todo |
| R3 | Seed rule in the manifest (same seed for all galaxies of a run; spot-check run uses seed + 1). Record jax/jaxlib/blackjax/numpy versions and CPU model per fit | two submissions of the same manifest on the same CPU type give bitwise-equal JSONs for 20 galaxies | S | no | todo |
| R4 | Convergence in each band set: the 21 galaxies × 20 seeds for BB only, BB+MB and BB+MB+MIRI | STATUS §5 criteria reported per band set; any setting change (e.g. more inner steps for BB only) decided by Wren | M + Blanca | — | todo |
| R5 | Representative two-seed test: ~2,000 random `use_phot = 1` galaxies stratified by field and F444W magnitude, 2 seeds, each band set | report: share with seed-to-seed \|Δz50\|/(1+z) > 0.15, solution-weight differences, Δ log M in posterior half-widths, by magnitude; Wren sets the pass line from the report | M + Blanca | — | todo |
| R6 | High-precision reference for ~5 galaxies, one of them at z > 7 (STATUS §6 A2), and the logZ error inflation factor (A3) recorded in the products | width criterion checked; factor written to catalogue header | S + Blanca | — | todo |
| R7 | Mass integral on real faint galaxies and BB-only fits (STATUS §6 A1 "next"), including the `A1 ≤ 0` path without an edge guard (`mass_marginal.py:312-326`) | integral within 0.01 nats of brute force where the posterior weight is, on ≥ 20 real galaxies with total S/N < 5 | S | maybe | todo |

### Z. Photo-z and the emulator

| ID | Task | Done when | Size | Numerics | Status |
|---|---|---|---|---|---|
| Z1 | Standing benchmark: every clean spec-z in UDS, COSMOS and EGS (grade-3, one-to-one matched, duplicates and close neighbours flagged, not dropped), **including every available spec-z above 7**, split by id hash into a tuning half and a confirmation half. One frozen metrics script: z50, outlier \|dz\|/(1+z) > 0.15, NMAD, bias, by z (with 7–9, 9–11, > 11 separate) and magnitude, raw and weighted to the parent pool, eazy non-fits and `spec_suspect` handled explicitly; eazy and production dynesty alongside. Fit in all three band sets, full sample files kept | script reproduces the 2026-11-16 numbers on specz1k; benchmark fits done (10,124 galaxies × 3 band sets, ~4.2k core-hours) | M + Blanca | — | **sample and script done 2026-11-17** (`nss/photoz_metrics.py`, `test_photoz_metrics.py` PASS, `nss/blanca/make_benchmark_sample.py`; private `nss_private_handoff/benchmark/`): 10,124 galaxies (all `use_phot = 1` spec-zs plus 6 at z > 7 with `use_phot = 0`), clean 4,998, 34 at z ≥ 9 (17 clean); specz1k numbers reproduced exactly; eazy on clean 6.4% / NMAD 0.013. The sample was built on DJA v3 and is **to be rebuilt on Z1b's v4.5 seg-map catalogue** once Wren has checked it (Wren chose v4.5). Fits not yet run |
| Z1b | Spec-z truth from DJA by segmentation map (Wren's decisions 2026-11-17): DJA `nirspec_public_v4.5` (grade-3 z identical to v4.4 except rubies-egs52/28470, 8.68 → 6.38), grade 3 only, spectrum position = the table's APT position (msaexp does not re-centre: `profcen` ≡ 0 in the Zenodo v4.4 table), no astrometric offsets (DJA and MINERVA both on Gaia via grizli; measured per-programme medians ≤ 0.02″ except excels-uds02 0.16″), owner = MINERVA segmentation id at that position, photo-zs never used to decide. Several grade-3 spectra of one source: highest resolution wins (H, M, prism); disagreements > 0.01(1+z) flagged. Possible blends flagged (any other segment with F444W light in the slit aperture). One spec-z goes to ONE source only, shreds included: the piece with most of the slit's light (meeting 2026-11-18). The slit is never more than 2″ from its target. Output: one catalogue of MINERVA id, ra, dec, z_spec (+ flags); Wren takes it from there | Wren's by-eye check of the change list (z ≥ 7 + random 50) | M | — | **catalogue built 2026-11-17, owner rule revised after Wren's verdicts of 2026-11-17 (13) and 2026-11-18 (18 more)** (private `nss_private_handoff/benchmark/specz_v45/`, script `dja_specz_catalog.py`; `veto_spectra.csv` = spectra Wren rejects, `overrides.csv` = her by-eye owner for single spectra (3), `verdicts.csv` = her 31 verdicts): owner = the segment with most F444W flux within 0.10″ of the spectrum position, except that a catalogue centroid within 0.10″ wins (compact source on a neighbour's segment); seg pixel where the mosaic has no data (EGS: 1,036 spectra, mostly CEERS/gto-wide-egs1, the v2.0 F444W mosaic on disk is zero there); owner centre > 2″ from the spectrum → unassigned (68 spectra, owners 2.1–66″ away: giant segments and sources missing from the catalogue; every one Wren checked was wrong). Shreds: z on the brightest piece only (no shred rule or SED similarity needed). Possible blend = any other segment with light in the aperture (165 sources; 2026-11-17's ≥ 20% cut gave 72). 10,396 sources; vs the catalogue z_spec: 260 extra copies, 10 moved, 14 z changed, 1,282 added. All 31 verdicts reproduced (3 through overrides) except COSMOS 2211016: its 2026-11-17 "fine" was for z = 1.21 from a spectrum 3.8″ from its centre, which the 2″ rule now drops; it keeps z = 0.903 from the spectrum on its centre (= the catalogue value), to confirm with Wren. **Open (Wren):** full EGS F444W mosaic; shutter outlines (msaexp) later if needed. The first pass (v3, nearest centroid) is superseded |
| Z2 | Per-outlier diagnosis: best χ² in a window at the spec-z against best χ² at the chosen z, by optimiser (STATUS §7's unrun comparison), per failure class (on specz1k: low-z trap 58, break confusion 57, ×1.3 line confusion 39, ×0.75 line confusion 19) | table per class: truth fits better / equal / worse; says which classes priors can fix | M | — | todo |
| Z3 | Levers testable by re-weighting, on the tuning half only: nebular and line priors (gas_logu, ionising-spectrum deltas), q_PAH tied to mass or metallicity, a floor on the high-z mass prior (STATUS §6 item 6). Not a candidate: any redshift prior that suppresses z ≥ 9 (the T8 taper of STATUS §6 item 6b loses 2 of the 14 real z ≥ 9 galaxies). Choose by pool-weighted outlier rate and NMAD with no z or magnitude bin made worse and z ≥ 9 recovery not reduced; confirm on the other half by re-weighting, then by refit. Two rounds at most | chosen prior set confirmed by refit on the confirmation half | L | yes | todo |
| Z4 | Levers that need refits: Lyα damping wing / CGM (priority: NSS bias +0.04 at z ≥ 9); the retrained emulator and its error term (J1), on which Z3's choice is re-confirmed | each tried on the benchmark; adopted only if it helps on the confirmation half without lowering z ≥ 9 recovery | M + Blanca | yes | todo |
| J1 | Requests to Joel for the retrain (D7). Message drafted 2026-11-16 for Wren to send. Asked for: one mixture component drawn from the Prospector-β SFH prior (the network was trained on SFR ratios uniform in ±4, where histories as smooth as the prior's are ~7 in 10⁹ of the draws; its error on such histories is 0.048 mag against 0.012 on its own held-out set); coverage kept out to z 15; an emulator error term that matches the new training; duste_gamma sampled up to 1; one cosmology for distances and ages (training used WMAP9, the fit's ages use Planck18; Planck18 preferred). To stay the same: the 29 parameters and ranges, the 42 bands, Chabrier IMF, w256, file layout. Not asked for: missing bands (D3). Nothing to do about float32 (the weights already are) | message sent; new md5 registered in the manifest; numerics tests pass on it | S | yes | todo (drafted) |

### S. Speed

| ID | Task | Expected gain | Done when | Size | Numerics | Status |
|---|---|---|---|---|---|---|
| S1 | Why Gold 6230 is 1.98× at full node load: time a full node per CPU type with and without `XLA_FLAGS=--xla_cpu_max_isa=AVX2` (STATUS §6 item 13) | up to ~15% of total if the 6230 returns to 1.35× | per-type table; flag adopted or type dropped | S + Blanca | 1e-13 | todo |
| S2 | Fixed chunk shapes so the last partial chunk does not recompile (`nss_jax.py:141-144`) | a few s per fit | bitwise-equal outputs | S | no | todo |
| S3 | Many galaxies per process: photometry passed as (padded) arguments, one compile per process; reuse the emulator outputs of the 10⁶ prior draws, which do not depend on the galaxy at a fixed seed (STATUS §6 item 10) | ~13–15% | numerics tests by z pass; 21-galaxy convergence unchanged; peak memory per process measured | M | yes | todo |
| S4 | Network hidden layers in float32 inside the float64 likelihood (STATUS §6 item 18). The weights are float32 on file (trained with x64 off), so this reproduces the training arithmetic. **Switch built 2026-11-16, uncommitted, off by default:** `jax_model.JaxLogLike(model, emu_dtype="float32")` or `NSS_EMU_DTYPE=float32`; the fit log prints a line when it is on (not yet in the fit JSON; add `emu_dtype` to the summary in `nss_jax.py` with R1). **Tested on the Mac** on six galaxies from z 0.12 to 8.5 (2204022, 1215494, 1212906, 1197855, 1152437, 1201575), 20,000–40,000 prior draws each over z 0.01–15: with the switch off, network fluxes, log L and mass-integrated log L′ are bitwise equal to the committed code and `test_jax_loglike.py` passes; with it on, the network output differs by 1e-6 mag median and ≤ 3e-4 arsinh mag in every z bin including 9–11 and 11–15 (emulator error: 0.02–0.05 mag), log L at posterior draws by 1–3e-4 nats median and ≤ 5e-3, mass-integrated log L′ within 1000 nats of the best prior draw by ≤ 0.08, with no finite/non-finite flips. **Costs:** fits are no longer bitwise equal to float64 ones or to the numpy reference, and log L gains ~1e-4 nats of fine-scale noise. **Gain:** only 3–15% in wall time and ±7% in CPU time on the Mac, which is multi-threaded and does not count. **One-core Blanca timing, 2026-11-16** (jobs 29643430 float64 and 29643432 float32 on bgpu-casa1, Gold 6230, 20 fits per node, 1197855 seeds 0–19; private `results/f32_timing/`): float32 is **8–9% faster** (time per iteration ×0.914 paired by seed, 16–84% 0.85–0.94; prior start ×0.95; total wall ×0.92: median 676 → 620 s). Same answer: iterations 125 in both, logZ 538.35 ± 0.11 against 538.30 ± 0.14 (difference +0.05 ± 0.04), z50 3.909 in both, log M 10.721 against 10.722. One galaxy and one CPU type; the AMD types are untimed. Proposed: make float32 the default at Gate A, validated in Gate B with everything else; Wren to confirm | ~8% (measured on Gold 6230) | Wren's go-ahead; default switched; setting recorded in the JSON (R1); Gate B on it | S | yes | doing (measured; default not switched) |
| S5 | Lockstep waste: several candidate points per slice step (STATUS §6 item 17). Prototype time-boxed to 3 days | 1.5–2× if it works | toys unbiased; ≥ 1.4× on one Blanca core; 21 × 20 convergence unchanged; else dropped | M–L | yes | todo |
| S6 | Job-level waste: no fit started that cannot finish before the limit; chunk and time limit set from the CPU type (the QoS allows 24 h); spectra stage packed 20 wide or run as its own array | ~10% | idle core-time per element < 5% in the pilot | S | no | todo |
| S7 | Do not refit a galaxy whose band set is unchanged from an earlier run (D4): no MB band → the BB fit serves BB+MB; no MIRI flux → the BB+MB fit serves BB+MB+MIRI | 25% of fits (2.16M → 1.63M) | driver skips them; the collector copies every product of the earlier fit into the later catalogue, with a source-run column (P4) | S | no | todo |

Not planned: fewer inner steps or live points (robustness first; STATUS §5), mid-fit
checkpointing (a preemption loses at most part of one fit per core).

### O. Operations

| ID | Task | Done when | Size | Status |
|---|---|---|---|---|
| O0 | On top of the committed restart logic (D5), fix what the review found: lb exit status ignored (mpirun death counts 20 fits as failed); no circuit breaker when most fits of a pass fail (bad node, missing catalogue, full disk); `exit 0` with work left; preemptions alone can use up `MAX_REQUEUE`; one killed spectra process charges a failure to every galaxy behind it; `_post.h5` validity is shallow and writes are not fsynced; a transient read error counts as "missing" and triggers a refit; status exit codes 1 and 2 collide with Python's own; a refit leaves the old JSON in place until the end | each has a unit test or a chaos-test case | M | **done 2026-11-17, committed 2026-11-19**: failures judged from each task's own exit-code file; circuit breaker (≥ 5 fits and ≥ 50% new failures → requeue once, exit 6 on a second trip); job exit codes 0/1/3–9; self-requeues counted apart from preemptions; spectra failures charged only to started galaxies; deep `_post.h5` checks with a 60 s hang guard (no false positives on 1,052 real files); `--clean` removes stale outputs before a refit, status by hand is read-only; one directory listing per scan (50k fits: 7.3 s); `RESULTS_ROOT`; per-element lock; malformed task lines reported. `test_pipeline_status.py` PASS, Mac chaos rehearsal 31/31. Known gap: arviz's daily stamp-file race can crash parallel spectra imports (efsps env, not fixed). Blanca restart test with chaos PASS 2026-11-17 (job 29666545, after a checker double-count fix) |
| O1 | Run manifest (§4.1) and `--band-set {bb,bb_mb,bb_mb_miri}` in `photometry.py`/`nss_jax.py` with the D3 lists, the mophongo MIRI columns and e = max(e_mophongo, e_aperture); the band set in the output name; `EXTRA` appends instead of replacing defaults; the band set passed to postprocessing reruns; the catalogue names `f275wu`/`f336wu` mapped to the network's UVIS bands | three band sets of one test galaxy give three distinct, correctly labelled outputs; MIRI fluxes and errors read from a current `_mophongo` catalogue match the rule | M | todo |
| O2 | The driver: `nss_run.py init` (catalogue + band set → manifest and task list from `use_phot = 1`, minus galaxies S7 skips), `submit` (builds the arrays, submits only elements with work, stays within 2,000 cores, refuses if that run already has jobs queued), `status` (read-only, fast, one line per element plus totals and failures with reasons), `collect`, `reweight`, `qa`. No hand-edited lists or array ranges | a field is launched, interrupted and finished using only `submit` and `status` | M–L | todo |
| O3 | Output layout (D6): sharded work directories on `/scratch/alpine`, per-element packing into shard files (§4.3), copy of finished shards to `/pl/active/suess-minerva/`, space check on both before submit, logs of successful fits dropped at packing | status of a 230k-galaxy run returns in seconds; file count per field-run in the thousands; nothing written under `/projects` | M | todo |
| O4 | Bundle and provenance: versions from `BUNDLE_INFO.txt`, library versions, host and CPU, Slurm job id into every JSON and product | no "unknown" code version in any Blanca product | S | **done 2026-11-17, committed 2026-11-19**: new `provenance.py`; commit, dirty flag and efsps commit from git or `BUNDLE_INFO.txt` (now key=value), library versions, host, CPU, Slurm ids, catalogue path/size/mtime, emulator md5 in every fit JSON, `_post.h5` and catalogue header ("MIXED" + warning when fits disagree); `make_catalog_nss.py` adds `fit_status`, `ess`, `timed_out` and masks non-`ok` rows. Bundle provenance checked by unpacking here; Linux paths (/proc/cpuinfo, RSS units, Slurm vars) not yet run on Blanca |
| O5 | Chaos test extended and run on `preemptable` with ≥ 2 elements: real preemption, mpirun death, missing catalogue, full disk, truncated-but-openable h5, duplicate submission, status during a run, the `MAX_REQUEUE` path; assert preempted fits have zero failure counts | `RESULT: PASS` on Blanca | M + Blanca | todo |
| O6 | Runbook (one page, exact commands) and a dry run: the postdoc runs the pilot (Gate C) from the runbook alone | pilot finished without help from the author | S | todo |
| O9 | Blanca facts | recorded in D6. One left: whether `/pl/active/suess-minerva` is visible inside a Blanca job (it is on the login node); if not, O3 copies from a login or compile node | S | done except the compute-node check |

### P. Products

| ID | Task | Done when | Size | Status |
|---|---|---|---|---|
| P1 | Weighted-sample store and `nss_run.py reweight`: new prior (or likelihood setting) in, re-weighted catalogue columns out, with the effective sample size per galaxy and a list of galaxies that need a refit, which `submit` accepts. Choose what to store (all dead points above a weight cut, quantised; or a thinned set) by test against the existing ground truth: 673 galaxies fitted under both the old and the production prior | re-weighted z weights match the refits within seed noise for galaxies above the ESS threshold; threshold and bytes per fit recorded | M | todo |
| P2 | One compact per-fit file and a single format bump before production: draws as one compressed array without derivable columns, no per-file wavelength grid, spectrum percentiles in quantised log (3 or 5 of them), float32 throughout, no separate 4000-draw file | ≤ 0.35 MB per fit with spectra and the P1 store (≤ 0.6 TB for 1.63M fits, inside PetaLibrary's 4.2 TB); every `test_postprocess_nss.py` check still passes | M | todo |
| P3 | Postprocessing regenerable from the P1 store, so a later format change is a postprocessing rerun, not a refit | regenerated file equals the original | S | todo |
| P4 | Collected products per (field, band set), each complete on its own (D4), run name in every file name: FITS catalogue with quality columns (`status`, χ², bands used, ESS, convergence flag, number of redshift solutions, second solution's z and weight, source run); p(z) on a common grid to z = 15 from the full weighted posterior; redshift-solution table; chains, SFH, spectra as now; full provenance in headers; units as attributes | built for the pilot; reviewed by Wren | M | todo |
| P5 | Catalogue builder: any field (from the manifest), incremental and resumable, atomic outputs, one job per field, deep validation that sends a bad file back to the fit queue | 230k-galaxy field built in one job; killed and resumed in a test | M | todo |
| P6 | Team documentation and a small reader module: columns and units, how to get p(z), solutions, draws, SFHs and spectra, what differs from spsv1.0 (stellar mass table, SFR definition, best fit, χ², 10 SFH bins, cumulative formed mass), one example notebook | a team member not involved in the fitting loads p(z) and an SFH from the README alone | S–M | todo |

## 6. Order of work, gates and expected timing

Dates are estimates made on 2026-11-16, assuming work starts 2026-11-17. The uncertain parts
are the Blanca round trips (Wren or the postdoc runs every command), the photo-z work (two
rounds at most), the lockstep prototype, Thanksgiving week, and the date E-FSPS is final.

| Step | Work | Estimate |
|---|---|---|
| 1. Foundations and operations | R1, R3, S2, O4, O0 at once; then O1 → P1 → P2, P3 → R2; then O2, O3, P4, P5, S6, S7 → O5 | Nov 17 – Dec 2 |
| 2. Speed and photo-z, in parallel with step 1 | S1 and S4's Blanca timings, S3, S5 prototype, R7; Z1 and Z1b (benchmark sample, slit-based truth), then its fits (today's code, full sample files kept, so P1 and Z2–Z3 have data), Z2, Z3; J1 sent this week | Nov 17 – Dec 5 |
| 3. **Gate A, configuration freeze** | Joel's retrained emulator swapped in (needed by ~Dec 4; D7); one commit that the manifest names; every numerics suite passes on it (HANDOFF §3 quick checks plus `test_mass_marginal`, `test_dust_prior`, `test_nss_priors`, `test_postprocess_nss`, `test_spectra_nss`, `test_pipeline_status`) | Dec 7 – 9 |
| 4. **Gate B, validation on the frozen configuration, per band set** | R4 (21 × 20 seeds), R5 (two-seed sample), R6, Z1 benchmark re-run with Z4's confirmation (10,124 × 3), O5 chaos test: ~45k fits, ~6k core-hours. Reported criterion by criterion, z 7–9 / 9–11 / > 11 separate; Wren decides pass | Dec 9 – 16 |
| 5. **Gate C, pilot** | ~10,000 random `use_phot = 1` galaxies per field, BB+MB, run by the postdoc from the runbook (O6) on `preemptable`: fit, spectra, packing, copy to PetaLibrary, catalogue, QA report, one re-weighting demonstration. Measures failure rate by status, wall time per CPU type, idle core share, storage per fit, time for status and collect. Fix what it shows; rerun if anything in the fit changed | Dec 16 – 23 |
| 6. Production | One field and band set at a time: BB+MB, then BB only, then BB+MB+MIRI, skipping unchanged galaxies (S7). During each run the 1% second-seed spot check (STATUS §6 item 14) through the same driver; at the end a QA report (failure counts, flags, seed-pair agreement, photo-z on that field's spec-z), packing, copy to PetaLibrary, scratch released | early January, 2–3 weeks |
| 7. After | Re-weighting studies (n(z) variants and others) from the stored samples; refit only the galaxies the tool lists. GOODS-N (and GOODS-S) through the same driver once their catalogues exist and D3's open point is settled | — |

## 7. Risks to keep in view

- **Photo-z statistics may stay short of eazy's with physically motivated priors.** The truth is
  usually in the posterior but down-weighted, and the likelihood barely separates the solutions.
  Z3–Z4 are limited to two rounds; if the gap at z 3–9 remains, Wren chooses between shipping at
  that level and delaying (D1).
- **High redshift is thinly tested.** specz1k had 14 spec-z at z ≥ 9; the Z1 benchmark
  has every one available (the v4.5 seg-map catalogue has 42 at z ≥ 9: 34 at 9–11, 8 above 11), and no change may lower their recovery.
- **Tuning priors on the test sample.** Hence the tuning/confirmation split in Z1.
- **The retrained emulator arrives late or behaves differently.** Steps 3–6 slip with it, and
  prior choices tuned on the current network are re-confirmed on the new one (Z4). An early
  draft from Joel reduces this. A change after Gate B is a new run, which the manifest makes
  clean.
- **Shared limits.** 2,000 preemptable cores per user sets the pace of production, and
  PetaLibrary's free space is shared with Ikki.
- **More than one session in one worktree.** It happened on 2026-11-16 (the restart work). If
  another session is active here, agree which files each owns before editing, and never
  unpack a new bundle on Blanca while a job is running from the code there.

## 8. Where the review material is

- This plan's numbers: STATUS.md (§3–§6), and the four reviews summarised in §2.
- Photo-z tables behind §2 and the Z tasks: private, `nss_private_handoff/prodprior_weekend/`
  (`analyze_specz1k.py`, `reweight_variants.py`, `specz1k_galaxies.csv`,
  `reweight_specz1k.csv`) and `nss_private_handoff/specz1k/`. The by-bin and pool-weighted
  numbers of 2026-11-16 were computed in a session scratch folder and are not saved; Z1's
  metrics script replaces them.
- Old pipeline for comparison: `postprocess_parrot.py`, `make_final_sps_catalog.py`,
  `submit.py` at the repo root.
