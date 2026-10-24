# Inference Latency as Beam-Training Overhead

### An Accuracy-Independent Budget for Learned Millimeter-Wave Beam Prediction

**Sooyoung Jang**<sup>1</sup>, **Sunmi Jun**<sup>2</sup>, **Sungwoo Choi**<sup>2</sup>, and **Eunkyung Kim**<sup>3</sup>
<sup>1</sup> Department of Computer Engineering, Hanbat National University, Daejeon 34158, Republic of Korea
<sup>2</sup> Mobile Communication Research Division, Electronics and Telecommunications Research Institute, Daejeon 34129, Republic of Korea
<sup>3</sup> Department of Artificial Intelligence Software, Hanbat National University, Daejeon 34158, Republic of Korea
Correspondence: `ekim@hanbat.ac.kr`

> **Status:** preprint, under review. This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) - ITRC (Information Technology Research Center) grant funded by the Korea government (MSIT) (IITP-2025-RS-2024-00437886, 50%) and by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2024-00397216, 50%); the authors declare no conflict of interest.

![Effective rate against the re-acquisition period in Chicago and Houston, under airtime-only accounting and with inference latency included](assets/fig1_accounting.png)

*Fig. 3 of the paper: the same methods, channels, users and noise realizations under
airtime-only accounting (a, c) and with inference latency included (b, d), in Chicago, where
the best of the ten methods changes at no period, and in Houston, where it changes in point
estimate at the shaded periods. The learned predictors' curves move down and to the right;
the measurement-only curves do not move.*

## Abstract

Learned predictors shorten millimeter-wave beam acquisition only if they infer fast enough, yet
accuracy is unknown before training. Counting inference latency as cold-start beam-training
overhead, we derive an exclusion condition from a narrow-beam candidate's completion time, a lower
bound on the exhaustive sweep's efficiency and a maximum period. If it holds, the candidate cannot
outperform the sweep at any period up to that maximum, even with perfect beam selection. We apply
it to eight ray-traced city scenarios with 20-microsecond probes of a 64-beam codebook, using
95th-percentile latencies of single-precision, unbatched, serialized inference on four hardware
profiles. The condition excludes two vision-model implementations finishing within the
10-millisecond maximum: an 18-layer residual network on eight processor threads and a base-size
vision transformer on the graphics processor. Both are excluded at sweep efficiencies above 60%,
with positive margins at adverse inputs and in all three repeated timing runs; five others cannot
finish. The residual network on the graphics processor is unresolved. Inference latency changes
the best method in point estimate at 10 of 72 scenario-period pairs, none statistically
established at ten seeds. A case-study controller scheduling inference misses its separate
pre-registered criterion against hindsight-selected fixed strategies. Designers can screen
predictor-processor pairs before training.

## Repository layout

```
src/
  sites/       DeepMIMO acquisition and outcome-blind scenario qualification (stage 1)
  latency/     predictor latency atlas across four hardware profiles (stage 2)
  budget/      accuracy-free compute budget, the T* crossover, the DP reference (stage 3)
  scheduler/   observation law, codebook, comparators, the scheduled controller, the
               analyses, and the protocol record protocol.json (stage 4)
paper/         manuscript generator (make_draft.py, to_latex.py, figures_pdf.py), front
               matter, figures, and the compiled manuscript and supplementary (stage 5)
scripts/       one runner per stage, in order
assets/        the paper's Fig. 3 as PNG, rendered from paper/figures/
```

Each stage folder holds its code and the results it writes (JSON and NPZ) side by side.
Not in the repository: the DeepMIMO scenarios (licensed by their owners, fetched by
`src/sites/acquire.py`), the built datasets and model checkpoints (their hashes are in
`src/scheduler/manifest.json`), the large solved tables of the sweeps and of the second
re-solve, the HTML draft and the LaTeX intermediates (`paper/build/`). All are regenerated
by the stages below.

## Installation

Two environments are required, because the DeepMIMO loader needs `numpy>=2` and the
rest of the pipeline is pinned to `numpy==1.26.4`.

```bash
# 1. experiment environment -- stages 2 to 5
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 2. DeepMIMO loader environment -- stage 1 only
python3 -m venv .venv-deepmimo && . .venv-deepmimo/bin/activate
pip install -r src/sites/requirements-pinned.txt
```

Results were produced with Python 3.13.12 and PyTorch 2.12.0+cu130 on an Intel Core
i9-14900KF and a Quadro RTX 8000, recorded in `src/latency/p_*.json`. `tectonic` builds the manuscript; nothing else needs a TeX
installation.

## Data preparation

Eight DeepMIMO v4 ray-traced city scenarios at 28 GHz. `src/sites/acquire.py`
downloads and extracts them; `qualify.py` admits a site without ever inspecting a beam
outcome. Roles are fixed before any result is computed:

| Scenario | Role | Receivers | LOS |
|---|---|---:|---:|
| `city_0_newyork_28` | training | 16303 | 4467 |
| `city_1_losangeles_28` | validation | 19911 | 4090 |
| `city_2_chicago_28` | test A | 10451 | 3609 |
| `city_3_houston_28` | test B | 19379 | 5912 |
| `city_4_phoenix_28` | held-out | 13489 | 2227 |
| `city_5_philadelphia_28` | held-out | 11302 | 3139 |
| `city_6_miami_28` | held-out | 27061 | 6739 |
| `city_7_sandiego_28` | held-out | 22681 | 8737 |

64-element ULA, 100 MHz over `K_F = 16` sub-bands, 30 dBm transmit power, 7 dB noise
figure, 20 μs per probe.

## Reproducing the results

Run the stages in order; each depends on the one before it.

```bash
./scripts/01_sites.sh           # DeepMIMO loader environment
./scripts/02_latency_atlas.sh   # timing stage -- see the warning below
./scripts/03_budget.sh
./scripts/04_experiment.sh      # includes a second timing stage
./scripts/05_paper.sh
```

**The two timing stages need a quiet machine.** Nothing else may run on the pinned
cores. A contended run in our own testing inflated the measured predictor latency
`τ₁` from 13.8 μs to 21.1 μs, which is larger than several of the effects the paper
reports. Each stage prints the load average and a re-check drift, and labels itself
`quiet` or `CONTENDED`; discard a `CONTENDED` run.

`scripts/04_experiment.sh` starts by verifying the protocol record
(`src/scheduler/protocol.py`): the frozen protocol's hash, the hashes of the inputs it
depends on, and those of the stored results it lists. The long runs of stage 4 are skipped
unless their flag is set (`RETIME`, `RERANK`, `EXACT`, `SCREEN`, `CALIB`, `NRCFG`,
`LEVELS`, `TRACE`; see the comments in the script), and every result the paper reads is
stored in the repository, so `./scripts/05_paper.sh` alone regenerates both PDFs.

## Results

| | |
|---|---|
| Accuracy-independent latency budget | if the sweep selects at least as well, a *K*-probe method can outperform an exhaustive *M*-beam sweep for some re-acquisition period if and only if its inference latency is below `τ* = (M−K)·t_p`: 1220 μs for a 4-probe method at `t_p` = 20.34 μs |
| No trained model needed | without that assumption, under block fading, for any predictor that selects among the 64 beams, the budget grows by at most `(1−η_S)(T_max−M·t_p)`: 6.1 μs for periods up to 10 ms, since the sweep reaches `η_S` ≥ 0.9993 in every non-training scenario; even a perfect predictor at an excluded latency could win only for periods above 1.8 s (5.0 s for the vision models), and by less than 0.071% at the measured efficiency; at its simultaneous lower bound (0.97218) the shortest such horizon is 48.6 ms, still beyond 10 ms, while an uncompiled ResNet-18 on the GPU, at 1.10τ*, would be excluded only up to 5.75 ms (Table S-XXIX); on a channel that varies within the period, Proposition 1(c) bounds the same budget by the sweep's integrated shortfall from the best beam, at most 42.6 μs under Doppler-only aging at 120 km/h (evaluated on the aging simulation's own grid) |
| Heavy architectures against the budget | as FP32, batch-one implementations under a fixed 95th-percentile latency, ViT-B/16 exceeds it on all four hardware profiles, its fastest runtime at 5.19`τ*` even on the GPU, and ResNet-18 on every CPU profile; five of these seven exclusions cannot finish within 10 ms, while ResNet-18 on eight threads (4.78 ms, its compiled runtime in the repeated timing runs, which rule R21 reads with the stored timings) and ViT-B/16 on the GPU (6.33 ms) finish within 10 ms yet cannot outperform the sweep even with perfect beam selection, for any sweep efficiency above 0.591 and 0.413; even their fastest timed call exceeds it by at least 3.6×; on the GPU, ResNet-18 lies at the budget and is unresolved (0.999–1.44× `τ*` across runtimes, 1.58 μs below `τ*` in the fastest); every exclusion of these two architectures would hold for a sweep efficiency `η_S` as low as 0.591 (0.413 for ViT-B/16 on the GPU), and the small transformer's on one E-core for one as low as 0.849, against 0.9993 measured |
| Equal-completion per-probe time and codebook size | `t_p* = τ/(M−K)` is the per-probe time at which a method finishes with the sweep, the threshold of the dominance case: on the GPU, ResNet-18 becomes viable above 21–30 μs and ViT-B/16 above 106–129 μs, both within the 250 μs studied; without that assumption, at a fixed `η_S > K/M`, Proposition 1(a) keeps an exclusion that finishes within 10 ms while `t_p ≤ (τ−(1−η_S)T_max)/(Mη_S−K)`, 79.6 and 105.4 μs for the two such exclusions at the measured efficiency (77.3 and 103.9 μs at its simultaneous lower bound), and one that cannot finish stays excluded at every per-probe time; at our per-probe time a 91-beam codebook makes ResNet-18 viable, while ViT-B/16 needs 316 beams; at NR's spacing of up to 64 synchronization signal blocks within 5 ms, 78 μs per beam on average, every vision exclusion still holds at the measured efficiency but the small transformer's on one E-core (up to 42.8 μs) does not, and ResNet-18 on eight threads then needs a sweep efficiency of at least 0.982 (ViT-B/16 on the GPU 0.672), above the efficiency's lower bound, and at one 125 μs slot per probe ViT-B/16 on the GPU and ResNet-18 on eight threads become viable too; both NR per-beam times give only an amortized comparison, and a scheduled procedure needs its own budget, the sweep's completion time less that of the reduced probing |
| A concrete NR procedure | for an aperiodic CSI-RS beam sweep in connected-mode beam reacquisition or refinement, at 120 kHz subcarrier spacing on a millimeter-wave target carrier that carries no data until its beam is chosen, its triggers and reports on another 120 kHz carrier of the same connection (one-port, one-symbol resources, 12 per 125 μs slot, inference at the base station, data from the target carrier's next slot boundary): NR allows one slot per aperiodic resource set and at most four aperiodic beam-report settings per bandwidth part, so the 64-beam sweep is six one-slot sets under two triggers and six L1-RSRP reports; it needs a user equipment that measures 64 aperiodic CSI-RS resources (four mandatory) and 12 in one slot (eight mandatory), and trigger states with a spatial quasi-co-location source on the target carrier (TS 38.214, TS 38.331 and TS 38.306; Table S-XXX). A report waits for a computation time after its trigger and another after its last resource (TS 38.214 5.4): at the report timing most favorable to the candidate the reduced set reports 267.9 μs and the sweep 803.6 μs after their triggers (Fig. S4), and, with its completion taken one 125 μs slot early in its favor (conservative inputs to the bound, not data-start times), ResNet-18 on the GPU is excluded for any lower bound η_S ≥ 0.940 (0.949 with a 30 kHz control carrier, 0.974 with eight resources per slot), finishing within 10 ms, while the trained four-probe networks pass. Simulated with each resource's own 198, 66 or 33 resource elements (density 3, 1, 0.5 over 66 resource blocks) and the six reports compared at 1 dB, ties broken against the sweep (rule R16), the sweep's efficiency is at least 0.996, 0.991 and 0.985 in every non-training scenario (0.982 with energy detection on each element); at the corrected threshold rule R12 establishes it in all seven at each density with either estimator uncalibrated (rule R19), but at no density once calibrated (rule R14; the coherent estimator needs 330 resource elements). At these efficiencies the exclusion holds with at least 0.22 ms to spare; at adverse inputs only at density 3 with a 120 kHz control carrier, by 32.8 μs, where the nominal 95% simultaneous lower bound 0.9485 clears the threshold only within its bootstrap error (0.69% of Phoenix's replicates at or below it, against the bound's 0.71%); lower bounds at 5% per scenario, the level a claim on the lowest efficiency needs (rule R18, post hoc, changing no verdict), clear it at every density (0.9883, 0.9755 and 0.9653). Unlike the two exclusions that finish within 10 ms, this one rests on point estimates. A camera-only model (K = 0) stays excluded only while the sweep's last report takes at most 261 μs to deliver, and with a 30 kHz control carrier at no timing |
| Exclusion margins at adverse inputs | with a 95% distribution-free lower bound on each latency percentile from the recorded calls, the stored ones and the repeated timing runs (rule R21), nominal because successive calls are correlated (the fastest seed's percentile for the stored TorchScript runs), the slowest probe processing measured and the sweep's nominal simultaneous lower bound 0.97218, every exclusion keeps a positive margin, at least 66% of its completion time's lower bound for the vision models, against up to 9.9% between the two passes of the full-count GPU timing; Table S-XXXIII of the supplementary gives every input, with the rank and count of each order statistic and which inputs are point values or measured maxima, so that each margin can be recomputed: 3.12 ms and 4.72 ms for the two exclusions that finish within 10 ms; these adverse inputs do not form a joint 95% confidence set |
| Against a fast reference | the same bound with a fixed reference that needs no trained model (Table II): at Top-32's lowest measured efficiency (0.844), ResNet-18 on the GPU cannot outperform Top-32 at any period up to 4.8 ms, whatever its accuracy, and the small transformer on one P-core up to 3.0 ms; at the references' simultaneous lower bounds (one-sided, nominal 95% over the seven scenarios and four references; rule R17), as low as 0.249 for Top-32, every horizon ends within 0.22 ms of the implementation's own completion time (1.5 ms for ResNet-18 on the GPU), because a partial sweep's efficiency varies far more from block to block than the exhaustive sweep's; at the level the claim needs (5% per scenario over the four references, rule R18, post hoc), Top-32's bound is 0.684 and the horizon for ResNet-18 on the GPU 2.7 ms |
| Channel aging | under per-path Doppler with fixed geometry, the assumption behind `τ*` holds at every sampled time for every fixed baseline up to 120 km/h (user-averaged rates on a `t_p/4` grid, margin ≥ 0.10 bit/s/Hz), a check that cannot cover the vision models; the training-free bound of Proposition 1(c) does cover them, and every exclusion holds at every speed; the initially selected beam loses at most 0.53% of the best beam's rate over 10 ms at 120 km/h and is still the best beam for at least 91.85% of users |
| Fixed strategies under airtime-only accounting and with inference latency included | reduced probing wins at short periods and the sweep at the longest under both, in all six evaluation scenarios (the convention is that of Graff et al., whose own assisted search leads at every coherence time; this comparison is ours): `T*` moves by 1.6–12.8%; 2 rank reversals in Chicago and 5 in Houston; the same 2 (method, `T`) pairs become infeasible in each test scenario, each having achieved only 10–12% of the best rate |
| What the latency changes within it | under block fading, a method that stays feasible loses the fraction `τ/(T−K·t_p)` of its rate whatever the channel, up to 85% (FC-16 at 0.35 ms) and at most 1.2% at 10 ms; the best method changes in point estimate at 10 of the 72 (scenario, period) pairs, all between 1 and 4 ms, where keeping the airtime-only choice costs 0.25–9.1% of the rate in point estimate; at none of them is airtime-only accounting's own preference significant, and none is resolved by rule R8, simultaneous intervals over every pairwise difference of the ten methods (nominal under the bootstrap, and also required in a t version inflated for ten seeds), which establish no difference of that family, so their verdict is no evidence that the changes are absent |
| Crossover out of sample | the closed form (4) underestimates every measured crossover, by factors up to 1.27 (Houston) and 2.10 (Chicago), because η shifts by up to 31% between scenarios; the validation-city envelope picks the best fixed strategy at 4 of 12 periods in Chicago and 5 in Houston |
| A selector frozen on the validation city | choosing the fixed method at each period with the latency included rather than under airtime-only accounting changes the choice at 2 of 12 periods among the nine methods of Section VI-B (+0.5% to +1.8% of the grid-averaged rate, positive in all 6 evaluation scenarios, established in none) and at 7 periods among the 36 fixed methods (+4.2% to +8.3%, positive in all 6, established in 1, Miami); post hoc, rule R11; Table III |
| Latency-aware vs latency-unaware planning | +1.1% to +2.2% of mean effective rate, positive in all 6 evaluation scenarios, established in none |
| The inference action | −1.9% to +24.8% over the same controller without it, positive in 5 of 6 scenarios, established in 1 |
| Pre-registered primary criterion | not met: on the three pre-registered seeds the controller wins for no value of `T` in Chicago (one in Houston), and its worst shortfall, 11.0% and 7.4% of the genie-aided rate, breaks the 1% guard in both |
| Calibration of the significance test | under a crossed seed-by-block model with each evaluation scenario's own blocks, the *t* interval covers 93.5–99.7% and the bootstrap interval 89.3–99.9%; requiring both to exclude zero rejects a true zero in at most 6.4% of replications at a nominal 5%, above 5% only where the seed-by-block interaction is small; the *t* interval is therefore an additional criterion beside the bootstrap, not a separately justified 95% interval (rule R13); every established difference lies where the interaction is small, at most 8% of the *t* interval's squared standard error coming from the seeds, so the test is about nominal there, not conservative; at the levels Holm's procedure uses, simulated with 20,000 null replications per scenario under twelve mixtures, the rule rejects a true zero up to 1.50 times as often as Holm's levels allow and Holm's proof bounds its family-wise error by 6.9%, so rule R14 re-reads every established verdict with a calibrated p-value (Holm's bound 5.4% on held-out replications): it removes one of 53 (the lead over the first-exit controller with the model held fixed, in Phoenix) and changes no verdict of Tables S-XII, S-XIII and S-XIV; rule R15 removes three more, whose calibrated p-values lie within two Monte Carlo standard deviations of Holm's level or which Holm's procedure reaches only through such a one (the same lead in Chicago and Philadelphia, and the lead over the guarded published adaptive baseline in Miami); Table S-XVII lists all 53; per-period results, which rest on the bootstrap alone (up to 6.4 times the level), are reported as nominal |
| The adaptive baseline, guarded | as published, neither form of Zhang et al. stops a descent at the period's end: at 0.25 ms, 16–36% of the simplified form's descents and 15–31% of the published form's overrun and deliver nothing; with a guard that ends the descent before a measurement that would not finish within the period, the simplified form loses 0.02–0.30% of its rate at every period and the published one changes by at most 0.7%, so the controller's lead does not come from the baseline's overruns; the controller gains +10.6% to +42.4% over the guarded simplified form (established in 1 scenario, San Diego) and +12.7% to +48.5% over the guarded published form (established in 1, San Diego; Miami's, at Holm's threshold within the bootstrap's Monte Carlo error, is not counted under rule R15); rule R10 |
| Myopic and blinkered rules | the myopic rule loses 55–77% (Chicago) and 57–75% (Houston) of the controller's value; the blinkered rule recovers 90% of that shortfall; all three deploy as the same 13.52 MB table of 65 blocks, one per probe count at the first inference call, read in 0.310 μs |
| Predictor quality | the predictor is weak: its top-1 accuracy is below the best measured beam's at every probe count, and its best exit's efficiency is too from 8 probes on; a per-K FC or CNN network, though more accurate at top-1 with 16 probes, lowers the rate in all 6 scenarios under the same restrictions, an established loss in San Diego; replacing the ranking with a genie-aided one for a random fraction q of users moves the inference action's median gain from +13.1% as measured to +10.7% at q = 0.125 and +67.4% at q = 1, established in 1, 1, 1 and 3 scenarios; the pre-registered rule needed 4 at some q ≤ 0.25 to call the case study predictor-limited, so that claim is not made, though at q = 1 even the smallest gain exceeds the largest as measured; latency awareness is established in at most 3 scenarios at any q (3 at q = 0.5 and at q = 1) and in none as measured, so the rule's second clause is not met and the paper does not say that it depends on predictor quality |
| The model held fixed | the refined comparisons let each variant re-estimate its model on its own policy; with the model held fixed instead, the inference action is positive in 3 of 6 scenarios (5 refined) and established in 1, the lead over the first-exit controller is established in 3 (1 refined; Phoenix's does not survive rule R14's calibration, nor Chicago's and Philadelphia's rule R15), and latency-aware planning is established in none (none refined); with the model fixed, the staged controller differs by −1.5% to +2.0% from one that repeats first-exit passes, and a controller that only repeats first-exit passes has 94–108% of the staged controller's lead over a single first-exit pass, so, in point estimate, a controller that never passes the first exit obtains that lead too |
| Measured predictor latency | 14.1 / 36.1 / 65.0 μs to reach exits 1 / 2 / 3 (sums of fresh invocations) on one dedicated P-core; with batch size one the GPU is slower, 3.0–3.2× for the FC classifiers and 1.15× for the CNN, but 21× and 37× faster for ResNet-18 and ViT-B/16, comparing each processor's fastest runtime; no GPU ratio is given for this network, whose GPU timings omit an input copy (the GPU was also shared) |

*Significant* means *established* in the paper's sense: both 95% intervals (seed-by-block
bootstrap and *t*) exclude zero and the larger of their p-values survives Holm correction
across the six evaluation scenarios after calibration (rule R14) by two Monte Carlo standard
deviations (rule R15). Table S-XIII of the supplementary lists every input to
each verdict.

Every inference call is accounted as a **fresh forward pass**: an intervening probe changes
the network's input, so the executor recomputes from the first block and the accounted
latency says so. Choosing the next codeword after an inference call, a masked argmax over the
exit's 64 output scores, is counted at the measured time of the same operation in the sweep's
selection (1.06 μs). The controller results exist in three versions, every controller solved and
evaluated in each: recorded (files without a suffix), without that time; the first re-solve
(files tagged `_rr`, `src/scheduler/rerank.py`), with it counted on every probe of a sequence that
invokes the predictor; and the second re-solve (files tagged `_rx`, `src/scheduler/exact.py`),
with it counted only on the probes after the first inference call, the probe count at that call
in the controller's state and a 13.52 MB policy table timed at 0.310 μs per lookup. The paper
reports the second; Appendix I-I of the supplementary gives each entry of Table S-XII under all
three.

## Protocol record and result files

The measurement model and the six-layer hierarchical codebook are frozen in
`src/scheduler/obs.py`. `src/scheduler/protocol.json` is the protocol record:

- the protocol as frozen on 2026-09-15, before any dataset was built, any model trained or any
  number seen (its fields hash to `self_sha256`);
- its three amendments, A1–A3;
- the changes made after results existed, each dated, with its reason;
- the interpretation rules R1–R21, which state what each result permits the paper to claim,
  each with the date it was set and whether it was set after the results it reads;
- the later analyses, each declared before it ran, with its scripts, its rules and the
  hashes of the stored files it wrote.

`python3 src/scheduler/protocol.py` checks the frozen hash, the inputs the protocol depends on
and those stored files. Stored outputs also record the hashes of the scripts that wrote them
(`scripts_sha256` in the record); the scripts here compute the same results and differ from
those only in comments, help texts and record keeping. The stored result files are as
those scripts wrote them, less record-keeping fields; the hashes that some of them record of
others refer to the files as released. The gate outcome computed with the predictor before
amendment A3 is kept in `src/scheduler/results_pre_A3.json` and `report_pre_A3.json`.

Result files in `src/scheduler/` are named after the run that wrote them:

| Name | Contents |
|---|---|
| `validation`, `results`, `heldout` | the validation scenario, the two test scenarios and the four held-out scenarios, ten seeds |
| `report` | the variant selection and the comparisons computed from those runs (`analyze.py`) |
| `*_3seed` | the pre-registered three-seed analysis (seeds 17, 29, 43) |
| no suffix, `_rr`, `_rx` | the controller results as recorded, after the first re-solve and after the second (above) |
| `sweep*_rho*`, `sweep*_tp*`, `sweep*_oq*` | the sensitivity runs: hardware ratio ρ, probe airtime, predictor quality q |
| `lat_*.json`, `../latency/p_*.json`, `o_*.json`, `o2_*.json` | the timings: the controller's charges and the latency atlas's profiles and compiled runtimes |
| `manifest.json` | hashes of the datasets and model checkpoints, which are regenerated, not tracked |

## Paper

`paper/manuscript.pdf` is the compiled 13-page IEEEtran manuscript, and
`paper/supplementary.pdf` is the 32-page supplementary carrying Appendices A–L: the proofs,
the design-rule verification, the validation-scenario rotation, the fixed-strategy
break-even, the re-acquisition period and the coherence time, the notation, predictor
accuracy, the dataset, timing environment and timed workloads, the pre-registered
three-seed analysis with its extensions (the decision rule cell by cell and the
comparisons with the model held fixed among them), the break-even map over every runtime,
the channel-aging analysis per scenario, and the uncertainty of the changes of the best
method.
`./scripts/05_paper.sh` regenerates both end to end: every number in the text is read
from the stage outputs, so changing an experiment propagates to the PDFs. The bibliography
is `paper/refs.bib`, typeset by BibTeX with `paper/IEEEtran.bst` (version 1.14, under the LaTeX
Project Public License); `tectonic` runs BibTeX itself. The author
biographies (`paper/backmatter.tex`) are left out of the initial submission; for the
accepted version, `INCLUDE_BIOS=1 python3 paper/to_latex.py` adds them.

## Citation

```bibtex
@unpublished{jang2026latency,
  title  = {Inference Latency as Beam-Training Overhead: An Accuracy-Independent
            Budget for Learned Millimeter-Wave Beam Prediction},
  author = {Jang, Sooyoung and Jun, Sunmi and Choi, Sungwoo and Kim, Eunkyung},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## Acknowledgement

The ray-traced channels come from the **DeepMIMO** dataset, whose license requires
both of the citations below. They are also recorded in `src/sites/site-manifest.json`.

- A. Alkhateeb, "DeepMIMO: A generic deep learning dataset for millimeter wave and
  massive MIMO applications," in *Proc. Information Theory and Applications Workshop
  (ITA)*, San Diego, CA, Feb. 2019.
- Remcom, *Wireless InSite* — the ray tracer behind these scenarios.

None of the eight scenario archives carries a further per-scenario citation
requirement. The scenario data itself is not redistributed here.

The comparator we reproduce is the adaptive hierarchical beam-training method of
Zhang et al., *Adaptive beam alignment method for millimeter-wave massive MIMO
communication systems*, Physical Communication 41 (2020) 101101.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

The DeepMIMO scenario data is not covered by it and is not redistributed here; it
carries its own terms, summarised under Acknowledgement.
