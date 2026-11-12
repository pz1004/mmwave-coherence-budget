## Recommendation

**Complete one focused final revision, then submit to IEEE Transactions on Wireless Communications as a stretch submission.** The revised paper has a defensible contribution: it identifies conditions under which a specified beam-prediction implementation can be excluded before its prediction accuracy is known. The remaining submission risk is **significance**: a senior reviewer could accept the mathematical argument and still judge the demonstrated capability too incremental for TWC.

The completed R26 audit strengthens the paper by showing how the screening rule behaves with limited calibration and different acquisition procedures. The remaining work should consolidate that evidence, resolve a reproducibility ambiguity, and correct several interpretations. This does not call for another broad experimental campaign.

The review below concerns the revised manuscript, particularly Proposition 1, Sections V-D and VI-C, Table VI, and Figure 3, together with the supplementary evidence map and analysis history.

:chatgpt-content-reference{index="3"}

:chatgpt-content-reference{index="4"}

The complete editable revision plan is available here:

::page-preview{page_id="page_e203a762b4948191b0b3a5693d923103"}

# 1. Corrections to the previous assessment

Several conclusions in the earlier assessment need narrower wording before they can withstand a senior reviewer’s scrutiny.

| Earlier interpretation | Corrected assessment |
|---|---|
| No fatal mathematical flaw or identical prior result was found, supporting TWC submission. | Conditional correctness and a distinct research question make submission defensible. They do not establish TWC-level significance. Significance must come from the capability enabled and its demonstrated usefulness. |
| A mean-completion-time comparison reproduces the audit decisions. | It reproduces the **18 endpoint classifications at 10 ms** under the recorded distributions. It does not replace the condition for exclusion at every period up to 10 ms. |
| Zero contradictions validate the screen’s reliability. | They establish consistency within this finite audit. The positive cases are too far from the relevant boundary to test erroneous exclusions caused by overestimating reference efficiency. |
| A pooled median of six exclusions is a reproducible headline. | That median and its range require the actual rule used to combine the city-specific calibration draws. Table VI’s marginal frequencies do not determine them. |
| Decisions remain stable over ten subsequent timing rounds. | The reported evidence directly establishes stability of the **full-population loss/opportunity labels**. Stability of every calibration-dependent operational screen output is a separate claim. |
| Phoenix demonstrates a general calibration bottleneck. | Phoenix demonstrates low exclusion yield under the **particular sampling and bound-construction procedure tested**. It does not establish an estimator-independent limitation. |

These corrections preserve the submission recommendation while making its basis more defensible.

# 2. State the contribution precisely

## 2.1 The strongest defensible novelty claim

Use one consistent description in the Abstract, Introduction, contribution list, and Conclusion:

> **The paper derives a conditional, accuracy-independent exclusion rule for a specified implementation of narrow-beam prediction within a declared cold-start acquisition procedure, and evaluates how measured runtimes, reference procedures, and limited calibration affect the resulting decisions.**

The capability is consequential because it answers a question available **before candidate accuracy is known**: can even perfect prediction compensate for this implementation’s acquisition delay?

Under the stated deterministic model, the central condition is

\[
d_L \ge d_S+(1-\eta_S)(H-d_S),
\qquad H=T_{\max}>d_S.
\]

Here \(d_L\) and \(d_S\) are candidate and reference completion times, and \(\eta_S\) characterizes the reference’s efficiency relative to the ideal rate. At exact inputs, the condition is tight for a perfect predictor. Correctly ordered conservative bounds yield a sufficient exclusion condition. The mathematical implication and the statistical validity of estimated inputs must remain separate.

Keep Proposition 1(a) central and give Proposition 1(d) suitable prominence as the extension to random completion times. The dominance case, crossover, envelope, and completion-time restatement support interpretation and use. Their short derivations do not need artificial expansion.

## 2.2 Acknowledge what prior work already establishes

A senior reviewer is likely to challenge any suggestion that inference-time accounting itself is new.

Zecchin *et al.* already include inference and signaling in beam-alignment time [3]. Salehi *et al.* combine a trained model’s inclusion probability with the fraction of the beam-validity interval remaining after acquisition-related delays [4]. Recent hardware-aware work also evaluates latency, accuracy, memory, and complexity [5]. The distinction to defend is the **accuracy-independent exclusion decision**, with its stated assumptions and scope. [www.eurecom.fr](https://www.eurecom.fr/publication/6552/download/comsys-publi-6552.pdf?utm_source=chatgpt.com)

In Section II, replace the sentence introducing Eq. (2) as “ours differs from it in one term” with:

> We use the latency-inclusive effective-rate model in (2), consistent with prior acquisition-time accounting that includes computation. Our question is what can be ruled out before the candidate’s accuracy is known.

Retain the scoped literature statement **“In the works we examined.”** The comparison supports a specific distinction; it does not establish an exhaustive priority claim.

## 2.3 Define the unit being excluded

An exclusion concerns a particular combination of:

- Architecture and precision.
- Runtime and hardware profile.
- Acquisition procedure and timing boundary.
- Reference procedure.
- Operating population and model assumptions.

This matters for claims about avoiding training. For a declared task, the screen alone supports omitting an architecture’s training **for the purpose of beating that reference** only when all implementations the designer is willing to deploy have been excluded.

The FP32 and FP16 ViT builds share weights. Their exclusions cannot be counted as independent avoided training jobs. Retain the calibration burden and describe the demonstrated benefit as implementation screening before candidate training; the present study does not quantify net training-cost savings.

# 3. Required revisions

## Priority 1 — Make screening yield the principal empirical message

**Locations:** Abstract, Contribution 3, Section VI-C, Conclusion.

**Likely reviewer objection:**  
“The audit reports no incorrect exclusions, but does that establish useful or reliable decision-making?”

### Required change

Lead with the additional information the screen provides beyond feasibility and the fraction of directly excludable cases recovered under limited calibration.

The revised manuscript supports the following account:

| Audit finding | Correct interpretation |
|---|---|
| Three GPU builds × three pilot durations × two scenarios | **18 experimental cells**, with shared timing and calibration inputs |
| All 18 pass the recorded expected-rate feasibility check at 10 ms | Feasibility alone does not eliminate these cases |
| Direct full-population calculation identifies 12 losses and six opportunities for perfect prediction at 10 ms | These are ideal-predictor endpoint classifications |
| Houston exclusion frequencies are 92.5–100% across the losing cells | High recovery under the tested eight-block calibration procedure |
| Phoenix exclusion frequencies are 7.5–31% across the losing cells | Low recovery under the same stated procedure |
| Changing pilot duration changes which builds have an ideal opportunity | The declared reference procedure materially affects the decision |

These results appear in Section VI-C and Table VI. Call the combinations **cases** or **experimental cells**. They are not 18 independently measured implementations or deployments.

### Why zero contradictions should remain secondary

Under the same timing record and reference completion time, overestimating reference efficiency, while capping the estimate at one, can suppress a positive ideal advantage only within the allowance

\[
(1-\eta_S)
\left(1-\frac{\overline d_S}{H}\right).
\]

Using Table VI’s rounded inputs, its largest value is approximately **0.0597% of the genie-aided rate \(G\)**. The smallest reported positive opportunity is **3.99% of \(G\)**.

Consequently, those positive cases do not test erroneous-exclusion risk close to this boundary. The zero-contradiction result is useful as an implementation consistency check, but it does not establish calibrated error control in future deployments.

**Suggested wording:**

> The audit quantifies the exclusion yield of the specified calibration procedure. Its absence of contradictions provides a consistency check on the recorded cases; the positive cases lie outside the region in which overestimating reference efficiency alone could reverse their classification.

**Completion criterion:** The opening communicates useful exclusion yield and reference-dependent opportunity, with explicit denominators and no implication of general deployment validation.

---

## Priority 2 — Separate the 10 ms endpoint result from the all-period theorem

**Locations:** Section VI-C and the R26 supplementary analysis.

**Likely reviewer objection:**  
“In these experiments, does the result reduce to comparing completion times?”

### Required change

Acknowledge exactly where that observation holds.

At the audit endpoint \(H=10\) ms, every recorded candidate completion is below \(H\). Under the stated channel-independence model, the normalized ideal-predictor advantage therefore reduces to

\[
A(H)=
\frac{\overline d_S-\mu_D}{H}
+
(1-\eta_S)
\left(1-\frac{\overline d_S}{H}\right),
\qquad
\mu_D=\mathbb E_{\mathrm{record}}[D_L].
\]

The first term measures the mean-completion advantage. The second is the allowance provided by the reference’s imperfect efficiency.

From Table VI’s rounded inputs, the equivalent completion-time allowance is only approximately **1.3–6.0 \(\mu\)s**, much smaller than the endpoint margins. Mean-completion ordering therefore reproduces the 18 endpoint classifications in these cases.

This is a limitation of the **empirically demonstrated regime**. It is not a reason to claim that the general theorem is redundant.

**Suggested discussion paragraph:**

> In the audited regimes, the full-population reference efficiencies are close to one. At the 10 ms endpoint, the resulting accuracy allowance is small relative to the measured completion-time margins, so mean-completion ordering reproduces the direct classifications. The audit therefore primarily demonstrates reference-dependent opportunities and the yield of limited calibration. Exclusion over the full period interval is established separately by the applicable uniform condition or the exact piecewise-linear verification.

### Preserve the uniform verification

For the recorded timing distribution, define

\[
\widehat\psi(T)=
\mathbb E_{\mathrm{record}}[(T-D_L)^+]
-\eta_S(T-\overline d_S)^+.
\]

The applicable sufficient condition is

\[
\widehat\psi(H)\le 0,
\qquad
\min_r D_r\ge\overline d_S.
\]

Here \(\eta_S\) is the full-population efficiency used by the direct audit. Substituting a valid lower efficiency bound gives a conservative sufficient check; whether an estimated bound is valid remains a separate statistical issue.

The exact piecewise-linear verification is another way to document the uniform comparison. Its evaluation set must include **zero, \(H\), the reference completion, and candidate completion times within the interval**. A grid check alone should not be described as verification over a continuum.

**Completion criterion:** Every claim distinguishes among “at 10 ms,” “on the evaluated grid,” and “at every period up to 10 ms,” with matching evidence.

---

## Priority 3 — Separate operational decisions from retrospective audit knowledge

**Locations:** Section VI-C’s interpretation of outcomes, later-round results, and Phoenix discussion.

**Likely reviewer objection:**  
“The proposed screen appears to use information that is available only to the full-population auditor.”

### A. Correct the interpretation of non-exclusion

The operational screen is one-sided. A non-exclusion does not reveal whether the candidate has a genuine ideal opportunity or whether calibration was too weak to certify an actual loss.

Replace the paragraph sorting outcomes into three actions with:

> The full-population audit distinguishes an ideal opportunity from a missed exclusion caused by an uninformative calibration bound. The operational screen returns excluded or not excluded, with a separate numerical unresolved category; a non-exclusion alone does not distinguish those two explanations.

Follow-up actions may still be discussed, but label them as **retrospective interpretations of the audit**.

### B. Name the label that remained stable

Replace ambiguous wording such as “every case kept its label” with:

> The full-population loss/opportunity labels were unchanged over ten subsequent timing rounds, and no previously issued exclusion was contradicted by those records.

This wording matches the reported direct comparisons. If saved outputs also establish that every frozen calibration bound produced the same operational screen result in every later round, that may be reported separately. Otherwise, remove that implication.

### C. Narrow the Phoenix conclusion

Replace the prescription to calibrate “where its rate is” with:

> Under the sampling and bound-construction procedure used here, Phoenix’s concentration of rate in a small number of spatial blocks was associated with weak lower bounds and low exclusion yield. Alternative sampling and bounding procedures were not evaluated.

Targeted calibration could require information unavailable to the operational screen. It would also need a sampling design that preserves the original user-averaged estimand. The current study supports a diagnostic finding, not a validated remedy.

**Completion criterion:** A reader can identify which claims use partial calibration, which require full-population knowledge, and which concern later recorded timings.

---

## Priority 4 — Resolve the pooled aggregation ambiguity

**Locations:** Section V-D, Section VI-C, and the supplementary output description.

**Likely reviewer objection:**  
“How were 200 calibrations per city combined to produce the pooled median of six exclusions and the range 3–12?”

This is the most concrete remaining reproducibility issue.

### Required change

Inspect the existing output generator and saved draw identities. Document the rule actually used to combine the Houston and Phoenix draws.

If the original calculation paired city draws by replicate index, state that rule and reproduce the statistic from the saved pairing. If it used another rule, report that rule. Do not choose a new pairing after examining which summary is more favorable.

Table VI’s marginal proportions reproduce a **mean count**, but they do not determine its median or range. Summing those proportions gives:

- Houston: **5.865** excluded cells per city calibration, on average.
- Phoenix: **1.150**.
- Combined mean across the two nine-cell evaluations: **7.015**, independent of how the city draws are paired.

This arithmetic is a descriptive consistency check. It should not replace the primary median merely because the resulting value is larger.

If the original aggregation cannot be recovered, remove the unsupported pooled median and range. Retain the reproducible cell-level frequencies and any city summaries supported by the saved records.

### Define every denominator

| Quantity | Required definition |
|---|---|
| Per-cell exclusion frequency | Exclusions divided by 200 draws for the specified city, block budget, build, and reference |
| Yield among feasible cases | Excluded cells divided by 18 in a documented combined replicate |
| Recovery of directly excludable cases | Captured losses divided by the 12 direct full-population losses |
| Contradictions among issued exclusions | Contradicted exclusions divided by issued exclusions; undefined when none were issued |
| Efficiency-bound coverage | Draws whose bound does not exceed population efficiency, for the stated city, pilot duration, and block budget |

Repeated draws and shared experimental cells should not be treated as independent deployments. Any Monte Carlo uncertainty statement should refer to the defined conditional sampling procedure.

**Completion criterion:** Every retained count, proportion, median, and range can be reconstructed without guessing the unit, weights, denominator, or cross-city aggregation.

---

## Priority 5 — Complete the R26 supplementary evidence trail

**Locations:** Supplementary Table S-I, a short R26 subsection, and Appendix M.

**Likely reviewer objection:**  
“The new central empirical contribution is not fully integrated into the evidence map and analysis history.”

The supplement introduces R1–R26, but Table S-I and Appendix M do not yet provide a corresponding R26 evidence entry and history.

### Required change

Add a concise R26 subsection and output index covering:

1. **Experimental construction:** three builds, three references, two reused scenarios, and the roles of the new and sensitivity noise realizations.
2. **Timing protocol:** timing boundary, invocation spacing, the first five calibration rounds, equal-session weighting, and ten subsequent rounds.
3. **Calibration design:** primary eight-block budget, four- and twelve-block sensitivities, endpoint, numerical tolerance, and actual city-draw aggregation.
4. **Information separation:** inputs available to the screen and the full population available to the separately implemented auditor.
5. **Results:** endpoint classifications, all-period verification, cell-level exclusion counts, bound coverage, and the later-round classifications actually checked.
6. **Provenance:** exact existing protocol and output files supporting the manuscript summaries.

Preserve the chronology: R26 was designed after the previous results and before its main measurements, with the disclosed smoke test already observed. “Prespecified” should retain that limited meaning; it should not imply independent preregistration.

A compact summary table and precise pointers to archived outputs are sufficient. Separate calculation paths are useful for checking implementation consistency, but the shared inputs do not provide independent environmental validation.

**Completion criterion:** A reviewer can trace Table VI and each R26 headline to the existing protocol, data, calculation, and interpretation.

# 4. Suggested replacement abstract

The following **199-word draft** foregrounds the reproducible cell-level findings and avoids relying on the pooled median before its aggregation is documented. It also preserves the distinction between profiled p95 findings, expected-rate findings, and ideal opportunities.

> Before training a learned beam predictor, its accuracy is unknown, but its implementation may be too slow to outperform a sweep. Under block fading with a common codebook, we derive a condition for cold-start acquisition that excludes an implementation at every period up to a maximum, using completion times and a lower bound on reference efficiency. The condition is tight for perfect prediction at exact inputs and sufficient with valid conservative bounds. We extend the comparison to random completion times and evaluate it using ray-traced channels and desktop measurements. At profiled p95 latencies, two FP32 implementations finish within 10 ms yet are excluded, and an optimized FP16 vision transformer remains excluded in expected rate against the baseline sweep under recorded timings. A prespecified audit combines three GPU builds, three pilot durations, and two scenarios. All 18 cases pass feasibility; direct full-population comparisons identify 12 losses for perfect prediction and six ideal opportunities at 10 ms. Across 200 eight-block calibrations per scenario, exclusion frequencies for the losing cases are 92.5–100% in Houston and 7.5–31% in Phoenix. These results quantify screening yield under the tested calibration procedure and show how changing the reference procedure changes the opportunity available to a predictor.

The proposed wording changes the presentation of the evidence; it does not alter the prespecified outcomes.

# 5. Presentation and scope control

## Preserve distinctions already handled correctly

The revised manuscript already contains qualifications that should remain intact:

- Cold-start acquisition and beam tracking have different starting conditions.
- Exhaustive search is not automatically the dominant reference in every formulation.
- A p95 plug-in comparison is distinct from an expected-rate comparison.
- Conditional validity at conservative inputs is distinct from joint statistical coverage.
- Recorded timing distributions do not establish future deployment coverage.
- A positive ideal opportunity does not demonstrate a trained predictor’s gain.
- The reported ranking changes are not established statistically.
- The controller did not meet its primary criterion.

These issues do not need to be reopened as requests for additional studies.

## Keep the empirical studies identifiable

The original four-probe workload, the image-only runtime study, and R26 have different candidate definitions or timing boundaries. Keep those differences visible. Their numbers should not be pooled into an apparent common end-to-end benchmark.

In Table VI and Figure 3, distinguish:

\[
100A \quad \text{as a percentage of }G
\]

from a percentage gain relative to the sweep. Positive Table VI values describe the opportunity available to a perfect predictor.

A Figure 3 change is optional. If the Houston–Phoenix yield contrast remains hard to see, a panel using existing per-cell exclusion frequencies or the four/eight/twelve-block sensitivity would help. Table VI already provides the core evidence.

## Stay within the initial-submission format

The current main manuscript is already 13 pages. TWC’s official limit is **13 double-column pages for an initial submission**, with **75–200 words for the abstract**. The 16-page revised-submission allowance does not apply simply because the authors revise their draft before first submission. Make replacements and consolidate repeated qualifications within the existing space. [IEEE Communications Society](https://www.comsoc.org/publications/journals/ieee-twc/policies-guidelines?utm_source=chatgpt.com)

# 6. Execution order and stopping criteria

## Recommended work sequence

| Step | Action | Deliverable |
|---|---|---|
| 1 | Verify the existing R26 summaries | Documented aggregation, identified later-round label, and located endpoint/all-period outputs |
| 2 | Revise Section VI-C and the supplement together | Consistent results, interpretations, evidence map, and history |
| 3 | Align Abstract, Introduction, contributions, and Conclusion | One novelty claim, one decision unit, consistent denominators |
| 4 | Tighten the nearest-prior-work comparison | Clear acknowledgment of established latency accounting and a precise exclusion-decision distinction |
| 5 | Regenerate affected tables and the manuscript PDF | Verified numbers, cross-references, captions, and final-size readability |
| 6 | Submit after the completion criteria are met | A technically bounded, reproducible TWC submission |

If an existing record contradicts a reported result, correct that result and every dependent statement. If a record needed only for an optional claim cannot be recovered, narrow or remove the claim. New measurements are justified only if a concrete inconsistency prevents support for a retained central result.

## Submission criteria

The final revision is complete when:

- The novelty claim centers on accuracy-independent exclusion and is accurately positioned against the nearest prior work.
- Every all-period statement has the appropriate analytical or exact verification support.
- The audit headline reports useful screening yield with explicit denominators.
- Every retained aggregate is reproducible.
- Later-record stability names the label actually checked.
- Non-exclusion is not presented as proof of an ideal opportunity.
- Phoenix’s conclusion is restricted to the procedure evaluated.
- Training-stage language does not imply unmeasured training-cost savings.
- R26 appears in the supplementary evidence map and analysis history.
- The manuscript fits the initial-submission limit and retains the relevant negative findings.

**Once these criteria are met, submit to TWC.** The completed audit has addressed the request for an operational evaluation. The remaining significance risk should be stated candidly and accepted as a publication decision: further polishing cannot guarantee that reviewers will consider the contribution sufficiently substantial. If a subsequent decision turns primarily on significance, retargeting would be more proportionate than continuing to add studies without a specific unresolved scientific question.

## References informing the novelty assessment

[3] M. Zecchin *et al.*, “LIDAR and position-aided mmWave beam selection with non-local CNNs and curriculum training,” *IEEE Trans. Veh. Technol.*, vol. 71, no. 3, pp. 2979–2990, Mar. 2022. [Author manuscript](https://www.eurecom.fr/publication/6552/download/comsys-publi-6552.pdf).

[4] B. Salehi *et al.*, “Deep learning on multimodal sensor data at the wireless edge for vehicular network,” *IEEE Trans. Veh. Technol.*, vol. 71, no. 7, pp. 7639–7655, Jul. 2022. [Author manuscript](https://arxiv.org/html/2201.04712v1).

[5] B. Sayitoğlu, A. Sateesan, M. Petrova, and L. Simić, “A hardware-aware performance analysis of machine learning models for sensor-aided millimeter-wave beam prediction,” in *Proc. IEEE Int. Conf. Commun. (ICC)*, 2026, doi: 10.1109/ICC59461.2026.11587005. [Institutional publication record](https://vbn.aau.dk/en/publications/a-hardware-aware-performance-analysis-of-machine-learning-models-/). The verified abstract supports its benchmarking objectives; it does not support a claim about the absence of particular results from its full text.

*The saved plan’s content and equation source were verified. A rendered Page preview was unavailable. The wording above constitutes proposed manuscript revisions; the manuscript PDF itself has not been edited.*
