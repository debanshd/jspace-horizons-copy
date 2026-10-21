# THE ARTICULATION BOUND — frozen multimodal J-space research protocol (October 2026)

(Verbatim from Nayan, 14 Oct 01:00. Governs everything after the E-pilot. Amendments go to the end, dated, with the reason.)

## 0. Mission
Investigate scaling laws for causally useful, verbally accessible representations across text, images, and video.
Working hypothesis: the relationship between information perceived, information causally used, and information accessible
through J-space depends on task demand and input representation, and changes predictably with model scale.
Rivals: H1 shared capacity (common bottleneck whose effective capacity grows with size); H2 modality-dependent access (similar
evidence incurs different costs of entering / being used through the candidate shared representation); H3 selective routing
(larger models improve mainly by recruiting the right evidence, not maintaining more at once); H4 distributed alternatives
(substantial computation bypasses the measured J-space; the common-workspace account is insufficient). None is the default.
At most three contributions: a phenomenon, a causal characterization, a predictive scaling law or its principled falsification.
A better J-Lens score is not a discovery.

## 1. Models
Primary Qwen/Qwen3.5-4B. Scale: post-trained Qwen3.5 0.8B, 2B, 9B, 27B only with verified model-to-lens compatibility.
Replication Qwen3.6-27B. Log architecture, training stage, checkpoint/lens/processor revisions, hidden dim, layer correspondence,
modalities. Use existing fitted J-Lens weights; compare J++ where compatible; never compare raw top-k counts across lens families
without calibration. 27B only in a memory-valid configuration.

## 2. Datasets
A. GQA (testdev-balanced): 64 exploratory images, ≥ 128 disjoint confirmatory. Select questions by program structure, independent
of model correctness: one atomic fact / one object-relation fact / two or more dependent facts. Record program depth and the number
of independently needed facts separately. Scene-graph conjunction questions with k = 1, 2, 3 verified facts, balanced yes/no and
false-clause positions, same image across loads (benchmark-derived controls, not official GQA questions).
B. Winoground (gated): 64 exploratory pairs + disjoint reserve; semantic roles with identical object vocabulary; poor accuracy is not
an internal-absence result.
C. TempCompass: 48 exploratory videos stratified by temporal type; original questions + validated contrasts; fixed frame budget with
recorded timestamps and token mappings; a temporal fact counts as available only if its event is visible in the sampled evidence.
D. Perception Test: 32 annotated clips; for each verified evidence unit compare text (human-verified description), image (still or
storyboard), video (original evidence); same question/answer where semantically possible; information-insufficient image conditions
are labelled separately, never as workspace failures.
E. OmniVideoBench (authorized): disjoint manageable-duration cases with atomic reasoning annotations; vision-sufficient vs
audio-dependent separated; Qwen vision-only never reported as audiovisual performance.

## 3. One measurement protocol
Per verified evidence unit: P perception (independent recognition query); U causal use (input-level evidence intervention changes the
answer in the predicted direction; paired counterfactuals, necessary-evidence ablations, language-prior controls); J candidate
workspace access (targeted intervention on a specified J-related residual component transports or removes the fact and selectively
changes downstream use); R reportability (the model states the fact when asked, beyond predicting the answer token). Never infer one
from another without a validity test. Save continuous log-probability effects.

## 4. Experimental structure
Each example: original evidence; matched counterfactual / evidence manipulation; target question; independently worded downstream
question; question-only or evidence-removed control; known answer and evidence annotation; pre-answer activation capture at source
positions and justified downstream positions. Readout: target concept/relation and matched absent, unrelated and answer-identity
controls. Causal arms: (1) clean original and clean counterfactual; (2) full source-state interchange (positive control); (3) candidate
J-subspace interchange; (4) equal-rank random-subspace interchange; (5) energy/norm-matched alternative-subspace interchange;
(6) off-target fact interchange; (7) targeted ablation where valid. Successful source-state patching is the positive control; the
failed receipt-only patch is not revived. A patch must affect the intended fact, not inject an answer token. Test transfer to a second
downstream question, ideally with a different answer vocabulary.

## 5. J-space subspace
From a frozen Jacobian map with documented orientation and output projection; orthonormal basis and projector at rank r; assert
orthogonality, idempotence, shapes, coordinates. Ranks r = 64 and 256 plus full state; matched dimensional and energy controls. Any
rank chosen on the pilot is frozen before confirmation. J++ is an alternative instrument, not the same subspace.

## 6. Estimands
Primary causal effect: change in log probability of the intended counterfactual response. E_full, E_J(r), E_control(r);
C(r) = [E_J(r) − E_control(r)] / [E_full − E_control(r)] on aggregate paired effects; report unnormalized effects and uncertainty.
A failed J-subspace effect proves nothing without sensitivity, power, patch fidelity and rival pathways tested. Access curves over
model size N, representation m, evidence complexity k, task type, rank r. Primary target: the complexity-dependent causal access curve,
with behaviour and reportability shown separately. Secondary: the rank needed to restore a fixed proportion of the full effect
("causal transfer cost", not memory capacity).

## 7. The frontier
No pair of curves is a Pareto frontier. A frontier needs two competing objectives under a controlled allocation policy; estimate an
empirical nondominated frontier only if a real trade-off appears. The paper does not depend on it.

## 8. The scaling law
Only after the principal causal metric passes its controls: freeze measurement and selection; ≥ 4 sizes if practical; rival simple
functions, no presupposed saturation; hold out a size and loads; predict held-out curves with uncertainty; compare with accuracy,
residual width, token budget, chance-adjusted competence, simpler non-J representations. No exponent from two sizes; a smooth
interpolation is not a law; a law that fails a held-out model has failed.

## 9. Execution order
Smoke (4 images, 2 text conditions, 2 videos, one verified patch) → Run 1 GQA 64 groups → Run 2 24–32 validated donor/receiver pairs
(GQA/Winoground) with all arms at frozen sites → Run 3 same frozen protocol on 2B/9B then the largest feasible → Run 4 TempCompass 48
videos → Run 5 Perception Test 32 clips, three presentations → Run 6 confirmatory group-disjoint repetitions + held-out-size
predictions → Run 7 OmniVideoBench replication. Image before a large video sweep; never redefine the hypothesis or metric per dataset.

## 10. Gates
Freeze before confirmation: estimand and criteria; model/lens revisions and layers; ranks and norms; sampling frame and grouped
splits; exclusions and failed-perception handling; CI method and multiplicity; effect-size and equivalence thresholds; power from
exploratory variance; rival predictions. Group uncertainty by image / caption pair / video. Keep unconditional and perception-stratified
results; never select the population on the reported statistic. Save prompts, option mappings, seeds, media ids, provenance, processor
configs, dtype, hashes, token boundaries, layer indices, raw predictions and interventions. Assert shapes, positions, orthonormality,
cache invalidation. Cached activations, bounded frame budgets, OOM halving; runtime estimates from the smoke.

## 11. Red-team objections every headline must survive
Decoding ≠ causation; patching ≠ ordinary mediation; source-token info ≠ broadcast workspace; absence may be instrument failure;
harder ≠ more capacity; image/storyboard/video may differ in information; answer words may come from the question/options; a swap
may inject answer bias; rank and width vary with size; low accuracy → selection/floor effects; a frontier may be a prompt/budget
artefact; a trend may fail held-out; the result may follow from existing multimodal-retrieval or workspace studies. Surviving
foundational objection → RED. Self-report and confidence baselines for any reportability claim. No consciousness, training-history or
alignment claims without experiments.

## 12. Output files
model_lens_manifest.json, dataset_sampling_ledger.csv, image_access.parquet, image_causal_interchange.parquet, video_access.parquet,
unified_modalities.parquet, scaling_fit_predictions.parquet, gate_ledger.csv, negative_results.md, figures/ (n, uncertainty,
causal/non-causal labelling on every figure).

## 13. Stop rules
Full-state transfer fails → no subspace interpretation. Perception fails → no capacity interpretation. J-subspace no more specific
than controls → not the mediator. Video effect vanishes under matched temporal coverage → no temporal-workspace claim. No matched lens
→ no cross-size law. Law fails held-out → abandon it, report the consistent account.

## 14. Deliverable
Report exactly which is supported: A shared causally effective workspace whose access capacity scales; B modality-dependent transport
cost; C task-dependent selection improves with scale without growth in simultaneous content; D the candidate J-space is not a
sufficient common mediator. Do not optimize toward A, B or C.

## Standing instructions added 14 Oct 01:05
- Answer correctness is judged by an LLM judge (Gemini Flash), never by regex heuristics, with the rubric's judge gates (frozen
  model and rubric, raw outputs preserved, failed calls never cached, a hand-label agreement sample). Regex stays only as a logged
  secondary.
- Both J-lens and J++ readouts wherever a J++ fit exists.
- All code, data, runs and caches on the network volume (/workspace); nothing of value on the container disk.
- Every hypothesis, result and verdict is appended to RECORD.md.
