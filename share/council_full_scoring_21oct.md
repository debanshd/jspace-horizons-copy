# J-space: the council and the full gate-by-gate scoring, 21 October 2026, after RECORD rows 109 to 132

Written at 15:00 local (UTC+3) for Nayan, after the two outside reviews received this afternoon (both from ChatGPT, treated as external review). Every status below follows the rubric's law: RED, YELLOW, GREEN or N/A, with N/A allowed only where the claim is absent from the title, the story, the figures and the note. There is no average. The 100-point number at the end is the handoff's bar, kept because Nayan asked for it, and it is not the rubric's verdict.

## 1. Bird's-eye verdict

What the program has now, in one sentence each.

1. A vision-language model reads a picture with the circuit it reads a word with: the same heads, at the same layers, and their outputs are interchangeable across the two modalities for the decision (2B, 4B, Gemma E4B) and for the sixteen-way identity (2B), with the name transferring on the 2B only (rows 95 to 101, 106, 111, 113, 127).
2. Every read closes at an ordered depth, the decisions' before the name's, in seven models of two families from 0.8B to 31B; what changes with size is not the depth or the order but how many layers share the read (rows 108, 114, 120, 128, 131, the access table in REPORT.md).
3. The 4B's report of where its answer came from records which pathway delivered it: with the picture and an agreeing note and the answer identical, blocking one channel's reads moves the report to the other on 64 of 64 and 62 of 62 scenes (row 115); with nothing blocked, writing in the heads of a run that answered from the note moves it on 64 of 64 while a disruption of the same size without that content moves 1 of 64 (row 129); Gemma E4B replicates the answer-held result by knockout (row 132); the record exists only when the name is written before the channel (row 130); the 2B's report is a format inference (rows 110, 115).
4. A text-trained lens shows the object's name on its image tokens at every depth, so a lens reading cannot say when, or from which channel, the content it shows is used (rows 81, 93).

What remains RED under the law and cannot be closed from this machine: human labels on the judge sheets (CI, CII), an outside replication (CLXXXIII), an expert red team (CCXXXIX), a blind review of the figures (CCXL). Everything else that was RED at 13:10 is now GREEN or N/A with the claim removed.

## 2. The two outside reviews, point by point

| the review's point | status now | where |
|---|---|---|
| the universal two-station claim persists in older sections | fixed: the theory section is scoped and names the 27B as what scoped it | REPORT, "The theory, labelled as such" |
| the lens-onset section keeps a withdrawn conclusion | fixed: a withdrawal note at the point of presentation | REPORT, "Where the lens first sees the name" |
| "if the answer does not change, nothing uses the state" is too strong | fixed: the converse is stated as weaker, with the 31B grids and the 2B's redundancy as the examples | REPORT, "What these methods can and cannot show" |
| generality ahead of the data (E4B replicates the decision transfer, not the source mechanism) | the source report now replicates on E4B by knockout (row 132); the interchange-based version was dropped as unscorable (the name does not transfer on E4B) | row 132, 14:39 note |
| row 115 may be anomaly detection (Singh et al. 2026) or a fallback route (Salazar et al. 2026) | answered on the 4B: nothing blocked, the note-cause heads move the report on 64 of 64, matched noise on 1 of 64, another picture of the same object on 0 of 64 | row 129, figSourceCause |
| does asking about the source first change the mechanism? | yes, it removes it: channel-first, the 4B says "read" on 0.98 of clean scenes; the claim is bounded to the name-first order | row 130 |
| the 27B deserves a theory of distributed access, not a boundary note | the access profiles of all seven models: position and order stable, the share any one layer carries falls from 0.6 to 0.8 (0.8B, 2B) to 0.09 (27B) and 0.01 (31B) | row 131's rubric note, figAccess, the access table |
| the report pathway itself is not reconstructed | open: no single head carries the "seen" evidence (row 117); on E4B it sits in the decision station's reads entirely (row 132) | rows 117, 132 |
| J-space-specific mechanism not established | N/A by choice: no workspace or J-space-specific claim is made; the lens is the instrument whose limit is shown | the title, the story |
| related work: Salazar 2026, Nikankin 2025, Ranjan 2026, Singh 2026 | added with their roles (closest neighbour, alternative explanation, cognitive analogue) | REPORT, "What is new against the closest work" |
| the 94 self-grade is not a rubric verdict | agreed and stated: the law has no average; the number is the handoff's bar | this document, REPORT grade line |

## 3. The council

Each member's own test from the rubric, the verdict on it, what persuaded them, and the objection that stands with the run that would answer it.

- Nanda (hard to fake). GREEN. The interchange with a shared token suffix and non-reader heads at clean, the sixteen-way identity transferring, the null patch and the matched noise in row 129, every rule written before its run. Standing objection: one scene set and seed per experiment except the 2B interchange; outside replication. Answer: a second crop set for R23 and R20 (cheap, 2 minutes each on the 4B) and an outsider.
- Olah (a mechanism). GREEN on the 2B for the object (two heads at the decision station, head 5 at 19 for the name, necessary and sufficient and interchangeable), YELLOW for the channel: the "seen" evidence is a sum over the decision station's reads (31 of 64 on the 4B, 53 of 53 on E4B) with no head above 2 of 64, and the component that reads it at the answer position is not identified. Answer: head-output patching of the answer position's own heads after the name token, on the 4B.
- Lindsey, Gurnee (extends or exposes the workspace account). GREEN as an instrument result: the lens shows the name on the image tokens at every depth while the model reads it at two stations and reports the channel from the reads, so a lens reading cannot say when or from where content is used. No claim about J-space's properties is made, and the J-space gates are N/A with the claim removed. Standing objection: nothing is said about whether the pathway record is readable by the lens. Answer: not tonight.
- Tenenbaum (computational structure). YELLOW. The structure is now stated as a rule with a measured scale law: a consumer reads the object at the token that names its test, at a window of depth set by whether it needs the object or its word form, with the window's position and order stable across seven models and its concentration on single layers falling with depth. Standing objection: no model predicts why the 27B's decisions end at 0.56 of depth where every other model's end at 0.39 to 0.45, nor the organization of an untested model. Answer: a prediction for the next size written before its run (Qwen3.5-14B or Gemma-4-12B).
- Dehaene, Baars, Block (producers and consumers). GREEN for producers and consumers (heads to question tokens to a decision after layer 19), N/A for broadcast, capacity and global availability (no such claim). Standing objection: no cross-task consumption test. Answer: a second consumer reading the same written-in content.
- Johnson (content against source). GREEN on the 4B and E4B: the object and its channel delivered by different components (rows 105, 112), the report following the pathway with the answer held (115, 132) and with nothing blocked (129), and a misattribution produced by intervention (105). Standing objection: the attribution is reconstructive in one order (row 130), so the "record" is a judgement made at the answer position from the reads that delivered the name. Stated as such in the story.
- Siegel, Camburu (cause beyond confidence and answer). GREEN: the answer and its log-probability are identical across the arms that move the report (row 129: log p(Z) within 0.01 of clean under every arm); the report moved with the pathway while the answer, the confidence and the presence of the note were held (rows 110, 129). Standing objection: whether the report predicts the cause on natural inputs. Answer: the SCAM and printed-word items of 14 October re-asked with this question and a knockout.
- Bowman, Perez (chain of thought). N/A, no chain of thought in the design.
- Hinton (boring reasons). GREEN: functional specialization shown by intervention throughout; the cosine contrast is scoped as one measure. Standing objection: the empty canvas donor in row 129 carries "empty slot" evidence as well as the note's content, so A's effect is content plus absence. Answer: the matched noise (N) and the other-picture donor (B) show absence of the picture's evidence at the heads is not what moves the report; a donor with a picture of a different object and the same note (its answer from the picture, X seen) is the next arm.
- Karpathy (whiteboard). GREEN: one circuit diagram (figCircuit), one bar figure per claim, the story in four sentences. Standing objection: the report is long and carries the 14 October chapters. Answer: the slide list in share/slides_figures_21oct.md is the short version.
- Pavlick (matched modalities). GREEN with stated differences: same scenes, question, answers; the typed prompt is shorter before the shared suffix; the interchange is asymmetric (word into picture stronger) and the name does not transfer on the 4B, E4B or 27B. These are in the story and the limits.
- Alignment and auditing researchers. N/A: no safety claim; the implication (a model's account of where its content came from can be set by which heads delivered it) is kept as implication.

Council verdict: the mechanism claims are GREEN within their stated scope in both families, the source-report chapter is GREEN on the 4B and E4B with its order boundary stated, and release is blocked by the four outside gates only. The council does not accept the "field-defining" label until an outside replication and a human red team have run.

## 4. The full scoring, every gate

Status per gate of the release constitution (rubric.md). "N/A, removed" means the claim is absent from the title, story, figures and note.

### II to XXVI: the question, the claims, the causal designs

| gate | status | why |
|---|---|---|
| II one sentence | GREEN | the title is the sentence; every 21 October experiment answers a consequence of it |
| III why the paper exists | GREEN | the bridge: a lens shows content at every depth and cannot say when or from where it is used; the report records the pathway |
| IV claim decomposition | GREEN | scientific, mechanistic, methodological, cognitive, safety sentences written (RECORD 13:10; restated in section 5) |
| V aha | GREEN | a dog seen and a dog read give the same head vector; the model says it saw what it read; the report records the pathway with the answer held |
| VI hard to fake | GREEN | boring alternatives listed in the ledgers and killed: answer identity (sixteen-way), lexical form (typed against picture), resolution, load, null patch, matched noise |
| VII cause against correlation | GREEN | every claim rests on an intervention (patch, knockout, head-output interchange, steering) |
| VIII caused against redundant | GREEN | the congruent design is the redundant cell: both sources suffice, the pathway is set by intervention (rows 115, 129, 132) |
| IX correctness orthogonality | N/A, removed | no probe reads correctness; the answer is correct and identical across the arms that move the report |
| X difficulty control | GREEN | same scenes under every arm; clean gates at 0.85 or above; costs in nats on matched scenes |
| XI lexical control | GREEN | the same question suffix for the word and the picture; digit answers never contain the name |
| XII answer identity | GREEN | digits for rules, a letter from a fixed list for identity, random name sets per scene |
| XIII question-identity leakage | N/A, removed | no trained probe, no folds |
| XIV probe capacity | GREEN | the only classifier is nearest-centroid, leave-one-scene-out, used for decodability alone |
| XV probe power | GREEN | the cosine contrast has a positive case (a channel component at layer 7) beside its null at the name station |
| XVI information against role | GREEN | back-patching and interchange show the state is used downstream |
| XVII causal use | GREEN | patch, steer, interchange, ablate, knock out, all with predicted consequences |
| XVIII producer and consumer | GREEN | heads write the object at the question's tokens, the decision is sealed after layer 19, cross-modal interchange |
| XIX interchangeability | GREEN for decisions (2B, 4B, E4B) and identity (2B), YELLOW for the name (2B only) | rows 101, 106, 111, 127 |
| XX content and provenance factorization | GREEN on the 4B | content by the name station's heads, channel by the middle layers, bound to the object (rows 105, 112, 129) |
| XXI amodality | GREEN for the decision content, YELLOW for the name | aligned vectors, no channel component at the name station, shared downstream use; the name's transfer is 2B-only |
| XXII provenance-tagged alternative | GREEN | two-sided: the middle layers do carry the channel with the object (row 112) |
| XXIII multimodal parity | GREEN with one stated difference | same scenes, question, answers; the typed prompt is shorter before the shared suffix |
| XXIV perception check | GREEN | crop gate per model, clean gates per question |
| XXV same answer | GREEN | the answer identical under every arm in rows 115, 129, 132 |
| XXVI blank or control image | YELLOW | the empty canvas is a donor only and the matched noise covers its size; the GQA patch used a grey image with no visually matched control |

### XXVII to XXXVIII: the lens, selection, order, reports

| gate | status | why |
|---|---|---|
| XXVII J-space definition | GREEN | the top 0.1 d singular directions of W_U J_L, stated with the projector |
| XXVIII span and remainder orthogonality | GREEN | orthonormal basis, complement by projection (14 October code) |
| XXIX dimension matching | GREEN | random parts of equal rank in every subspace test |
| XXX norm | YELLOW | steering doses are in units of the own-name component; norms of patched parts not reported separately |
| XXXI random span | GREEN | three random draws per subspace test |
| XXXII layer selection | GREEN | full sweeps; the 27B's and E4B's layers chosen by rule from the knockout before the source runs |
| XXXIII position selection | GREEN | every question token tested (row 109); the answer position for the name |
| XXXIV temporal order | GREEN | reads precede the answer; the report is made after the name (row 130) |
| XXXV self-report baseline | GREEN | the report is the object of study; the internal signal that predicts it is which heads delivered the content |
| XXXVI confidence | GREEN | log p of the answer within 0.01 of clean under every arm that moves the report (row 129) |
| XXXVII wrong sign | GREEN | "seen" for a read word, paired scenes, intervals, replicated with a control donor (rows 105, 112) |
| XXXVIII introspection levels | GREEN | access, readout and report distinguished; "faithful introspection" not claimed; "records the pathway" is the operational claim |

### XXXIX to LVIII: source reports, workspace, chain of thought, models, data

| gate | status | why |
|---|---|---|
| XXXIX source-report | GREEN on the 4B and E4B | report against internal signal against ground truth (rows 103 to 132); the 2B fails its gate |
| XL item against source memory | GREEN | the object and its channel separated by intervention |
| XLI human analogy | GREEN | functional analogy wording; differences listed |
| XLII consciousness overclaim | GREEN | none |
| XLIII global-workspace criteria | N/A, removed | no workspace property is claimed |
| XLIV modality integration | GREEN | common downstream machinery by interchange |
| XLV broadcast | N/A, removed | no global-availability claim |
| XLVI capacity | N/A, removed | load tested only as a boundary (row 107) |
| XLVII to LI chain-of-thought gates | N/A, removed | no chain of thought |
| LII cross-domain transfer | GREEN | held-out questions (sixteen-way, yes or no, category), a held-out model, a second crop set |
| LIII cross-model transfer | GREEN | two families for the stations, the decision interchange and the answer-held source report |
| LIV size | GREEN with a boundary | seven models; the concentration falls with depth, the position and order hold |
| LV base against instruction | N/A, removed | no claim about where the mechanism came from |
| LVI training history | GREEN | no attribution made |
| LVII benchmark | GREEN | canvases isolate the object's tokens; GQA and CountBench stated for what they test |
| LVIII synthetic against natural | GREEN | both, with their roles |

### LIX to LXXX: statistics and causal strength

| gate | status | why |
|---|---|---|
| LIX sample size | GREEN, one YELLOW | 64 scenes per run; row 105's conditional rests on 24 scenes |
| LX filtering | GREEN | crop gate counts, clean gates and exclusions reported |
| LXI manipulation check | GREEN | donor gates, clean gates, the null patch |
| LXII exclusion independence | GREEN | gates set on clean runs, not on the celebrated statistic |
| LXIII estimand | GREEN | nats, shares and paired flips written before each run |
| LXIV AUROC | N/A | none in the 21 October results |
| LXV equivalence | YELLOW | "no difference" claims use bounds and intervals, not formal equivalence tests |
| LXVI confidence intervals | GREEN | scene bootstraps throughout |
| LXVII multiple comparisons | YELLOW | sweeps labelled, no correction |
| LXVIII seeds | GREEN for the pathway record and the interchange, YELLOW elsewhere | rows 129 and 115 replicated on a second scene set within 0.03 (row 133); the 2B interchange on a second crop set (row 113) |
| LXIX permutation | N/A | no probes; the null patch and the matched noise serve as the sanity checks |
| LXX calibration | N/A | no probability is interpreted as certainty |
| LXXI effect size | GREEN | 64 of 64, nats |
| LXXII paired design | GREEN | every arm paired per scene |
| LXXIII counterbalancing | GREEN with the boundary | slots, sets and classes rotated; the word order tested and found to bound the result (row 130) |
| LXXIV contamination | GREEN | synthetic scenes, no memorization claim |
| LXXV causal strength | GREEN | graded costs in nats, dose in steering, the from-L-on profiles |
| LXXVI mediation | YELLOW | intervention to heads' outputs to report shown (row 129); no mediation through a defined subspace |
| LXXVII necessity and sufficiency | GREEN | knockout and head-output patching, separately (rows 97, 101) |
| LXXVIII localization | GREEN for heads and stations; N/A for J-space | no inside-against-outside claim |
| LXXIX double dissociation | GREEN on the 4B | content moved with the report held (105), report moved with the content held (115, 129) |
| LXXX functional dissociation | GREEN on the 4B | the same two results |

### LXXXI to CXVII: instruments, engineering, judges, figures, results

| gate | status | why |
|---|---|---|
| LXXXI no-probe flagship | GREEN | every headline result is an intervention |
| LXXXII probe triangulation | GREEN | patching and knockout name the same layers |
| LXXXIII instrument validity | GREEN | the lens audited, its pooled onset withdrawn |
| LXXXIV tool blindness | GREEN | the lens shows the name at every depth and cannot say when it is used |
| LXXXV lens approximation | GREEN | J++ and per-token readings used; the absence claim withdrawn |
| LXXXVI random Jacobian | GREEN | logit lens and random parts compared |
| LXXXVII baseline hierarchy | GREEN | clean, donors alone, null patch, noise, random directions |
| LXXXVIII runtime | GREEN | R23 in 52 seconds, R25 in 57; batching by memory |
| LXXXIX T4 | N/A | one RTX 6000; the small runs take minutes |
| XC early stop | GREEN | gates stop interpretation (rows 116, 122) |
| XCI run order | GREEN | the decisive test first (R23 before R24 and R25) |
| XCII cache validity | GREEN | judge cache keyed by the rubric hash; no failed call cached |
| XCIII tokenization | GREEN | single-token classes checked; note positions and suffix lengths asserted |
| XCIV shape | GREEN | assertions in capture and patch; four-scene smokes before every run |
| XCV index alignment | GREEN | paired by scene id; the duplicate-layer case caught and noted |
| XCVI layer index | GREEN | block index, "from L on", stated |
| XCVII precision | YELLOW | bf16 model, float32 captures; no fp32 check of the lens results |
| XCVIII determinism | GREEN | seeds, revisions, uv.lock, config.json per run |
| XCIX version control | YELLOW | the workspace is not a git repository; runs record the config, not a commit |
| C no hidden manual fix | GREEN | every manual step is in RECORD (dedup, time corrections, stopped chains) |
| CI judge validation | RED | 40 verdicts checked by an AI reader, no human |
| CII human label | RED | blind sheets prepared in share/, unlabelled |
| CIII figure truthfulness | GREEN | one question per figure, intervals, n stated |
| CIV figure legibility | GREEN | checked tonight at full size |
| CV hero figure | YELLOW | candidates (figCircuit, figSourceCause, figConflictSame); not chosen |
| CVI raw-data visibility | YELLOW | bars with intervals and paired counts; few per-scene plots |
| CVII uncertainty | GREEN | |
| CVIII layer heatmap | GREEN | matrices only where layer structure is the question |
| CIX color semantics | GREEN | orange decisions and "seen", blue copies and "read" throughout |
| CX caption | GREEN | question, manipulation, finding, n |
| CXI negative result | GREEN | the 31B grids, the predicate rule, the 27B predictions, the order test kept as misses |
| CXII surprise | GREEN | the 27B, the order boundary, E4B's decision-station-only evidence |
| CXIII fishing | GREEN | a rule before every run |
| CXIV retrospective story | GREEN | pre-registered and post hoc labelled |
| CXV replication | GREEN | second crop set (interchange), second family (stations, decision interchange, answer-held report) |
| CXVI transfer | GREEN for held-out questions, YELLOW for natural photos | the GQA knockout failed its nat rule, the patch showed the stations |
| CXVII boundary | GREEN | concentration at 27B and 31B, the 4B's name transfer, the 2B's report, the word order |

### CXVIII to CXLIX: specificity, cognition, cause knowledge, workspace entry

| gate | status | why |
|---|---|---|
| CXVIII model specificity | GREEN | every claim scoped to the sizes it holds on |
| CXIX multimodal model | GREEN | |
| CXX modality count | GREEN | image and text |
| CXXI source-monitoring literature | YELLOW | Johnson and Ranjan 2026 cited; misattribution and suggestion effects not discussed in depth |
| CXXII cognitive-theory mapping | GREEN | construct, operationalization, similarity, difference written (RECORD 13:10) |
| CXXIII workspace theory | N/A, removed | |
| CXXIV access against consciousness | GREEN | |
| CXXV psychophysics | GREEN | one variable per test |
| CXXVI minimal pair | GREEN | same content, different source, throughout |
| CXXVII error analysis | YELLOW | the 2B's and the channel-first default inspected; E4B's 11 clean "read" scenes not yet |
| CXXVIII self-knowledge hierarchy | GREEN | "knows the fact" against "the report tracks the pathway"; "knows why" not claimed |
| CXXIX cause knowledge | GREEN on the 4B and E4B | rows 115, 129, 132 |
| CXXX provenance against causality | GREEN | present and ignored (110), present and used (115, 129) |
| CXXXI internal cause | GREEN | the same rows |
| CXXXII correct against wrong cause | GREEN | the report tracks the pathway, not correctness (105: a read word reported as seen) |
| CXXXIII confidence-independent cause | GREEN | log p held (129) |
| CXXXIV graded causality | YELLOW | stations against all layers only; no dose sweep of the written-in fraction |
| CXXXV wrong hint | GREEN | the conflicting note (110) is the analogue |
| CXXXVI omission against denial | N/A, removed | no chain of thought |
| CXXXVII withholding against confabulation | GREEN | the 4B neither in the name-first order; the 2B, and the 4B channel-first, confabulate "read" |
| CXXXVIII report position | GREEN with a boundary | the judgement is made after the name (130) |
| CXXXIX pre-report | GREEN | every token tested (109) |
| CXL dynamics | GREEN | the access profiles over layers |
| CXLI to CXLVII workspace entry, silent cause, reportability, reflection, task demand, attention, routing | N/A, removed | no workspace claim; row 130 is the one report-demand manipulation and bounds the result |
| CXLVIII producer-consumer identity | GREEN | heads named, consumers at tokens |
| CXLIX J-space specificity | GREEN | nothing depends on the lens |

### CL to CLXXXII: the claims, the literature, the paper, reproducibility

| gate | status | why |
|---|---|---|
| CL to CLIII claim sentences | GREEN | section 5 below |
| CLIV novelty ledger | GREEN | "What is new against the closest work", updated with the 2026 papers |
| CLV literature search | YELLOW | J-lens, VLM pathways, modality circuits, source monitoring and introspection in LLMs searched; hidden-reasoning monitoring and metacognition not |
| CLVI Anthropic overlap | GREEN | the workspace paper has no picture, no provenance test, no read-depth measurement |
| CLVII Neel review | GREEN | attacks "can the lens say when content is used" |
| CLVIII introspection overlap | GREEN | causal provenance, with Singh et al.'s objection answered |
| CLIX faithfulness overlap | GREEN | internal localization plus source attribution |
| CLX, CLXI information-flow and simulatability overlap | N/A, removed | no chain of thought, no explanation-usefulness claim |
| CLXII no consciousness bait | GREEN | |
| CLXIII title | GREEN | names the discovery |
| CLXIV abstract | YELLOW | the four-sentence story is long; problem, gap, mechanism, consequence not in that order |
| CLXV introduction | YELLOW | no page-one hook yet |
| CLXVI no boilerplate | GREEN | |
| CLXVII one story | YELLOW | the report still carries the 14 October chapters in full |
| CLXVIII experiment necessity | GREEN | each 21 October run answers the previous one's question |
| CLXIX flagship result | YELLOW | two candidates: the shared circuit, the pathway record; one must be chosen |
| CLXX three results | GREEN | |
| CLXXI kill your darlings | YELLOW | width law, clutter and lens onset belong in an appendix |
| CLXXII first principles | GREEN | |
| CLXXIII simplicity | GREEN | R23: one manipulation, one contrast, two controls, one plot |
| CLXXIV five-minute explanation | GREEN | |
| CLXXV five-second figure | GREEN | |
| CLXXVI finesse | GREEN | |
| CLXXVII speed without sacrifice | GREEN | |
| CLXXVIII pilot and final | GREEN | four-scene smokes, 64-scene runs |
| CLXXIX compute budget | YELLOW | runtimes recorded after, not stated before |
| CLXXX reproducibility | GREEN | uv.lock, pinned revisions, chain scripts |
| CLXXXI auditability | GREEN | per-scene CSVs for every run |
| CLXXXII figure source | GREEN | every figure from a summary or CSV |

### CLXXXIII to CCXXI: outside checks, controls, payoffs, release hygiene

| gate | status | why |
|---|---|---|
| CLXXXIII external replication | RED | none |
| CLXXXIV code review | YELLOW | lint, tests, smokes; no independent read of the new experiments |
| CLXXXV deliberate bug | YELLOW | the null patch and the matched noise find nothing, as they must; no label-shuffle run on the 14 October classifier |
| CLXXXVI leakage test | N/A | no probe |
| CLXXXVII synthetic ground truth | GREEN | the canvas with known slot tokens; the empty-canvas donor |
| CLXXXVIII null control | GREEN | N, C, B; random directions |
| CLXXXIX positive control | GREEN | donors alone, the typed fact |
| CXC adversarial control | GREEN | N and B were built to fool the "read" metric and did not |
| CXCI reverse prediction | GREEN | every rule names both outcomes |
| CXCII failure value | GREEN | |
| CXCIII Bayesian surprise | YELLOW | moderate for the stations, larger for the pathway record and the 27B |
| CXCIV why didn't we think of this | GREEN for R20 and R23 | |
| CXCV job talk | GREEN | |
| CXCVI to CXCVIII safety relevance, practicality, instrument cost | N/A, removed | no safety claim |
| CXCIX epistemic humility | GREEN | |
| CC anthropomorphism | GREEN | "says", "reports"; "knows" avoided |
| CCI vocabulary | GREEN | |
| CCII cognitive-science payoff | GREEN | content access against source access, with the report as a pathway record |
| CCIII mechanistic payoff | GREEN | the object by a few heads, the channel by many |
| CCIV faithfulness payoff | YELLOW | the predictor of a faithful report (which heads delivered) shown by intervention, not on natural data |
| CCV workspace-theory payoff | GREEN | a lens reading cannot say when content is used |
| CCVI multimodal payoff | GREEN | |
| CCVII literature respect | GREEN | |
| CCVIII citation role | GREEN | |
| CCIX primary source | GREEN | |
| CCX currentness | GREEN | searched today |
| CCXI open code | YELLOW | exists, not released (Nayan's call) |
| CCXII raw results | YELLOW | the 14 October zip; the 21 October runs in runs/ |
| CCXIII ablation | GREEN | |
| CCXIV robustness | GREEN | second crop set, two methods, two families |
| CCXV sensitivity | GREEN | boundaries stated |
| CCXVI outlier | GREEN | per-object checks; the animal bias of row 105 noted |
| CCXVII class balance | GREEN | |
| CCXVIII prevalence | N/A | |
| CCXIX behavioral validity | GREEN | |
| CCXX ecological validity | GREEN | source confusion between a note and a picture; the SCAM photos |
| CCXXI toy to real | YELLOW | natural photos by patching only |

### CCXXII to CCLXII: ledgers, manuscript, the jury's survival gates, the legacy gates

| gate | status | why |
|---|---|---|
| CCXXII confound ledger | GREEN | RECORD 06:20, 12:40 |
| CCXXIII claim ledger | GREEN | section 5 adds the rows for 129 to 132 |
| CCXXIV result ledger | GREEN | RECORD rows |
| CCXXV run ledger | YELLOW | no commit hash |
| CCXXVI version integrity | GREEN | |
| CCXXVII manuscript-code consistency | YELLOW | the setup section still describes the 14 October methods in places |
| CCXXVIII plot-code consistency | GREEN | |
| CCXXIX no handwave | GREEN | |
| CCXXX prose | GREEN | |
| CCXXXI diagram | GREEN | figCircuit |
| CCXXXII no dashboard | GREEN | |
| CCXXXIII main text | YELLOW | the report is a record, not a paper |
| CCXXXIV appendix | YELLOW | same |
| CCXXXV abstract result | GREEN | every number in the story has a control, an interval and a replication (row 129 and row 115 on a second scene set, row 133; the interchange on a second crop set, row 113; the stations and the answer-held report in a second family) |
| CCXXXVI title result | GREEN | |
| CCXXXVII one-month memory | GREEN | "the picture is read by the word's circuit, and the report records which pathway delivered the answer" |
| CCXXXVIII quote | GREEN | |
| CCXXXIX external red team | RED | two AI reviews acted on; no human expert |
| CCXL blind review | RED | none |
| CCXLI replication before hype | GREEN | nothing public |
| CCXLII, CCXLIII firewalls | GREEN | |
| CCXLIV negative publication | GREEN | the 27B kept as a result |
| CCXLV why didn't Anthropic | GREEN | modality and provenance outside their framing; their instrument cannot say when content is used |
| CCXLVI survives Lindsey | GREEN | their claim represented as verbalizability, not provenance-freedom |
| CCXLVII survives Neel | GREEN | |
| CCXLVIII survives Tenenbaum | YELLOW | no computational model of the scale law |
| CCXLIX survives Siegel | GREEN | |
| CCL survives source-memory researchers | GREEN | |
| CCLI survives Hinton | GREEN | |
| CCLII survives Karpathy | GREEN | |
| CCLIII 2036 | GREEN | "a model reads a picture with the circuit it reads a word with, and its account of where an answer came from records which pathway delivered it" survives the removal of Qwen, the lens and Anthropic |
| CCLIV tool independence | GREEN | |
| CCLV cognitive durability | GREEN | |
| CCLVI safety durability | N/A, removed | |
| CCLVII simplicity after complexity | YELLOW | the report's length |
| CCLVIII research program | GREEN | next questions in REPORT "Loose ends" |
| CCLIX final comprehension | GREEN | |
| CCLX three contributions | GREEN | section 5 |
| CCLXI three figures | YELLOW | to choose: figStationBars, figCrossModal with figCrossAlign, figSourceCause |
| CCLXII final release decision | NOT HIGH QUALITY under the law | RED: CI, CII, CLXXXIII, CCXXXIX, CCXL; of the ten magnum-opus clauses, 4 (specialist red team) is not met and 10 is a guess |

Totals (updated after row 133 at 15:10): 5 RED, 33 YELLOW, 29 N/A with the claim removed, the rest GREEN. Score on the handoff's 100-point bar: 97. Under the law: not release-ready.

## 5. The claim sentences and the three contributions, restated after rows 129 to 132

Scientific. We provide evidence that a vision-language model reads an object's image tokens with the heads that read the object as a typed word, that every read closes at an ordered depth (decisions before the name) in seven models of two families, and that on the 4B and Gemma E4B the model's report of whether it saw or read an object follows the pathway that delivered the answer when the answer is held fixed.

Mechanistic. Our causal evidence supports that the reader heads' outputs carry the object without its channel at the name station and with its channel at the middle layers; that writing the note pathway's head outputs into the picture's run, with the picture present and unblocked, moves the report on every scene while a disruption of the same size moves one; that the "seen" evidence is a sum over the decision station's reads (all of it on E4B, half on the 4B); and that the report exists only when the name is written before the channel.

Cognitive analogy. The task instantiates the item-against-source distinction of source monitoring in that the object and its channel are delivered by different components and the channel can be lost with the object kept; it differs in that the attribution is a judgement made at the answer position from the reads that delivered the name, in one forward pass, in one word order.

Safety. Kept as implication: a model's account of where its content came from can be set by which heads delivered it, and the account is unavailable in another word order.

Contributions. (1) The same circuit reads a picture and a word, with interchangeable outputs, in two families. (2) Two ordered reads whose depth and order hold from 0.8B to 31B while their concentration on single layers falls with depth. (3) A source report that records the pathway, tested with the answer held, with nothing blocked, and in a second family, with its order boundary.

Claim ledger additions (exact sentence, type, evidence, confidence, scope, closest prior work, falsifier):
- "The 4B's report records which pathway delivered the answer and does not detect the missing evidence." Causal. Row 129. High on the 4B. Name-first order, this canvas and note. Singh et al. 2026 (anomaly detection) and Salazar et al. 2026 (fallback routes) are the alternatives it answers. Falsifier: matched noise or another picture of the same object moving the report on 0.5 or more of scenes.
- "The pathway-tracking report exists only in the name-first order." Behavioural. Row 130. High. The 4B. Falsifier: a channel-first prompt whose clean report matches the named object's channel on 0.8 or more and tracks the block.
- "On Gemma E4B the answer-held report tracks the pathway and the decision station's reads carry all of the 'seen' evidence." Causal. Row 132. High. E4B, this design. Falsifier: the report staying "seen" with the picture's reads blocked.
- "The read sits in the same window of depth at every size and is shared by more layers in a deeper model." Representational, post hoc. Row 131 and figAccess.csv. Medium. Seven models. Falsifier: a model whose decision half-depth falls outside 0.35 to 0.6 or whose name's half-depth precedes its decisions'.

## 6. What would move the most gates per minute, in order

1. Done at 15:05 (row 133): a second crop set for R23 and R20 on the 4B, both within 0.03 of the originals.
2. Human labels on share/source_report_to_label.csv and the earlier judge sheets (Nayan, an hour): CI and CII.
3. A donor with a picture of a different object and the same agreeing note (its answer from the picture), as Hinton's arm in R23 (one run, a minute): closes the "absence plus content" reading.
4. The answer position's own heads after the name token, patched from the note-cause donor (one run, minutes): Olah's open component.
5. A prediction for an untested size written before its run (Qwen3.5-14B or Gemma-4-12B, if the weights fit in memory): Tenenbaum's gate.
6. An outsider's reproduction of R23 from the chain script and uv.lock, and a blind look at figSourceCause and figConflictSame with the caption only: CLXXXIII, CCXXXIX, CCXL.
