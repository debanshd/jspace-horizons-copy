Yes. For this project the previous rubric would be nowhere near enough.

This project is trying to sit at the intersection of **mechanistic interpretability, cognitive science, metacognition, chain-of-thought faithfulness, source monitoring, multimodal representations, and global-workspace theory**. The standard therefore has to be different from the taste paper: here, **one uncontrolled confound can kill the headline**, because the contribution depends on knowing what an internal representation actually means.

Your current experimental thinking already contains the right instinct. The five-cell probe design exists specifically because “caused vs redundant” alone could secretly be detecting correctness, difficulty, answer identity, or lexical form rather than causal influence; the correct/wrong caused contrast is intended to break the correctness explanation. Pasted text And the multimodal W5 idea asks something substantially deeper than “can we decode a concept?”: whether the same semantic content arriving through image versus text loses or retains **provenance inside the proposed workspace itself**. Pasted text

That is the level the rubric needs to enforce.

# J-SPACE / WORKSPACE PSYCHOPHYSICS — COMPLETE RELEASE CONSTITUTION

## 0. Release law

There are four statuses:

**RED** — false, confounded, unsupported, or unclear.

**YELLOW** — promising but incompletely established.

**GREEN** — survives the strongest obvious alternative explanation.

**N/A** — permitted **only if the corresponding claim is removed** from the abstract, title, figures, introduction, conclusion, and publicity.

There is **no weighted average**.

A beautiful result with one fatal confound remains RED.

A novel phenomenon with invalid statistics remains RED.

A perfect probe with no causal relevance remains RED.

A causal intervention with no construct validity remains RED.

A 100-hour run that cannot be debugged or reproduced is not a virtue.

A viral plot is irrelevant if a domain expert can destroy it in thirty seconds.

The paper becomes **HIGH QUALITY only when every applicable gate is GREEN.**

The higher target—**field-defining candidate**—requires the final legacy gates too.

Your own stated criteria were already correct: novelty, “aha,” speed, correctness, completeness, finesse, simplicity, strong visuals, and something that makes experts ask “why didn't we think of this?” Pasted text (2)

But the order matters:

\[
\boxed{\text{correctness}}
>
\boxed{\text{identification}}
>
\boxed{\text{novelty}}
>
\boxed{\text{simplicity}}
>
\boxed{\text{finesse}}
>
\boxed{\text{speed}}
>
\boxed{\text{virality}}.
\]

If correctness fails, everything beneath it is worthless.

---

# I. The jury

These are not endorsements. They are adversarial mental models.

## Neel Nanda

Neel currently runs Google DeepMind's mechanistic interpretability team and explicitly frames the job as reverse-engineering learned algorithms and structures. [Neel Nanda](https://www.neelnanda.io/about?utm_source=chatgpt.com) His review of Anthropic's global-workspace paper separates four claims—scientific, methodological, pragmatic, and philosophical—and says the **scientific claim** is by far the most interesting because the paper provides enough “hard-to-fake” evidence that something important is genuinely happening. [LessWrong](https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S?utm_source=chatgpt.com)

The Neel test:

> **What observation would be genuinely hard to fake under every boring alternative explanation?**

If the answer is “AUROC = .62,” RED.

---

## Chris Olah

The Olah test:

> **Have you actually discovered a mechanism, or merely found a correlating direction?**

A useful probe is not automatically mechanistic interpretability.

A representation is not automatically a circuit.

A causal intervention is not automatically an algorithm.

If the paper uses “mechanism” loosely, RED.

---

## Jack Lindsey / Wes Gurnee / the Anthropic workspace team

Their paper argues that J-space contains verbalizable intermediate representations that can be reported, summoned, held, used during silent reasoning, and passed to downstream computations. [arXiv](https://arxiv.org/abs/2607.15495?utm_source=chatgpt.com)

Their test:

> **Does your result genuinely extend, refine, falsify, or expose a hidden assumption of the workspace account?**

Not merely:

> “We ran J-Lens on another benchmark.”

RED if so.

---

## Josh Tenenbaum

Tenenbaum's work is fundamentally about **structured representations and inductive inference**—what latent structures support flexible cognition and generalization. [Cocosci](https://cocosci.mit.edu/josh-tenenbaum/?utm_source=chatgpt.com)

His test:

> **What computational structure have you discovered, and what does it let the system infer or fail to infer?**

A cloud of decodable features is not a cognitive theory.

---

## Stanislas Dehaene / Bernard Baars / Ned Block

The cognitive-workspace jury.

Their questions:

> What exactly do you mean by “global”?

> What is being broadcast?

> Who are the producers and consumers?

> Is accessibility being confused with consciousness?

> Is verbalizability being confused with global availability?

> Is this a workspace or merely a subspace decodable near the output?

If “global workspace” is philosophical decoration, RED.

---

## Melissa Johnson / source-monitoring tradition

The classic source-monitoring framework studies how agents attribute remembered content to origins such as perception, suggestion, imagination, or another speaker, and explicitly treats source attributions as fallible inference over perceptual, contextual, semantic and cognitive evidence. [PubMed](https://pubmed.ncbi.nlm.nih.gov/8346328/?utm_source=chatgpt.com)

Their test:

> **Have you separated knowing the content from knowing where the content came from?**

This is central to W5.

---

## Noah Y. Siegel / Oana Camburu / explanation-faithfulness researchers

Siegel and collaborators explicitly study whether natural-language explanations track the actual factors responsible for model decisions, using counterfactual faithfulness tests and showing that apparent faithfulness can be confounded by properties such as explanation verbosity. [Google DeepMind](https://deepmind.google/research/publications/78755/?utm_source=chatgpt.com)

Their test:

> **Does your internal signal predict the actual cause of a decision rather than correlate with the explanation, correctness, confidence, or answer?**

---

## Sam Bowman / Ethan Perez / CoT-faithfulness lineage

Chain-of-thought can vary substantially in faithfulness, sometimes influencing answers heavily and sometimes being largely bypassed. [arXiv](https://arxiv.org/abs/2307.13702?utm_source=chatgpt.com)

Their test:

> **When CoT omits a cause, can you establish that the cause still influenced computation?**

Decodability alone is insufficient.

---

## Geoffrey Hinton

The representation test:

> **Could the information simply exist in a representation for boring statistical reasons without playing the cognitive role you claim?**

If yes, the paper needs more.

---

## Andrej Karpathy

The compression/clarity test:

> **Can the whole discovery be drawn on a whiteboard in sixty seconds?**

A world-class idea can have complex evidence.

Its core mechanism should still be simple.

---

## Ellie Pavlick / multimodal cognition researchers

The multimodal test:

> **Are “image” and “text” actually matched as information sources, or did tokenization, perceptual difficulty, modality-specific processing, or task asymmetry create the effect?**

This is the major risk in W5.

---

## Alignment / auditing researchers

Their test:

> **Would this actually help detect hidden reasoning, misleading explanations, prompt injection, eval awareness, or deceptive cognition better than asking the model or reading CoT?**

If “practical safety relevance” is claimed, demonstrate it.

Otherwise remove the claim.

---

# II. The one-sentence hypothesis gate

The entire project must have one deep sentence.

Not five projects.

Not “studying J-space.”

A candidate form might eventually be:

> **The model's reportable workspace carries the content needed for reasoning while causal provenance can remain elsewhere, creating systematic limits on what the model can faithfully report about why it knows what it knows.**

That is only an example.

The actual gate:

All major experiments must answer **different consequences of the same sentence**.

If Experiment 1, 5, 6 and W5 look like unrelated cool interpretability hacks, RED.

---

# III. The “why does this paper exist?” gate

It cannot merely establish:

> J-space exists.

Anthropic already argues that extensively. [arXiv](https://arxiv.org/abs/2607.15495?utm_source=chatgpt.com)

It cannot merely establish:

> models can introspect somewhat.

Anthropic already reports limited functional introspective abilities. [Anthropic](https://www.anthropic.com/research/introspection?_bhlid=10d512478c7d831c58ebdda6008ed456d661b21e\&utm_source=chatgpt.com)

It cannot merely establish:

> CoT can be unfaithful.

Established literature.

It cannot merely establish:

> source monitoring exists in humans.

Decades old.

The paper needs the **new bridge**.

Possible candidate:

> **The architecture that makes internal content globally available may abstract away exactly the provenance information needed for faithful self-explanation.**

That would be interesting.

But you must earn it.

---

# IV. Claim decomposition gate

Every major claim gets typed.

At minimum separate:

### Scientific claim

Something exists internally.

### Mechanistic claim

It plays a causal computational role.

### Methodological claim

Your instrument reliably reveals it.

### Cognitive analogy

It resembles some human cognitive construct.

### Safety/pragmatic claim

It helps monitoring or oversight.

Never let evidence for one silently support another.

This follows exactly the separation Neel used in evaluating the Anthropic paper. [LessWrong](https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S?utm_source=chatgpt.com)

---

# V. Aha gate

At least one result must survive this question:

> **If true, is the natural expert response “oh wow,” rather than “sure, I guess”?**

Good aha:

> The model can access the causal source internally while confidently reporting the opposite.

Potentially great aha:

> J-space preserves semantic content while systematically discarding which modality supplied it.

Weak:

> A linear probe can predict whether a hint was present.

Weak:

> Different layers encode different things.

Weak:

> Reasoning changes residual activations.

No flagship paper around predictable results.

---

# VI. Hard-to-fake gate

For every headline result ask:

> What boring mechanism could produce this plot?

List them before running.

Then kill them.

For a causal-suggestion probe:

- correctness;
- difficulty;
- answer confidence;
- answer letter;
- hint words;
- question identity;
- transcript length;
- position;
- model uncertainty;
- refusal style.

For modality provenance:

- different token counts;
- image embedding statistics;
- blank-image artifacts;
- different task difficulty;
- visual recognition errors;
- prompt wording;
- residual norm;
- dimensionality;
- modality-specific token positions.

A result is headline-worthy only once boring alternatives become implausible.

---

# VII. Cause-versus-correlation gate

The word **cause** may appear only if:

\[
\text{intervene on factor}
\rightarrow
\text{decision changes}
\]

or an equivalently justified causal design exists.

Decoding:

\[
P(y\mid h)
\]

is not causality.

A representation can encode:

- input;
- output;
- difficulty;
- nuisance structure;

without causing behavior.

---

# VIII. Caused-vs-redundant gate

If Experiment 5 stays, a simple:

\[
\text{caused vs redundant}
\]

classifier is insufficient.

Because:

\[
\text{caused}
\approx
\text{model initially uncertain}
\]

while:

\[
\text{redundant}
\approx
\text{model already knew}.
\]

Your own five-cell redesign exists precisely to kill this confound. Pasted text

Do not regress.

---

# IX. Correctness orthogonality gate

Your best control may be:

\[
\text{caused-correct}
\quad \text{vs} \quad
\text{caused-wrong}.
\]

If both were genuinely caused by the suggestion, a **causal-influence** signal should not strongly distinguish them merely because the final answer is right versus wrong.

But important statistical correction:

**failure to reject AUROC \(\neq .5\) is not enough.**

You need an **equivalence test**.

Predefine something like:

\[
0.45 < \text{AUROC} < 0.55
\]

or another justified practical-equivalence region.

Then show the confidence interval lies inside it.

Otherwise:

> “we got \(p>.05\), therefore chance”

is invalid.

---

# X. Difficulty-control gate

Caused items are often harder.

Therefore compare against:

- lexical hard controls;
- uncertainty-matched controls;
- confidence-matched controls;
- baseline-error-rate matched controls.

If the probe separates caused from redundant but equally separates hard from easy, RED.

---

# XI. Lexical-control gate

The same surface words must appear without causal suggestion.

Your current “professor wrote this question” style lexical controls are exactly aimed at this problem. Pasted text

A hint probe that merely reads “professor” is worthless.

---

# XII. Answer-identity gate

Counterbalance:

- A/B/C/D;
- correct option positions;
- hinted option positions;
- caused and redundant classes.

Never let label identity predict class.

---

# XIII. Question-identity leakage gate

Every transcript from a given question belongs to the **same cross-validation fold**.

You already designed grouped OOF evaluation for this reason. Pasted text

No train/test variants of the same question.

---

# XIV. Probe-capacity gate

Use the simplest probe that answers the question.

Then add:

- shuffled-label control;
- random-direction baseline;
- dimensionality-matched random subspace;
- regularization;
- held-out selection;
- no test-set hyperparameter tuning.

A giant nonlinear probe proving “information exists somewhere” is not impressive.

---

# XV. Probe-power gate

Chance decoding does **not** establish absence.

For W5 especially:

> provenance not decodable inside J-space

can mean:

1. provenance absent;
2. probe underpowered;
3. wrong layer;
4. J-space estimate noisy;
5. poor stimuli.

Therefore require positive controls demonstrating the same probe can recover:

- source outside the span;
- semantic content inside the span;
- source in a condition where it is intentionally made relevant.

Without power controls, “amodal” is unsupported.

---

# XVI. Information-vs-role gate

Distinguish:

> information is decodable

from:

> information is represented in a format used by downstream computation.

A serious workspace claim needs the second.

---

# XVII. Causal-use gate

When practical, intervene:

- patch;
- steer;
- interchange;
- ablate;
- project out.

Then show predicted behavioral consequence.

If provenance is outside J-space but removing it does nothing, perhaps it is merely residue.

---

# XVIII. Producer-consumer gate

A true workspace story implies:

\[
\text{producer}
\rightarrow
\text{workspace content}
\rightarrow
\text{consumer}.
\]

Ask whether the same internal content can be inserted by one process and consumed by another.

Cross-task or cross-modality causal interchange would be especially strong.

---

# XIX. Workspace-interchangeability gate

Strong version:

If two routes encode the **same workspace content**, swapping that content should preserve downstream use even when their provenance differs.

That is vastly more mechanistic than a probe alone.

---

# XX. Content/provenance factorization gate

W5's flagship question should be formalized as:

\[
h = h_{\text{content}} + h_{\text{provenance}} + h_{\text{other}}
\]

conceptually—not necessarily literally additive.

The experiment asks whether:

\[
\text{content} \in J
\]

while:

\[
\text{provenance} \notin J
\]

or whether both are encoded inside.

Make the alternatives explicit.

---

# XXI. Amodality gate

Chance provenance decoding **does not by itself prove amodality**.

A modality-independent representation requires evidence that:

- content representations align across modalities;
- modality/source information is reduced relative to controls;
- the common representation plays a shared downstream role.

Otherwise all you've shown is lost source information.

---

# XXII. Provenance-tagged workspace alternative

The experiment must be genuinely two-sided.

If provenance is strongly represented within J-space, that is not “failure.”

It means:

> the workspace may carry content-source bindings.

Potentially equally interesting.

A good experiment has two interesting outcomes.

---

# XXIII. Multimodal parity gate

Picture and note conditions must be matched for:

- semantic content;
- task relevance;
- answer;
- confidence;
- difficulty;
- token count where feasible;
- exposure;
- distractors;
- information timing.

Otherwise source decoding is trivial.

---

# XXIV. Perception-check gate

Before interpreting internal differences, confirm the model actually perceived the image correctly.

If it misread the red swatch, that item cannot test amodal representation of “red.”

---

# XXV. Same-answer gate

At the report position the model should be poised to emit the same semantic answer across modality conditions.

Otherwise internal differences could simply reflect output differences.

Your W5 design explicitly aims at this. Pasted text

---

# XXVI. Blank/control-image gate

Blank or control images cannot themselves create a trivially separable distribution.

Use visually matched noninformative controls where possible.

---

# XXVII. J-space-definition gate

State exactly what “inside J-space” means computationally.

Is it:

- top singular directions of Jacobian?
- token-aligned span?
- verbalizable subspace?
- projector produced by J-Lens?

Readers must be able to reproduce the projection.

---

# XXVIII. Span/remainder orthogonality gate

If residual is decomposed into:

\[
h_J=P_Jh
\]

and:

\[
h_\perp=(I-P_J)h,
\]

verify numerical/projector assumptions.

If the Jacobian basis is not orthonormal, handle it correctly.

No casual subtraction.

---

# XXIX. Dimension-matching gate

If J-space and complement differ dramatically in dimensionality, raw probe accuracy is unfair.

Compare to:

- dimensionality-matched random spans;
- norm-matched random spans;
- randomly rotated subspaces.

---

# XXX. Norm gate

Check whether class information is encoded merely in:

\[
\lVert h\rVert.
\]

Normalize or explicitly report.

---

# XXXI. Random-span gate

Every “J-space versus outside” claim requires random-span controls of equal dimension.

Your own W5 design already identified this as necessary. Pasted text

---

# XXXII. Layer-selection gate

No cherry-picked layer.

Use:

- preregistered band;
- training/validation selection only;
- or full layer sweep with multiplicity control.

Never select the best test layer.

---

# XXXIII. Position-selection gate

The report/answer position must be theoretically justified.

If claim is:

> causal content survives until report,

measure the exact computational position before answer emission.

No arbitrary token.

---

# XXXIV. Temporal-order gate

The representation must occur **before** the behavior it allegedly explains.

Post-decision decodability does not establish decision mechanism.

---

# XXXV. Self-report gate

Always compare the internal metric to:

> just ask the model.

Your file explicitly requires beating the ask baseline and confidence baseline for the causal probe. Pasted text

If internal interpretability does not beat self-report, the practical story weakens sharply.

---

# XXXVI. Confidence gate

Compare against:

- answer probability;
- margin;
- entropy;
- calibration;
- baseline uncertainty.

If a probe mostly reconstructs confidence, say so.

---

# XXXVII. Wrong-sign gate

One potentially striking result is:

> self-report moves in the wrong direction while internal causal signal moves correctly.

If that appears:

- reproduce it;
- rule out prompt effects;
- show paired items;
- show uncertainty;
- show effect size.

Do not oversell a noisy correlation reversal.

---

# XXXVIII. Introspection gate

Distinguish:

### access

internal information exists;

### readout

model behavior correlates with it;

### introspection

model can report it because it has functional access;

### faithful introspection

report tracks actual internal cause.

Those are different achievements.

---

# XXXIX. Source-report gate

For multimodal provenance:

Ask:

> “Did this information come from the image or the note?”

Then compare:

\[
\text{source report}
\]

against:

\[
\text{internal source encoding}
\]

and:

\[
\text{ground truth}.
\]

Errors may be more informative than successes.

---

# XL. Item-memory vs source-memory gate

Human source-monitoring theory explicitly distinguishes remembering content from remembering where it came from. [PubMed](https://pubmed.ncbi.nlm.nih.gov/8346328/?utm_source=chatgpt.com)

Your model experiment should make exactly that separation.

Content accuracy ≠ provenance accuracy.

---

# XLI. Human-analogy gate

Do not say:

> “the model has source memory like humans.”

Say:

> “the task instantiates an analogous computational distinction between item content and source attribution.”

Functional analogy only unless stronger evidence exists.

---

# XLII. Consciousness-overclaim gate

J-space evidence does not establish consciousness.

Anthropic itself describes functional hallmarks of conscious access, not proof of phenomenal consciousness. [arXiv](https://arxiv.org/abs/2607.15495?utm_source=chatgpt.com)

Your paper should be even more restrained.

---

# XLIII. Global-workspace criteria gate

If the project uses “global workspace,” specify which functional properties are under test:

- reportability;
- controllability;
- maintenance;
- flexible downstream consumption;
- broadcasting;
- capacity limitation;
- modality integration;
- source abstraction.

Do not treat the label as indivisible.

---

# XLIV. Modality-integration gate

A genuine multimodal workspace claim should show that representations generated from different modalities become usable by common downstream machinery.

Not merely that probes can classify both.

---

# XLV. Broadcast gate

If you claim “global availability,” show broad downstream access or causal use.

Otherwise call it a workspace-like subspace, not globally broadcast.

---

# XLVI. Capacity gate

If you make capacity claims, manipulate load.

Do not inherit “tens of concepts” from another paper as your result.

---

# XLVII. Faithfulness gate

For CoT experiments define faithfulness operationally.

Possible:

> the visible reasoning reports a causal factor responsible for the answer.

Do not conflate:

- plausibility;
- correctness;
- informativeness;
- faithfulness.

---

# XLVIII. Omission gate

If the CoT omits a cause while internal state retains it:

1. establish the factor truly caused the answer;
2. establish the CoT omitted it;
3. establish the internal representation predicts the cause;
4. establish it is not merely prompt memory;
5. preferably show causal relevance.

Only then say:

> internal causal information outlives verbal explanation.

---

# XLIX. CoT-judge gate

If an LLM judges whether a chain mentions a cause:

- hand-label a substantial random sample;
- report precision/recall/agreement;
- inspect disagreements;
- freeze rubric;
- never cache failed calls as valid.

Your existing suite already treats hand-validation of judges as mandatory. Pasted text

---

# L. Explanation-length gate

Faithfulness results can be confounded by verbosity. Siegel et al. explicitly report that apparent instruction-tuning differences can often reflect explanation verbosity and true/false-positive tradeoffs. [arXiv](https://arxiv.org/abs/2503.13445?utm_source=chatgpt.com)

Control or model explanation length.

---

# LI. Hidden-cause gate

The strongest chain-of-thought case is one where:

- cause is experimentally known;
- visible explanation excludes it;
- internal representation preserves it;
- behavioral answer still depends on it.

That triangulation is far stronger than any one component.

---

# LII. Cross-domain transfer gate

If probe trained on MMLU transfers to ARC:

- freeze weights;
- freeze layer;
- freeze threshold;
- no tuning on ARC;
- ensure answer/format vocabularies do not leak domain identity.

Then transfer actually means something.

---

# LIII. Cross-model transfer gate

If claiming architecture-level phenomenon, show at least one independent model family.

Otherwise claim:

> “we observe this in Qwen X.”

No shame in narrow claims.

---

# LIV. Size gate

A 4B T4 model is excellent for rapid discovery.

It is weak evidence for universal cognitive architecture.

For a field-level claim either:

- replicate on larger/different models;
- or narrow the conclusion.

---

# LV. Base-vs-instruction gate

If post-training is relevant, compare:

- base;
- instruction-tuned;
- reasoning-tuned if possible.

Otherwise do not infer where the mechanism came from.

---

# LVI. Training-history gate

Do not attribute a phenomenon to:

- RLHF;
- instruction tuning;
- reasoning training;

without a model comparison that supports it.

---

# LVII. Benchmark gate

Use datasets because they isolate the question.

Not because they are standard.

For each benchmark state:

> what property makes this benchmark diagnostic?

---

# LVIII. Synthetic-vs-natural gate

Synthetic tasks provide clean causality.

Natural tasks provide ecological validity.

The strongest paper often uses both.

Do not confuse the roles.

---

# LIX. Sample-size gate

Power before aesthetics.

Report:

- \(N\);
- surviving items after filters;
- positives/negatives;
- effective groups;
- exclusions.

Your earlier Experiment 1 had only 13 items in a subgroup, motivating an increase to 480 candidates. Pasted text

Small-\(N\) headline results = RED.

---

# LX. Filtering gate

Every filter is a potential selection bias.

Report:

- order-invariance filter;
- correctness filter;
- causal-following filter;
- single-token constraints;
- confidence filters.

Show how many items disappear.

---

# LXI. Manipulation-check gate

Every intervention needs a manipulation check.

If a hint is supposed to cause an answer, confirm it did.

If steering is supposed to alter workspace content, verify it.

If modality is supposed to convey “red,” verify perception.

---

# LXII. Exclusion-independence gate

Never define the analyzed population using the same statistic you then celebrate.

Avoid collider/selection artifacts.

---

# LXIII. Statistical-estimand gate

Before computing anything, write:

> What quantity do we want to estimate?

Then choose metric.

Not reverse.

---

# LXIV. AUROC gate

AUROC is useful only when the question is rank discrimination.

Report:

- AUROC;
- uncertainty;
- class counts;
- baseline.

No giant decorative AUROC bars with no effect interpretation.

---

# LXV. Equivalence gate

Whenever the claim is **no difference**, use equivalence testing or sufficiently tight interval bounds.

Never:

\[
p>.05 \Rightarrow \text{same}.
\]

---

# LXVI. Confidence-interval gate

Headline numbers get confidence intervals.

Prefer participant/question-group bootstrap when dependence requires it.

Not iid row bootstrap if rows share questions.

---

# LXVII. Multiple-comparison gate

If sweeping:

- layers;
- positions;
- scales;
- task arms;

control multiplicity or clearly label exploratory analysis.

---

# LXVIII. Seed gate

Repeat stochastic training/probing across seeds where material.

One lucky probe fit is not science.

---

# LXIX. Permutation gate

Useful probe results should survive label-permutation sanity checks.

---

# LXX. Calibration gate

If interpreting probabilities/confidence, check calibration.

Raw softmax is not epistemic certainty.

---

# LXXI. Effect-size gate

Statistical significance is not interesting.

Ask:

> Is the effect large enough to matter scientifically?

---

# LXXII. Paired-design gate

When possible use the same question under multiple manipulations.

Paired contrasts are more convincing and efficient.

---

# LXXIII. Counterbalancing gate

Systematically rotate:

- hints;
- labels;
- options;
- source modality;
- wording;
- order.

---

# LXXIV. Contamination gate

Ensure benchmark answers weren't merely memorized if claim depends on reasoning.

Your caused/redundant design partly handles this by conditioning on baseline behavior, but be explicit.

---

# LXXV. Causal-strength gate

Do not binarize causal influence unnecessarily.

Measure:

\[
\Delta p(\text{hinted option})
\]

or other graded intervention strength where useful.

Then ask whether representation tracks causal magnitude.

---

# LXXVI. Mediation gate

A very strong result would establish:

\[
\text{intervention}
\rightarrow
\text{workspace state}
\rightarrow
\text{answer}
\]

with workspace manipulation mediating behavior.

Hard, but much stronger than correlation.

---

# LXXVII. Necessity / sufficiency gate

For any proposed mechanism ask separately:

### Necessary?

Does removing it impair the behavior?

### Sufficient?

Does injecting it induce the behavior?

A causal story is stronger when both converge.

---

# LXXVIII. Localization gate

If saying “this lives outside J-space,” establish specificity:

- outside > inside for provenance;
- inside > outside for content;
- random spans do neither equivalently.

Double dissociation is the goal.

---

# LXXIX. Double-dissociation gate

The dream W5 pattern:

\[
\text{semantic content: J-space} > \text{remainder}
\]

while:

\[
\text{provenance: remainder} > \text{J-space}.
\]

That is much more compelling than:

> provenance probe is 0.61 outside.

---

# LXXX. Functional-dissociation gate

Even better:

intervening on content in J-space changes answer semantics;

intervening on provenance outside changes source report but not content answer.

That would be extremely strong.

---

# LXXXI. No-probe flagship gate

At least one key result should ideally be visible **without trusting a trained probe**.

For example:

- activation patching;
- steering;
- direct J-Lens readout;
- interchange;
- source-report error relation.

Neel explicitly values hard-to-fake evidence; reducing probe dependence helps. [LessWrong](https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S?utm_source=chatgpt.com)

---

# LXXXII. Probe triangulation gate

Where probes are necessary, triangulate with another method.

Never let the whole paper depend on one classifier family.

---

# LXXXIII. Instrument-validity gate

J-Lens is an instrument.

Therefore ask:

> What can it see?

> What can it systematically miss?

Your proposed 5b question—whether causal information lives inside the span the lens can read or outside—is exactly the kind of instrument audit that could matter. Pasted text

---

# LXXXIV. Tool-blindness gate

If important causal information systematically lies outside J-space/J-Lens:

that is not just a curiosity.

It directly limits J-Lens as a forensic instrument.

But prove:

- information is causal;
- genuinely outside;
- not recoverable because of approximation error;
- not merely another layer/position issue.

---

# LXXXV. Lens-approximation gate

J-Lens is approximate.

Any “absence” claim must account for instrument sensitivity.

---

# LXXXVI. Random-Jacobian gate

Where applicable compare against:

- random linear projections;
- logit lens;
- PCA;
- linear regression;
- same-rank subspaces.

Otherwise how do we know J-space is special?

---

# LXXXVII. Baseline hierarchy gate

Every flagship experiment gets:

### trivial baseline

majority/random.

### behavioral baseline

model self-report/confidence.

### representation baseline

raw residual/PCA/random span.

### interpretability baseline

logit lens or comparable method.

Then J-space earns its place.

---

# LXXXVIII. Runtime gate

Speed is explicitly one of this project's constraints. Pasted text (2)

Every experiment should have:

- expected runtime;
- GPU memory;
- number of forward passes;
- cached reusable activations;
- batch policy;
- fallback batch halving on OOM.

But:

> speed may optimize waste, never validity.

---

# LXXXIX. T4 gate

A T4 pilot should:

- finish quickly;
- reveal whether hypothesis is alive;
- produce scientifically valid small-scale evidence.

No “cheap mode” that changes the question.

---

# XC. Early-stop gate

Before running expensive arms, define conditions under which the project stops.

Example:

If positive control fails, do not interpret negative provenance decoding.

This saves compute and avoids motivated interpretation.

---

# XCI. Run-order gate

Run experiments in order of **information value per minute**, not numbering.

Your own current design prioritizes the probe/faithfulness chain because it can determine the entire story before slower experiments. Pasted text

Keep that philosophy.

---

# XCII. Cache-validity gate

Cache keys must include every field that affects output.

Failed API calls must never be cached as successes.

---

# XCIII. Tokenization gate

Assert every assumption about:

- single-token answers;
- special tokens;
- prompt prefixes;
- multimodal placeholders;
- position indices.

Token mistakes silently invalidate interpretability work.

---

# XCIV. Shape gate

Every tensor transformation gets shape assertions during development.

No silent broadcasting.

---

# XCV. Index-alignment gate

Verify joins between:

- item tables;
- transcripts;
- activations;
- labels;
- questions.

One off-by-one can fabricate a result.

---

# XCVI. Layer-index gate

Clearly distinguish:

- block index;
- residual stream index;
- model layer numbering;
- lens layer numbering.

---

# XCVII. Precision gate

Check whether fp16 changes delicate Jacobian/projection results.

If it does, use higher precision where needed.

---

# XCVIII. Determinism gate

Save:

- seeds;
- package versions;
- model revision hashes;
- dataset revisions;
- prompts;
- filters;
- activation locations.

---

# XCIX. Version-control gate

Every result links to exact code commit/notebook version.

Your own project already suffered old/new-cell confusion. Pasted text

That must never happen in the final research record.

---

# C. No-hidden-manual-fix gate

Any manual intervention is documented.

No silently deleting ugly points.

---

# CI. Judge-validation gate

If an external model judge is used:

- freeze model/version;
- freeze rubric;
- hand-label;
- calculate agreement;
- inspect systematic disagreement;
- preserve raw outputs.

---

# CII. Human-label gate

The person doing hand validation should be blind to experimental condition where feasible.

---

# CIII. Figure truthfulness gate

Every plot answers one question.

No visual embellishment that hides uncertainty.

No y-axis manipulation.

No bold giant headline sitting above tiny \(N\).

---

# CIV. Figure legibility gate

A reader should understand:

- what is manipulated;
- what is measured;
- what result supports hypothesis;

within 10–15 seconds.

---

# CV. Hero-figure gate

There should be one figure that carries the central discovery.

Not five mediocre dashboards.

Possible structures:

**input routes → workspace decomposition → content/provenance readouts → behavioral source report**

or:

**causal hint → omitted CoT → internal cause retained**.

Whichever result actually lands.

---

# CVI. Raw-data visibility gate

Where possible show distributions/paired points, not only bars.

---

# CVII. Uncertainty-visualization gate

Headline plots show uncertainty.

---

# CVIII. Layer-heatmap gate

Only use heatmaps if layer structure itself matters.

No rainbow wallpaper.

---

# CIX. Color-semantics gate

Use consistent semantics across all figures.

Same condition = same visual encoding.

---

# CX. Caption gate

Captions state:

- question;
- manipulation;
- finding;
- relevant \(N\).

Not merely “Results of experiment 5.”

---

# CXI. Negative-result gate

A null must still answer something.

Do not hide failures.

They constrain theory.

---

# CXII. Surprise gate

If result exactly matches prediction:

great.

If opposite:

ask whether it reveals a more interesting architecture.

Do not force every result into the planned narrative.

---

# CXIII. Fishing gate

Do not keep adding experiments until one produces \(p<.05\).

Each experiment must have a question before results are read.

---

# CXIV. Retrospective-story gate

Separate:

**preregistered/predicted**

from:

**post hoc interpretation**.

Both are allowed.

Do not blur them.

---

# CXV. Replication gate

Headline result reproduced:

- across seed;
- across held-out sample;
- preferably across model or task.

---

# CXVI. Transfer gate

A field-defining mechanism should generalize farther than the exact stimulus set used to discover it.

---

# CXVII. Boundary gate

Equally important:

where does it **not** generalize?

Boundaries make the claim stronger.

---

# CXVIII. Model-specificity gate

If only Qwen3.5-4B shows it:

say so.

Do not turn one model into “language models.”

---

# CXIX. Multimodal-model gate

Any multimodal workspace claim requires a genuinely multimodal model.

Not a text-only simulation of provenance.

---

# CXX. Modality-count gate

Image/text is enough for a strong first multimodal result.

Do not add audio merely for breadth unless it answers a new question.

---

# CXXI. Source-monitoring literature gate

The paper should know:

- source memory;
- content/source dissociations;
- misattribution;
- suggestion effects;
- source similarity;
- confidence/source dissociation.

Classic source-monitoring theory directly motivates your manipulation. [PubMed](https://pubmed.ncbi.nlm.nih.gov/8346328/?utm_source=chatgpt.com)

Not decorative citation.

---

# CXXII. Cognitive-theory mapping gate

For every human cognitive construct, write:

> **human construct**

> **machine operationalization**

> **similarity**

> **difference**

No anthropomorphic hand waving.

---

# CXXIII. Workspace-theory gate

If testing the “amodal workspace” notion, explicitly locate that claim in workspace theory.

Do not invent a stronger theory than the literature supports.

---

# CXXIV. Access-vs-consciousness gate

Ned Block-style distinction matters:

functional access is not phenomenal experience.

Do not imply consciousness.

---

# CXXV. Cognitive-psychophysics gate

Stimuli should manipulate one cognitive variable at a time.

W5 should feel more like a good psychophysics experiment than a benchmark sweep.

---

# CXXVI. Minimal-pair gate

Prefer:

\[
\text{same content}
+
\text{different source}
\]

over uncontrolled datasets.

That is scientific finesse.

---

# CXXVII. Error-analysis gate

Inspect errors qualitatively.

Especially:

- source-report failures;
- probe/behavior disagreements;
- CoT omissions;
- cross-modal confusions.

Error structure may contain the discovery.

---

# CXXVIII. Model-self-knowledge gate

Distinguish:

> model knows fact

from:

> model knows it knows

from:

> model knows why it knows

from:

> model can report why it knows.

This hierarchy could be central.

---

# CXXIX. Cause-knowledge gate

The strongest introspection question is arguably:

> **Does the model know which information caused its answer?**

Your experiments should distinguish this from simple source recall.

---

# CXXX. Provenance-versus-causality gate

The source of information and the cause of the answer are not identical.

An image may supply a fact that the model ignores.

A note may exist but be redundant.

Therefore orthogonally manipulate:

\[
\text{source}
\times
\text{causal influence}.
\]

This is a very important possible extension.

---

# CXXXI. Internal-cause gate

If a representation claims “suggestion caused this,” then it should distinguish:

- suggestion present but ignored;
- suggestion present and causally decisive.

That is exactly why the redundant cell matters.

---

# CXXXII. Correct-cause / wrong-cause gate

The internal signal should track **being caused**, not normative correctness.

Your current five-cell logic explicitly aims to establish this. Pasted text

---

# CXXXIII. Confidence-independent cause gate

If a suggestion strongly changes answer despite high final confidence, the cause signal should still work.

Otherwise it may just be uncertainty.

---

# CXXXIV. Graded-causality gate

Where feasible sweep intervention strength and ask whether internal causal signal changes monotonically.

This is stronger than binary class discrimination.

---

# CXXXV. Wrong-hint gate

Use wrong hints as positive causal interventions.

But ensure they are not systematically more linguistically surprising.

---

# CXXXVI. CoT omission-vs-denial gate

Distinguish:

- omitted cause;
- explicitly denied cause;
- falsely claimed cause.

Those are different faithfulness failures.

---

# CXXXVII. Withholding-vs-confabulation gate

Important safety distinction:

**withholding:** internal cause exists but is omitted.

**confabulation:** reported cause was not internally responsible.

Your design should ideally discriminate both.

---

# CXXXVIII. Report-position gate

Why “report position”?

Because that is the moment the model must convert internal computation into explicit output.

State this theoretically.

---

# CXXXIX. Pre-report gate

Check earlier positions if claim concerns accumulation or access.

Avoid only examining the final token and calling it architecture.

---

# CXL. Dynamics gate

If the representation appears at a specific layer/time:

does it:

- emerge;
- persist;
- disappear?

Dynamic trajectory can distinguish workspace entry from passive encoding.

---

# CXLI. Workspace-entry gate

A particularly strong question:

> What determines whether causal information enters J-space?

That could connect difficulty, explicit reasoning, task demand, report requirements.

---

# CXLII. Silent-cause gate

Causal information that affects behavior but never enters J-space would be conceptually important.

But prove causal influence first.

---

# CXLIII. Reportability gate

Can forcing explicit reflection pull previously outside information into J-space?

That directly tests workspace accessibility.

---

# CXLIV. Reflection-intervention gate

Compare:

- ordinary answer;
- forced reflection;
- source-report demand.

Does provenance migrate into J-space when task-relevant?

That would be a powerful global-workspace prediction.

---

# CXLV. Task-demand gate

If provenance is absent when irrelevant but enters when source is queried, that is more interesting than simple “provenance lives outside.”

It suggests dynamic workspace routing.

---

# CXLVI. Attention-vs-workspace gate

Do not confuse attention with workspace entry.

Probe both if needed.

---

# CXLVII. Routing gate

If the project eventually identifies a gating/router mechanism controlling workspace entry, that is much stronger than content decoding alone.

---

# CXLVIII. Producer-consumer identity gate

Identify which modules/layers write to workspace and which read.

Even coarse evidence matters.

---

# CXLIX. J-space-specificity gate

Does the same phenomenon appear in:

- raw residual;
- PCA subspace;
- logit space?

If yes, J-space may not be special.

---

# CL. Scientific-claim gate

At final submission, write one sentence beginning:

> **We provide evidence that...**

It must contain no unsupported philosophical words.

---

# CLI. Mechanistic-claim gate

Write separately:

> **Our causal evidence supports...**

If you cannot, do not call it mechanistic.

---

# CLII. Cognitive-claim gate

Write separately:

> **This is analogous to human source monitoring because...**

Then list differences.

---

# CLIII. Safety-claim gate

Write separately:

> **This matters for monitoring because...**

If no practical demonstration exists, keep it as implication.

---

# CLIV. Novelty ledger gate

For every headline claim, maintain:

**claim**  
**closest prior work**  
**what they showed**  
**what you add**  
**why difference matters**

No novelty by ignorance.

---

# CLV. Literature-search gate

Search at least:

- J-Lens/J-space;
- global workspace in transformers;
- source monitoring LLMs;
- introspection LLMs;
- CoT faithfulness;
- hidden reasoning monitoring;
- activation probes;
- causal tracing;
- multimodal representation alignment;
- modality provenance;
- metacognition;
- confidence/calibration.

---

# CLVI. Anthropic-overlap gate

For each experiment ask:

> Would Anthropic's workspace paper already count this as evidence?

If yes, what new claim do you get?

No derivative replication disguised as novelty.

---

# CLVII. Neel-review gate

Neel's review explicitly identifies open issues and practical limitations of J-Lens. [LessWrong](https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S?utm_source=chatgpt.com)

Your work should ideally attack one of those unresolved boundaries.

---

# CLVIII. Introspection-overlap gate

Anthropic's introspection work already investigates whether models can access/report internal states. [Anthropic](https://www.anthropic.com/research/introspection?_bhlid=10d512478c7d831c58ebdda6008ed456d661b21e\&utm_source=chatgpt.com)

Your novelty should be **causal provenance/faithfulness**, not merely introspection again.

---

# CLIX. Faithfulness-overlap gate

Faithfulness literature already uses counterfactual interventions.

Your novelty must be:

- internal localization;
- workspace relation;
- source attribution;
- or stronger causal mechanism.

---

# CLX. Information-flow overlap gate

Recent work explicitly defines CoT faithfulness through mediated versus shortcut information flow. [arXiv](https://arxiv.org/abs/2605.24286?utm_source=chatgpt.com)

Do not reinvent:

> “faithful reasoning means information went through the CoT.”

Position against it.

---

# CLXI. Simulatability overlap gate

Recent work also evaluates whether explanations help predict future model behavior. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v306/mayne26a.html?utm_source=chatgpt.com)

Know that literature if making claims about explanation usefulness.

---

# CLXII. No-consciousness-bait gate

Do not use consciousness language merely to get attention.

Experts will punish it.

Workspace analogy is already interesting.

---

# CLXIII. Title gate

The title should name the discovery, not the apparatus.

Bad final-title pattern:

> “Probing J-Space in Qwen.”

Better pattern depends on result:

> source, cause, provenance, workspace, faithful introspection.

Do not decide final title before result.

---

# CLXIV. Abstract gate

Abstract contains:

1. problem;
2. precise gap;
3. experiment/manipulation;
4. flagship result;
5. mechanism;
6. consequence.

No literature dump.

No adjectives like “groundbreaking.”

---

# CLXV. Introduction gate

Page 1 should make the reader ask:

> **Can a model know a fact without knowing where it came from—or know why it answered without being able to say so?**

Something this clean.

Then explain why J-space makes that question testable.

---

# CLXVI. No-boilerplate gate

Delete:

> “LLMs have demonstrated remarkable capabilities…”

Immediately.

---

# CLXVII. One-story gate

The paper should have one narrative.

Potentially:

\[
\text{cause}
\rightarrow
\text{internal availability}
\rightarrow
\text{report}
\]

with source/provenance as the key missing binding.

Experiments should form chapters of that story.

---

# CLXVIII. Experiment-necessity gate

Every experiment answers a question the previous experiment created.

No “also we tried…”

---

# CLXIX. Flagship-result gate

One result must carry the paper.

The rest:

- establishes controls;
- mechanism;
- generalization;
- consequence.

If you cannot point to **the** result, story is diffuse.

---

# CLXX. Three-results gate

Your current notes envision three potentially standalone results:

- altered-record noticing/internal gap;
- self-report wrong sign vs internal causal probe;
- CoT omits cause while residual retains it;

plus the potentially deeper span-localization result. Pasted text

The final paper does **not** need all to succeed.

It needs one deep result and enough supporting evidence.

Do not dilute the best result to preserve old experiments.

---

# CLXXI. Kill-your-darlings gate

If experiment is cool but unrelated to headline:

appendix or separate paper.

---

# CLXXII. First-principles gate

Before every experiment ask:

> If I knew nothing about current tooling, what would be the cleanest way to answer the scientific question?

Then use J-Lens only if it helps.

---

# CLXXIII. Simplicity gate

Best experiment:

one manipulation;

one theoretically diagnostic contrast;

one obvious control;

one plot.

Complexity is tolerated only when necessary.

---

# CLXXIV. Five-minute-explanation gate

Could Neel understand the whole experiment in five minutes without reading code?

If not, simplify.

---

# CLXXV. Five-second-figure gate

Could he understand the figure in five seconds after reading caption?

If not, redesign.

---

# CLXXVI. Finesse gate

Finesse means:

- minimal manipulation;
- orthogonal controls;
- paired comparisons;
- no unnecessary judge;
- no fragile heuristics;
- no giant benchmark sweep;
- clear alternative outcomes.

Not “many experiments.”

---

# CLXXVII. Speed-without-sacrifice gate

The project explicitly demands blazing-fast T4 experiments without sacrificing scientific validity. Pasted text (2)

The acceptable optimization hierarchy:

1. cache activations;
2. batch intelligently;
3. reduce redundant forward passes;
4. paired designs;
5. diagnostic subsamples for pilot;
6. vectorize;
7. only then reduce \(N\).

Never delete controls to save time.

---

# CLXXVIII. Pilot/final separation gate

Fast pilot can answer:

> worth continuing?

Final experiment must answer:

> worth publishing?

Do not confuse them.

---

# CLXXIX. Compute-budget gate

Every proposed experiment should state:

- expected T4 minutes;
- expected memory;
- forward passes;
- number of examples;
- rerun cost.

This forces good experimental design.

---

# CLXXX. Reproducibility gate

A strong outsider can rerun flagship experiment from:

- one environment file;
- one command/notebook;
- documented model;
- documented dataset;
- fixed seeds.

---

# CLXXXI. Auditability gate

Save intermediate artifacts so a reviewer can inspect:

- item selection;
- labels;
- model answers;
- manipulations;
- probe predictions;
- judge outputs.

---

# CLXXXII. Figure-source gate

Every plot can be traced to one machine-readable results table.

No manual Excel edits.

---

# CLXXXIII. External-replication gate

Before “field-defining” language, get at least one competent outsider to reproduce the main effect if feasible.

---

# CLXXXIV. Code-review gate

Before final runs:

- static read;
- smoke test;
- shape assertions;
- tiny synthetic known-answer test;
- cache test;
- fold-leakage test;
- seed repeat.

---

# CLXXXV. Deliberate-bug gate

Inject a known label shuffle or impossible condition.

Pipeline should fail.

If analysis still finds an effect, pipeline is broken.

---

# CLXXXVI. Leakage test gate

Explicitly test whether probe can infer:

- question ID;
- answer letter;
- hint type;
- batch;
- transcript length.

Unexpected leakage requires investigation.

---

# CLXXXVII. Synthetic-ground-truth gate

Where possible create a toy condition where the location/cause of information is known exactly.

Validate apparatus there before interpreting ambiguous natural data.

---

# CLXXXVIII. Null-control gate

Include a condition where theory predicts **no** effect.

A method that “finds something” everywhere is worthless.

---

# CLXXXIX. Positive-control gate

Include a condition where theory predicts a clear effect.

A null flagship result with failed positive control is uninterpretable.

---

# CXC. Adversarial-control gate

Design at least one control specifically to fool your own metric while leaving the target mechanism absent.

If metric gets fooled, revise it.

---

# CXCI. Reverse-prediction gate

For every key result state:

> what exact pattern would support the opposite architecture?

Example W5:

Inside provenance decoding **high** with semantic alignment preserved:

> provenance-tagged workspace.

Inside provenance **equivalent to chance** with strong power controls and outside provenance high:

> provenance stripped from workspace.

Both scientifically informative.

---

# CXCII. Failure-value gate

A great experiment should be valuable even if hypothesis is wrong.

If a null gives you nothing, design is weak.

---

# CXCIII. Bayesian-surprise gate

Ask:

> How much would a knowledgeable expert update from this result?

Not:

> Is \(p<.05\)?

A field-changing result creates a large posterior shift.

---

# CXCIV. “Why didn't we think of this?” gate

This phrase only applies if the experiment is:

- obvious in hindsight;
- not obvious before seeing it;
- simple;
- theoretically motivated;
- technically clean;
- revealing something previously hidden.

Complexity usually reduces this effect.

---

# CXCV. Job-talk gate

Imagine one slide in an Anthropic/DeepMind research interview.

Could you say:

> “Here's the question. Here's the intervention. Here's the control. Here's the result. Here's why it changes what we think J-space is.”

If yes, good.

---

# CXCVI. Safety-relevance gate

If you claim monitoring significance, connect directly to:

- hidden causal influence;
- deceptive explanations;
- prompt injection;
- eval awareness;
- unfaithful CoT;
- source confusion.

No generic:

> “This improves AI safety.”

---

# CXCVII. Practicality gate

Could the method reasonably run in an audit?

If it needs 1,000 GPU-hours per prompt, practical claims fail.

---

# CXCVIII. Instrument-cost gate

Compare cost to simply:

- asking model;
- inspecting CoT;
- confidence;
- logit lens.

If J-Lens is 100× costlier but only marginally better, say so.

---

# CXCIX. Epistemic-humility gate

Never say:

> “we read the model's thoughts.”

Say exactly what the instrument measures.

---

# CC. Anthropomorphism gate

Words like:

- knows;
- believes;
- notices;
- remembers;
- introspects;

must either be operationalized or clearly treated as shorthand.

---

# CCI. Internal/external vocabulary gate

Use:

> representation predicts source

before:

> model remembers source.

Precision first.

---

# CCII. Cognitive-science payoff gate

Tenenbaum-type reader should gain something too.

Not just:

> human theory inspired our benchmark.

Potential payoff:

> source attribution and content availability factorize internally;

or:

> global availability does not imply provenance availability.

That is a computational cognitive claim.

---

# CCIII. Mechanistic-interpretability payoff gate

Neel/Olah reader should gain:

> a new internal factorization or causal pathway.

Not merely behavior.

---

# CCIV. Faithfulness payoff gate

Siegel/Camburu reader should gain:

> an internal predictor of causal faithfulness unavailable from text alone.

---

# CCV. Workspace-theory payoff gate

Lindsey/Gurnee/Dehaene reader should gain:

> a newly tested property or limitation of workspace representations.

---

# CCVI. Multimodal payoff gate

Multimodal researcher should gain:

> evidence about whether semantic content becomes modality-agnostic while source remains elsewhere.

---

# CCVII. Literature-respect gate

Do not imply you invented:

- source monitoring;
- unfaithful CoT;
- global workspace;
- introspection;
- linear probing;
- activation patching.

Own the intersection.

---

# CCVIII. Citation-role gate

Each central citation has one job:

**foundation**

**method**

**closest neighbor**

**alternative explanation**

**cognitive analogue**

No citation piles.

---

# CCIX. Primary-source gate

Use original papers for technical claims.

The LessWrong review is excellent for critique/interpretation, not as the source for Anthropic's experiments.

---

# CCX. Currentness gate

This is a fast-moving field.

Rerun novelty search immediately before posting preprint.

A paper six months old can invalidate a novelty claim.

---

# CCXI. Open-code gate

If possible release code.

Interpretability credibility increases enormously with inspectability.

---

# CCXII. Raw-results gate

Release non-sensitive processed result tables.

---

# CCXIII. Ablation gate

Every complex pipeline component gets an ablation or reason it is indispensable.

---

# CCXIV. Robustness gate

Headline result survives reasonable:

- layer choices;
- probe regularization;
- sample subsets;
- seeds.

Not every conceivable parameter.

Just enough to show it isn't brittle.

---

# CCXV. Sensitivity gate

Show when it breaks.

A method with known boundaries is more credible.

---

# CCXVI. Outlier gate

Inspect whether result is driven by a few extreme questions.

---

# CCXVII. Class-balance gate

Do not let AUROC hide tiny positive populations.

Report class sizes.

---

# CCXVIII. Prevalence gate

If causal-hint cases are rare, say so.

Effect strength and prevalence are different.

---

# CCXIX. Behavioral-validity gate

Internal effect should connect back to actual model behavior.

Otherwise it risks being representation archaeology.

---

# CCXX. Ecological-validity gate

At least one task should resemble a real failure mode:

- misleading hint;
- prompt injection;
- explanation omission;
- source confusion.

---

# CCXXI. Toy-to-real gate

Use toy designs to identify mechanism.

Use natural tasks to show relevance.

---

# CCXXII. Confound ledger gate

Maintain one internal table:

| Experiment | headline interpretation | strongest confound | control | status |

No experiment is GREEN until every serious confound has a disposition.

---

# CCXXIII. Claim ledger gate

For every manuscript claim:

- exact sentence;
- type: behavioral / representational / causal / cognitive / safety;
- evidence;
- confidence;
- scope;
- closest prior work;
- falsifier.

---

# CCXXIV. Result ledger gate

For every result:

- hypothesis;
- preregistered metric;
- outcome;
- effect;
- uncertainty;
- unexpected observations;
- status.

---

# CCXXV. Run ledger gate

Record:

- model;
- commit;
- GPU;
- seed;
- runtime;
- dataset hash;
- date;
- output path.

---

# CCXXVI. Version-integrity gate

No confusion between old and new notebook cells.

Your prior workflow already encountered exactly that risk, including explicit checks for whether the old suite was still executing. Pasted text

Before every major run:

print version hash.

---

# CCXXVII. Manuscript-code consistency gate

Every methodological statement in paper matches actual code.

No stale paragraph describing Experiment 5 v1 while code runs v3.

---

# CCXXVIII. Plot-code consistency gate

Every caption matches plotted rows/filters.

---

# CCXXIX. No-handwave gate

Phrases that require immediate justification:

> “the workspace knows”

> “the model noticed”

> “the cause lives here”

> “this is introspection”

> “this proves amodality”

> “this is global”

> “this is faithful.”

Each needs an operational meaning.

---

# CCXXX. Prose gate

Mechanistic papers benefit from plain prose.

One idea per sentence where possible.

No hype.

No:

> groundbreaking

> revolutionary

> unprecedented

in manuscript body.

Let the result do it.

---

# CCXXXI. Diagram gate

The core experimental design should fit into a tiny diagram.

For W5:

**same concept**

↙︎ image  text ↘︎

**same answer**

↓

**J-space / complement**

↓

content? source?

↓

**source report**

That is good scientific compression.

---

# CCXXXII. No-dashboard gate

A figure with 14 tiny panels usually means the scientific question is unclear.

Primary paper:

few strong panels.

Appendix:

everything else.

---

# CCXXXIII. Main-text gate

Only results needed for central inference belong in main text.

---

# CCXXXIV. Appendix gate

Move:

- extra layer sweeps;
- secondary tasks;
- extra baselines;
- extensive hyperparameters.

Do not let them bury the aha.

---

# CCXXXV. Abstract-result gate

Every quantitative claim in abstract must have:

- full control;
- replication;
- interval.

No pilot headline.

---

# CCXXXVI. Title-result gate

Final title cannot depend on a result still marked exploratory.

---

# CCXXXVII. One-month-memory gate

A reader should remember one thing:

Potentially:

> **the workspace knows what, but not necessarily where from.**

or:

> **the model's explanation can lose the cause before the internal state does.**

Whichever your data earns.

---

# CCXXXVIII. Quote gate

You probably need **one** memorable sentence.

Not fifteen.

Mechanistic work is memorable because of the experiment.

---

# CCXXXIX. External-red-team gate

Before release, get:

- one mech-interp expert;
- one cognitive scientist;
- one faithfulness/metacognition researcher;
- one multimodal researcher;

to try to kill the interpretation.

---

# CCXL. Blind-review gate

Give them:

- figure;
- caption;
- methods summary;

without telling them your preferred interpretation.

Ask:

> what could generate this?

Their answers become your final controls.

---

# CCXLI. Replication-before-hype gate

Do not tweet the most exciting result before checking:

- seed;
- indexing;
- leakage;
- N;
- confounds;
- rerun.

---

# CCXLII. Career-motivation firewall

You explicitly want this to get Neel's attention and help your career. That's fine as motivation. Your files say so plainly. Pasted text

But experimental decisions must pass:

> **Would I make the same decision if my name were removed from the paper?**

If not, stop.

---

# CCXLIII. Virality firewall

A “viral” effect that doesn't survive review hurts more than a smaller clean discovery.

Novelty ≠ sensationalism.

---

# CCXLIV. Negative-publication gate

If the flagship hypothesis fails cleanly but reveals:

> J-space actually preserves provenance,

that could be a paper.

Design for truth, not desired direction.

---

# CCXLV. “Why didn't Anthropic do this?” gate

For every proposed experiment, ask:

> Why wasn't this already obvious to the workspace authors?

Possible acceptable answers:

- modality was unavailable;
- their instrument could not test provenance;
- their scientific question differed;
- it required a new control decomposition;
- source monitoring was outside their framing.

If answer is:

> “they probably just didn't try,”

novelty is weak.

---

# CCXLVI. “Would this survive Jack Lindsey?” gate

He should not be able to say:

> “J-space was only claimed to be verbalizable, not provenance-free; your critique attacks a claim we never made.”

Your paper must represent prior claims exactly.

---

# CCXLVII. “Would this survive Neel?” gate

He should not immediately say:

> “Your probe is reading correctness.”

> “This is just difficulty.”

> “Your random span has different dimension.”

> “You chose the layer on the test set.”

> “Decodability isn't causality.”

Those objections should already be handled.

---

# CCXLVIII. “Would this survive Tenenbaum?” gate

He should not say:

> “You borrowed a cognitive term but didn't give me a computational model.”

Make the representation/function connection explicit.

---

# CCXLIX. “Would this survive Siegel?” gate

He should not say:

> “Your explanation-faithfulness measure is actually confidence or verbosity.”

Controls.

---

# CCL. “Would this survive source-memory researchers?” gate

They should recognize a genuine analogue of:

\[
\text{item information}
\neq
\text{source information}.
\]

Not superficial borrowed vocabulary.

---

# CCLI. “Would this survive Hinton?” gate

He should not say:

> “Of course source is linearly decodable; transformers preserve all sorts of irrelevant information.”

Show functional specialization.

---

# CCLII. “Would this survive Karpathy?” gate

He should be able to restate the idea simply.

If he needs your custom jargon, not ready.

---

# CCLIII. 2036 gate

Delete:

- Qwen;
- J-Lens;
- Anthropic;
- current CoT conventions.

Is the finding still interesting?

For example:

> global accessibility can preserve semantic content while losing causal provenance.

That could last.

“Probe X gets .72 AUROC on Qwen” cannot.

---

# CCLIV. Tool-independence gate

If J-Lens is replaced tomorrow by a better interpretability instrument, does the scientific question survive?

It should.

---

# CCLV. Cognitive-durability gate

Does the paper expose a general distinction such as:

\[
\text{content access}
\neq
\text{source access}
\]

or:

\[
\text{internal cause}
\neq
\text{verbal explanation}?
\]

That is more durable than a method benchmark.

---

# CCLVI. Safety-durability gate

Would the result remain relevant for future systems whose visible CoTs are hidden?

If internal source/cause monitoring matters independently of exposed CoT, yes.

---

# CCLVII. Simplicity-after-complexity gate

The paper can contain sophisticated experiments.

But after reading, the reader's model should become simpler.

Not more cluttered.

---

# CCLVIII. Research-program gate

The work should generate obvious next questions:

- what enters workspace?
- when does provenance enter?
- how does task demand route it?
- can source information be induced into workspace?
- does provenance loss cause introspection failures?
- does post-training alter source binding?
- do multimodal systems differ from text-only models?
- can workspace-cause monitoring improve audits?

That is how a paper becomes fertile.

---

# CCLIX. Final-comprehension gate

An expert should be able to say:

> “They separated causal influence from correctness, found where that information is represented relative to J-space, and showed whether the model's verbal explanation has access to it.”

Or equivalent based on eventual results.

If summaries vary wildly, paper is diffuse.

---

# CCLX. Final contribution gate

Exactly **three contributions** maximum.

Example shape:

1. **scientific discovery**;
2. **mechanistic localization/explanation**;
3. **consequence for faithfulness/workspace theory**.

Not eight experiments as contributions.

---

# CCLXI. Final figure gate

A person should be able to learn the paper from:

**Figure 1:** phenomenon / manipulation.

**Figure 2:** internal localization / mechanism.

**Figure 3:** consequence/generalization.

Three is probably enough.

---

# CCLXII. Final release decision

The paper is **NOT HIGH QUALITY** if any of these remains true:

- the flagship probe can read correctness or difficulty;
- negative decoding lacks a positive-control/power argument;
- decodability is called causal;
- causal information is claimed without manipulation;
- J-space absence claims ignore lens approximation;
- multimodal source conditions differ in difficulty/content;
- the workspace claim does not specify which GWT property is tested;
- source monitoring is used merely as metaphor;
- CoT faithfulness is judged only by plausibility;
- self-report/confidence baselines are missing;
- question variants leak across folds;
- random-span/dimensionality controls are missing;
- layer selection uses test performance;
- null results are interpreted from \(p>.05\);
- judge outputs are unvalidated;
- model-general claims come from one 4B model;
- pilot \(N\) is treated as final evidence;
- code/version ambiguity remains;
- figures obscure sample size or uncertainty;
- an expert can explain result as answer identity, lexical form, difficulty, confidence, norm, or position;
- the central experiment takes hours because implementation is wasteful rather than scientifically necessary;
- the paper is a collection of neat experiments without one hypothesis;
- novelty is merely “J-Lens on X”;
- the headline cannot survive without words like consciousness or thoughts;
- the safety implication is generic;
- the key result cannot be explained on one whiteboard;
- a negative outcome would make the experiment worthless;
- the result is impressive only because nobody saw the controls.

And for the **magnum-opus / field-defining candidate** bar, all of the above must be green **plus**:

1. There is one clean result that changes the conceptual model of J-space/workspace or faithful self-report.
2. At least two independent methodological routes converge on it.
3. The result survives an independent task or model.
4. The interpretation survives specialist red-teaming.
5. The hero figure is immediately legible.
6. The scientific question survives replacement of J-Lens itself.
7. The paper links mechanistic evidence to a genuine cognitive distinction rather than borrowed terminology.
8. The result admits an obvious falsifier and survived it.
9. The entire story can be told in under two minutes.
10. An expert's response is plausibly:

> **“Of course we should have tested that.”**

## The master question

For this project, the ultimate rubric is:

> **Can we identify a representation that tracks what actually caused the model's decision, distinguish that from what the model says caused it, locate that information relative to the workspace, and rule out every simpler reason the signal could be there?**

If the answer is **yes**, you have something substantial.

If W5 then additionally shows that **semantic content becomes workspace-accessible while its provenance remains outside—or, just as interestingly, that provenance is explicitly tagged inside—**you have a result that genuinely touches mechanistic interpretability, source monitoring, global-workspace theory, multimodal cognition, and explanation faithfulness at once.

That is the bar I would use on every new experiment, every figure, and every claim in this project from now on.