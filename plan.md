# plan.md: which layers steer a video, and a sound?

A toy check, 5 videos and 5 sounds, before any real experiment. Nothing has been run.

- **Version 1**, 15 Nov 2026: the first plan (commit `504e7bb` holds it as written).
- **Version 2**, 15 Nov 2026: this file. The council pass at the bottom found a fatal flaw in version 1 and six gaps; the design and the rule below are the corrected ones.

Built on two commits: `88b0b98` (the Karpathy guidelines: state assumptions, simplest thing, surgical changes, a check per step) and `bb38cc3` (the master rubric, applied in full in the council pass).

## The question

On an image, adding the J-lens direction of the word "bird" to an airplane's image tokens makes the model say "bird". Does the same work on a clip of an airplane taking off, and on the sound of an airplane, and from which layers?

## What the image runs already say

These numbers set the design. Version 1 ignored the first one.

| image result (Qwen3.5-4B, 64 scenes) | number | source |
|---|---|---|
| steering one 4-layer window, any depth, 4 sigma | says Y on 0.00 | `results/depth_sweep`, `inject` curve; same on Gemma-31B |
| steering every image token, layers -1 to 10 | says Y on 1.00; animal rule follows on 0.23 | RECORD row 93 |
| steering every image token, layers 11 to 22 | says Y on 1.00; animal rule follows on 0.03 | RECORD row 93 |
| random direction, same dose | 0.00 to 0.03 | RECORD row 93 |
| steering only the object's 49 slot tokens | never says Y at any dose | `results/name_steering`, R4 |
| steering the answer position | says Y on 0.98, rules 0.05; plain unembedding direction also says Y on 0.80 | REPORT, "Steering the picture with a name" |

So steering works over a block of layers on every media token, and one window does nothing.

## Short answers

**Are 3 Qwen and 3 Gemma enough?** Six models is more than a toy needs; the limit is the 5 clips. But the grid cannot be filled for sound:

| | video | sound | J-lens on Neuronpedia |
|---|---|---|---|
| Qwen3.5 (any size) | yes | **no audio input** | 0.8B, 2B, 4B, 27B (9B: J++ only) |
| Gemma-4 E2B | yes (frames) | yes | yes |
| Gemma-4 E4B | yes (frames) | yes | yes |
| Gemma-4 12B | yes | yes | **no lens** |
| Gemma-4 31B | yes (frames) | **no audio input** | yes |

So: video on six models, sound on two.

**Are 5 clips enough?** Enough to see whether naming follows the steering, from which block of layers, and whether the J-lens direction is needed. Not enough for the rule question: the image effect there is 0.23 against 0.03, which is one clip in five. Not enough for any interval. The toy is a go or no-go for a real run of 64 clips.

## Models

| role | model | layers | download | note |
|---|---|---|---|---|
| Qwen small | `Qwen/Qwen3.5-0.8B` | 24 | 2 GB | pinned; cannot answer the digit rules, so naming only |
| Qwen medium | `Qwen/Qwen3.5-4B` | 32 | 9 GB | pinned; the image numbers above are its own |
| Qwen large | `Qwen/Qwen3.5-27B` | 64 | 54 GB | pinned; its patching runs crashed in the linear-attention kernel (RECORD row 67), steering hooks untried. Fallback: `Qwen3.5-9B`, J++ lens only |
| Gemma small | `google/gemma-4-E2B-it` | 35 | 10 GB | **not pinned yet**: add its revision and lens sha256 to `config.py` |
| Gemma medium | `google/gemma-4-E4B-it` | 42 | 16 GB | pinned |
| Gemma large | `google/gemma-4-31B-it` | 60 | 63 GB | pinned; video only; needs a GPU of about 90 GB |

Start with the two mediums (4B and E4B). Add the others only if the mediums pass their gates.

## Data

One source for both modalities: **VGGSound** (CC BY 4.0, 10-second YouTube clips where the labelled object is both visible and audible). The same five clips give the video arm (frames, sound dropped) and the sound arm (audio track, no frames).

- Five classes, one clip each: airplane, bird, dog, car, train. Y is of the other kind (airplane to bird, bird to airplane, dog to car, car to dog, train to bird). Z is of the same kind (airplane to train, bird to dog, and so on).
- Each name must be one token in every model's tokenizer (asserted, as `stimuli.load_crops` does).
- Draw 4 candidate clips per class with a fixed seed; keep the first that every model in the run names correctly when clean. Save the candidates and which were dropped.
- Source: `Loie/VGGSound` on Hugging Face is 20 shards of about 16 GB; one shard is enough. A small mirror (`harpreetsahota/VGGSound-1000`) may do and is not yet checked.
- Sound fallback: ESC-50 (5-second clean clips; CC BY-NC).

ImageNet VID (92 GB, boxes, no sound) stays the choice for a real video run.

## Design (version 2)

One new script, `experiments/media_steer.py`, run name `toy_media_steer`.

**Inputs**, four per clip:

| input | why |
|---|---|
| the clip, 8 frames at 224 x 224 | the question |
| a static video: the middle frame 8 times | same token format, no motion |
| the middle frame as an image | ties the toy to the image numbers above |
| the audio track (Gemma E2B and E4B only) | the question |

**Questions**, in separate prompts: say ("What is the object in the video? Reply with the noun only.") and the animal rule ("if the object in the video is an animal, answer 7; otherwise answer 3"). Sound: "What is making this sound?", placed before the audio, as Gemma's card asks.

**Where the direction is added**: every media token (the main site), and the last prompt token (the control for an output push).

**Which layers**, as shares of depth so the models compare:

| layer set | layers on the 4B | purpose |
|---|---|---|
| early block | -1 to 10 (up to 0.33) | row 93's early steer |
| middle block | 11 to 22 (0.33 to 0.70) | row 93's late steer |
| late block | 23 to 31 (after 0.70) | after naming stops reading the tokens |
| prefix sweep | -1 to L, L in steps of 4 | the smallest stretch from the input that steers |
| suffix sweep | L to the last layer, L in steps of 4 | the latest start that still steers |

**Directions**, each of unit length at dose 4 sigma_L: Y's J-lens direction; Z's (same kind, so the rule must not flip); a random direction; Y's plain unembedding direction. The three blocks are also run at doses 2 and 8, all three reported.

Before any steering the script prints, per layer, the J-lens and logit-lens rank of the clip's own name on its media tokens, with another object's name as the null.

Token counts to assert: 196 video tokens on Qwen3.5 (one token covers 32 x 32 pixels of two frames), 8 x 70 = 560 on Gemma-4, 250 audio tokens for a 10-second sound (40 ms each).

## Predictions, written before the run

From the image numbers, for every model and modality that passes its gate:

1. Naming follows Y from the early block and from the middle block on 5 of 5 clips.
2. Naming does not follow Y from the late block (at most 1 of 5).
3. A random direction and the plain unembedding direction on the media tokens move at most 1 of 5.
4. The clip, the static video and the middle frame give the same blocks.
5. Sound gives the same blocks as the middle frame on the same model.

Prediction 5 is the one most likely to fail, and the most informative if it does.

## The rule, written before the run

- **Gate.** A question on a model and modality is read only if 5 of 5 clean clips are answered correctly. Otherwise the cell reads "not perceived, untestable". It never reads "no pathway" (rubric E2).
- **A layer set steers naming** when Y is said on at least 4 of 5 clips and the random direction gives Y on at most 1 of 5.
- **J-specific** when the plain unembedding direction on the media tokens gives Y on at most 1 of 5.
- **Not told apart from an output push** when the media-token steer passes only where the same steer at the last prompt token also passes and the unembedding direction passes too.
- **Nothing temporal tested** when the clip, the static video and the middle frame pass in the same layer sets.
- **Same as the image** when the prefix and suffix edges are within one step (4 layers) of the middle frame's on the same model.
- **The animal rule is reported as counts with no verdict.** Five clips cannot read it.
- No intervals, and no claim across sizes or families.

## Steps

1. Environment: `uv sync`, `uv run pytest -q` on a GPU machine (this cloud session has none). Verify: tests pass.
2. Pin Gemma E2B in `config.py`; add the clips to `scripts/download.py` with sha256. Verify: `download.py` passes its hash checks.
3. Input smoke, no steering: each model encodes one clip and one sound; print the token counts, the answer and the lens ranks. Verify: the counts above, asserted media-token positions, the 27B loads.
4. Write `media_steer.py` with the docstring carrying this design, the predictions and the rule. Video and audio encoding live in the script, so `Session` is untouched. If `Session.encode` must change, run `depth_sweep --n 4` before and after and compare with `check.py`. Verify: `ruff`, tests, matching folders.
5. The 4B, middle frame only. Verify: it reproduces row 93 (early and middle blocks say Y, random does not). If it does not, stop: the script is wrong.
6. The 4B and E4B, clip and static video. Verify: gate, then read the CSV row by row.
7. Sound on E4B, then E2B. Verify: gate first; stop there if it fails.
8. Only if steps 6 and 7 pass: the small and large models.
9. One row per model and modality in `docs/RECORD.md`, with each prediction marked kept or missed.

Cost: about 2,000 short generations per model. Minutes per model; the downloads dominate.

## RunPod: time and cost

Estimates made on 15 Nov 2026, before any run. The prices are from RunPod's pricing page and third-party trackers, which disagree by tier and date; check the live page before renting.

| card | models that fit | pod time | price per hour | compute cost |
|---|---|---|---|---|
| RTX 6000 Ada, 48 GB | Qwen 0.8B and 4B, Gemma E2B and E4B | 3 to 5 hours | about $1 ($0.84 to $1.09 listed) | $3 to $5 |
| RTX PRO 6000, 96 GB | all six | 5 to 7 hours | about $2.49 ($1.69 to $2.49 listed) | $12 to $17 |

- The 48 GB card cannot hold Qwen3.5-27B (54 GB of weights) or Gemma-4-31B (up to 88 GB in the original runs). The original runs used the 96 GB card.
- Storage is extra: about 55 GB for four models and one VGGSound shard, about 170 GB for all six. A few dollars for a month of network volume.
- Where the time goes: downloads (20 to 60 minutes), then writing and debugging the video and audio encoding on the pod (most of it). The steering itself is a few minutes per model: about 2,000 short generations, against the 21,000 forwards in 15 minutes the notes record for the 4B on this card type.
- Because compute is the small part, 64 clips cost nearly the same as 5: about ten times the generations, still under an hour per model.

## Does running it raise the score?

- **Not the proposal score.** The 25.5 judges the design. It rises with a prior-art audit (item 02), an accepted thesis (01), a figure design (29) and an outside reviewer (G10), none of which needs a GPU.
- **Running starts the evidence score (DE)**, which is separate and currently not assessed.
- **Five clips earn almost nothing there.** Item 20 scores a run without intervals at zero, and the rivals cannot be separated on the rule question.
- **Recommendation: run 64 clips, not 5.** Same pod, nearly the same cost. That is what can lift statistics (20), rival discrimination (07), frozen prediction (22) and, if the predictions hold, genuine audio and video (24). The 5-clip toy then becomes step 5 and step 6's first look, not the result.

## Known risks

- **Sound may fail its gate.** The one earlier audio attempt (RECORD row 6, Gemma E4B, ESC-50) did not perceive the sounds (hits 0.88, false alarms 0.41), and Gemma-4's card lists audio for speech recognition and translation only. Fallback: the object's name spoken aloud, which tests a spoken word, not a sound, and must be labelled so.
- **The earlier video attempt failed its control** (RECORD rows 65 and 68: a still canvas repeated four times, patching). This toy uses real motion and steering, so that failure neither predicts nor excuses this one.
- **The lens was fitted on text.** It reads image tokens; audio tokens are unchecked, hence the rank printout.
- **Steering every media token is a blunt intervention.** It cannot say which object in a clip was changed. One object per clip hides that; a real run needs boxes.
- **Outputs stay in `runs/`.** `results/` is never edited.

## Not in the toy

Patching a donor clip into a receiver, ablation, several objects per clip, any statistics. Each is the next experiment if the toy says go, with its own rule written first.

---

# The council pass (master rubric v2.0)

Scored 15 Nov 2026. Stage: proposal quality (PQ). Discovery evidence (DE) is not assessed, since nothing has run.

**What this pass is.** One reviewer, the assistant that wrote the plan, applying the rubric's lenses one after another. The named researchers are lines of scrutiny taken from the rubric; none of them reviewed, voted or endorsed anything. The rubric does not count a model-written review as independent verification, so gate G10 is RED by construction and no score here is a clearance.

## 1. Thesis and status

Version 1 had a question and no thesis, so it failed Stage 0 admission. This pass proposes one, to be accepted or replaced by the authors:

> A word's lens direction written on an input's tokens is said when it is written anywhere before the model stops reading those tokens for the name, and this window sits at the same share of depth for a photo, a clip and a sound.

- Scope: Qwen3.5 and Gemma-4, single-object inputs, one-token names.
- What kills it: a perceived modality where naming does not follow from the early or middle block, or follows from the late block.
- What survives a kill: "the window is set by the token type", which is a result about modality.

**PQ: version 1, 13 / 100. Version 2, 25.5 / 100. DE: not assessed.**

## 2. The thirty items

Each item gets 0, half or all of its weight. "v1" is the first plan, "v2" this file.

| ID | weight | v1 | v2 | evidence, and the strongest objection left | cheapest test that kills it |
|---:|---:|---:|---:|---|---|
| 01 revelation | 5 | 0 | 2.5 | v1 had no proposition. v2 has one with a falsifier; it extends the image result and may read as a minor extension | an expert predicts the outcome unaided |
| 02 originality | 8 | 0 | 0 | no prior-art matrix. The ledger lists the close work (below) but no paper was read for this pass | one earlier J-lens or steering result on audio tokens |
| 03 epiphany | 6 | 0 | 3 | v2 names three rivals with different forecasts; none is yet a forecast a skeptic made blind | all rivals predict the same cells |
| 04 field-changing | 10 | 0 | 0 | a go or no-go toy changes no belief | none |
| 05 elegance, durability | 6 | 0 | 0 | nothing here survives without the model names | none |
| 06 mathematical exactness | 4 | 2 | 2 | the operator, the dose unit sigma_L, the layer sets and the pass rule are defined; no uncertainty at n = 5 | a second draw of 5 clips changes a verdict |
| 07 decisive discriminations | 4 | 0 | 2 | v2 has the output-push arm and the static-video arm; n = 5 separates only all-or-none effects | the answer-position steer matches the media-token steer everywhere |
| 08 natural writer, state, reader | 5 | 0 | 0 | steering forces a state the model never writes; no natural path is traced | not addressed |
| 09 necessity, sufficiency | 3 | 0 | 0 | sufficiency only; no ablation, no bypass measured | not addressed |
| 10 content, task, answer split | 3 | 0 | 1.5 | v1 asked only for the name, which is the steering word itself. v2 adds the digit rule and Z; not a factorial | the rule never moves while the name always does |
| 11 lens validity | 1 | 0.5 | 0.5 | the published lens, with random and unembedding controls; no PCA or complement arm; the 9B fallback is J++ | the logit lens reads audio tokens as well as the J-lens |
| 12 construct validity | 2 | 1 | 1 | the claim is about naming, with a perception gate; "see" in the question is looser than what is measured | the gate passes on the audio's file-level cues, not the sound |
| 13 computational explanation | 3 | 0 | 0 | no model of why a window would be shared | none |
| 14 dissociations | 2 | 0 | 1 | v2 has say against rule, readable only as counts | none at n = 5 |
| 15 source monitoring | 2 | 0 | 0 | not claimed; no substitute test was declared, so the weight is lost, not waived | none |
| 16 claim discipline | 1 | 1 | 1 | no workspace, thought or consciousness claim | a write-up that says the model "hears a bird" |
| 17 competence, scaffolding | 2 | 1 | 1 | gate, asserted token counts, the card's modality order; no published benchmark reproduced | the gate fails on sound, as in row 6 |
| 18 intervention controls | 3 | 1.5 | 1.5 | clean, random, unembedding, Z, three doses; no reverse direction, no matched-confidence check | random direction at dose 8 also flips the name |
| 19 baselines, off-manifold | 2 | 1 | 1 | random and unembedding; no norm report for the steered state, no PCA | the steered norm is far outside the clean range |
| 20 statistics | 2 | 0 | 0 | five clips, no interval, by design | any |
| 21 falsification, negatives | 1 | 0.5 | 1 | v2 has dated predictions, a kill condition, misses kept in the ledger | a rule changed after step 6 |
| 22 out-of-sample prediction | 4 | 0 | 2 | v2 freezes forecasts from the image runs for two unseen modalities; direction only, no magnitudes | prediction 1 or 2 fails on a perceived modality |
| 23 transfer across models | 2 | 1 | 1 | two families, three sizes, one assay, blocks as shares of depth; no forecast per size | blocks differ by family |
| 24 genuine audio and video | 2 | 1 | 1 | real motion and real sound with a gate; the static-video arm added; no benchmark-matched pipeline | clip, static video and frame give identical cells |
| 25 learning origin | 1 | 0 | 0 | not addressed | none |
| 26 natural ecology | 1 | 0 | 0 | five single-object clips with known class words | none |
| 27 implementation | 3 | 1.5 | 1.5 | position and count assertions, the before and after check, step 5 as a reference check; no unit test for the new script, no repeated-run check | step 5 does not reproduce row 93 |
| 28 provenance | 2 | 1 | 1 | sha256, seeds, the candidate list, `runs.Run` configs; YouTube clips can vanish, no outside rerun planned | a clip's hash changes |
| 29 Figure 1 | 4 | 0 | 0 | no figure is designed | none |
| 30 prose, signature | 6 | 0 | 0 | a plan, not a paper | none |
| | **100** | **13** | **25.5** | | |

Section subtotals, v1 then v2: I 0 and 5.5 of 35; II 2.5 and 6 of 20; III 2 and 3 of 10; IV 4 and 4.5 of 10; V 2 and 4 of 10; VI 2.5 and 2.5 of 5; VII 0 and 0 of 10.

## 3. The ten gates

| gate | v1 | v2 | reason |
|---|---|---|---|
| G1 revelation | RED | RED | a candidate thesis now exists; no insight a reader could not have guessed |
| G2 field transformation | RED | RED | nothing follows from a toy |
| G3 novelty | RED | RED | no prior-art audit |
| G4 exactness, rivals | RED | YELLOW | v1 confounded the answer with the steering word; v2 names rivals and arms, without the power to separate them |
| G5 natural mechanism | RED | RED | forced steering only |
| G6 empirical integrity | RED | YELLOW | v1's main arm was contradicted by the repository's own data; v2 has gates, controls and a prior rule, and no uncertainty |
| G7 cognitive explanation | RED | RED | no account of why |
| G8 frozen prediction | RED | RED | forecasts are frozen in v2; no confirmation exists |
| G9 public clarity | RED | RED | no figure, no paper |
| G10 council integration | RED | RED | a single self-review |

## 4. Three verdicts

- Mechanistic identification: **RED**. Steering shows that a state can force an answer, which the rubric lists as a typical zero for item 08.
- Cognitive theory: **RED**. No computational account is offered or tested.
- Interdisciplinary consequence: **RED**. None claimed.

These three are one reviewer's readings under three lenses. They are not independent.

## 5. Prior art

Not audited. The titles below come from this repository's ledger and report; none was read for this pass, which is why item 02 scores 0.

| work (as the ledger names it) | overlap with this plan | what would be new |
|---|---|---|
| Gurnee et al. 2026, the J-lens paper | the lens and its steering, on text | non-text tokens |
| This repository, RECORD rows 51, 53, 93 | the same steering on image tokens, with the layer blocks used here | video and audio tokens, same assay |
| *Do AV-LLMs really see and hear?* (CVPR 2026); AudioLens; *Can we read the mind of an audio LLM?* | logit lens on audio tokens: audio semantics at intermediate and late layers | the J-lens, and an intervention where they only read |
| Neo et al., ICLR 2025 | logit lens on visual tokens, object information local to its patches | steering, and video |
| Nikankin 2025; Salazar 2026 (named in REPORT) | cross-modal circuits and pathways | not yet compared |

The cheapest way to raise item 02: read the three audio-lens papers in full and write the overlap line by line.

## 6. Evidence and contradiction ledger

- **Proven in this repository (images):** every row of "What the image runs already say".
- **Inferred, untested:** that Qwen3.5's video tokens and Gemma-4's frame and audio tokens behave like image tokens under steering.
- **Speculative:** the proposed thesis.
- **Contradicted:** version 1's main arm, one 4-layer window at 4 sigma, by the `inject` curve (0.00 at every window on two models).
- **Earlier failures kept:** audio perception on Gemma E4B (row 6); the four-frame video donor (rows 65, 68); the 27B under patching hooks (row 67).

## 7. The rival tournament

| rival | what it says | forecast for the toy | decided by |
|---|---|---|---|
| A, a shared schedule | tokens of any modality are read for the name in one window of depth | early and middle blocks steer naming for clip and sound; the late block does not | predictions 1, 2, 5 |
| B, entry by modality | audio becomes word-like deeper than pictures, as the audio-lens work reports | for sound the early block fails, or the late block passes; the lens ranks on audio tokens rise later | prediction 5, and the rank printout |
| C, an output push | the direction only raises the next word's score | the answer-position steer and the unembedding direction do the same as the media-token steer; the rule never moves | the push rule; needs 64 clips for the rule part |
| D, a photo in disguise (video only) | the frames are read as pictures | clip, static video and frame are identical in every cell | prediction 4 |

Stop rule: if step 5 does not reproduce row 93, nothing after it is read.

## 8. Experiment spec

In the sections above: gate, inputs, sites, layer sets, directions, doses, predictions, rule, cost. Power: the toy detects an effect near 1.00 against a control near 0.00, and nothing smaller.

## 9. Figures and claims

None. One figure is worth designing if the toy says go: a grid with layer blocks down and photo, static video, clip and sound across, one dot per clip.

## 10. Objections and revisions, lens by lens

Each lens gives its strongest objection and one revision. "Adopted" means it is in version 2.

| lens (the rubric's reference names) | strongest objection | revision | adopted |
|---|---|---|---|
| search and selection bias (Nanda) | the main arm repeats a recorded null; sweeping 60 windows on 5 clips is layer fishing | steer the blocks the image runs used; fix the layer sets as shares of depth before the run | yes |
| lens fidelity (Gurnee) | a text-fitted lens is unchecked on audio tokens, so a failed steer says nothing | print J-lens and logit-lens ranks on the media tokens first, with a null word | yes |
| site and dose robustness (Bau) | one dose turns a threshold into a false negative | doses 2, 4, 8 on the blocks, all reported; prefix and suffix sweeps for the edges | yes |
| answer plan against content (Siegel, P11) | the question asks for the very word that was added | the digit rule, the same-kind word Z, and the answer-position steer | yes, as counts |
| causal abstraction (Geiger, Mueller, P10) | steering shows sufficiency of a forced state; the natural reader may bypass it | attention knockout from the question to the media tokens, as `attention_knockout` does for images | no: the next experiment |
| circuits (Olah, Lindsey, Ameisen) | "which layers" is a position, not an algorithm | head-level reads at the layers the toy finds | no: the next experiment |
| multimodal pipeline (Agrawal, Krishna) | eight frames of one object need no motion to name | the static-video arm now; a question only motion can answer (taking off or landing) in the real run | partly |
| probes against use (Belinkov) | a readable name on the tokens is not use | keep readout and steering as separate columns, never merged into one verdict | yes |
| object identity over time (Scholl) | one object per clip never tests tracking | two objects with boxes, ImageNet VID | no: the real run |
| word form against concept (Fedorenko, Kanwisher) | naming is taken for recognition | the claim is stated about naming only | yes |
| computational account (Tenenbaum, Gershman) | no reason is given for a shared window | write the size-by-modality forecast before any second model runs | no: open |
| statistics auditor | five clips give no interval; verdicts will flip on a redraw | 64 clips for anything reported as a number | no: the user's choice of a toy stands, and the limit is stated |
| code auditor | new media encoding is where hooks land on the wrong tokens | step 5 as a reference check; assert counts and positions | yes |
| replication auditor | YouTube clips disappear | hash the five files and keep them with the run | yes |
| seat P01, output pullback | the lens direction may be whatever the output prefers | the unembedding control at both sites | yes |
| seat P15, serial depth | "early for the rule, late for the name" may be ordinary depth bookkeeping | not answerable at n = 5 | no: open |
| seats P02 to P09, P12 to P14, P16 | no claim in this plan touches them | none | not applicable |

Left open, not settled by argument: whether a real run should be 64 clips of one object or fewer clips with boxes; whether sound should be dropped if the gate fails or replaced by spoken names.

## The output contract

```text
CANDIDATE: A word's lens direction written on an input's tokens is said when it is written
  before the model stops reading those tokens for the name, at the same share of depth for
  a photo, a clip and a sound. (Proposed in this pass; not yet the authors' claim.)
VERSION / DATE: plan.md version 2 / 15 Nov 2026
PQ: 25.5/100 (version 1: 13/100)       DE: not yet assessed
GATES: G1 [RED]; G2 [RED]; G3 [RED]; G4 [YELLOW]; G5 [RED]; G6 [YELLOW]; G7 [RED]; G8 [RED]; G9 [RED]; G10 [RED]
INDEPENDENT VERDICTS (one reviewer, three lenses, not independent):
  Mechanistic identification: RED
  Cognitive theory:          RED
  Interdisciplinary impact:  RED
FATAL FLAW: version 1 steered one 4-layer window, which the image runs record as doing
  nothing at any depth. Fixed in version 2. Remaining: five clips cannot separate the rivals
  on the rule question.
MOST IMPORTANT PRIOR ART: the audio logit-lens papers in the ledger, unread for this pass.
STRONGEST RIVAL EXPLANATION: an output push (rival C).
ONE DECISIVE / CHEAPEST KILL EXPERIMENT: step 5, then the answer-position steer on one model.
WHAT WOULD SURPRISE AN INFORMED EXPERT: sound steering from the same blocks as a photo,
  through a lens fitted on text only.
ONE-LINE REVELATION THAT MAKES AN EXPERT SAY WOW: none yet.
THREE NEW RESEARCH PROGRAMS UNLOCKED: none yet.
SINGLE RESULT WHICH CHANGES A 2036 TEXTBOOK: none yet.
MICHELANGELO TEST: fails; this is a pilot.
PUBLIC EXPLAINER: not written; there is no result.
AUTHORS' UNIQUE CONTRIBUTION: the image steering assay carried to video and audio tokens.
WHAT THE DATA CURRENTLY ALLOW US TO SAY: nothing about video or sound.
RELEASE: NO
```

**Ruling.** Not a release candidate, and not meant as one. As a pilot under the rubric's Stage 2, version 2 may run: its gates, controls and predictions are written, and step 5 stops it early if the script is wrong. What it can return is a go or no-go for a 64-clip run, with the audit of prior art (item 02) and an outside reviewer (G10) owed before that run is planned.
