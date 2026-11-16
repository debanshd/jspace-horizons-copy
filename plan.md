# plan.md: which layers steer a video, and a sound?

A pilot on video clips and sounds: a 5-clip smoke check, then a 64-clip run. Nothing has been run.

| version | date | what changed | proposal score |
|---|---|---|---|
| 1 | 15 Nov 2026 | the first plan, 5 clips (commit `504e7bb`) | 13 / 100 |
| 2 | 15 Nov 2026 | the council pass found a fatal flaw; steering moved from single windows to layer blocks; controls added | 25.5 / 100 |
| 3 | 15 Nov 2026 | 64 clips with intervals, a knockout arm, an ablation arm, a figure design, a partial prior-art audit | 37 / 100 |
| 3.1 | 16 Nov 2026 | details the code fixed where version 3 left them open (data source, word assignment, calibration block, knockout queries, lens rank, the ablation rule); written before any run, no prediction or threshold changed | 37 / 100 |
| 3.2 | 16 Nov 2026 | this file: video and sound only. The middle frame as a picture and the step that reproduced RECORD row 93 are dropped; the image runs' numbers still set the predictions but are not rerun. Predictions 7 and 8 and the comparison rule are restated without the picture; written before any run | 37 / 100 |

Built on two commits: `88b0b98` (the Karpathy guidelines: state assumptions, simplest thing, surgical changes, a check per step) and `bb38cc3` (the master rubric, applied in full in the council pass at the bottom).

## The question

On an image, adding the J-lens direction of the word "bird" to an airplane's image tokens makes the model say "bird". Does the same work on a clip of an airplane taking off, and on the sound of an airplane, and from which layers?

## What the image runs already say

These numbers set the design and the predictions. They are not rerun: this plan runs only video and sound.

| image result (Qwen3.5-4B, 64 scenes) | number | source |
|---|---|---|
| steering one 4-layer window, any depth, 4 sigma | says Y on 0.00 | `results/depth_sweep`, `inject` curve; same on Gemma-31B |
| steering every image token, layers -1 to 10 | says Y on 1.00; animal rule follows on 0.23 | RECORD row 93 |
| steering every image token, layers 11 to 22 | says Y on 1.00; animal rule follows on 0.03 | RECORD row 93 |
| random direction, same dose | 0.00 to 0.03 | RECORD row 93 |
| steering only the object's 49 slot tokens | never says Y at any dose | `results/name_steering`, R4 |
| steering the answer position | says Y on 0.98, rules 0.05; the plain unembedding direction also says Y on 0.80 | REPORT, "Steering the picture with a name" |
| removing the own name's direction, or the 256-direction lens part | naming 0.95 to 0.91, and 0.98; rules unchanged | `results/name_steering`, accuracy |
| attention knockout | decisions stop reading the image at 0.26 to 0.33 of depth (Qwen), 0.38 to 0.48 (Gemma); naming at 0.67 to 0.75 | REPORT, the stations |

So steering works over a block of layers on every media token, one window does nothing, and removing the name direction costs nothing on images.

## Short answers

**Are 3 Qwen and 3 Gemma enough?** Yes for video. The grid cannot be filled for sound:

| | video | sound | J-lens on Neuronpedia |
|---|---|---|---|
| Qwen3.5 (any size) | yes | **no audio input** | 0.8B, 2B, 4B, 27B (9B: J++ only) |
| Gemma-4 E2B | yes (frames) | yes | yes |
| Gemma-4 E4B | yes (frames) | yes | yes |
| Gemma-4 12B | yes | yes | **no lens** |
| Gemma-4 31B | yes (frames) | **no audio input** | yes |

So: video on six models, sound on two.

**Are 5 clips enough?** Only as a smoke check that the script and the gates work. Every number comes from the 64-clip run, which costs nearly the same compute.

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

One source for both modalities: **VGGSound** (CC BY 4.0, 10-second YouTube clips where the labelled object is both visible and audible). Each clip gives the video arm (frames, sound dropped) and the sound arm (audio track, no frames).

- **64 clips: 8 classes by 8 clips.** Four animals (dog, cat, bird, horse) and four vehicles (car, bus, train, airplane), so the animal rule is balanced. The class labels are to be confirmed against `vggsound.csv` before the draw.
- **The smoke check** uses the first clip of five classes (airplane, bird, dog, car, train).
- Y is a name of the other kind, Z a name of the same kind. They are assigned in a fixed rotation over the 8 clips of a class rather than drawn at random, so that every ordered pair appears and each pair runs both ways (each Y twice per class, each Z two or three times).
- Each name must be one token in every model's tokenizer (asserted, as `stimuli.load_crops` does).
- A label belongs to a class when it names exactly one class word ("dog barking" is dog; "railroad car, train wagon" names two and is not used).
- Draw 16 candidates per class with a fixed seed; keep the first 8 that every model in the run names correctly when clean (the clip, asked "What is the object in the video?"). Save the candidates and which were dropped.
- The first 16 kept clips are the calibration split; the other 48 carry the verdicts.
- Source: the first shard of VGGSound (about 10,000 clips) as single mp4 files from `harpreetsahota/VGGSound`, with the labels from `Loie/VGGSound`'s `vggsound.csv`, both at pinned commits. This avoids downloading a 16 GB archive for 128 clips.
- Hashes: the mp4 of every candidate and its decoded frames and sound are hashed, and the list is committed with the code (`scripts/vggsound_manifest.json`), since YouTube clips disappear. A later download is checked against it; each run's `clips.json` records the hashes of its own clips.
- Sound fallback: ESC-50 (5-second clean clips; CC BY-NC).

ImageNet VID (92 GB, boxes, no sound) stays the choice for a run with several objects per clip.

## Design (version 3)

One new script, `experiments/media_steer.py`. Run names: `smoke_media_steer` (5 clips), `media_steer` (64 clips).

**Inputs**, three per clip:

| input | why |
|---|---|
| the clip, 8 frames at 224 x 224 | the question |
| a static video: the middle frame 8 times | same token format, no motion |
| the audio track (Gemma E2B and E4B only) | the question |

**Questions**, in separate prompts: say ("What is the object in the video? Reply with the noun only.") and the animal rule ("if the object in the video is an animal, answer 7; otherwise answer 3"). Sound: "What is making this sound?", placed before the audio, as Gemma's card asks.

**Layer sets**, as shares of depth so the models compare:

| layer set | layers on the 4B | purpose |
|---|---|---|
| early block | -1 to 10 (up to 0.33) | row 93's early steer |
| middle block | 11 to 22 (0.33 to 0.70) | row 93's late steer |
| late block | 23 to 31 (after 0.70) | after naming stops reading the tokens |
| prefix sweep | -1 to L, L in steps of 4 | the smallest stretch from the input that steers |
| suffix sweep | L to the last layer, L in steps of 4 | the latest start that still steers |

**Arm 1, steering** (can a written name be said, and used?). Sites: every media token, and the last prompt token as the control for an output push. Directions, each of unit length: Y's J-lens direction; Z's (the rule must not flip); a random direction; Y's plain unembedding direction; the top principal direction of the clip's media tokens (the energy control). The dose is picked on the 16 calibration clips as the smallest of 1, 2, 4, 8 sigma_L at which Y is said on half of them, with Y on the media tokens in the middle block (row 93's late steer, which said Y on every scene); all four doses are reported. If no dose passes, the sweeps run at 8 and the input is marked as having no calibrated dose. The pairs run both ways (airplane to bird and bird to airplane), so each direction has its reverse.

**Arm 2, attention knockout** (where does the unmodified model read the media tokens?). Attention from every text token to the media tokens is blocked (the question, the answer position, and the timestamp text between frames, so that no text token carries a frame forward), as `attention_knockout.py` does: all layers from L on, and all layers up to L, in steps of 4, on clean clips with no steering. In Qwen3.5 only the one layer in four with standard attention can be blocked; in Gemma-4 every layer can.

**Arm 3, ablation** (is the name direction needed?). The own name's J-lens direction, another name's (Y's) J-lens direction, a random direction, the lens part of rank r = 0.1 d (256 directions on the 4B, 512 on the 27B, as AGENTS.md fixes the rank) and a random part of the same rank, mean-ablated on every media token at the read layers. Y's direction is the control `name_steering`'s rule compares against.

**Checks printed before any arm**: per layer, the J-lens and logit-lens rank of the clip's own name on its media tokens, with another object's name as the null; and the norm of the steered state against the clean range.

Token counts to assert: 196 video tokens on Qwen3.5 (one token covers 32 x 32 pixels of two frames), 8 x 70 = 560 on Gemma-4, 250 audio tokens for a 10-second sound (40 ms each).

## Predictions, written before the run

From the image numbers, for every model and modality that passes its gate:

1. Naming follows Y from the early block and from the middle block, on at least 0.8 of clips.
2. Naming does not follow Y from the late block (at most 0.1).
3. A random direction, the unembedding direction and the principal direction on the media tokens move at most 0.1.
4. The animal rule follows Y more from the early block than from the middle block (image: 0.23 against 0.03), and never reaches 0.5.
5. The knockout puts the last naming read at 0.67 to 0.75 of depth, and the latest suffix start that still steers naming falls at or before it, within one step.
6. Ablating the own name's direction or the lens part costs naming at most 0.10, as on images.
7. The clip and the static video agree in every block.
8. Sound agrees with the clip on the same model.

Prediction 5 links two independent arms: the knockout runs on clean clips and forecasts where steering stops working. Prediction 8 is the most likely to fail and the most informative if it does.

## The rule, written before the run

Thresholds are those of the frozen image rules (`name_steering` R0 and R4, row 93), unchanged.

- **Gate.** A question is read only if clean accuracy is at least 0.90 for naming and 0.85 for the rule. Otherwise the cell reads "not perceived, untestable". It never reads "no pathway" (rubric E2).
- **A layer set steers naming** when Y is said on at least 0.5 of the 48 held-out clips and the lower bound is above the random direction's upper bound.
- **Seen, or only said.** "Seen" when the rule follows Y by at least 0.30 more than the random direction, with the interval above 0 and Z flipping the rule on at most 0.10. "Only said" when the rule follows Y by at most 0.10 more than random.
- **J-specific** when the unembedding direction on the media tokens gives Y on at most 0.10.
- **Not told apart from an output push** when the media-token steer passes only where the same steer at the last prompt token passes and the unembedding direction passes too.
- **Same as the clip** when the difference between the sound's rate and the clip's (paired by clip, at the clip's calibrated dose) has its interval inside plus or minus 0.15, per block. Outside it on any block: "differs by modality".
- **Nothing temporal tested** when the clip and the static video are the same by that test (at the static video's calibrated dose) in every block.
- **Knockout** is read by `attention_knockout.py`'s rule: a cost is clean minus knocked-out log p(X's answer) − log p(Y's answer); the last reader is the largest L whose from-L-on knockout costs at least 1 nat with the interval above 0.
- **Ablation** is read by `name_steering.py`'s R1, per question: D = accuracy with Y's direction removed minus accuracy with the own name's removed. "Necessary" when D ≥ 0.25 with the interval above 0; "not needed" when D ≤ 0.10 with the interval inside (−0.15, 0.15); otherwise undetermined. The same in nats (R1_nats): log p of the correct first token with the own name removed minus with Y's removed; "necessary" at ≤ −0.5 with the interval below 0, "not needed" at ≥ −0.2 with the interval inside (−0.3, 0.3).
- **Intervals**: 95% bootstrap over clips, 2000 resamples, seed 0 (`analysis.boot_scene`). With 48 clips a rate's interval is about plus or minus 0.14 at worst, enough for 1.00 against 0.00 and for 0.23 against 0.03 as a paired difference, not for effects under 0.15.
- **The smoke check** carries no verdict: it reports counts out of 5.

## Figure 1, designed before the run

One grid per model. Columns: static video, clip, sound. Rows: early, middle and late block. Each cell holds two bars with intervals, "says the steered name" and "applies the rule to it", and a dashed line for the random direction. A strip under the grid shows the knockout's last naming read and last rule read as two ticks on a depth axis from 0 to 1. A reader should see in one look whether the columns match and where the ticks fall against the blocks.

## Steps

1. Environment: `uv sync`, `uv run pytest -q` on a GPU machine (this cloud session has none). Verify: tests pass.
2. Pin Gemma E2B in `config.py`; add the clips to `scripts/download.py` with sha256. Verify: `download.py` passes its hash checks.
3. Input smoke, no steering: each model encodes one clip and one sound; print the token counts, the answer and the lens ranks. Verify: the counts above, asserted media-token positions, the 27B loads.
4. Write `media_steer.py` with the docstring carrying this design, the predictions and the rule, and a CPU test in `tests/` for the media-token position helper on made-up token ids. Video and audio encoding live in the script, so `Session` is untouched. If `Session.encode` must change, run `depth_sweep --n 4` before and after and compare with `check.py`. Verify: `ruff`, tests, matching folders.
5. The smoke check, 5 clips, 4B and E4B, all inputs. Verify: gates, token counts, read the CSV row by row. Run it twice and compare the two runs. If the token counts differ between clips or the two runs differ, stop: the script is wrong.
6. The 64-clip run on the 4B and E4B: clip and static video. Then sound on E4B and E2B; stop sound if its gate fails.
7. Only if step 6 passes: the small and large models.
8. One row per model and modality in `docs/RECORD.md`, with each prediction marked kept or missed.

## The code, and how to run it on RunPod

Written 16 Nov 2026; step 4 is done, nothing has run on a GPU. It was checked on a CPU only: 12 new tests (`tests/test_media.py`: the label rules, the scoring, the clip assignment, the layer blocks, the row-wise hooks, the knockout mask, and the whole summary on made-up answers), ruff, PyAV decoding of a synthetic 10-second clip, and the Qwen3.5 and Gemma-4 video and audio processors of transformers 5.19 on a test clip (196 video tokens on Qwen, 8 x 70 on Gemma, the true frame times, 250 audio tokens for 10 s).

| file | what it does |
|---|---|
| `scripts/pin_media.py` | resolves the commits of the two VGGSound repositories and of Gemma-4-E2B with its lens sha256, and writes them into `config.py` (they could not be resolved from the session that wrote the code) |
| `scripts/prepare_media.py` | downloads 16 candidate clips per class from the first VGGSound shard and decodes each once: 8 frames at 224 x 224 and 10 s of mono 16 kHz sound, with a manifest of hashes |
| `src/jspace/media.py` | the classes, the label rule, the prompts, the scoring, the clip-to-word assignment, the layer blocks |
| `src/jspace/batched.py` | the engine: row-wise steering, ablation and knockout in shared batches |
| `experiments/media_steer.py` | the experiment, in three stages: `gate`, `smoke`, `main` |
| `scripts/media_steer.sh` | steps 2 to 8 in order, skipping what is done |

On a fresh pod, from the repository:

```bash
uv sync
uv run scripts/pin_media.py                     # once; then commit src/jspace/config.py
STAGE=smoke scripts/media_steer.sh              # steps 5 and 6: the 5-clip smoke check on the 4B and E4B
scripts/media_steer.sh                          # step 7: gate for both models, then the 64-clip run
MODELS="Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-4B Qwen/Qwen3.5-27B google/gemma-4-E2B-it google/gemma-4-E4B-it google/gemma-4-31B-it" RUN=media_steer_all scripts/media_steer.sh   # step 8, 96 GB card
```

How it keeps the GPU busy. No code can promise 100% utilization; this code removes the usual causes of idle time and measures what is left:
- **No padding.** Every clip has the same frame times and the same sound length, so every prompt of one input and question has the same length. Session's padding check had found that left padding moved Gemma-4-E4B's and Qwen3.5-27B's numbers and ran them one prompt at a time (October configs: `batch 1`); with no padding they batch fully.
- **Many arms per batch.** The 54,000 rows on the 4B (105 arms in phase A and 36 sweep arms in phase B, for 2 questions, 3 inputs and 64 clips) and about 80,000 on Gemma-4-E4B (more knockout and sweep starts, and the sound) are mixed in batches of up to 512 prompts; each layer's hook edits only its own rows.
- **The encoder once per clip.** Vision-encoder outputs are cached and passed as `mm_encoder_outputs`. The cache is switched on only if a start-up check gives identical answers and first-token log-probabilities within 0.05 nats; otherwise the encoder runs in every batch. Gemma's audio tower takes no cached outputs and runs in every batch.
- **The processor once per prompt**, and the next batches assembled and pinned on a CPU thread while the GPU runs.
- **Batch size from free memory**, halved on out-of-memory and kept.
- **Measured.** `gpu.csv` holds nvidia-smi's utilization every 2 s; `summary.json` gives its mean and median and the share of wall time spent in generation.

Changes outside the new files: `config.py` gains the VGGSound constants (`None` until pinned) and a `vggsound` entry in `DATA_FILES`; `download.py` gains `--models` and `--skip-data`. No existing experiment reads either, so their outputs cannot change; per AGENTS.md they should still be run at `--n 4` before and after on the pod to confirm it.

What can still go wrong on the GPU, with the guard for each:

| risk | guard |
|---|---|
| a processor or model call that the CPU checks could not reach | the smoke stage runs every arm on 5 clips first |
| the cached encoder outputs differ from the encoder's | the start-up comparison; the cache turns itself off |
| hooks landing on the wrong tokens | equal-layout assertion per batch; step 5 must reproduce row 93 |
| the 27B's linear-attention kernel crashing under hooks, as in RECORD row 67 | none in code: run it last, and fall back to the 9B |
| a class with fewer than 8 clips every model names | the run stops and says so; rerun `prepare_media.py --candidates 32` |

## RunPod: time and cost

Estimates made on 15 Nov 2026, before any run. The prices are from RunPod's pricing page and third-party trackers, which disagree by tier and date; check the live page before renting.

| card | models that fit | pod time, 64 clips | price per hour | compute cost |
|---|---|---|---|---|
| RTX 6000 Ada, 48 GB | Qwen 0.8B and 4B, Gemma E2B and E4B | 5 to 8 hours | about $1 ($0.84 to $1.09 listed) | $5 to $8 |
| RTX PRO 6000, 96 GB | all six | 9 to 12 hours | about $2.49 ($1.69 to $2.49 listed) | $22 to $30 |

- The 48 GB card cannot hold Qwen3.5-27B (54 GB of weights) or Gemma-4-31B (up to 88 GB in the original runs). The original runs used the 96 GB card.
- Storage is extra: about 55 GB for four models and one VGGSound shard, about 170 GB for all six. A few dollars for a month of network volume.
- Where the time goes: downloads (20 to 60 minutes), then writing and debugging the video and audio encoding on the pod (most of it). The 64-clip run is about 40,000 short generations per model, roughly half an hour to an hour each on the medium models, against the 21,000 forwards in 15 minutes the notes record for the 4B on this card type.
- The 5-clip smoke check alone is 3 to 5 hours on the 48 GB card, almost all of it setup. The 64 clips add about two hours.

## What still raises the score, and who can do it

| what | items | who |
|---|---|---|
| accept or replace the thesis | 01 | the authors |
| blind forecasts for the rivals from a skeptic, before the run | 03, G3 | an outside researcher |
| read the methods and results of the audio papers; find the one the ledger cites that was not found | 02, G3 | anyone, no GPU |
| a question only motion can answer, with labels | 24 | needs a labelled set (taking off or landing is not in VGGSound) |
| a declared substitute for the source-monitoring item | 15 | the authors |
| blind readers for Figure 1 | 29, G9 | three people outside the project |
| an outside review of this plan | G10 | not the assistant |
| a result that surprises someone | 04, 05, G1, G2 | only a run can supply it |

Running the plan does not raise the proposal score. It starts the separate evidence score.

## Known risks

- **Sound may fail its gate.** The one earlier audio attempt (RECORD row 6, Gemma E4B, ESC-50) did not perceive the sounds (hits 0.88, false alarms 0.41), and Gemma-4's card lists audio for speech recognition and translation only. Fallback: the object's name spoken aloud, which tests a spoken word, not a sound, and must be labelled so.
- **The earlier video attempt failed its control** (RECORD rows 65 and 68: a still canvas repeated four times, patching). This run uses real motion and steering, so that failure neither predicts nor excuses this one.
- **The lens was fitted on text.** It reads image tokens; audio tokens are unchecked, hence the rank printout.
- **Steering every media token is blunt.** It cannot say which object in a clip was changed. One object per clip hides that; a later run needs boxes.
- **Qwen3.5's knockout is partial.** Only one layer in four can be blocked, so prediction 5 is sharpest on Gemma-4.
- **Outputs stay in `runs/`.** `results/` is never edited.

## Not in this plan

Patching a donor clip into a receiver, head-level reads, several objects per clip. Each is the next experiment, with its own rule written first.

---

# The council pass (master rubric v2.0)

Scored 15 Nov 2026, three times: version 1, version 2, version 3. Stage: proposal quality (PQ). Discovery evidence (DE) is not assessed, since nothing has run.

**What this pass is.** One reviewer, the assistant that wrote the plan, applying the rubric's lenses one after another. The named researchers are lines of scrutiny taken from the rubric; none of them reviewed, voted or endorsed anything. The rubric does not count a model-written review as independent verification, so gate G10 is RED by construction and no score here is a clearance.

## 1. Thesis and status

Version 1 had a question and no thesis, so it failed Stage 0 admission. This pass proposes one, to be accepted or replaced by the authors:

> A word's lens direction written on an input's tokens is said when it is written anywhere before the model stops reading those tokens for the name, and this window sits at the same share of depth for a clip and a sound as the image runs found for a photo.

- Scope: Qwen3.5 and Gemma-4, single-object inputs, one-token names.
- What kills it: a perceived modality where naming does not follow from the early or middle block, or follows from the late block, or where the knockout's last naming read and the steering edge disagree by more than one step.
- What survives a kill: "the window is set by the token type", which is a result about modality.

**PQ: version 1, 13. Version 2, 25.5. Version 3, 37 of 100. DE: not assessed.**

## 2. The thirty items

Each item gets 0, half or all of its weight. The last two columns describe version 3.

| ID | weight | v1 | v2 | v3 | evidence, and the strongest objection left | cheapest test that kills it |
|---:|---:|---:|---:|---:|---|---|
| 01 revelation | 5 | 0 | 2.5 | 2.5 | a thesis with a falsifier, proposed by the reviewer and not yet the authors'; may read as a minor extension | an expert predicts the outcome unaided |
| 02 originality | 8 | 0 | 0 | 4 | a partial audit (section 5): three papers read at abstract and introduction, one not found, methods unread; the novelty is close to "not done with this lens" | a paper that steers audio or video tokens with a word direction |
| 03 epiphany | 6 | 0 | 3 | 3 | four rivals with different forecasts; none is a forecast a skeptic made blind | all rivals predict the same cells |
| 04 field-changing | 10 | 0 | 0 | 0 | a pilot changes no belief | none |
| 05 elegance, durability | 6 | 0 | 0 | 0 | nothing here survives without the model names | none |
| 06 mathematical exactness | 4 | 2 | 2 | 2 | operator, dose unit, layer sets, estimands, nulls and intervals are defined; nothing is independently checked | a second draw of clips changes a verdict |
| 07 decisive discriminations | 4 | 0 | 2 | 2 | the output-push, static-video and knockout arms separate the rivals at n = 48; no natural task decides between them | the answer-position steer matches the media-token steer everywhere |
| 08 natural writer, state, reader | 5 | 0 | 0 | 2.5 | the knockout finds where the unmodified model reads, and prediction 5 ties it to the steering edge; no writer, no heads, no rescue | steering works after the knockout's last read |
| 09 necessity, sufficiency | 3 | 0 | 0 | 1.5 | steering for sufficiency, knockout and ablation for necessity, four doses; the ablation is expected to be null, and bypass routes are not measured | ablation and knockout both cost nothing |
| 10 content, task, answer split | 3 | 0 | 1.5 | 1.5 | the digit rule, Z, the answer-position site; not a factorial | the rule never moves while the name always does |
| 11 lens validity | 1 | 0.5 | 0.5 | 0.5 | the published lens with random, unembedding and principal-direction controls; no complement arm; the 9B fallback is J++ | the logit lens reads audio tokens as well as the J-lens |
| 12 construct validity | 2 | 1 | 1 | 1 | the claim is about naming, with a gate; "see" in the question is looser than what is measured | the gate passes on a cue that is not the sound |
| 13 computational explanation | 3 | 0 | 0 | 0 | no model of why a window would be shared | none |
| 14 dissociations | 2 | 0 | 1 | 1 | say against rule, now with intervals; one dissociation, single direction | none |
| 15 source monitoring | 2 | 0 | 0 | 0 | not claimed; no substitute test declared, so the weight is lost, not waived | none |
| 16 claim discipline | 1 | 1 | 1 | 1 | no workspace, thought or consciousness claim | a write-up that says the model "hears a bird" |
| 17 competence, scaffolding | 2 | 1 | 1 | 1 | gate, asserted token counts, the card's modality order; no published benchmark reproduced | the gate fails on sound, as in row 6 |
| 18 intervention controls | 3 | 1.5 | 1.5 | 1.5 | clean, random, unembedding, principal direction, Z, reverse pairs, a dose picked on a calibration split; confidence is not matched | the random direction at dose 8 also flips the name |
| 19 baselines, off-manifold | 2 | 1 | 1 | 1 | the norm of the steered state is now reported; no complement or full-state arm | the steered norm is far outside the clean range |
| 20 statistics | 2 | 0 | 0 | 1 | effect, unit, clip bootstrap, equivalence margins and thresholds fixed beforehand; power is stated roughly, not computed; many layer sets, no correction | an interval that straddles its threshold |
| 21 falsification, negatives | 1 | 0.5 | 1 | 1 | dated predictions, kill conditions, misses kept in the ledger | a rule changed after step 5 |
| 22 out-of-sample prediction | 4 | 0 | 2 | 2 | forecasts frozen from the image runs, now with magnitudes and a cross-arm forecast; no competing forecaster, no calibration | prediction 1, 2 or 5 fails on a perceived modality |
| 23 transfer across models | 2 | 1 | 1 | 1 | two families, three sizes, one assay, blocks as shares of depth; no forecast per size | blocks differ by family |
| 24 genuine audio and video | 2 | 1 | 1 | 1 | real motion and real sound with a gate and a static-video arm; no question that needs motion | clip and static video identical in every cell |
| 25 learning origin | 1 | 0 | 0 | 0 | not addressed | none |
| 26 natural ecology | 1 | 0 | 0 | 0.5 | 64 clips in the wild; one object each, known class words | none |
| 27 implementation | 3 | 1.5 | 1.5 | 1.5 | position and count assertions, a CPU test, the smoke check run twice; all planned, none written | the smoke check's two runs differ |
| 28 provenance | 2 | 1 | 1 | 1 | sha256, seeds, the candidate list, `runs.Run` configs, the clips kept with the run; no outside rerun planned | a clip's hash changes |
| 29 Figure 1 | 4 | 0 | 0 | 2 | a figure is designed; no blind reader has seen it | a reader cannot say what was changed |
| 30 prose, signature | 6 | 0 | 0 | 0 | a plan, not a paper | none |
| | **100** | **13** | **25.5** | **37** | | |

Section subtotals for version 3: I 9.5 of 35; II 10 of 20; III 3 of 10; IV 5.5 of 10; V 4.5 of 10; VI 2.5 of 5; VII 2 of 10.

## 3. The ten gates

| gate | v1 | v2 | v3 | reason for the version 3 colour |
|---|---|---|---|---|
| G1 revelation | RED | RED | RED | no insight a reader could not have guessed |
| G2 field transformation | RED | RED | RED | nothing follows from a pilot |
| G3 novelty | RED | RED | RED | the audit is partial and no skeptic has forecast |
| G4 exactness, rivals | RED | YELLOW | YELLOW | rivals and arms are named and powered for large effects; nothing is checked by a second person |
| G5 natural mechanism | RED | RED | YELLOW | the knockout tests the unmodified model's reads; no writer or reader is identified |
| G6 empirical integrity | RED | YELLOW | YELLOW | gates, controls, a prior rule, intervals; the pipeline matches no published benchmark |
| G7 cognitive explanation | RED | RED | RED | no account of why |
| G8 frozen prediction | RED | RED | RED | forecasts are frozen; no confirmation exists |
| G9 public clarity | RED | RED | RED | a figure design, no paper |
| G10 council integration | RED | RED | RED | a single self-review |

## 4. Three verdicts

- Mechanistic identification: **RED**. The knockout adds a test of natural reads, but no component is identified.
- Cognitive theory: **RED**. No computational account is offered or tested.
- Interdisciplinary consequence: **RED**. None claimed.

These three are one reviewer's readings under three lenses. They are not independent.

## 5. Prior art

A partial audit, done for version 3. "Read" below means the abstract and introduction, through the Hugging Face papers pages. No methods or results section was read. Two searches for steering on video tokens found no precedent, which is weak evidence of absence.

| work | what it does (from its abstract) | overlap | what this plan adds |
|---|---|---|---|
| Gurnee et al. 2026, the J-lens paper (not re-read; described in the ledger) | the lens and its steering, on text | the instrument | non-text tokens |
| This repository, RECORD rows 51, 53, 93 | the same steering on image tokens, in the layer blocks used here | the assay | video and audio tokens |
| AudioLens ([2506.05140](https://arxiv.org/abs/2506.05140)), read | logit lens on three audio-language models; attribute information often drops in deep layers; the models query the audio tokens directly; earlier-layer states written into deeper layers raise accuracy | readout on audio tokens, and one intervention across layers | a word-specific direction, a text-fitted Jacobian lens, the same assay on photo, clip and sound |
| *Do Audio-Visual LLMs Really See and Hear?* ([2604.02605](https://arxiv.org/abs/2604.02605)), read | probing shows audio semantics at intermediate layers; deeper fusion layers favour vision | where audio content sits by depth; a rival forecast for sound (rival B) | an intervention, and a naming-against-use split |
| *Are Audio-Language Models Listening?* ([2603.06854](https://arxiv.org/abs/2603.06854)), read; found in this audit | audio-specialist attention heads; an audio-against-silence steering direction applied at the final representation | **the closest neighbour: activation steering in audio-language models** | their direction is modality-level and placed at the end; here it is one word's direction, on the audio tokens, localised by layer block |
| *Can we read the mind of an audio LLM?* (cited in the ledger as 2608.24958) | not found on Hugging Face papers | unknown | unread |
| Neo et al., ICLR 2025; Nikankin 2025; Salazar 2026 (from the ledger and REPORT) | logit lens on visual tokens; cross-modal circuits | not re-read | not yet compared |

Honest reading of the novelty: a known assay carried to two new token types. That is the rubric's typical zero for item 02 unless the result contradicts a forecast, which is why item 02 stays at half and G3 stays RED.

## 6. Evidence and contradiction ledger

- **Proven in this repository (images):** every row of "What the image runs already say".
- **Inferred, untested:** that Qwen3.5's video tokens and Gemma-4's frame and audio tokens behave like image tokens under steering and knockout.
- **Speculative:** the proposed thesis.
- **Contradicted:** version 1's main arm, one 4-layer window at 4 sigma, by the `inject` curve (0.00 at every window on two models).
- **Expected null, declared:** the ablation arm (prediction 6).
- **Earlier failures kept:** audio perception on Gemma E4B (row 6); the four-frame video donor (rows 65, 68); the 27B under patching hooks (row 67).

## 7. The rival tournament

| rival | what it says | forecast | decided by |
|---|---|---|---|
| A, a shared schedule | tokens of any modality are read for the name in one window of depth | predictions 1, 2, 5, 7, 8 hold | the blocks, and the knockout against the steering edge |
| B, entry by modality | audio becomes word-like deeper than pictures, as the audio papers report | for sound the early block fails or the late block passes; the knockout's reads sit deeper for sound | prediction 8, the knockout, the rank printout |
| C, an output push | the direction only raises the next word's score | the answer-position steer and the unembedding direction match the media-token steer; the rule never moves in any block | the push rule; prediction 4 |
| D, a photo in disguise (video only) | the frames are read as pictures | clip and static video are the same in every block | prediction 7, read as a limit on what was tested |

Stop rule: if the smoke check's token counts differ between clips or its two runs differ, nothing after it is read.

## 8. Experiment spec

In the sections above: gate, inputs, sites, layer sets, three arms, doses, the calibration split, predictions, rule, intervals, cost.

## 9. Figures and claims

Figure 1 is designed above. It maps to predictions 1, 2, 4, 5, 7 and 8. No abstract or headline exists to audit.

## 10. Objections and revisions, lens by lens

Each lens gives its strongest objection and one revision.

| lens (the rubric's reference names) | strongest objection | revision | status |
|---|---|---|---|
| search and selection bias (Nanda) | the first main arm repeated a recorded null; sweeping windows on 5 clips is layer fishing | the blocks the image runs used, fixed as shares of depth before the run | in v2 |
| lens fidelity (Gurnee) | a text-fitted lens is unchecked on audio tokens | J-lens and logit-lens ranks on the media tokens first, with a null word | in v2 |
| site and dose robustness (Bau) | one dose turns a threshold into a false negative | four doses, the dose picked on a calibration split; prefix and suffix sweeps | in v3 |
| answer plan against content (Siegel, P11) | the question asks for the very word that was added | the digit rule, Z, the answer-position steer, now with intervals | in v3 |
| causal abstraction (Geiger, Mueller, P10) | steering shows a forced state is sufficient; the natural reader may bypass it | the attention knockout on clean clips, and prediction 5 | in v3 |
| statistics auditor | five clips give no interval | 64 clips, 48 held out, a clip bootstrap, equivalence margins | in v3 |
| baselines (Belinkov) | a direction may work through its energy | the principal-direction control and the norm report; readout and steering kept as separate columns | in v3 |
| figures (Siegel, Olah) | nothing shows what was changed | Figure 1, designed before the run | in v3, unread by anyone |
| code auditor | new media encoding is where hooks land on the wrong tokens | the smoke check run twice, count and position assertions, a CPU test | in v3 |
| replication auditor | YouTube clips disappear | hash the files and keep them with the run | in v2 |
| circuits (Olah, Lindsey, Ameisen) | "which layers" is a position, not an algorithm | head-level reads at the layers the run finds | open: the next experiment |
| multimodal pipeline (Agrawal, Krishna) | eight frames of one object need no motion to name | a question only motion can answer | open: needs labels |
| object identity over time (Scholl) | one object per clip never tests tracking | two objects with boxes, ImageNet VID | open: a later run |
| computational account (Tenenbaum, Gershman) | no reason is given for a shared window | a size-by-modality forecast before any second model runs | open |
| word form against concept (Fedorenko, Kanwisher) | naming is taken for recognition | the claim is stated about naming only | in v2 |
| seat P01, output pullback | the lens direction may be whatever the output prefers | the unembedding control at both sites | in v2 |
| seat P15, serial depth | "early for the rule, late for the name" may be ordinary depth bookkeeping | prediction 4 tests the direction; no rival model of depth is fitted | open |
| seats P02 to P09, P12 to P14, P16 | no claim in this plan touches them | none | not applicable |

Left open, not settled by argument: whether sound should be dropped if the gate fails or replaced by spoken names; whether the large models are worth the 96 GB card before the mediums are read.

## The output contract

```text
CANDIDATE: A word's lens direction written on an input's tokens is said when it is written
  before the model stops reading those tokens for the name, at the same share of depth for
  a clip and a sound as the image runs found for a photo. (Proposed in this pass; not yet the authors' claim.)
VERSION / DATE: plan.md version 3 / 15 Nov 2026
PQ: 37/100 (version 1: 13; version 2: 25.5)       DE: not yet assessed
GATES: G1 [RED]; G2 [RED]; G3 [RED]; G4 [YELLOW]; G5 [YELLOW]; G6 [YELLOW]; G7 [RED]; G8 [RED]; G9 [RED]; G10 [RED]
INDEPENDENT VERDICTS (one reviewer, three lenses, not independent):
  Mechanistic identification: RED
  Cognitive theory:          RED
  Interdisciplinary impact:  RED
FATAL FLAW: none found in version 3. Version 1's was the single-window steer.
MOST IMPORTANT PRIOR ART: Are Audio-Language Models Listening? (arXiv 2603.06854), read at
  abstract and introduction only.
STRONGEST RIVAL EXPLANATION: an output push (rival C); for sound, entry by modality (rival B).
ONE DECISIVE / CHEAPEST KILL EXPERIMENT: the smoke check, then the knockout against the steering edge
  on Gemma E4B.
WHAT WOULD SURPRISE AN INFORMED EXPERT: sound steering from the same blocks as a clip,
  through a lens fitted on text only.
ONE-LINE REVELATION THAT MAKES AN EXPERT SAY WOW: none yet.
THREE NEW RESEARCH PROGRAMS UNLOCKED: none yet.
SINGLE RESULT WHICH CHANGES A 2036 TEXTBOOK: none yet.
MICHELANGELO TEST: fails; this is a pilot.
PUBLIC EXPLAINER: not written; there is no result.
AUTHORS' UNIQUE CONTRIBUTION: the image steering and knockout assays carried to video and
  audio tokens, with the forecasts fixed from the image runs.
WHAT THE DATA CURRENTLY ALLOW US TO SAY: nothing about video or sound.
RELEASE: NO
```

**Ruling.** Not a release candidate. As a pilot under the rubric's Stage 2, version 3 may run: its gates, controls, predictions and intervals are written, and the smoke check stops it early if the script is wrong. The 63 points still missing sit mostly where a plan cannot reach: 25.5 in revelation and field consequence, 8 in figures and prose, and the rest in work that needs other people (a skeptic's forecasts, blind readers, an outside review) or a labelled motion question.
