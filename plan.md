# plan.md: which layer steers a video, and a sound?

A toy check, 5 videos and 5 sounds, before any real experiment. Written 15 Nov 2026, nothing has been run.

Built on two commits: `88b0b98` (the Karpathy guidelines: state assumptions, simplest thing, surgical changes, a check per step) and `bb38cc3` (the master rubric: items 17, 18, 19, 20, 21, 24, 27 and gate G6 apply to this toy).

## The question

On an image, adding the J-lens direction of the word "bird" to an airplane's image tokens makes the model say "bird", and only in some layers (`depth_sweep`, `name_steering`). Does the same work on a clip of an airplane taking off, and on the sound of an airplane, and in which layers?

## Short answers

**Are 3 Qwen and 3 Gemma enough?** Six models is more than a toy needs; the limit is the 5 clips, not the models. But the 3 + 3 grid cannot be filled for sound:

| | video | sound | J-lens on Neuronpedia |
|---|---|---|---|
| Qwen3.5 (any size) | yes | **no audio input** | 0.8B, 2B, 4B, 27B (9B: J++ only) |
| Gemma-4 E2B | yes (frames) | yes | yes |
| Gemma-4 E4B | yes (frames) | yes | yes |
| Gemma-4 12B | yes | yes | **no lens** |
| Gemma-4 31B | yes (frames) | **no audio input** | yes |

So: video on six models, sound on two.

**Are 5 clips enough?** Enough to see whether steering works at all and roughly where. Not enough for a number: each clip moves a rate by 0.2, and the rubric scores a one-run anecdote at zero (item 20). The toy's output is a go or no-go for a real run, nothing more.

## Models

| role | model | layers | download | note |
|---|---|---|---|---|
| Qwen small | `Qwen/Qwen3.5-0.8B` | 24 | 2 GB | pinned in `config.py` |
| Qwen medium | `Qwen/Qwen3.5-4B` | 32 | 9 GB | pinned; the image results to compare with |
| Qwen large | `Qwen/Qwen3.5-27B` | 64 | 54 GB | pinned; its patching runs crashed in the linear-attention kernel (RECORD row 67), steering hooks untried. Fallback: `Qwen3.5-9B`, which has only a J++ lens |
| Gemma small | `google/gemma-4-E2B-it` | 35 | 10 GB | **not pinned yet**: add its revision and lens sha256 to `config.py` |
| Gemma medium | `google/gemma-4-E4B-it` | 42 | 16 GB | pinned |
| Gemma large | `google/gemma-4-31B-it` | 60 | 63 GB | pinned; video only; needs a GPU of about 90 GB |

Start with the two mediums (4B and E4B). Add the others only if the mediums pass their gates.

## Data

One source for both modalities: **VGGSound** (CC BY 4.0, 10-second YouTube clips where the labelled object is both visible and audible). The same five clips give the video arm (frames, sound dropped) and the sound arm (audio track, no frames).

- Five classes, one clip each: airplane, bird, dog, car, train. Steering targets are of the other kind: airplane to bird, bird to airplane, dog to car, car to dog, train to bird.
- Each name must be one token in every model's tokenizer (asserted, as `stimuli.load_crops` does).
- Draw 4 candidate clips per class with a fixed seed; keep the first that every model in the run names correctly when clean (the crop gate, for clips). Save the list of candidates and which were dropped.
- Source: `Loie/VGGSound` on Hugging Face is 20 shards of about 16 GB; one shard is enough. A small mirror (`harpreetsahota/VGGSound-1000`) may do and is not yet checked.
- Sound fallback: ESC-50 (5-second clean clips: dog, rooster, airplane, train, chirping birds; CC BY-NC).

Not ImageNet VID for the toy: it is 92 GB and has no sound. It stays the choice for a real video run, because it has boxes.

## Design

One new script, `experiments/media_steer.py`, run name `toy_media_steer`. It reuses `depth_sweep`'s inject arm unchanged in logic.

Prompt: the clip, then "What is the object in the video? Reply with the noun only." (sound: "What is making this sound? Reply with the noun only.", after the audio as Gemma's card asks).

| arm | what it does | why |
|---|---|---|
| clean | no hook | the perception gate |
| steer Y | adds 4 sigma_L of Y's J-lens direction on every media token, one 4-layer window at a time, from layer -1 to the last | the question |
| steer random | a random unit direction, same dose and windows | the norm alone must not flip the answer |
| steer Y, W_U only | Y's plain unembedding direction | is the Jacobian needed? |
| middle frame as an image | the same sweep on one frame of the clip | ties the toy to the image results; shows whether video differs from a photo |

The dose is fixed at 4 sigma_L, as in `depth_sweep`. No dose tuning on 5 clips.

Fixed inputs: 8 frames sampled evenly at 224 x 224. On Qwen3.5 that is 196 video tokens (one token covers 32 x 32 pixels of two frames). On Gemma-4 it is 8 x 70 = 560 tokens. A 10-second sound is 250 audio tokens on Gemma-4 (40 ms each).

## The rule, written before the run

- **Gate.** A model and modality is tested only if it names 5 of 5 clean clips. Below that the cell reads "not perceived, untestable". It never reads "no pathway" (rubric E2).
- **A window steers** when the answer is Y on at least 3 of 5 clips and the random direction gives Y on at most 1 of 5.
- **The steer layers** of a model and modality are the window starts that pass. Reported as a list and as a share of depth, beside the middle-frame list.
- **No verdict beyond that.** No intervals, no claim across sizes or families from n = 5.

## Steps

1. Environment: `uv sync`, `uv run pytest -q` on a GPU machine (this cloud session has none). Verify: tests pass.
2. Pin Gemma E2B in `config.py`; add the clips to `scripts/download.py` with sha256. Verify: `download.py` passes its hash checks.
3. Input smoke, no steering: each model encodes one clip and one sound; print the token counts and the answer. Verify: counts match the numbers above, media-token positions are asserted, the 27B loads.
4. Write `media_steer.py` with the docstring carrying this design and rule. Video and audio encoding live in the script, so `Session` is untouched. If `Session.encode` must change, run `depth_sweep --n 4` before and after and compare with `check.py`. Verify: `ruff`, tests, the before and after folders match.
5. Run the mediums, video: 4B, then E4B. Verify: gate, then read the CSV row by row.
6. Run sound on E4B, then E2B. Verify: gate first; stop there if it fails.
7. Only if steps 5 and 6 pass: the small and large models.
8. Add one row per model and modality to `docs/RECORD.md`, misses included.

Cost: about 150 to 300 short forwards per model per modality. Minutes per model; the downloads dominate.

## Known risks

- **Sound may fail its gate.** The one earlier audio attempt (RECORD row 6, Gemma E4B, ESC-50) did not perceive the sounds (hits 0.88, false alarms 0.41). Gemma-4's card lists audio for speech recognition and translation only. Fallback for the toy: the object's name spoken aloud, which tests a spoken word, not a sound, and must be labelled so.
- **Video may be a photo in disguise.** The rubric scores "static images mislabeled video" at zero (item 24). The middle-frame arm shows it: identical steer layers mean the toy has not yet tested anything temporal.
- **The earlier video attempt failed its control** (RECORD rows 65 and 68: a still canvas repeated four times, patching). This toy uses real motion and steering, so that failure neither predicts nor excuses this one.
- **The lens was fitted on text.** It reads image tokens well; audio tokens are unchecked. Step 3 should also print the lens rank of the clip's own name on its media tokens.
- **Outputs stay in `runs/`.** `results/` is the reference and is never edited.

## Not in the toy

Patching a donor clip into a receiver, the animal and ride rules, ablation, slots with several objects, any statistics. Each is the next experiment if the toy says go, with its own rule written first.
