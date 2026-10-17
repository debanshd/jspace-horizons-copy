# Method

The question: when the J-lens reads an object's name on its image tokens, is that the content the model reasons with? The experiments move, remove or add information in the image tokens' residual stream and read the model's answers to questions about the object.

## Models and lenses

| model | layers | width | lens | read layers |
|---|---|---|---|---|
| Qwen3.5-4B | 32 | 2560 | J-lens (Neuronpedia, fitted on 1000 wikitext prompts) | 9, 11, ..., 29 |
| Gemma-4-31B | 60 | 5376 | J-lens and J++ | 18, 23, ..., 58 |

The J-lens of layer L reads a state h as W_U J_L h, with J_L the average Jacobian of the last layer's state with respect to layer L's, fitted on text. A word's lens direction at layer L is J_L^T W_U[t], scaled to length one. The lens subspace of layer L is spanned by the top r right singular directions of W_U J_L, the directions the lens reads most strongly; r is 256 on the 4B (a tenth of the width) and 537 on the 31B. The read layers start at 0.3 of the depth and step by a twelfth of it (`Session.layers`).

## Stimuli

A canvas is a grid of numbered 224-pixel slots, three by two. On Qwen3.5 each slot covers exactly 7 x 7 = 49 image tokens, so the tokens of an object are known. One slot holds a crop of a single object from COCO val2017, from 16 classes with one-token names (dog, cat, bird, horse, car, bus, train, bicycle, airplane, boat, chair, clock, cup, bed, pizza, laptop). Per class the 36 largest single, non-crowd boxes are shuffled once with seed 0; crop set 0 is the first 12 of that order and crop set 1, the fresh crops of the later experiments, the next 12. A crop is kept only if the model names it correctly when shown alone (the crop gate, `stimuli.load_crops`). The same object also appears as its name printed in the slot (the printed word) and as the sentence "Image slot q contains a dog" with no picture (the typed sentence). The natural photos are SCAM (a note naming another object, and a twin where the note is blank) and GQA testdev-balanced, resized so that each side is a multiple of 32 pixels and the longer side is at most 448.

## Questions and scoring

The canvas questions, asked in separate prompts after "Study the scene. It has an image grid with numbered slots.":

| task | question | scored by |
|---|---|---|
| say | What is the object in image slot q? Reply with the noun only. | the first vocabulary word |
| letter | What is the first letter of the name of the object in image slot q? | the first character |
| animal | Rule: if the object in image slot q is an animal, answer 7; otherwise answer 3. | the first character |
| ride | the same rule for "something people ride or drive" | the first character |
| set | the same rule for "a {a}, a {b} or a {c}", a random set of three names per scene | the first character |

The rules answer with a digit, so the answer never contains the name. On the natural photos the LLM judge scores the answers (`src/jspace/judge.py`: Gemini 2.5 Flash, temperature 0, frozen rubrics, raw outputs kept, failed calls never cached), with a regex score kept beside it. Each judged experiment writes a stratified sample of 40 verdicts to label by hand (`*_to_label.csv`; `judge.agreement` gives Cohen's kappa).

## Interventions

All of them act on the residual stream of the image tokens and are built in `src/jspace/hooks.py`.

- **Residual-stream patching**: a donor run sees object X under a neutral prompt with no question; its image-token states are recorded and written into a receiver that sees object Z of the other kind in the same slot and is asked a question. Window patching writes four consecutive layers only and slides the window from the input (layer -1) to the last layer. Subspace patching writes only part of the state (the lens subspace, its complement, or a random subspace of the same rank).
- **Back-patching**: states recorded at later layers written into earlier ones, here from the donor run into the receiver. Restoration writes the receiver's own clean states back in.
- **Directional ablation**: removes a direction (rank 1) or a subspace from the state; mean-ablation puts back its component in the average image-token state instead of zero.
- **Activation steering**: adds a word's lens direction at a dose alpha * sigma_L, where sigma_L is the typical size of a token's own-name component at layer L.

The controls in every experiment: a random direction or subspace of the same rank and norm, another word's direction, the plain unembedding direction W_U[t] (no Jacobian), and the full-state patch as the positive control.

## Statistics

Each estimate comes with a 95% bootstrap interval over scenes (2000 resamples, seed 0; rows that share a scene, image or item are resampled together). Every test had a rule, a threshold and a prediction written in the project's ledger before it ran; the comments in the code cite the ledger by its time stamps, and `summary.json` holds the verdict the frozen rule gives. Misses are kept as misses.

## The experiments

| script | question | design | main estimate and rule |
|---|---|---|---|
| `depth_sweep` | until which depth is the object still named, and still used? | window patching of photo, printed-word and typed-sentence donors; Z's direction steered in per window; the lens direction of the word "animal" at the read layers; `--spectrum` adds letter, set and ride | per question, the horizon is the last window start at which the answer follows the donor on at least half the scenes; L_use and L_say per input type |
| `back_patching` | is the fact still in the late state? | the donor's early layers, the middle layers (between the two horizons), early with the receiver restored in the middle, cross-layer back-patching; a four-frame video donor; a leave-one-out nearest-centroid decoder per layer | "two horizons, in both directions" when the restore arm names Z and classifies X on at least 0.6 and the middle-only arm the reverse |
| `subspace_patching` | does part of the state carry the name without the fact? | the lens subspace, its complement, three random subspaces, J++, a placebo donor that saw the empty canvas, every-layer arms, the typed fact; a joint question; an optional rank sweep | D = P(names X and gives Z's animal answer); D(lens) - D(full) >= 0.4 and lens - random >= 0.3 on naming |
| `name_steering` | does the model use what it sees through the direction that names it? | removal of the own name's direction and of eight controls on the slot's tokens; steering with another object's name at three sites and four doses, the dose picked on 16 calibration scenes, verdicts on the other 48; steering on the empty canvas | rules R0 to R6 (`summary.json`, `verdict`) |
| `gqa_ablation` | is the lens-readable part of a natural photo's tokens necessary? | 48 GQA images disjoint from the pilot's 16, one question per program-length bin; ablation of the lens subspace (256, 64), random, PCA and W_U subspaces, the answer word's own direction, mean-ablation | J - random (r = 256) in log p of the reference: "specifically necessary" when <= -0.5 nats with the interval below 0 |
| `printed_word` | does one word direction decide whether the printed word or the object is named? | a COCO crop with another word printed on it; rank-1 ablation of the word's, the object's, another word's and a random direction; the lens subspace removed on the label's rows or the object's rows; the model's own report of its source | rescue: says-X rises by >= 0.3 under the word's ablation with the controls within 0.05 |
| `scam_notes` | the same on real photos | SCAM photos with a hand-written note; the same ablations; steering the blank-note twin toward the note's word at four doses | rescue as above; induction: says-Y rises with the dose and is >= 0.3 above a random direction at the top dose |

## Reproducibility

Decoding is greedy, every random draw has a fixed seed (the crop shuffle 0, scene draws 1, 11 and 17, random subspaces 100 + layer, random directions 7 + layer, 200 + layer, 900 + layer), and `torch.manual_seed(0)` is set after the model loads, because the PCA control in `gqa_ablation` draws from that generator. What can still change the numbers: the GPU type (floating-point order in the kernels), the CPU threads (the random and lens subspaces are orthonormalized by a QR on the CPU, which rounds differently with another thread count, so `Session` sets the 16 threads of the original runs), the batch size (it depends on GPU memory: 96 prompts and 65,536 tokens per batch from 80 GB up), the flash-linear-attention kernels (Qwen3.5's linear-attention layers use them when installed), and the drawing of the stimuli (Pillow and the DejaVu fonts). The package versions, model, lens and data revisions are pinned (README, "Reproducibility"); each run records them in its `config.json`.

The code in this repository was checked against the code that produced the original runs, on the same GPU type, at small size (4 scenes, 3 GQA images). No full run was repeated.

- `printed_word`, `scam_notes`, `depth_sweep --spectrum`, `back_patching` and `gqa_ablation`: every answer, verdict and number, the summaries and the stimulus sheets are identical.
- `name_steering`: every answer is identical. The log probabilities of the two arms that remove a 256-dimensional subspace differed, by up to 0.15 nats, because that run used 64 CPU threads. With the 16 threads `Session` now sets, the subspace bases are byte-identical to the original code's, and `gqa_ablation`, which uses such subspaces, matched exactly. The experiment was not run again.
- `subspace_patching`: run before that fix, 3 of 360 answers differed, for the same reason. It was not run again.
- The Gemma code path (its model class, and its processor taking one list of pictures per prompt) was not run.

The figures drawn from the original outputs are pixel-identical to the original figures, `scripts/download.py` reproduces the original data and lens files byte for byte, and the pinned models, tokenizers, processors and lenses load with the hub offline.

The original runs read the lens repository's main branch on 13 and 14 October 2026, at commit `9fa267f`, which a later rewrite of that repository's history removed. The Qwen3.5-4B lens at the pinned commit `e166deb` has the same sha256 as the file the runs read; the Gemma-4-31B J-lens and J++ files were last changed on 17 June and 8 October 2026, before the runs. `scripts/download.py` checks every lens file against these hashes (`src/jspace/config.py`).

The judge's verdicts are cached by a hash of the model and the filled rubric. The verdicts of the original runs are in `results/judge_cache.json`, so a reproduced answer identical to the original reuses its verdict and needs no API key; a new answer is sent to Gemini (`GEMINI_API_KEY` in `.env`, see `.env.example`), and without a key the run stops.

## The original runs

The original runs used the project's internal names. For tracing a number back to the ledger:

| this repository | original run | experiment id | date |
|---|---|---|---|
| `printed_word/Qwen3.5-4B` | `run1/Qwen3.5-4B` | A1 | 14 Oct 03:20 |
| `scam_notes/Qwen3.5-4B` | `run1/Qwen3.5-4B` | A2 | 14 Oct 03:22 |
| `name_steering/Qwen3.5-4B` | `run3/Qwen3.5-4B` | G1 | 14 Oct 04:07 |
| `subspace_patching/Qwen3.5-4B` | `run4/Qwen3.5-4B` | F4 | 14 Oct 04:34 |
| `depth_sweep/Qwen3.5-4B` | `run4/Qwen3.5-4B` | D1 | 14 Oct 04:55 |
| `depth_sweep_spectrum/Qwen3.5-4B` | `run5/Qwen3.5-4B` | D1, spectrum | 14 Oct 05:31 |
| `back_patching/Qwen3.5-4B` | `run5/Qwen3.5-4B` | D2 | 14 Oct 06:03 |
| `gqa_ablation/Qwen3.5-4B` | `n1confirm5/Qwen3.5-4B` | N1 | 14 Oct 02:35 |
| `subspace_patching/gemma-4-31B-it` | `run4/gemma-4-31B-it` | F4 | 14 Oct 05:23 |
| `depth_sweep/gemma-4-31B-it` | `run4/gemma-4-31B-it` | D1 | 14 Oct 06:13 |

Two known issues of the original outputs, kept as they were in `results/`:

- `subspace_patching/*/summary.json`, `crops`: the original code recorded the COCO ids of the crop candidates by their position before the crop gate, so a scene whose class lost an earlier crop to the gate lists the wrong crop. The answers are unaffected. This repository's code records the crops actually shown.
- `back_patching/*/summary.json`, `decodable_by_layer`: the decoder's accuracy for the last donor run, the video donor. Its picture-donor values are printed in the original run's log.
