# Atlas: everything that was tried, by model, dataset and method

One page on the breadth of the program, so a reader can see what ran where before reading a single result. Every row of RECORD.md (133 rows on 21 October 2026, after the 14 October rows) has a run folder under results/, a rule written before the run, and a verdict; the examples in results/examples show one scene of each headline result with its answers verbatim.

## The models

| model | layers | width | what ran on it |
|---|---|---|---|
| Qwen3.5-0.8B | 24 | 1024 | the 18-question knockout battery, the lens onset, the width law (held out) |
| Qwen3.5-2B | 24 | 2048 | everything: the patching battery, knockouts by layer, head and query token, head-output patching, the interchange in four directions and on a second crop set, the sixteen-way identity, the source report, both conflict designs, load, natural photos (GQA, CountBench), the lens onset, the width law |
| Qwen3.5-4B | 32 | 2560 | the main model: all of the above, the answer-held source tests (row 115), the matched-disruption test (129), the order counterbalance (130), the replication on crop set 0 (133), the 14 October subspace patching, steering, back-patching, restoration, printed words, SCAM notes, GQA ablation |
| Qwen3.5-9B | 32 | 4096 | the six-question knockout (row 131) |
| Qwen3.5-27B | 64 | 5120 | the six-question knockout (120), the interchange and the source report with layers chosen by rule (121, 122), the width law and GQA ablation of 14 October |
| Gemma-4-E4B | 42 | 2560 | the knockout at every layer (108), the patching battery (114), the interchange (127), the answer-held source test by knockout (132), the depth sweep of 14 October |
| Gemma-4-31B | 60 | 5376 | three knockout grids (123, 124, 128), the depth sweep, subspace patching and width law of 14 October |

## The datasets

| dataset | what it is | where it is used |
|---|---|---|
| COCO canvases, crop set 1 | six-slot canvases with one COCO crop of 16 classes, 64 scenes, the crops gated by each model | every 21 October experiment |
| COCO canvases, crop set 0 | the original crops, new scenes | the interchange (113) and the pathway record (133) replicated |
| printed words | the object's name printed in the slot | the depth sweep (14 October), the printed-word test |
| typed sentences | "Image slot q contains a dog." with no picture | the depth sweep, the typed knockouts (95, 98, 99), every interchange and source-report donor |
| canvases with distractors | two or five other objects in the other slots | load (107) |
| GQA testdev | 48 photos, three questions each, judge-scored | knockout and blank-state patching (87 to 91, 96), the 14 October ablation and width law |
| CountBench | 491 photos, "how many X", 400 kept | knockout by count group (88, 90) |
| SCAM | photos with a hand-written note | the 14 October note test and source report (ex7, ex8) |
| MathVision | dropped: 2 of 60 right in the direct-answer setting | row 84 |

## The methods, and what each found

| method | script | rows | finding |
|---|---|---|---|
| single-layer residual patching with log-probabilities | experiments/read_battery.py | 80, 86, 96, 107, 114 | two stations: decisions at one depth, copies of the word form at another, in 0.8B, 2B, 4B and E4B |
| attention knockout by layer | experiments/attention_knockout.py | 82, 108, 120, 123, 124, 128, 131 | the same stations by a second method; spread reads at 27B and 31B; the access profiles across seven models |
| attention knockout by head | attention_knockout.py --heads | 85, 92, 99 | two heads carry the decisions on the 2B, one head serves both stations |
| attention knockout by query token | experiments/query_knockout.py | 109, 116, 118, 119 | a decision reads the picture at the question's tokens, never at the answer position; the name at the answer position |
| attention knockout on photos | experiments/gqa_knockout.py, bench_knockout.py | 87, 88, 90 | the order holds on GQA; counting reads deeper the more there is to count |
| blank-state patching on photos | gqa_knockout.py --mode patch | 91 | decisions back by layer 11, names by 19, attributes by 15 to 23 |
| head-output patching | experiments/head_patch.py | 97, 100, 101 | necessity and sufficiency sit at different stations |
| cross-modal interchange | experiments/cross_modal_patch.py | 101, 106, 111, 113, 121, 127 | a word's read written into the picture's run moves the decision (2B, 4B, E4B) and the identity (2B); the name on the 2B only |
| alignment cosines by position | cross_modal_patch.py --alignment-only | 101, 127 | the same vector for a dog seen and read at the name station; the object deposited on the rule's test word |
| source report after interchange | experiments/source_report.py | 103, 105, 112, 122 | given only the name station's output the model says it saw what it read; the channel is bound to the delivered object |
| two sources in conflict | experiments/source_conflict.py | 110 | the report follows the channel that carried the answer, the other present and ignored |
| the same answer from either source | source_conflict.py --congruent | 115, 117, 132, 133 | the report follows the pathway with the answer held, on the 4B and E4B; no single head carries the evidence |
| matched-disruption source test | experiments/source_cause.py | 129, 133 | nothing blocked: the note pathway's heads move the report on 64 of 64, matched noise on 1 |
| order counterbalance | source_conflict.py --order channel-first | 130 | the record exists only in the name-first order |
| lens onset, J and J++ lenses, token by token | experiments/lens_onset.py | 81 | the name is visible at every depth; the 14 October onset claim withdrawn |
| subspace patching, steering, directional ablation, back-patching, restoration, width law, clutter | jbound/ (14 October) | the 14 October rows | a state can be named without being used; the name direction's cost falls as one over the width |

## Where to look

- results/oct21/<run>/<model>/: config, per-scene CSV (gzipped), summary, stimuli, crops, for every run above (80 folders, ten of them four-scene smokes).
- results/tables/: the same results as flat CSVs (every summary value with its interval; knockout costs and stations; battery depths and curves; access profiles).
- results/figures/: every figure of the 21 October write-up, as PNG, PDF and SVG, each from a summary or a CSV (scripts/fig_battery.py).
- results/oct14_figures/<run>/<model>/figures/: every figure and stimulus sheet of the 13 and 14 October runs.
- results/examples/: ex1 to ex17, one scene each with the model's answers verbatim and, from ex9 on, each cell's count over the 64 scenes.
- results/samples/: every tile of the 2B and 4B crop banks, every receiver canvas of the 64 scenes, the 48 GQA photos, the 400 CountBench photos and the SCAM photos at the size the models received them, with their questions.
- docs/RECORD.md: the ledger; docs/REPORT.md: the write-up; share/council_full_scoring_21oct.md: the gate-by-gate scoring.
