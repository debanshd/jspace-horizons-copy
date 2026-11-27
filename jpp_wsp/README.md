# jpp_wsp: the 50% OLMo 3 7B J++ lens written from the 100% lens, by weight-space programming

The 100% checkpoint's J++ lens is edited into a lens for the 50% checkpoint (`stage1-step725000`) using only the two checkpoints'
weights: no prompts, no forward or backward passes, no fitting. The edits are scored with the J++ post's evals beside the 50%
checkpoint's own fitted J++ lens. Like `jpp_evals/` (PR 18), this runs in the J++ harness's environment and touches nothing in
`src/`, `experiments/` or `results/`.

| file | what |
|---|---|
| `RULE.md`, `RULE2.md` | the two experiments' rules, committed before their edits were computed or scored |
| `edit_lens.py` | experiment 1: least-squares basis transport, J50 = A_L J100 B_l (V1; V0 to V3 with the control and ablations) |
| `edit_lens2.py` | experiment 2: the same with each map constrained to a rotation (P1 to P3) |
| `frozen.json`, `frozen2.json` | sha256 of every edited lens, committed before any score |
| `diagnostics.json`, `diagnostics2.json` | per layer, how much of the weight change the maps explain (from weights alone) |
| `cosine.py`, `cosine.json` | per-layer cosine of every lens with the source and the target (exploratory) |
| `pod.py`, `score_pod2.sh` | the scoring run: a RunPod H200 driven through Jupyter, every lens checked against the frozen hashes |
| `score_pod.sh` | the first scoring attempt, superseded: its pod stalled at start-up and scored nothing |
| `score_files.json`, `inputs/` | the lens files scored (pinned Hugging Face commits) and the 50% fit's correctness files |
| `verdicts.py` | the rules' verdicts with the paired bootstrap; it reproduces the harness's recall exactly |
| `out/`, `RESULTS.md` | the outputs of the scoring run, the verdicts and a summary |

The lenses are in `debanshd/jspace-omni-jpp-runs`, under `wsp/`.

```bash
python jpp_wsp/edit_lens.py  --lens olmo7b_jpp_n64.pt --m100 <Olmo-3-1025-7B@a81bae4> --m50 <@6bfdeb0> --out <dir>   # CPU, ~45 min, 15 GB
python jpp_wsp/edit_lens2.py --lens olmo7b_jpp_n64.pt --m100 <...> --m50 <...> --out <dir>
python jpp_wsp/pod.py create && python jpp_wsp/pod.py start --pr18 <jpp_evals at 3d8ccd6> --urls jpp_wsp/score_files.json
python jpp_wsp/pod.py status; python jpp_wsp/pod.py fetch --dest jpp_wsp/out; python jpp_wsp/pod.py terminate
python jpp_wsp/verdicts.py --out jpp_wsp/out/olmo-3-7b-50pct --stamp100 <olmo7b_jpp_n64_stamp.json> --primary wspV1_jpp_n64 --name verdicts1.json
```
