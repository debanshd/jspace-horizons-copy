"""Audio sweep, question first: audio_sweep on the q_audio order alone, the order in which both audio models pass the clean gates.

Why. audio_clean (PR #11) stopped audio_sweep before any patching: its rule requires the clean answers to pass in both prompt
orders, and with the audio first both Gemma models answer the 7-or-3 rules wrongly (animal 0.75 and 0.73 against 0.85). With the
question first (Gemma-4's card's order) both pass every gate: say 1.00, animal 1.00, set 0.98 and 1.00, ride 0.93 and 1.00. This
run patches in that order only, so it measures the name and rule horizons on spoken input as video_sweep2 does on video. Running
one order makes it a new experiment with its own run name; audio_sweep and its two-order rule stand as they are, and
experiments/audio_rules.py asks why the audio-first order fails.

What is audio_sweep's, unchanged and imported: everything (the 64-scene schedule, the donor and self states captured under each
row's own question with the alignment check, the patching arms, media_steer's interventions, strict.py's whole-answer scoring of
first answers with hit_limit, dclean rows, controls_pass, the summaries) except the two points below.

What changes (written before the run):
- Inputs: q_audio only (the question, then the audio). There is no order comparison: order_compare reads "untestable".
- Schedule: from the audio_clean run (--gate-run), audio_clean.order_ready for q_audio: the scenes whose receiver and donor
  recordings are named exactly with the question first. The run stops unless at least 5 such scenes exist and, on them, the
  receivers and the donors (under the receiver's question) pass say >= 0.90 and animal >= 0.85 in that order. The check and its
  numbers are saved in scenes.json and summary.json under gate.
The rule is otherwise video_sweep's, as audio_sweep applies it: the patching gate, the machinery check (self patch >= 0.95), the
positive control (every-layer donor patch, say follows the donor >= 0.9), the horizons, the cut and verdicts, P1 to P3, and
media_steer's frozen rule and predictions 1 to 6 for the other interventions. Recordings recur across scenes (15 recordings, one
voice), so intervals bootstrapped over scenes are narrower than intervals over recordings would be; any write-up says so.
Models: Gemma-4-E2B and E4B, and Gemma-4-12B (added 24 Nov, with the lens config.py pins for it) once scripts/check_lens.py
passes for its lens; its own audio_clean run decides its question-first schedule like the others'. Qwen2.5-Omni-3B and 7B and
Qwen3-Omni-30B-A3B (added 25 Nov, with this project's lenses, PR #13): likewise, each once check_lens passes for its lens.

    uv run experiments/audio_sweep_qfirst.py --model google/gemma-4-E4B-it --parts patch --n 5 --run audio_qfirst_pilot
    uv run experiments/audio_sweep_qfirst.py --model google/gemma-4-E4B-it --run audio_qfirst

Outputs in runs/<run>/<model>/: audio_sweep's (audio_sweep.csv, summary.json, scenes.json, stimuli.png, gpu.csv, config.json,
run.log).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_clean as ac  # noqa: E402  the one-order readiness
import audio_sweep as sw  # noqa: E402  the sweep

from jspace import spoken  # noqa: E402

ORDER = "q_audio"


def q_first(args, gate, gate_dir):
    """the schedule: audio_clean.order_ready for q_audio on the gate run's answers; stops unless ready"""
    csv = gate_dir / "audio_clean.csv"
    if not csv.exists():
        raise SystemExit(f"{csv} not found: run audio_clean for {args.model} first")
    S = spoken.scenes(args.data_dir)
    schedule = [(r["scene_id"], r["receiver_audio_id"], r["donor_audio_id"]) for r in S]
    ok = ac.order_ready(pd.read_csv(csv), schedule, spoken.recordings(S), ORDER)
    print(f"{ORDER}: {ok['named']} recordings named, {len(ok['gated_scenes'])} scenes; clean {ok['clean']}; ready {ok['ready']}")
    if not ok["ready"]:
        raise SystemExit(f"{csv}: the clean answers with the question first do not pass the gates; nothing to patch")
    sc = ok["gated_scenes"]
    return (sc[: args.n] if args.n else sc), dict(order=ORDER, order_ready=ok)


if __name__ == "__main__":
    sw.main(experiment="audio_sweep_qfirst", doc=__doc__, schedule=q_first, default_inputs=ORDER)
