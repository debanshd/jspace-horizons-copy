#!/usr/bin/env python
"""The launcher's checks between the audio steps (scripts/audio_sweep.sh): exit 0 only when a step's gates pass.

    uv run scripts/audio_gate.py clean runs/audio_clean/gemma-4-E2B-it/summary.json
    uv run scripts/audio_gate.py pilot runs/audio_sweep_pilot/gemma-4-E2B-it/summary.json
    uv run scripts/audio_gate.py lens runs/check_lens/gemma-4-12B-it/summary.json

clean: audio_clean's ready (at least 5 gated scenes; on them, in both audio orders, receivers and donors pass the say >= 0.90
and animal >= 0.85 gates). pilot: the patch-only pilot's controls_pass (in every input: the clean say and animal gates, the self
patch, the full donor patch) and the donors' clean say and animal answers under the receiver's question at the same gates. A
counterpart input of audio_sweep_static (q_frozen, q_typed) whose receivers or donors fail the clean say or animal gate is not
perceived: as audio_sweep_static's rule (written before its runs) and video_sweep's say, it is untestable, reported as a
result, and does not stop the chain; one that is perceived must pass every control. The spoken input always must.
lens: scripts/check_lens.py's passes (a lens from outside Neuronpedia, against the reference model's).
Repeating the same answers twice is checked apart (scripts/compare_media.py) and is not enough on its own.
"""

import json
import sys
from pathlib import Path

GATE = {"say": 0.90, "animal": 0.85}  # video_sweep.GATE
STATIC = ("q_frozen", "q_typed")  # audio_sweep_static's counterpart inputs (audio_sweep.STATIC)


def clean(s):
    print(f"gated scenes (both orders): {len(s['gated_scenes'])}")
    for inp, rc in s["recognition"].items():
        print(f"  {inp}: named {rc['named']} of {s['n_recordings']} exactly; missed {[(m['label'], m['answer'], m['scenes_lost']) for m in rc['missed'].values()]}")
    for inp, by in s["clean"].items():
        cells = []
        for t, roles in by.items():
            cells.append(f"{t} " + "/".join("-" if v["gated"] is None else f"{v['gated'][0]:.2f}" for v in roles.values()))
        print(f"  {inp} clean on gated scenes (receiver/donor): " + "  ".join(cells))
    return bool(s["ready"])


def perceived(s, inp):
    """the input's clean say and animal answers pass the patching gate, for the receivers and for the donors"""
    return all(s["patch"]["gate"][inp][t]["passes"] for t in GATE) and all(s["donor_clean"][inp][t] >= g for t, g in GATE.items())


def pilot(s):
    """every input's controls pass and its donors are named; a counterpart input (STATIC) the model does not perceive is
    untestable instead, as audio_sweep_static's rule and video_sweep's say, and does not stop the chain"""
    ok, cp = True, s.get("controls_pass", {})
    for inp, c in s["patch"]["controls"].items():
        g = {t: round(v["clean"], 2) for t, v in s["patch"]["gate"][inp].items()}
        d = {t: round(v, 2) for t, v in s["donor_clean"][inp].items()}
        print(f"  {inp}: clean {g}; donors under the receiver's question {d}; self patch identical {c['self_patch_same']:.2f}; full donor say follows {c['every_layer_say_follows_donor'][0]:.2f}")
        if bool(cp.get(inp)) and all(s["donor_clean"][inp][t] >= v for t, v in GATE.items()):
            continue
        if inp in STATIC and not perceived(s, inp):
            print(f"  {inp}: not perceived (its clean gates fail): untestable, reported as a result; the chain goes on")
            continue
        print(f"  {inp}: its controls fail")
        ok = False
    print(f"  answers at the token limit: {s['hit_limit']:.3f}")
    return ok


def lens(s):
    print(f"  lens {s['lens']['repo']}/{s['lens']['file']} ({s['lens']['n_prompts']} prompts): agreement with the model by layer {s['jlens']}")
    if s.get("rule") == "qwen":
        print(f"  logit lens {s['logit']} (reported, decides nothing); the Qwen rule's bar by layer {s['calibration']['bar']}")
    else:
        print(f"  logit lens {s['logit']}; deepest {s['deepest']:.3f} against the reference's {s['reference']['deepest']:.3f}")
    return bool(s["passes"])


def main():
    step, path = sys.argv[1], Path(sys.argv[2])
    if not path.exists():
        raise SystemExit(f"{path} not found")
    s = json.loads(path.read_text())
    ok = {"clean": clean, "pilot": pilot, "lens": lens}[step](s)
    print(f"{step}: {'passes' if ok else 'FAILS'} ({path})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
