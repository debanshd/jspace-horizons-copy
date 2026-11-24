#!/usr/bin/env python
"""The launcher's checks between the audio steps (scripts/audio_sweep.sh): exit 0 only when a step's gates pass.

    uv run scripts/audio_gate.py clean runs/audio_clean/gemma-4-E2B-it/summary.json
    uv run scripts/audio_gate.py pilot runs/audio_sweep_pilot/gemma-4-E2B-it/summary.json
    uv run scripts/audio_gate.py lens runs/check_lens/gemma-4-12B-it/summary.json

clean: audio_clean's ready (at least 5 gated scenes; on them, in both audio orders, receivers and donors pass the say >= 0.90
and animal >= 0.85 gates). pilot: the patch-only pilot's controls_pass (in every input: the clean say and animal gates, the self
patch, the full donor patch) and the donors' clean say and animal answers under the receiver's question at the same gates.
lens: scripts/check_lens.py's passes (a lens from outside Neuronpedia, against the reference model's).
Repeating the same answers twice is checked apart (scripts/compare_media.py) and is not enough on its own.
"""

import json
import sys
from pathlib import Path

GATE = {"say": 0.90, "animal": 0.85}  # video_sweep.GATE


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


def pilot(s):
    ok = bool(s.get("controls_pass", {}).get("all"))
    for inp, c in s["patch"]["controls"].items():
        g = {t: round(v["clean"], 2) for t, v in s["patch"]["gate"][inp].items()}
        print(f"  {inp}: clean {g}; self patch identical {c['self_patch_same']:.2f}; full donor say follows {c['every_layer_say_follows_donor'][0]:.2f}")
    for inp, d in s["donor_clean"].items():
        print(f"  {inp}: donor clean under the receiver's question { ({t: round(v, 2) for t, v in d.items()}) }")
        ok = ok and all(d[t] >= g for t, g in GATE.items())
    print(f"  answers at the token limit: {s['hit_limit']:.3f}")
    return ok


def lens(s):
    print(f"  lens {s['lens']['repo']}/{s['lens']['file']} ({s['lens']['n_prompts']} prompts): agreement with the model by layer {s['jlens']}")
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
