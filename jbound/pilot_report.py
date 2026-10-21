#!/usr/bin/env python
"""Write runs/<tag>/<model>/pilot_report.md from eE1/eE2/eE2P results and the e*.csv files (PILOT.md rules).
    venv/bin/python jbound/pilot_report.py --tag epilot1 --model Qwen3.5-4B
"""
import argparse, json, os, datetime
import pandas as pd

RUNS = "/workspace/nayan/runs"


def ci(t, pct=False):
    try:
        return (f"{t[0]:.0%} [{t[1]:.0%}, {t[2]:.0%}]" if pct else f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]")
    except Exception:
        return "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    d = f"{RUNS}/{a.tag}/{a.model}"
    cfg = json.load(open(f"{d}/config.json")) if os.path.exists(f"{d}/config.json") else {}
    L = [f"# Pilot report — {a.model} — tag `{a.tag}`", "", f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}. Protocol: PILOT.md (frozen before the run). Ledger: jbound/CHANGES.md.",
         f"Model revision {cfg.get('model_revision')} · lens `{cfg.get('lens_file')}` · J++ loaded: {cfg.get('jpp_loaded')} · read layers {cfg.get('layers')} · batch check |Δp| {cfg.get('batch_check', {}).get('max_dp')} · smoke: {cfg.get('smoke')}", ""]
    if cfg.get("smoke"):
        L.append("**SMOKE RUN (4 scenes): machinery check only, not evidence.**\n")
    decisions = []
    e1 = json.load(open(f"{d}/eE1_results.json")) if os.path.exists(f"{d}/eE1_results.json") else None
    if e1:
        L += ["## E1 — does relevance change what J-space holds?", "", f"{e1['n_scenes']} scenes, k = 4, conditions A (one position), B (any position), C (joint). Readout = J-lens top-10 at any read layer; logit lens and J++ beside it.", "",
              "| rendering | gate: acc A (≥0.85 img / ≥0.95 txt) | acc B | acc C | target @decision A / B | **other @decision A / B / C** | absent @decision A | other on own tokens A / B (logit lens) | P1 other B−A | P2 target B−A | verdict |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for rend, v in e1["by_rendering"].items():
            c = v["conditions"]
            L.append(f"| {rend} | {c['A']['accuracy'][0]:.2f} {'ok' if v['seen'] else '**FAIL**'} | {c['B']['accuracy'][0]:.2f} | {c['C']['accuracy'][0]:.2f} | {c['A']['target_dec'][0]:.2f} / {c['B']['target_dec'][0]:.2f} | **{c['A']['other_dec'][0]:.2f} / {c['B']['other_dec'][0]:.2f} / {c['C']['other_dec'][0]:.2f}** | {c['A']['absent_dec'][0]:.2f} | {c['A']['other_src'][0]:.2f} / {c['B']['other_src'][0]:.2f} ({c['A']['other_src_logit'][0]:.2f}) | {ci(v['P1_other_B_minus_A'])} | {ci(v['P2_target_B_minus_A'])} | {v['verdict']} |")
            decisions.append(f"E1 {rend}: {v['verdict']}")
        L.append("")
    e2 = json.load(open(f"{d}/eE2_results.json")) if os.path.exists(f"{d}/eE2_results.json") else None
    if e2:
        L += ["## E2 — cross-modal competition", "", f"{e2['n_scenes']} scenes; arms V0 (image only), V (image + distractor list), T0, T, VT (joint), R (redundant list). Interference I = paired accuracy(alone) − accuracy(with the other source).", "",
              "| k | acc V0 / V / T0 / T / VT / R | I text→vision | I vision→text | redundancy R−V | visual target @decision V0 / V / VT | on its own slot V0 / V / VT |", "|---|---|---|---|---|---|---|"]
        for k, v in e2["by_k"].items():
            acc = v["accuracy"]
            L.append(f"| {k} | {' / '.join(f'{acc[x][0]:.2f}' for x in ('V0', 'V', 'T0', 'T', 'VT', 'R'))} | {ci(v['I_text_on_vision'])} | {ci(v['I_vision_on_text'])} | {ci(v['redundancy'])} | {v['vis_dec']['V0'][0]:.2f} / {v['vis_dec']['V'][0]:.2f} / {v['vis_dec']['VT'][0]:.2f} | {v['vis_src']['V0'][0]:.2f} / {v['vis_src']['V'][0]:.2f} / {v['vis_src']['VT'][0]:.2f} |")
        L.append(f"\nVerdict (k = {list(e2['by_k'])[-1]}): **{e2['verdict']}**\n")
        decisions.append(f"E2: {e2['verdict']}")
    e2p = json.load(open(f"{d}/eE2P_results.json")) if os.path.exists(f"{d}/eE2P_results.json") else None
    if e2p:
        L += ["## E2P — causal interchange (one image slot's state swapped from a donor scene)", "", f"{e2p['n_pairs']} donor/receiver pairs. Δp(donor object) on the single-fact question; joint question: Δp(yes) in the predicted direction.", "",
              "| k | positive control: flips / Δp full every layer (≥0.8, ≥0.5) | full read layers | **J part** | random part | complement | non-target slot (≤0.05) | joint: full / J / random | verdict |", "|---|---|---|---|---|---|---|---|---|"]
        for k, v in e2p["by_k"].items():
            e, ev = v["effects"], v.get("effects_VT", {})
            g = lambda n: e[n]["dp_donor"]
            L.append(f"| {k} | {e['full, every layer']['flipped']:.2f} / {ci(g('full, every layer'))} {'ok' if v['control_ok'] else '**FAIL**'} | {ci(g('full, read layers'))} | **{ci(g('J part, read layers'))}** | {ci(g('random part, read layers'))} | {ci(g('complement, read layers'))} | {ci(g('non-target slot, read layers'))} | "
                     f"{ci(ev.get('full, every layer', (float('nan'),) * 3))} / {ci(ev.get('J part, read layers', (float('nan'),) * 3))} / {ci(ev.get('random part, read layers', (float('nan'),) * 3))} | {v['verdict']} |")
            decisions.append(f"E2P k={k}: {v['verdict']}")
        L.append("")
    L += ["## Decision rule (PILOT.md)", ""]
    L += [f"- {x}" for x in decisions]
    L += ["", "Next stage: replicate E2P on Qwen3.6-27B (with J++) only if the 4B positive control passed; E3 (Gemma audio gates) only after E1/E2 are interpretable; no scaling fit before a stable phenomenon.", "",
          "Files: e1_relevance.csv, e2_competition.csv, e2_patching.csv, eE1/eE2/eE2P_results.json, figures/figE1_relevance.png, figE2_competition.png, figE2_patching.png"]
    open(f"{d}/pilot_report.md", "w").write("\n".join(L))
    print("wrote", f"{d}/pilot_report.md")


if __name__ == "__main__":
    main()
