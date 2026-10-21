#!/usr/bin/env python
"""Write runs/<tag>/REPORT.md from the saved results of every model in the tag.
    venv/bin/python jbound/report.py --tag run1 [--extra run1s]      (extra tags are merged in, later tags win per model/cell)
"""
import argparse, glob, json, os, datetime

RUNS = "/workspace/nayan/runs"


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def ci(t, pct=False):
    try:
        if pct:
            return f"{t[0]:.0%} [{t[1]:.0%}, {t[2]:.0%}]"
        return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    except Exception:
        return "—"


def load(tag):
    out = {}
    for d in sorted(glob.glob(f"{RUNS}/{tag}/*/")):
        m = os.path.basename(d.rstrip("/"))
        res = {}
        for f in glob.glob(f"{d}/e*_results.json") + glob.glob(f"{d}/config.json") + glob.glob(f"{d}/verdicts.json"):
            try:
                res[os.path.basename(f).replace("_results.json", "").replace(".json", "")] = json.load(open(f))
            except Exception as e:
                res[os.path.basename(f)] = {"error": str(e)}
        res["_dir"] = d.rstrip("/")
        res["_figs"] = sorted(os.path.basename(p) for p in glob.glob(f"{d}/figures/*.png"))
        out[m] = res
    return out


def model_section(m, r, rel):
    L = [f"## {m}", ""]
    cfg = r.get("config", {})
    if cfg:
        L.append(f"Config: {cfg.get('model_class')} · lens `{cfg.get('lens_file')}` · read layers {cfg.get('layers')} · BATCH {cfg.get('batch')} · batch check |Δp| {g(cfg, 'batch_check', 'max_dp', default=float('nan')):.4f} · transformers {cfg.get('transformers')} · cells sha {cfg.get('cells_sha')} · {cfg.get('date')}")
        if cfg.get("smoke"):
            L.append("**SMOKE RUN — not evidence.**")
        L.append("")
    v = r.get("verdicts", {})
    if g(v, "cells"):
        L.append("Cell times (s): " + ", ".join(f"{k} {t}" for k, t in v["cells"].items()))
        L.append("")
    b1 = r.get("eB1")
    if b1:
        L += ["### B1 — the bound (lens top-10 at the report, by modality and load)", "",
              f"Rule: bound by modality when K_text − K_picture has its 95% interval above 1; amodal when inside [−1, 1]. Observed text − picture: {b1.get('text_minus_picture')} → **{b1.get('verdict')}**. {b1.get('copying')}.",
              "", "| rendering | seen (k=1 hits/FA) | K (top-10) | K 2nd wording | K either lang. | **K asked how many** | relevance | in J-space by k | absent null by k | use (hits) by k |", "|---|---|---|---|---|---|---|---|---|---|"]
        for rend, row in b1["by_rendering"].items():
            bk = row["by_k"]
            ks = sorted(bk, key=int)
            L.append(f"| {rend} | {'ok' if row['seen'] else 'NO'} ({bk[ks[0]]['hit'][0]:.2f}/{bk[ks[0]]['fa'][0]:.2f}) | {ci(row['capacity']['K'])} | {g(row, 'capacity_alt_wording', 'K', default=[float('nan')])[0]:.1f} | {g(row, 'capacity_either_language', 'K', default=[float('nan')])[0]:.1f} | "
                     f"{g(row, 'capacity_count_question', 'K', default=[float('nan')])[0]:.1f} | {g(b1, 'relevance', rend, default='—').split(':')[0]} | " + " ".join(f"{k}:{bk[k]['present'][0]:.1f}" for k in ks) + " | " + " ".join(f"{bk[k]['absent'][0]:.1f}" for k in ks) + " | " + " ".join(f"{bk[k]['hit'][0]:.2f}" for k in ks) + " |")
        k8 = sorted(b1["by_rendering"][next(iter(b1["by_rendering"]))]["by_k"], key=int)[-1]
        L.append("")
        L.append(f"Threshold sweep at k = {k8} (present / absent): " + "; ".join(f"{rend} top-5 {g(row, 'by_k', k8, 'present_top5', default=['—'])[0]}/{g(row, 'by_k', k8, 'absent_top5', default=['—'])[0]}, top-10 {row['by_k'][k8]['present'][0]:.1f}/{row['by_k'][k8]['absent'][0]:.1f}, top-20 {g(row, 'by_k', k8, 'present_top20', default=['—'])[0]}/{g(row, 'by_k', k8, 'absent_top20', default=['—'])[0]}, null-corrected {ci(g(row, 'by_k', k8, 'present_null_corrected', default=(float('nan'),) * 3))}" for rend, row in b1["by_rendering"].items()))
        L.append("Per-item rank AUROC (present vs absent) at k = " + k8 + ": " + ", ".join(f"{rend} {ci(row['by_k'][k8]['auroc'])}" for rend, row in b1["by_rendering"].items()))
        L.append("Serial position at k = " + k8 + " (share read at the report by place): " + "; ".join(f"{rend} " + " ".join(f"{p}:{v[0]:.2f}" for p, v in row.get("serial_position_k8", {}).items()) for rend, row in b1["by_rendering"].items()))
        L.append("")
    b2 = r.get("eB2")
    if b2 and b2.get("where"):
        L += ["### B2 — the collapse (where the dropped items are; the additive ceiling)", ""]
        w = b2["where"]
        ks = sorted(w, key=int)
        L.append("| k | readable on own patches | anywhere on the card | at the report | absent shape somewhere |")
        L.append("|---|---|---|---|---|")
        for k in ks:
            L.append(f"| {k} | {ci(w[k]['own'], True)} | {ci(w[k]['anywhere'], True)} | {ci(w[k]['report'], True)} | {w[k]['absent_anywhere'][0]:.2f} |")
        L.append(f"\nVerdict: **{b2.get('where_verdict')}**. Additive ceiling: " + "; ".join(f"{rend} → {v.get('verdict')}" for rend, v in b2.get("additive", {}).items()))
        L.append("")
    for key, title in (("eB3", "B3 — carried from the receipt (question-token state), as preregistered"), ("eB3s", "B3s — carried from the scene state (the carry test moved to the scene positions)")):
        b3 = r.get(key)
        if not b3 or not b3.get("by_rendering"):
            continue
        L += [f"### {title}", ""]
        arm = "receipt, every layer" if key == "eB3" else "scene state, every layer"
        L.append(f"| rendering | control (k=1 carry hits / blank d′) | K_carried | at k=8: full / {arm.split(',')[0]} / blank hits | carried items k=8 | read at report k=8 | silent share k=8 | lens→carry AUROC (perm. 95th) | carried content |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for rend, row in b3["by_rendering"].items():
            ks = sorted(row["by_k"], key=int)
            c1, b8 = row["by_k"][ks[0]], row["by_k"][ks[-1]]
            read8 = g(b1, "by_rendering", rend, "by_k", ks[-1], "present", default=[float("nan")])[0] if b1 else float("nan")
            split = g(b3, "split", rend, "verdict", default="—")
            L.append(f"| {rend} | {'ok' if row['control_ok'] else '**FAILED**'} ({c1[arm]['hit'][0]:.2f} / {c1['blank frame']['dprime'][0]:.2f}) | {ci(row['K_carried']['K'])} | {b8['full scene']['hit'][0]:.2f} / {b8[arm]['hit'][0]:.2f} / {b8['blank frame']['hit'][0]:.2f} | {ci(b8['carried_items'])} | {read8:.1f} | {ci(b8['shares']['carried_unread'], True)} | {ci(row['lens_predicts_carry_pooled'])} ({row.get('permutation_95', float('nan')):.2f}) | {split} |")
        if b3.get("verdict"):
            L.append(f"\nVerdict: **{b3['verdict'].get('bound')}**")
        if key == "eB3s":
            for rend, sp in b3.get("split", {}).items():
                k8 = max(int(k) for k in sp["full scene"]) if "full scene" in sp else None
                if k8:
                    k8 = str(k8)
                    L.append(f"- {rend} d′ at k = {k8}: " + ", ".join(f"{n} {sp[n][k8]['dprime'][0]:.2f}" for n in ("full scene", "blank frame", "scene state, read layers", "lens part, read layers", "random part, read layers", "complement, read layers") if n in sp) + f" → {sp['verdict']}")
        L.append("")
    b4 = r.get("eB4")
    if b4 and b4.get("by_k"):
        L += ["### B4 — photographs (COCO val2017, POPE negatives)", ""]
        ks = sorted(b4["by_k"], key=int)
        L.append(f"Seen {'ok' if b4['seen'] else 'NO'}; carried control (receipt) {'ok' if b4['control_ok'] else '**FAILED**'}; K_lens {ci(b4['capacity']['K'])}; in J-space by k: " + " ".join(f"{k}:{b4['by_k'][k]['entry']['present'][0]:.1f}" for k in ks) + "; absent null: " + " ".join(f"{b4['by_k'][k]['entry']['absent'][0]:.1f}" for k in ks))
        L.append("False alarms (full scene) by negative type: " + ", ".join(f"{kd} {v['full scene']['fa'][0]:.2f}" for kd, v in b4["by_negative"].items()) + f". Verdict: **{b4['verdict']}**")
        L.append("")
    b7 = r.get("eB7")
    if b7 and b7.get("by_rendering"):
        L += ["### B7 — omni (one vocabulary through text, speech, picture, video, sound)", "",
              "| rendering | lexical | seen (k=1 hits/FA) | carry control | K_said | K_carried (receipt) | said k=8 (null) | carried k=8 | silent share k=8 |", "|---|---|---|---|---|---|---|---|---|"]
        for rend, row in b7["by_rendering"].items():
            ks = sorted(row["by_k"], key=int)
            c1, b8 = row["by_k"][ks[0]], row["by_k"][ks[-1]]
            L.append(f"| {rend} | {'yes' if row['lexical'] else 'no'} | {'ok' if row['seen'] else 'NO'} ({c1['full scene']['hit'][0]:.2f}/{c1['full scene']['fa'][0]:.2f}) | {'ok' if row['control_ok'] else '**FAILED**'} | {ci(row['K_said']['K'])} | {ci(row['K_carried']['K'])} | {b8['said'][0]:.1f} ({b8['said_absent'][0]:.1f}) | {b8['carried_items'][0]:.1f} | {ci(b8['silent'], True)} |")
        L.append(f"\nVerdict: **{b7.get('verdict')}**")
        L.append("")
    b6 = r.get("eB6")
    if b6 and b6.get("fits"):
        L += ["### B6 — the frontier (fitted plateaus)", ""]
        for mm, fits in b6["fits"].items():
            L.append(f"- {mm}: " + "; ".join(f"{rend} said K {f['K_said']:.1f} (κ {f['kappa_said']:.1f}), used K {f['K_used']:.1f} (κ {f['kappa_used']:.1f}), ratio {f['ratio']:.2f}" for rend, f in fits.items()))
        L.append("")
    if r["_figs"]:
        L.append("Figures: " + ", ".join(f"[{f}]({os.path.relpath(r['_dir'], rel)}/figures/{f})" for f in r["_figs"]))
        L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()
    models = load(args.tag)
    for t in args.extra:
        for m, r in load(t).items():
            base = models.setdefault(m, {"_dir": r["_dir"], "_figs": []})
            for k, v in r.items():
                if k not in ("_dir", "_figs"):
                    base[k] = v
            base["_figs"] = sorted(set(base["_figs"]) | set(r["_figs"]))
            base["_dir"] = r["_dir"]
    out = f"{RUNS}/{args.tag}/REPORT.md"
    L = [f"# Articulation bound — results for tag `{args.tag}`" + (f" (+ {', '.join(args.extra)})" if args.extra else ""), "",
         f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}. Rules and thresholds as in notes.md; implementation ledger in jbound/CHANGES.md; design and findings in PLAN.md.", ""]
    # cross-model table
    L += ["## Across models", "", "| model | K_said text / zh / picture (top-10 at the report) | K asked-how-many | relevance | carried (scene state) text / picture, control | silent share k=8 picture | carried content (split) | B2 where | B7 verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for m, r in models.items():
        b1, b3s, b2, b7 = r.get("eB1", {}), r.get("eB3s", {}), r.get("eB2", {}), r.get("eB7", {})
        K = g(b1, "K", default={})
        Kc = g(b1, "K_count_question", default={})
        rel = g(b1, "relevance", default={})
        car = " / ".join(f"{g(b3s, 'by_rendering', rend, 'K_carried', 'K', default=[float('nan')])[0]:.1f}{'' if g(b3s, 'by_rendering', rend, 'control_ok') else '†'}" for rend in ("text", "picture")) if b3s else "—"
        L.append(f"| {m} | {K.get('text', float('nan')):.1f} / {K.get('text_zh', float('nan')):.1f} / {K.get('picture', float('nan')):.1f} | " + (" / ".join(f"{v:.1f}" for v in Kc.values()) if Kc else "—") + f" | {', '.join(v.split(':')[0] for v in rel.values()) if rel else '—'} | {car} | {ci(g(b3s, 'verdict', 'picture_silent_k8', default=(float('nan'),) * 3), True)} | {', '.join(f'{k} {v}' for k, v in g(b3s, 'verdict', 'split', default={}).items()) or '—'} | {(g(b2, 'where_verdict', default='—') or '—').split(':')[0]} | {(g(b7, 'verdict', default='—') or '—').split(':')[0]} |")
    L.append("\n† control failed (uninterpretable)\n")
    for m, r in models.items():
        L += model_section(m, r, f"{RUNS}/{args.tag}")
    open(out, "w").write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
