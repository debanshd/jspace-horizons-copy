"""Diagnostic for B3's carry machinery on k = 1 scenes: does the patch hook work at all, and where does the model read the item from?

Arms on the same 6 k = 1 scenes, picture and text rendering, second turn "Is there a {shape}?" for the present shape and 2 absent ones:
  full scene        the real scene, no patch (ceiling)
  blank             the blank frame, no patch (floor)
  all positions     every first-prompt position patched from the real run, every layer (must match the full scene: the hook check)
  scene positions   only the scene tokens patched (image pads / list tokens): is the item read from the scene tokens?
  receipt           only the question tokens + report position patched (B3's arm)
  receipt, report   the report position alone
Prints hits per arm. Run: jbound/jb-diag  (or: venv/bin/python jbound/diag_carry.py)
"""
import os, sys, re, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as R

ns = {"__name__": "__diag__", "MODEL_NAME": os.environ.get("DIAG_MODEL", "Qwen/Qwen3.5-4B"), "RUN_TAG": "diag_carry", "RUN_ROOT": "/workspace/nayan/runs", "LENS_REVISION": None, "BATCH_OVERRIDE": 0, "B_OTHER_PATHS": [], "SMOKE": True}
exec(compile(open(R.SETUP).read(), R.SETUP, "exec"), ns)
cells = R.split_cells(open(R.CELLS).read())
exec(compile(cells["M0"], R.CELLS + ":M0", "exec"), ns)                    # with the preflight
for dep in ("B1", "B3"):
    ns["save"] = lambda *a, **k: None
    exec(compile(R.definitions(cells[dep]), R.CELLS + ":" + dep, "exec"), ns)
g = ns
torch, pd = g["torch"], g["pd"]
B_ASK, B_OK, B_YESNO, B_SHAPES = g["B_ASK"], g["B_OK"], g["B_YESNO"], g["B_SHAPES"]
chat, mm_chat, mm_forward, ws_probs, YES_IDS, NO_IDS, tokenizer = g["chat"], g["mm_chat"], g["mm_forward"], g["ws_probs"], g["YES_IDS"], g["NO_IDS"], g["tokenizer"]
b_receipt_hooks, b_receipt_len, n_tokens, b_card, B_COLOURS = g["b_receipt_hooks"], g["b_receipt_len"], g["n_tokens"], g["b_card"], g["B_COLOURS"]
Image = g["Image"]
FINAL, MM_W, MM_H, MM_VS, MM_VE = g["FINAL"], g["MM_W"], g["MM_H"], g["MM_VS"], g["MM_VE"]
B_ALL = list(range(FINAL + 1))
rng = np.random.default_rng(3)
shapes = ["star", "circle", "triangle", "heart", "square", "diamond"]
colours = ["red", "blue", "green", "purple", "orange", "pink"]
B_ASSIST = chat([("user", "x")])[chat([("user", "x")]).rindex("<|im_start|>assistant"):]


def turn2(prompt, shape):
    return prompt + B_OK + "<|im_end|>\n<|im_start|>user\n" + B_YESNO.format(shape=shape) + "<|im_end|>\n" + B_ASSIST


def arms(rend):
    if rend == "text":
        real = [chat([("user", f"Here is a list: a {c} {s}.\n{B_ASK}")]) for s, c in zip(shapes, colours)]
        blank = [chat([("user", f"Here is a list: a plain item.\n{B_ASK}")]) for _ in shapes]
        imgs, bimgs = None, None
    else:
        real = [mm_chat([("user", B_ASK)]) for _ in shapes]
        blank = list(real)
        imgs = [b_card([(s, c, 5)], np.random.default_rng(i))[0] for i, (s, c) in enumerate(zip(shapes, colours))]
        bimgs = [Image.new("RGB", (MM_W, MM_H), (238, 238, 238))] * len(shapes)
    ids_r = [tokenizer.encode(t, add_special_tokens=False) for t in real]
    ids_b = [tokenizer.encode(t, add_special_tokens=False) for t in blank]
    if rend == "text":
        assert all(len(a) == len(b) for a, b in zip(ids_r, ids_b)), [(len(a), len(b)) for a, b in zip(ids_r, ids_b)]
    nrec = b_receipt_len(real[0])
    # every position of the first prompt, from the real run (text) or the real picture run; the encoded length with pictures is read from the processor
    if rend == "text":
        n_all = len(ids_r[0])
    else:
        n_all = len(g["mm_encode"](real[:1], imgs[:1])["input_ids"][0])
    _, RR = mm_forward(real, imgs, B_ALL, where=lambda ids, n=n_all: list(range(len(ids) - n, len(ids))))
    REC = {L: RR[L].half().numpy() for L in B_ALL}
    print(f"{rend}: first prompt {n_all} tokens, receipt {nrec} tokens, captured {REC[0].shape}")
    # scene positions: text = tokens between 'Here is a list:' and the question; picture = the image pads
    if rend == "text":
        pre = len(tokenizer.encode(real[0][:real[0].index("Here is a list:")], add_special_tokens=False))
        scene_slice = (pre, n_all - nrec)
    else:
        row = g["mm_encode"](real[:1], imgs[:1])["input_ids"][0].tolist()
        scene_slice = (row.index(MM_VS) + 1, row.index(MM_VE))
    Q = pd.DataFrame([dict(scene=i, shape=s2, present=s2 == s) for i, s in enumerate(shapes) for s2 in [s] + [x for x in shapes if x != s][:2]])
    t_full = [turn2(real[i], s) for i, s in zip(Q.scene, Q["shape"])]
    t_blank = [turn2(blank[i], s) for i, s in zip(Q.scene, Q["shape"])]
    na = np.array([n_tokens(t) - n_tokens(blank[i]) for t, i in zip(t_blank, Q.scene)])
    qi = [imgs[i] for i in Q.scene] if imgs else None
    qb = [bimgs[i] for i in Q.scene] if bimgs else None

    def sub(REC, lo, hi):
        """a receipt that covers positions lo..hi of the first prompt, as the last (n_all - lo) positions with the ones after hi left unpatched by masking to the blank's own value"""
        return {L: REC[L][:, lo:hi] for L in B_ALL}

    out = {}
    for name, hooks in (("full scene", None), ("blank", None),
                        ("all positions", b_receipt_hooks(B_ALL, REC, Q.scene.values, na, n_all)),
                        ("receipt (B3 arm)", b_receipt_hooks(B_ALL, sub(REC, n_all - nrec, n_all), Q.scene.values, na, nrec)),
                        ("report position only", b_receipt_hooks(B_ALL, sub(REC, n_all - 1, n_all), Q.scene.values, na, 1)),
                        ("scene positions only", b_receipt_hooks(B_ALL, sub(REC, scene_slice[0], scene_slice[1]), Q.scene.values, na + (n_all - scene_slice[1]), scene_slice[1] - scene_slice[0]))):
        texts, media = (t_full, qi) if name == "full scene" else (t_blank, qb)
        P, mass = ws_probs(texts, media, [YES_IDS, NO_IDS], hooks=hooks)
        p = P[:, 0] / np.maximum(P.sum(1), 1e-9)
        pr = Q.present.values
        out[name] = (float((p[pr] >= 0.5).mean()), float((p[~pr] >= 0.5).mean()), float(mass.mean()))
        print(f"  {name:22s} hits {out[name][0]:.2f}  false alarms {out[name][1]:.2f}  yes/no mass {out[name][2]:.2f}   p(yes) present: " + " ".join(f"{v:.2f}" for v in p[pr]))
    return out


for rend in ("text", "picture"):
    arms(rend)
