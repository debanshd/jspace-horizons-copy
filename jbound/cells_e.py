# Pilot cells E0–E2P: task-dependent access and cross-modal competition in J-space. Protocol frozen in PILOT.md.
# Run with jbound/jb --cells-file jbound/cells_e.py --cells E0,E1,E2,E2P (the setup and the M0/B1/B3 definitions load first).
# Every name starts with e_, E_ or resE. Nothing in the B cells is touched.

# %% ---------------------------------------------------------------- E0: stimuli. COCO object crops in fixed numbered slots, and the same scene as text
import zipfile, collections, itertools
E_VOCAB = ["dog", "cat", "car", "bus", "chair", "clock", "cup", "bird", "train", "horse", "bicycle", "airplane", "boat", "bed", "pizza", "laptop"]
E_N_SCENES = 64
E_SEED = 0
E_CELL, E_CROP, E_MARGIN = 224, 184, 0.10
E_BG, E_EMPTY = (235, 235, 235), (210, 210, 210)
E_COCO_DIR = "/workspace/nayan/data/coco2017"
E_PER_CLASS = 12
E_RNG = np.random.default_rng(E_SEED)
t0 = time.time()


def e_single(word):
    """ids of the single-token forms of a word (word, ' word', Word, ' Word'); the first token of ' word' when none is single"""
    ids = single_ids([word, " " + word, word.capitalize(), " " + word.capitalize()])
    return ids, bool(ids)


def e_font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


E_BANK_META = {}


def e_crop_bank():
    """{class: [PIL crops]} from COCO val2017: single instances, not crowd, both sides >= 96 px, 10% margin, fitted into E_CROP on grey"""
    ann = json.load(open(f"{E_COCO_DIR}/annotations/instances_val2017.json"))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    files = {im["id"]: im["file_name"] for im in ann["images"]}
    per = collections.defaultdict(list)
    for a in ann["annotations"]:
        n = cats[a["category_id"]]
        if n in E_VOCAB and not a.get("iscrowd", 0):
            x, y, w, h = a["bbox"]
            if w >= 96 and h >= 96:
                per[n].append((w * h, a["image_id"], (x, y, w, h)))
    zf = zipfile.ZipFile(f"{E_COCO_DIR}/val2017.zip")
    bank = {}
    for n in E_VOCAB:
        rows = sorted(per[n], reverse=True)[:E_PER_CLASS * 3]
        E_RNG.shuffle(rows)
        crops = []
        _set = int(os.environ.get("E_CROP_SET", 0))                                 # 0: the first 12 of the shuffle (every run before F4); 1: the next 12, never used before (F4)
        E_BANK_META[n] = [(int(iid), [int(v) for v in bb]) for _, iid, bb in rows[_set * E_PER_CLASS:(_set + 1) * E_PER_CLASS]]
        for _, iid, (x, y, w, h) in rows[_set * E_PER_CLASS:(_set + 1) * E_PER_CLASS]:
            im = Image.open(zf.open(f"val2017/{files[iid]}")).convert("RGB")
            mx, my = w * E_MARGIN, h * E_MARGIN
            box = (max(0, int(x - mx)), max(0, int(y - my)), min(im.width, int(x + w + mx)), min(im.height, int(y + h + my)))
            c = im.crop(box)
            c.thumbnail((E_CROP, E_CROP))
            tile = Image.new("RGB", (E_CROP, E_CROP), E_BG)
            tile.paste(c, ((E_CROP - c.width) // 2, (E_CROP - c.height) // 2))
            crops.append(tile)
        bank[n] = crops
    return bank


def e_canvas(n_cols, n_rows, slots):
    """a fixed grid of numbered cells; slots = {slot_number: PIL tile or None}. Slot numbers run row-major from 1"""
    img = Image.new("RGB", (n_cols * E_CELL, n_rows * E_CELL), E_BG)
    d = ImageDraw.Draw(img)
    font = e_font(30)
    for s in range(1, n_cols * n_rows + 1):
        r, c = divmod(s - 1, n_cols)
        x0, y0 = c * E_CELL, r * E_CELL
        if slots.get(s) is None:
            d.rectangle([x0 + 4, y0 + 4, x0 + E_CELL - 5, y0 + E_CELL - 5], fill=E_EMPTY)
        else:
            img.paste(slots[s], (x0 + (E_CELL - E_CROP) // 2, y0 + (E_CELL - E_CROP) // 2))
        d.rectangle([x0 + 6, y0 + 6, x0 + 44, y0 + 44], fill="white", outline="black")
        d.text((x0 + 14, y0 + 8), str(s), fill="black", font=font)
    return img


def e_slot_tokens(n_cols, n_rows, s):
    """row-major image-token indices (within the image's tokens) covered by slot s on a grid of n_cols x n_rows cells of 7 x 7 tokens"""
    r, c = divmod(s - 1, n_cols)
    W = n_cols * (E_CELL // 32)
    return [(r * 7 + i) * W + (c * 7 + j) for i in range(7) for j in range(7)]


def e_text_scene(items):
    """items = {slot: class}; the text rendering of the same scene"""
    return "; ".join(f"Position {s}: {items[s]}" if items.get(s) else f"Position {s}: empty" for s in sorted(items)) + "."


def e_chat_media(parts):
    """one user turn from a list of parts: str (text) or 'IMAGE'"""
    content = [{"type": "image"} if p == "IMAGE" else {"type": "text", "text": p} for p in parts]
    return MM_PROC.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def e_word_positions(prompt, word, occurrence_after):
    """token indices of `word` in the (text-only) prompt, the first occurrence after character index occurrence_after"""
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    i = prompt.index(word, occurrence_after)
    return [t for t, (a, b) in enumerate(enc["offset_mapping"]) if a < i + len(word) and b > i]


def e_gen_correct(outs, targets, words):
    """greedy continuation judged by the first vocabulary word it contains: correct when that word is the target"""
    import re
    ok = []
    for o, t in zip(outs, targets):
        toks = re.findall(r"[a-z]+", o.lower())
        first = next((w for w in toks if w in words or w.rstrip("s") in words), None)
        first = first.rstrip("s") if first and first not in words else first
        ok.append(bool(first == t))
    return np.array(ok)


# --- reads
try:
    MM_PROC
except NameError:
    MM_PROC = None
if MM_PROC is None:                                                            # M0's picture preflight (sets MM_PROC, MM_VISION_PATH, MM_N_IMAGE)
    _problem = mm_preflight()
    assert not _problem, _problem
    print(f"vision path '{MM_VISION_PATH}' reads the picture")
E_SID = {}
for w in E_VOCAB:
    ids, ok = e_single(w)
    if ok:
        E_SID[w] = ids
E_WORDS = [w for w in E_VOCAB if w in E_SID]
print(f"vocabulary: {len(E_WORDS)} single-token classes" + (f"; dropped (multi-token): {[w for w in E_VOCAB if w not in E_SID]}" if len(E_WORDS) < len(E_VOCAB) else ""))
E_BANK = e_crop_bank()



# the crop gate: every crop shown alone on a 224 x 224 canvas must be named by the model (greedy, 6 tokens); kept crops per class are logged
E_GATE_Q = "What object is this? Reply with the noun only."
_crops = [(w, j, im) for w in E_WORDS for j, im in enumerate(E_BANK[w])]
_canv = [e_canvas(1, 1, {1: im}) for _, _, im in _crops]
_outs = mm_generate([e_chat_media(["IMAGE", E_GATE_Q])] * len(_crops), _canv, 6)
_ok = e_gen_correct(_outs, [w for w, _, _ in _crops], set(E_WORDS))
E_CROP_GATE = {w: dict(kept=int(sum(o for (ww, _, _), o in zip(_crops, _ok) if ww == w)), total=len(E_BANK[w])) for w in E_WORDS}
E_BANK = {w: [im for (ww, _, im), o in zip(_crops, _ok) if ww == w and o] for w in E_WORDS}
print("crop gate (kept/total): " + ", ".join(f"{w} {v['kept']}/{v['total']}" for w, v in E_CROP_GATE.items()))
E_WORDS = [w for w in E_WORDS if len(E_BANK[w]) >= 4]
save(dict(vocab=E_WORDS, crop_gate=E_CROP_GATE, gate_question=E_GATE_Q), "eE0_stimuli")
mm_contact_sheet([(w, E_BANK[w][0]) for w in E_WORDS], 8, "figE0_crops")
print(f"crop bank: {len(E_WORDS)} classes x {E_PER_CLASS} crops  [{time.time() - t0:.0f}s]")

# %% ---------------------------------------------------------------- E1: does relevance change what J-space holds? A (one position), B (any position), C (joint)
E1_N_SLOTS, E1_COLS, E1_ROWS = 4, 2, 2
E1_CUE = {"A": "Study the scene. You will be asked about position {p}.", "B": "Study the scene. Any position may be queried.", "C": "Study the scene. You will need to combine two positions."}
E1_Q = {"A": "What object is in position {p}? Reply with the noun only.", "B": "What object is in position {p}? Reply with the noun only.", "C": "Do positions {p} and {q} contain the same type of object? Reply yes or no."}
E1_CHUNK = 32
t0 = time.time()


def e1_scenes():
    rows = []
    for i in range(E_N_SCENES):
        cls = [E_WORDS[j] for j in E_RNG.choice(len(E_WORDS), E1_N_SLOTS, replace=False)]
        p = i % E1_N_SLOTS + 1
        q = [s for s in range(1, E1_N_SLOTS + 1) if s != p][i // E1_N_SLOTS % (E1_N_SLOTS - 1)]
        same = i % 2 == 1                                                     # half the scenes plant a duplicate so C's answer is balanced
        if same:
            cls[q - 1] = cls[p - 1]
        items = {s: cls[s - 1] for s in range(1, E1_N_SLOTS + 1)}
        tiles = {s: E_BANK[items[s]][int(E_RNG.integers(len(E_BANK[items[s]])))] for s in items}
        rows.append(dict(scene=i, items=items, p=p, q=q, same=same, img=e_canvas(E1_COLS, E1_ROWS, tiles), text=e_text_scene(items)))
    return pd.DataFrame(rows)


@torch.no_grad()
def e_ranks_at(texts, images, where, n_pos, id_sets, chunk):
    """J-lens and logit-lens best-variant ranks of each id set at n_pos captured positions (where) plus the last token, every read layer:
    two int arrays [n, len(LAYERS), n_pos + 1, len(id_sets)], computed chunk by chunk on the GPU. Also the last-token logits [n, vocab]."""
    out_j, out_l, out_p, logits = [], [], [], []
    for s in range(0, len(texts), chunk):
        tx = texts[s:s + chunk]
        im = images[s:s + chunk] if images is not None else None
        lg, R = mm_forward(tx, im, LAYERS, where=where)
        n = lg.shape[0]
        H = R
        assert H[LAYERS[0]].shape[1] == n_pos + 1, (H[LAYERS[0]].shape, n_pos + 1)
        rj = b_ranks({L: H[L].reshape(n * (n_pos + 1), -1).numpy() for L in LAYERS}, id_sets, jac=True).reshape(n, len(LAYERS), n_pos + 1, len(id_sets))
        rl = b_ranks({L: H[L].reshape(n * (n_pos + 1), -1).numpy() for L in LAYERS}, id_sets, jac=False).reshape(n, len(LAYERS), n_pos + 1, len(id_sets))
        rp = b_ranks({L: H[L].reshape(n * (n_pos + 1), -1).numpy() for L in LAYERS}, id_sets, jac="jpp").reshape(n, len(LAYERS), n_pos + 1, len(id_sets)) if JPP is not None else rl * 0 - 1
        out_j.append(rj)
        out_l.append(rl)
        out_p.append(rp)
        logits.append(lg)
        del R, H
    E_RP.clear()
    E_RP.append(np.concatenate(out_p))                                        # the J++ ranks of the last call (-1 when no J++ lens is loaded)
    return np.concatenate(out_j), np.concatenate(out_l), torch.cat(logits)


E_RP = []


def e_where_image(n_image):
    """the image tokens (all of them, in order) of a row, then the last token (the decision position). mm_forward returns exactly these positions"""
    def f(ids):
        ve = ids.index(MM_VE)
        return list(range(ve - n_image, ve)) + [len(ids) - 1]
    return f


def e_answer(logits, words):
    """p of each word's forms at the last token: [n, len(words)]; the restricted argmax; and the raw mass on the vocabulary"""
    pr = torch.softmax(logits.float(), -1)
    P = np.stack([sum(pr[:, t].numpy() for t in E_SID[w]) for w in words], 1)
    return P, P.sum(1)


# --- reads
S1 = e1_scenes()
E1_N_IMAGE = E1_COLS * E1_ROWS * 49
ids_probe = mm_encode([e_chat_media(["x", "IMAGE", "y"])], [S1.img.iloc[0]])["input_ids"][0].tolist()
assert ids_probe.index(MM_VE) - ids_probe.index(MM_VS) - 1 == E1_N_IMAGE, "the canvas does not give the expected number of image tokens"
mm_contact_sheet([(f"scene {i}: {S1.text.iloc[i]}", S1.img.iloc[i]) for i in range(min(4, len(S1)))], 2, "figE1_stimuli")
resE1 = dict(by_rendering={}, n_scenes=int(len(S1)))
rows_out = []
present = np.array([[w in r.items.values() for w in E_WORDS] for r in S1.itertuples()])
for rend in ("image", "text"):
    R = {}
    for cond in ("A", "B", "C"):
        if rend == "image":
            texts = [e_chat_media([E1_CUE[cond].format(p=r.p), "IMAGE", E1_Q[cond].format(p=r.p, q=r.q)]) for r in S1.itertuples()]
            images = S1.img.tolist()
            rj, rl, lg = e_ranks_at(texts, images, e_where_image(E1_N_IMAGE), E1_N_IMAGE, [E_SID[w] for w in E_WORDS], E1_CHUNK)
            slot_tok = {s: e_slot_tokens(E1_COLS, E1_ROWS, s) for s in range(1, E1_N_SLOTS + 1)}
        else:
            texts = [chat([("user", E1_CUE[cond].format(p=r.p) + "\n" + r.text + "\n" + E1_Q[cond].format(p=r.p, q=r.q))]) for r in S1.itertuples()]
            images = None
            # the token positions of each slot's word in each prompt
            pos_rows = []
            for t, r in zip(texts, S1.itertuples()):
                base = t.index("Position 1:")
                pos_rows.append({s: e_word_positions(t, r.items[s], t.index(f"Position {s}:", base)) for s in r.items})
            maxlen = max(len(v) for pr_ in pos_rows for v in pr_.values())
            where_rows = [sum(([v[0]] * (maxlen - len(v)) + v for v in [pr_[s] for s in sorted(pr_)]), []) for pr_ in pos_rows]   # padded per slot to maxlen
            n_pos = E1_N_SLOTS * maxlen
            enc_ids = [tokenizer.encode(t, add_special_tokens=False) for t in texts]

            def where_text(ids, _rows=where_rows, _enc=enc_ids):
                """the row is identified by its own tokens at the end of the (left-padded) row; positions are offset by the padding"""
                for wr, e in zip(_rows, _enc):
                    if len(ids) >= len(e) and ids[-len(e):] == e:
                        return [len(ids) - len(e) + p for p in wr] + [len(ids) - 1]
                raise AssertionError("row not matched")
            rj, rl, lg = e_ranks_at(texts, None, where_text, n_pos, [E_SID[w] for w in E_WORDS], E1_CHUNK)
            slot_tok = {s: list(range((s - 1) * maxlen, s * maxlen)) for s in range(1, E1_N_SLOTS + 1)}
        P, mass = e_answer(lg, E_WORDS)
        gen = mm_generate(texts, images if images is not None else [None] * len(texts), 6) if cond != "C" else [""] * len(texts)
        gen_ok = e_gen_correct(gen, [r.items[r.p] for r in S1.itertuples()], set(E_WORDS)) if cond != "C" else None
        rp = E_RP[-1]
        best_dec_j, best_dec_l, best_dec_p = rj[:, :, -1, :].min(1), rl[:, :, -1, :].min(1), rp[:, :, -1, :].min(1)   # [n, words] at the decision position
        best_src_p = np.stack([rp[:, :, slot_tok[s], :].min(axis=(1, 2)) for s in range(1, E1_N_SLOTS + 1)], 1)
        best_src_j = np.stack([rj[:, :, slot_tok[s], :].min(axis=(1, 2)) for s in range(1, E1_N_SLOTS + 1)], 1)   # [n, slots, words] on own slot
        best_src_l = np.stack([rl[:, :, slot_tok[s], :].min(axis=(1, 2)) for s in range(1, E1_N_SLOTS + 1)], 1)
        pyes = None
        if cond == "C":
            pr = torch.softmax(lg.float(), -1)
            y = sum(pr[:, t].numpy() for t in YES_IDS)
            n_ = sum(pr[:, t].numpy() for t in NO_IDS)
            pyes = y / np.maximum(y + n_, 1e-9)
        for i, r in enumerate(S1.itertuples()):
            wi = {w: j for j, w in enumerate(E_WORDS)}
            target = r.items[r.p]
            for s in range(1, E1_N_SLOTS + 1):
                w = r.items[s]
                rows_out.append(dict(scene=r.scene, rendering=rend, condition=cond, slot=s, word=w, is_target=(s == r.p), is_q=(s == r.q), role=("target" if s == r.p else "other"),
                                     dec_rank_j=int(best_dec_j[i, wi[w]]), dec_rank_l=int(best_dec_l[i, wi[w]]), dec_rank_p=int(best_dec_p[i, wi[w]]), src_rank_j=int(best_src_j[i, s - 1, wi[w]]), src_rank_l=int(best_src_l[i, s - 1, wi[w]]), src_rank_p=int(best_src_p[i, s - 1, wi[w]]),
                                     p_correct=float(P[i, wi[target]]) if cond != "C" else np.nan, correct=bool(gen_ok[i]) if cond != "C" else bool((pyes[i] >= 0.5) == r.same), restricted_correct=bool(E_WORDS[int(P[i].argmax())] == target) if cond != "C" else np.nan, generation=(gen[i] if cond != "C" else ""),
                                     p_yes=float(pyes[i]) if cond == "C" else np.nan, same=r.same, mass=float(mass[i])))
            for w in E_WORDS:                                                                            # absent classes: the null at the decision position
                if w not in r.items.values():
                    rows_out.append(dict(scene=r.scene, rendering=rend, condition=cond, slot=0, word=w, is_target=False, is_q=False, role="absent",
                                         dec_rank_j=int(best_dec_j[i, wi[w]]), dec_rank_l=int(best_dec_l[i, wi[w]]), dec_rank_p=int(best_dec_p[i, wi[w]]), src_rank_j=np.nan, src_rank_l=np.nan, src_rank_p=np.nan, p_correct=np.nan, correct=np.nan, p_yes=np.nan, same=r.same, mass=float(mass[i])))
        acc = np.mean([x["correct"] for x in rows_out if x["rendering"] == rend and x["condition"] == cond and x["role"] == "target"])
        print(f"{rend:5s} {cond}: accuracy {acc:.2f}  [{time.time() - t0:.0f}s]")
E1 = pd.DataFrame(rows_out)
E1["dec_in_j"], E1["dec_in_l"], E1["src_in_j"], E1["src_in_l"] = E1.dec_rank_j <= B_TOP, E1.dec_rank_l <= B_TOP, E1.src_rank_j <= B_TOP, E1.src_rank_l <= B_TOP
E1["dec_in_p"], E1["src_in_p"] = (E1.dec_rank_p >= 1) & (E1.dec_rank_p <= B_TOP), (E1.src_rank_p >= 1) & (E1.src_rank_p <= B_TOP)      # J++ (all False when no J++ lens)
E1.to_csv(f"{RUN_DIR}/e1_relevance.csv", index=False)
# --- the preregistered reads
for rend in ("image", "text"):
    d = E1[E1.rendering == rend]
    r = {}
    for cond in ("A", "B", "C"):
        dc = d[d.condition == cond]
        r[cond] = dict(accuracy=b_boot_scene(dc[dc.role == "target"].correct.astype(float).values, dc[dc.role == "target"].scene.values),
                       target_dec=b_boot_scene(dc[dc.role == "target"].dec_in_j.values, dc[dc.role == "target"].scene.values),
                       other_dec=b_boot_scene(dc[dc.role == "other"].dec_in_j.values, dc[dc.role == "other"].scene.values),
                       absent_dec=b_boot_scene(dc[dc.role == "absent"].dec_in_j.values, dc[dc.role == "absent"].scene.values),
                       target_src=b_boot_scene(dc[dc.role == "target"].src_in_j.values, dc[dc.role == "target"].scene.values),
                       other_src=b_boot_scene(dc[dc.role == "other"].src_in_j.values, dc[dc.role == "other"].scene.values),
                       other_dec_logit=b_boot_scene(dc[dc.role == "other"].dec_in_l.values, dc[dc.role == "other"].scene.values),
                       other_src_logit=b_boot_scene(dc[dc.role == "other"].src_in_l.values, dc[dc.role == "other"].scene.values),
                       other_dec_jpp=b_boot_scene(dc[dc.role == "other"].dec_in_p.values, dc[dc.role == "other"].scene.values) if JPP is not None else None,
                       other_src_jpp=b_boot_scene(dc[dc.role == "other"].src_in_p.values, dc[dc.role == "other"].scene.values) if JPP is not None else None)
    # P1: non-target items at the decision position, B minus A, paired by scene
    oa = d[(d.condition == "A") & (d.role == "other")].groupby("scene").dec_in_j.mean()
    ob = d[(d.condition == "B") & (d.role == "other")].groupby("scene").dec_in_j.mean()
    diff = (ob - oa).dropna()
    rng = np.random.default_rng(0)
    bs = [rng.choice(diff.values, len(diff)).mean() for _ in range(2000)]
    P1 = (float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    ta = d[(d.condition == "A") & (d.role == "target")].groupby("scene").dec_in_j.mean()
    tb = d[(d.condition == "B") & (d.role == "target")].groupby("scene").dec_in_j.mean()
    d2 = (tb - ta).dropna()
    bs2 = [rng.choice(d2.values, len(d2)).mean() for _ in range(2000)]
    P2 = (float(d2.mean()), float(np.percentile(bs2, 2.5)), float(np.percentile(bs2, 97.5)))
    seen = r["A"]["accuracy"][0] >= (0.85 if rend == "image" else 0.95)
    verdict = ("not seen: single-fact accuracy below the gate" if not seen else
               "relevance admits items beyond the forthcoming answer" if P1[0] >= 0.15 and P1[1] > 0 else
               "the decision-position readout is answer preparation (equivalence)" if -0.05 <= P1[1] and P1[2] <= 0.05 else "undetermined at this n")
    resE1["by_rendering"][rend] = dict(conditions=r, P1_other_B_minus_A=P1, P2_target_B_minus_A=P2, seen=bool(seen), verdict=verdict)
    print(f"{rend:5s} acc A {r['A']['accuracy'][0]:.2f} B {r['B']['accuracy'][0]:.2f} C {r['C']['accuracy'][0]:.2f} | decision position in J-space: target A {r['A']['target_dec'][0]:.2f} B {r['B']['target_dec'][0]:.2f}; "
          f"other A {r['A']['other_dec'][0]:.2f} B {r['B']['other_dec'][0]:.2f} C {r['C']['other_dec'][0]:.2f}; absent A {r['A']['absent_dec'][0]:.2f} | own-slot: other A {r['A']['other_src'][0]:.2f} B {r['B']['other_src'][0]:.2f} (logit lens {r['A']['other_src_logit'][0]:.2f}) | "
          f"P1 other B-A {b_fmt(P1)}  P2 target B-A {b_fmt(P2)} -> {verdict}")
resE1["seconds"] = time.time() - t0
save(resE1, "eE1_results")
# --- figure
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(wspace=0.65))
conds = ["A", "B", "C"]
for j, rend in enumerate(("image", "text")):
    r = resE1["by_rendering"][rend]["conditions"]
    c = B_COLOR["picture"] if rend == "image" else B_COLOR["text"]
    xs = np.arange(len(conds)) + (j - 0.5) * 0.36
    for a_, key, lab in ((ax[0], "other_dec", "non-target items at the decision position"), (ax[1], "other_src", "non-target items on their own tokens")):
        v = [r[cd][key] for cd in conds]
        a_.bar(xs, [t[0] for t in v], 0.36, color=c, alpha=0.85)
        a_.errorbar(xs, [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
        a_.set(xticks=np.arange(len(conds)), xticklabels=["A: one position", "B: any position", "C: joint"], ylabel=f"share in the J-lens top-{B_TOP} (any read layer)\n{lab}", ylim=(0, 1.02))
        a_.text(0.98, 0.97 - 0.07 * j, rend, color=c, fontsize=8.5, transform=a_.transAxes, va="top", ha="right")
    v = [r[cd]["absent_dec"] for cd in conds]
    ax[0].plot(np.arange(len(conds)) + (j - 0.5) * 0.36, [t[0] for t in v], "_", color="0.4", ms=14, mew=1.5)
    v = [r[cd]["accuracy"] for cd in conds]
    ax[2].bar(xs, [t[0] for t in v], 0.36, color=c, alpha=0.85)
    ax[2].errorbar(xs, [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
ax[2].set(xticks=np.arange(len(conds)), xticklabels=["A: one position", "B: any position", "C: joint"], ylabel="accuracy (C: same/different)", ylim=(0, 1.02))
ax[0].text(0.02, 0.97, "dash: absent classes (the null)", color="0.4", fontsize=7.5, transform=ax[0].transAxes, va="top")
fig.suptitle("E1. " + "; ".join(f"{rend}: non-target items at the decision position, B − A = {b_fmt(resE1['by_rendering'][rend]['P1_other_B_minus_A'])} → {resE1['by_rendering'][rend]['verdict']}" for rend in ("image", "text")), fontsize=9.5, y=1.0)
ws_finish(fig, "figE1_relevance", f"E1 · {len(S1)} scenes · k = {E1_N_SLOTS}")

# %% ---------------------------------------------------------------- E2: cross-modal competition. Six image slots and a six-item text list; arms V0 V T0 T VT R; loads k
E2_N_SLOTS, E2_COLS, E2_ROWS = 6, 3, 2
E2_KS = [1, 2, 4, 6]
E2_INTRO = "Study the scene. It has an image grid with numbered slots and a lettered text list."
E2_Q = {"V": "What object is in image slot {q}? Reply with the noun only.", "T": "What object is text item ({rl})? Reply with the noun only.",
        "VT": "Does image slot {q} contain the same class of object as text item ({rl})? Reply yes or no."}
E2_CHUNK = 24
t0 = time.time()


E2_LETTER = "abcdef"


def e2_text_list(items):
    """the text items are lettered a-f so that a slot number never refers to both sources (smoke 5: 'image slot 4' was answered with text item 4)"""
    return "Text list: " + " ".join(f"({E2_LETTER[s - 1]}) {items[s]}" if items.get(s) else f"({E2_LETTER[s - 1]}) empty" for s in range(1, E2_N_SLOTS + 1))


def e2_scenes():
    rows = []
    for i in range(E_N_SCENES):
        cls = [E_WORDS[j] for j in E_RNG.choice(len(E_WORDS), 2 * E2_N_SLOTS, replace=False)] if len(E_WORDS) >= 2 * E2_N_SLOTS else \
              [E_WORDS[j] for j in E_RNG.choice(len(E_WORDS), E2_N_SLOTS, replace=False)] + [E_WORDS[j] for j in E_RNG.choice(len(E_WORDS), E2_N_SLOTS, replace=False)]
        img_cls, txt_cls = cls[:E2_N_SLOTS], cls[E2_N_SLOTS:]
        order = list(E_RNG.permutation(E2_N_SLOTS) + 1)                       # the order in which slots fill as k grows (the first k are occupied)
        q, rr = order[0], order[0]                                             # the queried image slot and text slot are always occupied
        same = i % 2 == 1
        if same:
            txt_cls[rr - 1] = img_cls[q - 1]
        tiles = {s: E_BANK[img_cls[s - 1]][int(E_RNG.integers(len(E_BANK[img_cls[s - 1]])))] for s in range(1, E2_N_SLOTS + 1)}
        rows.append(dict(scene=i, img_cls=img_cls, txt_cls=txt_cls, order=order, q=q, r=rr, same=same, tiles=tiles))
    return pd.DataFrame(rows)


def e2_render(r, k, arm):
    """prompt and image for scene r at load k under an arm; returns (prompt, image, image_items, text_items)"""
    occ = set(r.order[:k])
    img_items = {s: (r.img_cls[s - 1] if s in occ else None) for s in range(1, E2_N_SLOTS + 1)}
    txt_items = {s: (r.txt_cls[s - 1] if s in occ else None) for s in range(1, E2_N_SLOTS + 1)}
    if arm == "R":
        txt_items = dict(img_items)
    img = e_canvas(E2_COLS, E2_ROWS, {s: (r.tiles[s] if s in occ else None) for s in range(1, E2_N_SLOTS + 1)})
    q_ = E2_Q["V" if arm in ("V0", "V", "R") else "T" if arm in ("T0", "T") else "VT"].format(q=r.q, rl=E2_LETTER[r.r - 1])
    if arm == "V0":
        return e_chat_media([E2_INTRO, "IMAGE", q_]), img, img_items, None
    if arm == "T0":
        return chat([("user", E2_INTRO + "\n" + e2_text_list(txt_items) + "\n" + q_)]), None, None, txt_items
    return e_chat_media([E2_INTRO, "IMAGE", e2_text_list(txt_items) + "\n" + q_]), img, img_items, txt_items


# --- reads
S2 = e2_scenes()
E2_N_IMAGE = E2_COLS * E2_ROWS * 49
rows_out = []
wi = {w: j for j, w in enumerate(E_WORDS)}
for k in E2_KS:
    for arm in ("V0", "V", "T0", "T", "VT", "R"):
        rend = [e2_render(r, k, arm) for r in S2.itertuples()]
        texts, images = [x[0] for x in rend], ([x[1] for x in rend] if rend[0][1] is not None else None)
        if images is not None:
            rj, rl, lg = e_ranks_at(texts, images, e_where_image(E2_N_IMAGE), E2_N_IMAGE, [E_SID[w] for w in E_WORDS], E2_CHUNK)
        else:
            lg, _ = mm_forward(texts, None, [])
            rj = rl = None
        P, mass = e_answer(lg, E_WORDS)
        pr = torch.softmax(lg.float(), -1)
        ymass = sum(pr[:, t].numpy() for t in YES_IDS) + sum(pr[:, t].numpy() for t in NO_IDS)
        pyes = sum(pr[:, t].numpy() for t in YES_IDS) / np.maximum(ymass, 1e-9)
        if arm != "VT":
            tg = [(r.img_cls[r.q - 1] if arm in ("V0", "V", "R") else r.txt_cls[r.r - 1]) for r in S2.itertuples()]
            gen = mm_generate(texts, images if images is not None else [None] * len(texts), 6)
            gen_ok = e_gen_correct(gen, tg, set(E_WORDS))
        for i, r in enumerate(S2.itertuples()):
            vis_target, txt_target = r.img_cls[r.q - 1], (r.img_cls[r.r - 1] if arm == "R" else r.txt_cls[r.r - 1])
            target = vis_target if arm in ("V0", "V", "R") else txt_target if arm in ("T0", "T") else None
            same = (vis_target == txt_target)
            row = dict(scene=r.scene, k=k, arm=arm, q=r.q, r=r.r, vis_target=vis_target, txt_target=txt_target, same=same, mass=float(mass[i]),
                       p_correct=float(P[i, wi[target]]) if target else np.nan, correct=(bool(gen_ok[i]) if target else bool((pyes[i] >= 0.5) == same)),
                       restricted_correct=(bool(E_WORDS[int(P[i].argmax())] == target) if target else np.nan), generation=(gen[i] if target else ""),
                       p_yes=float(pyes[i]) if arm == "VT" else np.nan, yesno_mass=float(ymass[i]) if arm == "VT" else np.nan)
            if rj is not None:
                st = e_slot_tokens(E2_COLS, E2_ROWS, r.q)
                rp = E_RP[-1]
                row.update(vis_src_rank_j=int(rj[i, :, st, wi[vis_target]].min()), vis_src_rank_l=int(rl[i, :, st, wi[vis_target]].min()), vis_src_rank_p=int(rp[i, :, st, wi[vis_target]].min()), vis_dec_rank_p=int(rp[i, :, -1, wi[vis_target]].min()),
                           vis_dec_rank_j=int(rj[i, :, -1, wi[vis_target]].min()), vis_dec_rank_l=int(rl[i, :, -1, wi[vis_target]].min()),
                           txt_dec_rank_j=int(rj[i, :, -1, wi[txt_target]].min()) if arm != "V0" else np.nan,
                           absent_dec_j=float(np.mean([rj[i, :, -1, wi[w]].min() <= B_TOP for w in E_WORDS if w not in r.img_cls and w not in r.txt_cls])))
            rows_out.append(row)
        acc = np.mean([x["correct"] for x in rows_out if x["k"] == k and x["arm"] == arm])
        print(f"k = {k} {arm:3s}: accuracy {acc:.2f}  [{time.time() - t0:.0f}s]")
E2 = pd.DataFrame(rows_out)
E2.to_csv(f"{RUN_DIR}/e2_competition.csv", index=False)
resE2 = dict(by_k={}, n_scenes=int(len(S2)))
for k in E2_KS:
    d = E2[E2.k == k]
    acc = {arm: b_boot_scene(d[d.arm == arm].correct.astype(float).values, d[d.arm == arm].scene.values) for arm in ("V0", "V", "T0", "T", "VT", "R")}
    def paired(a, b):
        x = d[d.arm == a].set_index("scene").correct.astype(float) - d[d.arm == b].set_index("scene").correct.astype(float)
        x = x.dropna().values
        rng = np.random.default_rng(0)
        bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
        return (float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    vis_dec = {arm: b_boot_scene((d[d.arm == arm].vis_dec_rank_j <= B_TOP).values, d[d.arm == arm].scene.values) for arm in ("V0", "V", "VT", "R")}
    vis_src = {arm: b_boot_scene((d[d.arm == arm].vis_src_rank_j <= B_TOP).values, d[d.arm == arm].scene.values) for arm in ("V0", "V", "VT", "R")}
    resE2["by_k"][k] = dict(accuracy=acc, I_text_on_vision=paired("V0", "V"), I_vision_on_text=paired("T0", "T"), redundancy=paired("R", "V"), vis_dec=vis_dec, vis_src=vis_src)
    print(f"k = {k}: acc V0 {acc['V0'][0]:.2f} V {acc['V'][0]:.2f} T0 {acc['T0'][0]:.2f} T {acc['T'][0]:.2f} VT {acc['VT'][0]:.2f} R {acc['R'][0]:.2f} | "
          f"I text->vision {b_fmt(resE2['by_k'][k]['I_text_on_vision'])} I vision->text {b_fmt(resE2['by_k'][k]['I_vision_on_text'])} | visual target at the decision position V0 {vis_dec['V0'][0]:.2f} V {vis_dec['V'][0]:.2f} VT {vis_dec['VT'][0]:.2f}; on its slot V0 {vis_src['V0'][0]:.2f} V {vis_src['V'][0]:.2f} VT {vis_src['VT'][0]:.2f}")
k8 = E2_KS[-1]
I = resE2["by_k"][k8]["I_text_on_vision"]
resE2["verdict"] = ("interference: the text list costs visual accuracy at the highest load" if I[0] >= 0.10 and I[1] > 0 else "no interference (equivalence)" if -0.05 <= I[1] and I[2] <= 0.05 else "undetermined at this n")
resE2["seconds"] = time.time() - t0
save(resE2, "eE2_results")
print(f"\n{resE2['verdict']}")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(wspace=0.65))
cols = {"V0": B_COLOR["picture"], "V": OKABE[4], "VT": OKABE[2], "R": OKABE[3], "T0": B_COLOR["text"], "T": OKABE[5]}
ends = []
for arm in ("V0", "V", "VT", "R", "T0", "T"):
    y = ws_band(ax[0], E2_KS, [resE2["by_k"][k]["accuracy"][arm] for k in E2_KS], cols[arm])
    ends.append((y, arm, cols[arm]))
ax[0].set(xlabel="occupied slots per source, k", ylabel="accuracy", xticks=E2_KS, ylim=(0, 1.04))
ws_end_labels(ax[0], ends, 0.05)
ends = []
for arm in ("V0", "V", "VT", "R"):
    y = ws_band(ax[1], E2_KS, [resE2["by_k"][k]["vis_dec"][arm] for k in E2_KS], cols[arm])
    ax[1].plot(E2_KS, [resE2["by_k"][k]["vis_src"][arm][0] for k in E2_KS], ":", color=cols[arm], lw=1.2)
    ends.append((y, arm, cols[arm]))
ax[1].set(xlabel="occupied slots per source, k", ylabel=f"visual target in the J-lens top-{B_TOP}\nat the decision position (dotted: on its own slot)", xticks=E2_KS, ylim=(0, 1.04))
ws_end_labels(ax[1], ends, 0.05)
for key, c, lab in (("I_text_on_vision", B_COLOR["picture"], "text list on visual accuracy"), ("I_vision_on_text", B_COLOR["text"], "image on text accuracy"), ("redundancy", OKABE[3], "redundant list (R − V)")):
    y = ws_band(ax[2], E2_KS, [resE2["by_k"][k][key] for k in E2_KS], c)
ax[2].axhline(0, color="0.6", lw=0.8)
ax[2].set(xlabel="occupied slots per source, k", ylabel="paired accuracy change (alone − with the other source)", xticks=E2_KS)
ws_end_labels(ax[2], [(resE2["by_k"][k8][key][0], lab, c) for key, c, lab in (("I_text_on_vision", B_COLOR["picture"], "text on vision"), ("I_vision_on_text", B_COLOR["text"], "vision on text"), ("redundancy", OKABE[3], "redundancy"))], 0.03)
fig.suptitle(f"E2. {resE2['verdict']}: I text→vision at k = {k8} = {b_fmt(I)}", fontsize=9.5, y=1.0)
ws_finish(fig, "figE2_competition", f"E2 · {len(S2)} scenes")

# %% ---------------------------------------------------------------- E2u: cross-modal competition with an unindexed text source (no slot number a text item can capture)
# Smoke 5/6 showed that an indexed text list captures the image slot reference ("image slot 4" is answered with text item 4 even when the
# items are lettered). Here the text source is a set: "The text mentions: airplane, horse, bus." The text question is a two-alternative
# membership probe (one mentioned item against one absent class), the joint question asks whether the image slot's object is mentioned.
# Arms: V0 image only; V image + set (question about the image); T0 set only; T set + image (question about the set); VT joint; R the set
# names the image's own objects. Loads k in E2_KS. Interference, readouts and rules as in E2 (PILOT.md), estimands named with a u suffix.
E2U_Q = {"V": "What object is in image slot {q}? Reply with the noun only.", "T": "Which of these objects is mentioned in the text: {x} or {y}? Reply with the noun only.",
         "VT": "Is the object in image slot {q} mentioned in the text? Reply yes or no."}
t0 = time.time()
try:
    S2
except NameError:                                                              # E2 not run in this session: the same scene generator (change #25)
    S2 = e2_scenes()
    E2_N_IMAGE = E2_COLS * E2_ROWS * 49


def e2u_text(items):
    return "The text mentions: " + ", ".join(items) + "."


def e2u_render(r, k, arm):
    occ = set(r.order[:k])
    img_items = {s: (r.img_cls[s - 1] if s in occ else None) for s in range(1, E2_N_SLOTS + 1)}
    txt_set = [r.txt_cls[s - 1] for s in sorted(occ)]
    if arm == "R":
        txt_set = [r.img_cls[s - 1] for s in sorted(occ)]
    img = e_canvas(E2_COLS, E2_ROWS, {s: (r.tiles[s] if s in occ else None) for s in range(1, E2_N_SLOTS + 1)})
    mentioned = r.txt_cls[r.r - 1] if arm != "R" else r.img_cls[r.q - 1]
    _abs = [w for w in E_WORDS if w not in r.img_cls and w not in r.txt_cls]
    absent = _abs[r.scene % len(_abs)]
    x, y = (mentioned, absent) if r.scene % 2 == 0 else (absent, mentioned)
    q_ = E2U_Q["V" if arm in ("V0", "V", "R") else "T" if arm in ("T0", "T") else "VT"].format(q=r.q, x=x, y=y)
    if arm == "V0":
        return e_chat_media([E2_INTRO_U, "IMAGE", q_]), img, mentioned
    if arm == "T0":
        return chat([("user", E2_INTRO_U + "\n" + e2u_text(txt_set) + "\n" + q_)]), None, mentioned
    return e_chat_media([E2_INTRO_U, "IMAGE", e2u_text(txt_set) + "\n" + q_]), img, mentioned


E2_INTRO_U = "Study the scene. It has an image grid with numbered slots, and a text."

# --- reads
rows_out = []
wi = {w: j for j, w in enumerate(E_WORDS)}
for k in E2_KS:
    for arm in ("V0", "V", "T0", "T", "VT", "R"):
        rend = [e2u_render(r, k, arm) for r in S2.itertuples()]
        texts, images, mentioned = [x[0] for x in rend], ([x[1] for x in rend] if rend[0][1] is not None else None), [x[2] for x in rend]
        if images is not None:
            rj, rl, lg = e_ranks_at(texts, images, e_where_image(E2_N_IMAGE), E2_N_IMAGE, [E_SID[w] for w in E_WORDS], E2_CHUNK)
        else:
            lg, _ = mm_forward(texts, None, [])
            rj = rl = None
        P, mass = e_answer(lg, E_WORDS)
        pr = torch.softmax(lg.float(), -1)
        ymass = sum(pr[:, t].numpy() for t in YES_IDS) + sum(pr[:, t].numpy() for t in NO_IDS)
        pyes = sum(pr[:, t].numpy() for t in YES_IDS) / np.maximum(ymass, 1e-9)
        if arm != "VT":
            tg = [(r.img_cls[r.q - 1] if arm in ("V0", "V", "R") else m) for r, m in zip(S2.itertuples(), mentioned)]
            gen = mm_generate(texts, images if images is not None else [None] * len(texts), 6)
            gen_ok = e_gen_correct(gen, tg, set(E_WORDS))
        for i, r in enumerate(S2.itertuples()):
            vis_target = r.img_cls[r.q - 1]
            target = vis_target if arm in ("V0", "V", "R") else mentioned[i] if arm in ("T0", "T") else None
            in_text = (vis_target in [r.txt_cls[s - 1] for s in r.order[:k]]) if arm != "R" else True
            row = dict(scene=r.scene, k=k, arm=arm, q=r.q, vis_target=vis_target, mentioned=mentioned[i], in_text=in_text, mass=float(mass[i]),
                       p_correct=float(P[i, wi[target]]) if target else np.nan, correct=(bool(gen_ok[i]) if target else bool((pyes[i] >= 0.5) == in_text)),
                       restricted_correct=(bool(E_WORDS[int(P[i].argmax())] == target) if target else np.nan), generation=(gen[i] if target else ""),
                       p_yes=float(pyes[i]) if arm == "VT" else np.nan, yesno_mass=float(ymass[i]) if arm == "VT" else np.nan)
            if rj is not None:
                st = e_slot_tokens(E2_COLS, E2_ROWS, r.q)
                rp = E_RP[-1]
                row.update(vis_src_rank_j=int(rj[i, :, st, wi[vis_target]].min()), vis_src_rank_l=int(rl[i, :, st, wi[vis_target]].min()), vis_src_rank_p=int(rp[i, :, st, wi[vis_target]].min()),
                           vis_dec_rank_j=int(rj[i, :, -1, wi[vis_target]].min()), vis_dec_rank_l=int(rl[i, :, -1, wi[vis_target]].min()), vis_dec_rank_p=int(rp[i, :, -1, wi[vis_target]].min()),
                           absent_dec_j=float(np.mean([rj[i, :, -1, wi[w]].min() <= B_TOP for w in E_WORDS if w not in r.img_cls and w not in r.txt_cls])))
            rows_out.append(row)
        acc = np.mean([x["correct"] for x in rows_out if x["k"] == k and x["arm"] == arm])
        print(f"k = {k} {arm:3s}: accuracy {acc:.2f}  [{time.time() - t0:.0f}s]")
E2U = pd.DataFrame(rows_out)
E2U.to_csv(f"{RUN_DIR}/e2u_competition.csv", index=False)
resE2u = dict(by_k={}, n_scenes=int(len(S2)))
for k in E2_KS:
    d = E2U[E2U.k == k]
    acc = {arm: b_boot_scene(d[d.arm == arm].correct.astype(float).values, d[d.arm == arm].scene.values) for arm in ("V0", "V", "T0", "T", "VT", "R")}
    def paired(a, b):
        x = (d[d.arm == a].set_index("scene").correct.astype(float) - d[d.arm == b].set_index("scene").correct.astype(float)).dropna().values
        rng = np.random.default_rng(0)
        bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
        return (float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    vis_dec = {arm: b_boot_scene((d[d.arm == arm].vis_dec_rank_j <= B_TOP).values, d[d.arm == arm].scene.values) for arm in ("V0", "V", "VT", "R")}
    vis_src = {arm: b_boot_scene((d[d.arm == arm].vis_src_rank_j <= B_TOP).values, d[d.arm == arm].scene.values) for arm in ("V0", "V", "VT", "R")}
    resE2u["by_k"][k] = dict(accuracy=acc, I_text_on_vision=paired("V0", "V"), I_vision_on_text=paired("T0", "T"), redundancy=paired("R", "V"), vis_dec=vis_dec, vis_src=vis_src)
    print(f"k = {k}: acc V0 {acc['V0'][0]:.2f} V {acc['V'][0]:.2f} T0 {acc['T0'][0]:.2f} T {acc['T'][0]:.2f} VT {acc['VT'][0]:.2f} R {acc['R'][0]:.2f} | "
          f"I text->vision {b_fmt(resE2u['by_k'][k]['I_text_on_vision'])} I vision->text {b_fmt(resE2u['by_k'][k]['I_vision_on_text'])} | visual target at the decision position V0 {vis_dec['V0'][0]:.2f} V {vis_dec['V'][0]:.2f} VT {vis_dec['VT'][0]:.2f}; on its slot V0 {vis_src['V0'][0]:.2f} V {vis_src['V'][0]:.2f} VT {vis_src['VT'][0]:.2f}")
k8 = E2_KS[-1]
I = resE2u["by_k"][k8]["I_text_on_vision"]
resE2u["verdict"] = ("interference: the text costs visual accuracy at the highest load" if I[0] >= 0.10 and I[1] > 0 else "no interference (equivalence)" if -0.05 <= I[1] and I[2] <= 0.05 else "undetermined at this n")
resE2u["seconds"] = time.time() - t0
save(resE2u, "eE2u_results")
print(f"\n{resE2u['verdict']}")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(wspace=0.65))
cols = {"V0": B_COLOR["picture"], "V": OKABE[4], "VT": OKABE[2], "R": OKABE[3], "T0": B_COLOR["text"], "T": OKABE[5]}
ends = []
for arm in ("V0", "V", "VT", "R", "T0", "T"):
    y = ws_band(ax[0], E2_KS, [resE2u["by_k"][k]["accuracy"][arm] for k in E2_KS], cols[arm])
    ends.append((y, arm, cols[arm]))
ax[0].set(xlabel="occupied slots per source, k", ylabel="accuracy", xticks=E2_KS, ylim=(0, 1.04))
ws_end_labels(ax[0], ends, 0.05)
ends = []
for arm in ("V0", "V", "VT", "R"):
    y = ws_band(ax[1], E2_KS, [resE2u["by_k"][k]["vis_dec"][arm] for k in E2_KS], cols[arm])
    ax[1].plot(E2_KS, [resE2u["by_k"][k]["vis_src"][arm][0] for k in E2_KS], ":", color=cols[arm], lw=1.2)
    ends.append((y, arm, cols[arm]))
ax[1].set(xlabel="occupied slots per source, k", ylabel=f"visual target in the J-lens top-{B_TOP}\nat the decision position (dotted: on its own slot)", xticks=E2_KS, ylim=(0, 1.04))
ws_end_labels(ax[1], ends, 0.05)
for key, c in (("I_text_on_vision", B_COLOR["picture"]), ("I_vision_on_text", B_COLOR["text"]), ("redundancy", OKABE[3])):
    ws_band(ax[2], E2_KS, [resE2u["by_k"][k][key] for k in E2_KS], c)
ax[2].axhline(0, color="0.6", lw=0.8)
ax[2].set(xlabel="occupied slots per source, k", ylabel="paired accuracy change (alone − with the other source)", xticks=E2_KS)
ws_end_labels(ax[2], [(resE2u["by_k"][k8][key][0], lab, c) for key, c, lab in (("I_text_on_vision", B_COLOR["picture"], "text on vision"), ("I_vision_on_text", B_COLOR["text"], "vision on text"), ("redundancy", OKABE[3], "redundancy"))], 0.03)
fig.suptitle(f"E2u (unindexed text). {resE2u['verdict']}: I text→vision at k = {k8} = {b_fmt(I)}", fontsize=9.5, y=1.0)
ws_finish(fig, "figE2u_competition", f"E2u · {len(S2)} scenes")

# %% ---------------------------------------------------------------- E2P: causal interchange. Swap one image slot's state from a donor scene: full / J part / random part / complement / non-target
E2P_KS = [1, 4]
E2P_N_PAIRS = 32
E2P_LAYERS_FULL = [-1] + list(range(FINAL + 1))   # -1 = the input to layer 0 (image embeddings): without it, later tokens still attend to the receiver's image at layer 0 (epilot1: full swap flipped only 0.62)
t0 = time.time()


def e2p_pos_hooks(layers, REC, pos, Q=None, part="all"):
    """hooks(enc, idx) -> [(L, fn)]: row i of a batch receives REC[L][idx[i]] (fp16 [n, P, d]) at the absolute positions pos[idx[i]] (list of P
    indices into the padded row, computed by the caller from the row length). part as in b_receipt_hooks."""
    def hooks(enc, idx):
        b, s = enc["input_ids"].shape
        idx = np.asarray(idx)
        Pm = torch.as_tensor(np.stack([pos[i](s) for i in idx]), device=DEV)
        rows = torch.arange(b, device=DEV)[:, None].expand(-1, Pm.shape[1])
        out_ = []
        for L in layers:
            D = torch.as_tensor(np.asarray(REC[L])[idx], device=DEV, dtype=torch.float32)
            Qd = None if Q is None else torch.as_tensor(Q[L], device=DEV, dtype=torch.float32)

            def fn(mod, inp, out, D=D, Qd=Qd):
                h = residual_of(out)
                if h.shape[1] == 1:
                    return out
                x = h.float()
                v = x[rows, Pm]
                assert v.shape == D.shape, (v.shape, D.shape)
                if part == "all":
                    v = D
                elif part == "lens":
                    v = v - (v @ Qd) @ Qd.T + (D @ Qd) @ Qd.T
                else:
                    v = (v @ Qd) @ Qd.T + (D - (D @ Qd) @ Qd.T)
                x[rows, Pm] = v
                x = x.to(h.dtype)
                return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
            out_.append((L, fn))
        return out_
    return hooks


@torch.no_grad()
def e_jpp_subspace(L, r):
    """the top-r right singular directions of W_U J++_L (the J++ analogue of b_lens_subspace)"""
    if "G" not in _B_WUTWU:
        b_lens_subspace(LAYERS[0], r)
    M = JPP[L].float().T @ _B_WUTWU["G"] @ JPP[L].float()
    return b_orth(np.ascontiguousarray(torch.linalg.eigh(M)[1].cpu().numpy()[:, -r:][:, ::-1].T))   # strongest first (change #24)


# --- reads
try:
    S2
except NameError:                                                              # E2 not run in this session: the same scene generator (change #25)
    S2 = e2_scenes()
    E2_N_IMAGE = E2_COLS * E2_ROWS * 49
    wi = {w: j for j, w in enumerate(E_WORDS)}
from judge import Judge
JUDGE = Judge(RUN_DIR)
assert JUDGE.available(), "GEMINI_API_KEY missing: the judge is required (PROTOCOL.md)"
Qs = {L: b_lens_subspace(L, B_R) for L in LAYERS}
Qr = {L: b_random_subspace(Qs[L].shape[0], B_R, 100 + L) for L in LAYERS}



Qpp = {L: e_jpp_subspace(L, B_R) for L in LAYERS} if JPP is not None else None
pairs = []
for i, r in enumerate(S2.head(E2P_N_PAIRS).itertuples()):
    others = [w for w in E_WORDS if w not in r.img_cls and w not in r.txt_cls]
    donor_cls = others[i % len(others)]
    q2 = r.order[1]                                                            # a second occupied slot (needs k >= 2): the non-target control
    donor2_cls = others[(i + 1) % len(others)]
    pairs.append(dict(scene=r.scene, q=r.q, q2=q2, receiver_cls=r.img_cls[r.q - 1], donor_cls=donor_cls, donor2_cls=donor2_cls, txt_target=r.txt_cls[r.r - 1]))
PP = pd.DataFrame(pairs)
rows_out = []
for k in E2P_KS:
    for question in ("V", "VT"):
        recv_txt, recv_img, don_img, don2_img, pos_fns, pos2_fns = [], [], [], [], [], []
        for r, p in zip(S2.head(E2P_N_PAIRS).itertuples(), PP.itertuples()):
            prompt, img, _, _ = e2_render(r, k, "V0" if question == "V" else "VT")   # V0, image only: with the indexed list (V) the list captures the answer half the time (#31)
            recv_txt.append(prompt)
            recv_img.append(img)
            occ = set(r.order[:k])
            tiles = {s: (r.tiles[s] if s in occ else None) for s in range(1, E2_N_SLOTS + 1)}
            d_tiles = dict(tiles)
            d_tiles[p.q] = E_BANK[p.donor_cls][p.scene % len(E_BANK[p.donor_cls])]
            don_img.append(e_canvas(E2_COLS, E2_ROWS, d_tiles))
            d2 = dict(tiles)
            if k >= 2:
                d2[p.q2] = E_BANK[p.donor2_cls][p.scene % len(E_BANK[p.donor2_cls])]
            don2_img.append(e_canvas(E2_COLS, E2_ROWS, d2))
        n_alls = [len(mm_encode(recv_txt[j:j + 1], recv_img[j:j + 1])["input_ids"][0]) for j in range(len(recv_txt))]
        vs_off = mm_encode(recv_txt[:1], recv_img[:1])["input_ids"][0].tolist().index(MM_VS) + 1
        img_pos = list(range(vs_off, vs_off + E2_N_IMAGE))                    # every image token: the donor differs from the receiver in one slot only, so the swap transfers one fact
        # donor states at every image token, every layer (one prompt per forward: no padding, positions are absolute)
        REC = {L: np.zeros((len(PP), E2_N_IMAGE, hf_model.config.get_text_config().hidden_size), np.float16) for L in E2P_LAYERS_FULL}
        REC2 = {L: np.zeros_like(REC[L]) for L in E2P_LAYERS_FULL}
        for j, p in enumerate(PP.itertuples()):
            _, R = mm_forward(recv_txt[j:j + 1], don_img[j:j + 1], E2P_LAYERS_FULL, where=lambda ids, _t=img_pos: list(_t))
            for L in E2P_LAYERS_FULL:
                REC[L][j] = R[L][0].half().numpy()
            _, R = mm_forward(recv_txt[j:j + 1], don2_img[j:j + 1], E2P_LAYERS_FULL, where=lambda ids, _t=img_pos: list(_t))
            for L in E2P_LAYERS_FULL:
                REC2[L][j] = R[L][0].half().numpy()
        pos = [(lambda s, _n=n_alls[j]: np.array([s - _n + t for t in img_pos])) for j in range(len(PP))]
        arms = [("none", None), ("full, every layer", e2p_pos_hooks(E2P_LAYERS_FULL, REC, pos)), ("full, read layers", e2p_pos_hooks(LAYERS, REC, pos)),
                ("J part, read layers", e2p_pos_hooks(LAYERS, REC, pos, Qs, "lens")), ("random part, read layers", e2p_pos_hooks(LAYERS, REC, pos, Qr, "lens")),
                ("complement, read layers", e2p_pos_hooks(LAYERS, REC, pos, Qs, "complement")), ("non-target slot, read layers", e2p_pos_hooks(LAYERS, REC2, pos))]
        if JPP is not None:
            arms.insert(4, ("J++ part, read layers", e2p_pos_hooks(LAYERS, REC, pos, Qpp, "lens")))
        for name, hooks in arms:
            lg, _ = mm_forward(recv_txt, recv_img, [], hooks=hooks)
            P, mass = e_answer(lg, E_WORDS)
            pr = torch.softmax(lg.float(), -1)
            pyes = sum(pr[:, t].numpy() for t in YES_IDS) / np.maximum(sum(pr[:, t].numpy() for t in YES_IDS) + sum(pr[:, t].numpy() for t in NO_IDS), 1e-9)
            gen = mm_generate(recv_txt, recv_img, 6, hooks=hooks)                 # judged generations: the first token is the noun only ~37% of the time (E2Pd, change #28)
            if question == "V":
                sd = JUDGE.name_match(gen, [p.donor_cls for p in PP.itertuples()])
                sr = JUDGE.name_match(gen, [p.receiver_cls for p in PP.itertuples()])
                sy = [None] * len(PP)
            else:                                                                 # VT: the first word of the generation, yes / no (the first token is neither: yes/no mass 0.00, change #32)
                sd, sr = [None] * len(PP), [None] * len(PP)
                sy = [(True if g.strip().lower().startswith("yes") else False if g.strip().lower().startswith("no") else None) for g in gen]
            for j, p in enumerate(PP.itertuples()):
                rows_out.append(dict(scene=p.scene, k=k, question=question, arm=name, q=p.q, receiver_cls=p.receiver_cls, donor_cls=p.donor_cls, txt_target=p.txt_target,
                                     p_receiver=float(P[j, wi[p.receiver_cls]]), p_donor=float(P[j, wi[p.donor_cls]]), p_yes=float(pyes[j]),
                                     same_receiver=(p.receiver_cls == p.txt_target), same_donor=(p.donor_cls == p.txt_target), mass=float(mass[j]),
                                     generation=gen[j], says_donor=sd[j], says_receiver=sr[j], says_yes=sy[j]))
        print(f"k = {k} {question}: " + "  ".join(f"{name.split(',')[0]} p(donor) {np.mean([x['p_donor'] for x in rows_out if x['k'] == k and x['question'] == question and x['arm'] == name]):.2f}" for name, _ in arms) + f"  [{time.time() - t0:.0f}s]")
E2P = pd.DataFrame(rows_out)
E2P.to_csv(f"{RUN_DIR}/e2_patching.csv", index=False)
resE2P = dict(by_k={}, n_pairs=int(len(PP)))
for k in E2P_KS:
    d = E2P[(E2P.k == k) & (E2P.question == "V")]
    base = d[d.arm == "none"].set_index("scene")
    eff = {}
    for name in d.arm.unique():
        if name == "none":
            continue
        x = d[d.arm == name].set_index("scene")
        dp = (x.p_donor - base.p_donor).dropna().values
        rng = np.random.default_rng(0)
        bs = [rng.choice(dp, len(dp)).mean() for _ in range(2000)]
        eff[name] = dict(dp_donor=(float(dp.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))), flipped=float(x.says_donor.astype(float).fillna(0).mean()),
                         says_receiver=float(x.says_receiver.astype(float).fillna(0).mean()), flipped_first_token=float(((x.p_donor > x.p_receiver) & (base.p_receiver > base.p_donor)).mean()))
    dvt = E2P[(E2P.k == k) & (E2P.question == "VT")]
    bvt = dvt[dvt.arm == "none"].set_index("scene")
    eff_vt = {}
    for name in dvt.arm.unique():
        if name == "none":
            continue
        x = dvt[dvt.arm == name].set_index("scene")
        # predicted direction: p(yes) should move toward the donor's truth (same_donor) and away from the receiver's (same_receiver)
        sign = np.where(x.same_donor.values, 1.0, -1.0) * np.where(x.same_receiver.values != x.same_donor.values, 1.0, 0.0)
        dy = ((x.says_yes.astype(float).fillna(0.5) - bvt.says_yes.astype(float).fillna(0.5)).values * sign)   # judged yes/no from the generation (change #32)
        dy = dy[sign != 0]
        if len(dy):
            rng = np.random.default_rng(0)
            bs = [rng.choice(dy, len(dy)).mean() for _ in range(2000)]
            eff_vt[name] = (float(dy.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    ok = eff["full, every layer"]["flipped"] >= 0.8                             # amended 02:20: judged flips only (the first-token Δp was the broken metric)
    jp, rp, cp = eff["J part, read layers"]["dp_donor"], eff["random part, read layers"]["dp_donor"], eff["complement, read layers"]["dp_donor"]
    sj = d[d.arm == "J part, read layers"].set_index("scene").says_donor.astype(float).fillna(0)
    sr_ = d[d.arm == "random part, read layers"].set_index("scene").says_donor.astype(float).fillna(0)
    fd = (sj - sr_.loc[sj.index]).values
    rng = np.random.default_rng(0)
    fbs = [rng.choice(fd, len(fd)).mean() for _ in range(2000)]
    eff["J minus random, judged flips"] = (float(fd.mean()), float(np.percentile(fbs, 2.5)), float(np.percentile(fbs, 97.5)))
    fdm = eff["J minus random, judged flips"]
    where = ("uninterpretable: the full-state swap does not transfer the fact" if not ok else "the J-readable part carries a causally effective fact" if fdm[0] >= 0.2 and fdm[1] > 0
             else "the J-readable part is not specifically effective" if fdm[2] < 0.2 else "undetermined at this n")
    resE2P["by_k"][k] = dict(effects=eff, effects_VT=eff_vt, control_ok=bool(ok), verdict=where)
    print(f"k = {k}: full swap flips {eff['full, every layer']['flipped']:.2f} (Δp donor {b_fmt(eff['full, every layer']['dp_donor'])}); read layers full {eff['full, read layers']['dp_donor'][0]:.2f}, J part {jp[0]:.2f}, random {rp[0]:.2f}, complement {cp[0]:.2f}, non-target {eff['non-target slot, read layers']['dp_donor'][0]:.2f} -> {where}"
          + (f" | VT (joint) predicted-direction Δp(yes): full {eff_vt.get('full, every layer', (np.nan,))[0]:.2f}, J part {eff_vt.get('J part, read layers', (np.nan,))[0]:.2f}, random {eff_vt.get('random part, read layers', (np.nan,))[0]:.2f}" if eff_vt else ""))
resE2P["seconds"] = time.time() - t0
save(resE2P, "eE2P_results")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(wspace=0.5))
names = ["full, every layer", "full, read layers", "J part, read layers"] + (["J++ part, read layers"] if JPP is not None else []) + ["random part, read layers", "complement, read layers", "non-target slot, read layers"]
short = ["full\nevery layer", "full\nread layers", "J part"] + (["J++ part"] if JPP is not None else []) + ["random\npart", "comple-\nment", "non-target\nslot"]
for j, k in enumerate(E2P_KS):
    e = resE2P["by_k"][k]["effects"]
    xs = np.arange(len(names)) + (j - 0.5) * 0.38
    v = [e[n]["dp_donor"] for n in names]
    ax[0].bar(xs, [t[0] for t in v], 0.38, color=[OKABE[1], OKABE[0]][j], alpha=0.85)
    ax[0].errorbar(xs, [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
    ax[0].text(0.98, 0.97 - 0.07 * j, f"k = {k}: {resE2P['by_k'][k]['verdict']}", color=[OKABE[1], OKABE[0]][j], fontsize=8, transform=ax[0].transAxes, va="top", ha="right")
    ev = resE2P["by_k"][k]["effects_VT"]
    v = [ev.get(n, (np.nan, np.nan, np.nan)) for n in names]
    ax[1].bar(xs, [t[0] for t in v], 0.38, color=[OKABE[1], OKABE[0]][j], alpha=0.85)
    ax[1].errorbar(xs, [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
ax[0].set(xticks=np.arange(len(names)), xticklabels=short, ylabel="Δ p(donor object) for \"what is in image slot q\"\n(n pairs = %d)" % len(PP))
ax[1].set(xticks=np.arange(len(names)), xticklabels=short, ylabel="joint question: Δ p(yes) in the predicted direction")
for a_ in ax:
    a_.axhline(0, color="0.6", lw=0.8)
    a_.tick_params(axis="x", labelsize=7.5)
fig.suptitle("E2P. Swapping one image slot's state at the read layers: " + "; ".join(f"k = {k}: {resE2P['by_k'][k]['verdict']}" for k in E2P_KS), fontsize=9.5, y=1.0)
ws_finish(fig, "figE2_patching", f"E2P · {len(PP)} donor/receiver pairs")

# %% ---------------------------------------------------------------- N1: is the J-readable component of the image tokens necessary for answering GQA?
# Natural images, the benchmark's own questions (GQA testdev-balanced, open "query" questions with a single-token answer), binned by the
# length of the question's semantic program (2 / 3 / >= 4 operations). Arms, all at the read layers over every image token:
# clean; no image (the language prior); ablate the J-readable r-dim part (the top-r right singular directions of W_U J_L, as in B3s);
# ablate an equal-rank random part; ablate the top-r principal components of the image-token residuals (the energy-matched alternative);
# keep only the J part (ablate the complement). Outcomes per question: log p(reference first token), the greedy answer judged by the
# Gemini judge against the reference (regex kept as a secondary), and the reference word's J-lens / logit-lens / J++ rank at the image
# tokens and at the decision position in the clean run. Rules (written before the run): evidence necessary when no-image accuracy is
# below clean by >= 0.3; the J part is specifically necessary when ablating it lowers log p(reference) more than the random part by
# >= 0.5 nats with the image-grouped interval excluding 0, and at least as much as the energy-matched PCA part.
N1_N_IMAGES = int(os.environ.get("N1_N_IMAGES", 16))                           # confirmatory runs set 48 (RECORD.md)
N1_EXCLUDE_CSV = os.environ.get("N1_EXCLUDE_CSV", "")                           # a pilot's n1_gqa_necessity.csv: its images are excluded (disjoint confirmatory sample)
N1_RANKS = [int(x) for x in os.environ.get("N1_RANKS", "256,64").split(",")]  # 256 always (the verdict's rank); the 27B confirmatory adds 512 = the 4B's r/d (RECORD.md)
assert 256 in N1_RANKS
N1_BINS = {"2": [2], "3": [3], "4+": [4, 5, 6, 7, 8, 9]}
N1_GQA = "/workspace/nayan/data/bench/gqa"
N1_MAX_SIDE = 448
t0 = time.time()


def n1_hooks_ablate(Q, part, mu=None):
    """ablation = patch with a zero donor: part='lens' removes the Q-part, part='complement' keeps only the Q-part. With mu (per layer,
    the mean image-token residual over the dataset) the removed Q-part is replaced by the mean's Q-part instead of zero (mean-ablation)"""
    def hooks(enc, idx):
        b, s = enc["input_ids"].shape
        ids = enc["input_ids"]
        rows_pos = [[j for j, t in enumerate(ids[i].tolist()) if t == B_PAD] for i in range(b)]
        P = max(len(r) for r in rows_pos)
        Pm = torch.as_tensor([r + [r[-1]] * (P - len(r)) for r in rows_pos], device=DEV)
        rows = torch.arange(b, device=DEV)[:, None].expand(-1, P)
        out_ = []
        for L in LAYERS:
            Qd = torch.as_tensor(Q[L], device=DEV, dtype=torch.float32)
            Md = None if mu is None else (torch.as_tensor(mu[L], device=DEV, dtype=torch.float32) @ Qd) @ Qd.T

            def fn(mod, inp, out, Qd=Qd, Md=Md):
                h = residual_of(out)
                if h.shape[1] == 1:
                    return out
                x = h.float()
                v = x[rows, Pm]
                v = (v - (v @ Qd) @ Qd.T) if part == "lens" else (v @ Qd) @ Qd.T
                if Md is not None:
                    v = v + Md
                x[rows, Pm] = v
                x = x.to(h.dtype)
                return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
            out_.append((L, fn))
        return out_
    return hooks


def n1_hooks_ablate_rows(Qrows, mu=None):
    """per-question directions (Gurnee et al.'s token-associated J-lens vectors): Qrows[L] is [n_questions, d, r] with orthonormal
    columns per row; removes each question's own part at its image tokens, at the read layers (content-specific ablation, change #33).
    With mu (per layer, [d]) the removed part is replaced by the mean's part (mean-ablation)"""
    def hooks(enc, idx):
        b, s = enc["input_ids"].shape
        ids = enc["input_ids"]
        rows_pos = [[j for j, t in enumerate(ids[i].tolist()) if t == B_PAD] for i in range(b)]
        P = max(len(r) for r in rows_pos)
        Pm = torch.as_tensor([r + [r[-1]] * (P - len(r)) for r in rows_pos], device=DEV)
        rows = torch.arange(b, device=DEV)[:, None].expand(-1, P)
        out_ = []
        for L in LAYERS:
            Qd = torch.as_tensor(Qrows[L][np.asarray(idx)], device=DEV, dtype=torch.float32)   # [b, d, r]
            Md = None if mu is None else torch.einsum("br,bdr->bd", torch.einsum("d,bdr->br", torch.as_tensor(mu[L], device=DEV, dtype=torch.float32), Qd), Qd)

            def fn(mod, inp, out, Qd=Qd, Md=Md):
                h = residual_of(out)
                if h.shape[1] == 1:
                    return out
                x = h.float()
                v = x[rows, Pm]                                                    # [b, P, d]
                v = v - torch.einsum("bpr,bdr->bpd", torch.einsum("bpd,bdr->bpr", v, Qd), Qd)
                if Md is not None:
                    v = v + Md[:, None, :]
                x[rows, Pm] = v
                x = x.to(h.dtype)
                return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
            out_.append((L, fn))
        return out_
    return hooks


@torch.no_grad()
def n1_word_directions(L, token_ids):
    """unit J-lens token directions at layer L for a list of token ids: J_L^T W_U[t], normalised — [n, d, 1]"""
    V = (JAC[L].float().T @ U_OUT[token_ids].float().T).T                        # [n, d]
    V = V / V.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return V.cpu().numpy()[:, :, None]


@torch.no_grad()
def n1_pca_subspace(H, r):
    """top-r principal directions of the image-token residuals H [n, d] (centred), orthonormal [d, r]"""
    X = torch.as_tensor(H, dtype=torch.float32, device=DEV)
    X = X - X.mean(0, keepdim=True)
    q = min(r + 8, X.shape[1] - 1, X.shape[0] - 1)                             # the rank cannot exceed the sample count (smoke sizes; change #25)
    if q < r + 8:
        print(f"PCA control: only {X.shape[0]} samples, rank capped at {q - 8}")
    U, S, V = torch.pca_lowrank(X, q=q, center=False)
    return b_orth(V[:, :r].T.cpu().numpy())


# --- reads
try:
    MM_PROC
except NameError:
    MM_PROC = None
if MM_PROC is None:
    _problem = mm_preflight()
    assert not _problem, _problem
import io as _io
from judge import Judge
JUDGE = Judge(RUN_DIR)
assert JUDGE.available(), "GEMINI_API_KEY missing: the judge is required (PROTOCOL.md)"
B_PAD = int(getattr(hf_model.config, "image_token_id", B_PAD))                   # the image token of this model (Qwen <|image_pad|>, Gemma <|image_soft_token|>)
e_where_image = lambda n_image: (lambda ids: [j for j, t in enumerate(ids) if t == B_PAD][-n_image:] + [len(ids) - 1])   # positions by token id, any family
gq = pd.read_parquet(f"{N1_GQA}/testdev_balanced_instructions/testdev-00000-of-00001.parquet")
gq["plen"] = gq.semantic.apply(len)
gq["structural"] = gq.types.apply(lambda t: t["structural"])
gq = gq[(gq.structural == "query") & (gq.answer.str.split().str.len() == 1)].copy()
gq["ans_ids"] = gq.answer.apply(lambda a: single_ids([a, " " + a, a.capitalize(), " " + a.capitalize()]))
gq = gq[gq.ans_ids.apply(len) > 0]
gq["bin"] = gq.plen.apply(lambda n: next((b for b, v in N1_BINS.items() if n in v), None))
gq = gq[gq.bin.notna()]
rng = np.random.default_rng(0)
imgs_ok = [i for i, g in gq.groupby("imageId") if g.bin.nunique() == 3]
if N1_EXCLUDE_CSV:
    _ex = set(pd.read_csv(N1_EXCLUDE_CSV).imageId.astype(str))
    imgs_ok = [i for i in imgs_ok if str(i) not in _ex]
    print(f"excluding {len(_ex)} pilot images; {len(imgs_ok)} eligible images remain")
pick = [imgs_ok[j] for j in rng.permutation(len(imgs_ok))[:N1_N_IMAGES]]
rows_q = []
for iid in pick:
    g = gq[gq.imageId == iid]
    for b in N1_BINS:
        gb = g[g.bin == b]
        rows_q.append(gb.iloc[int(rng.integers(len(gb)))])
GQ = pd.DataFrame(rows_q).reset_index(drop=True)
imtab = pd.read_parquet(f"{N1_GQA}/testdev_balanced_images/testdev-00000-of-00001.parquet")
imtab = imtab[imtab.id.isin(pick)].set_index("id")


def n1_image(iid):
    im = Image.open(_io.BytesIO(imtab.loc[iid, "image"]["bytes"])).convert("RGB")
    im.thumbnail((N1_MAX_SIDE, N1_MAX_SIDE))
    w, h = (im.width // 32) * 32, (im.height // 32) * 32
    return im.resize((max(w, 64), max(h, 64)))


IM = {iid: n1_image(iid) for iid in pick}
print(f"GQA: {len(pick)} images x 3 questions (program length bins {list(N1_BINS)}), {len(gq)} eligible query questions in the pool  [{time.time() - t0:.0f}s]")
mm_contact_sheet([(f"{r.imageId}: {r.question} -> {r.answer}"[:70], IM[r.imageId]) for r in GQ.head(6).itertuples()], 3, "figN1_stimuli")
Qs = {L: b_lens_subspace(L, max(N1_RANKS)) for L in LAYERS}
Qr = {L: b_random_subspace(Qs[L].shape[0], max(N1_RANKS), 100 + L) for L in LAYERS}
Qpp = {L: e_jpp_subspace(L, max(N1_RANKS)) for L in LAYERS} if JPP is not None else None
for L in LAYERS:                                                                 # Q[:, :r] must be the top-r (change #24)
    b_assert_descending(L, Qs[L])
    if Qpp is not None:
        b_assert_descending(L, Qpp[L], JPP)
Qu = {L: b_unembed_subspace(max(N1_RANKS)) for L in LAYERS}                       # output-sensitivity-matched control: top-r directions of W_U alone (change #30)
# clean run with readouts, and the image-token residuals for the PCA control
n1_prompt = lambda q: e_chat_media(["IMAGE", f"{q} Answer with one word."])
n1_prompt_noimg = lambda q: chat([("user", f"Answer the question about a photo as best you can. {q} Answer with one word.")])
texts = [n1_prompt(r.question) for r in GQ.itertuples()]
images = [IM[r.imageId] for r in GQ.itertuples()]
rows_out, H_img = [], {L: [] for L in LAYERS}
for iid in pick:                                                                 # one image at a time: its questions share the token count
    sel = np.flatnonzero((GQ.imageId == iid).values)
    n_image = len([t for t in mm_encode([texts[sel[0]]], [images[sel[0]]])["input_ids"][0].tolist() if t == B_PAD])
    rj, rl, lg = e_ranks_at([texts[i] for i in sel], [images[i] for i in sel], e_where_image(n_image), n_image, [GQ.ans_ids.iloc[i] for i in sel], chunk=8)
    rp = E_RP[-1]
    _, R = mm_forward([texts[i] for i in sel], [images[i] for i in sel], LAYERS, where=e_where_image(n_image))
    for L in LAYERS:
        H_img[L].append(R[L][:, :-1].reshape(-1, R[L].shape[-1]).numpy())
    for j, i in enumerate(sel):
        rows_out.append(dict(i=int(i), img_rank_j=int(rj[j, :, :-1, j].min()), img_rank_l=int(rl[j, :, :-1, j].min()), img_rank_p=int(rp[j, :, :-1, j].min()),
                             dec_rank_j=int(rj[j, :, -1, j].min()), dec_rank_l=int(rl[j, :, -1, j].min()), dec_rank_p=int(rp[j, :, -1, j].min()), n_image=n_image))
RO = pd.DataFrame(rows_out).set_index("i").sort_index()
Qpca = {L: n1_pca_subspace(np.concatenate(H_img[L]), max(N1_RANKS)) for L in LAYERS}
def n1_energy(Q, r):
    """mean share of the image-token residual's squared norm inside the first r directions of Q, over read layers (what an ablation removes)"""
    out = []
    for L in LAYERS:
        X = torch.as_tensor(np.concatenate(H_img[L]), dtype=torch.float32, device=DEV)
        Qd = torch.as_tensor(Q[L][:, :r], dtype=torch.float32, device=DEV)
        out.append(float(((X @ Qd) ** 2).sum(1).mean() / (X ** 2).sum(1).mean()))
    return float(np.mean(out))
N1_ENERGY = {f"J r={r}": n1_energy(Qs, r) for r in N1_RANKS} | {f"random r={r}": n1_energy(Qr, r) for r in N1_RANKS} | {f"PCA r={r}": n1_energy(Qpca, r) for r in N1_RANKS} | {f"W_U r={r}": n1_energy(Qu, r) for r in N1_RANKS}
if Qpp is not None:
    N1_ENERGY.update({f"J++ r={r}": n1_energy(Qpp, r) for r in N1_RANKS})
print("energy share of the image-token residual removed by each ablation: " + ", ".join(f"{k} {v:.3f}" for k, v in N1_ENERGY.items()))
# how much of the J-readable subspace lies inside the top-PCA span (and inside a random span): mean over read layers of ||Qpca^T Qj||_F^2 / r
N1_OVERLAP = {}
for r in N1_RANKS:
    N1_OVERLAP[f"J in PCA r={r}"] = float(np.mean([np.linalg.norm(Qpca[L][:, :r].T @ Qs[L][:, :r]) ** 2 / r for L in LAYERS]))
    N1_OVERLAP[f"J in random r={r}"] = float(np.mean([np.linalg.norm(Qr[L][:, :r].T @ Qs[L][:, :r]) ** 2 / r for L in LAYERS]))
    N1_OVERLAP[f"J in W_U r={r}"] = float(np.mean([np.linalg.norm(Qu[L][:, :r].T @ Qs[L][:, :r]) ** 2 / r for L in LAYERS]))
    if Qpp is not None:
        N1_OVERLAP[f"J in J++ r={r}"] = float(np.mean([np.linalg.norm(Qpp[L][:, :r].T @ Qs[L][:, :r]) ** 2 / r for L in LAYERS]))
print("subspace overlap (share of the J basis inside the other span; random = chance level r/d): " + ", ".join(f"{k} {v:.3f}" for k, v in N1_OVERLAP.items()))
MU = {L: np.concatenate(H_img[L]).mean(0) for L in LAYERS}                        # the mean image-token residual per read layer (mean-ablation arms)
del H_img
arms = [("clean", None, None), ("no image", None, "noimg")]
for r in N1_RANKS:
    arms += [(f"ablate J part r={r}", n1_hooks_ablate({L: Qs[L][:, :r] for L in LAYERS}, "lens"), None),
             (f"ablate random part r={r}", n1_hooks_ablate({L: Qr[L][:, :r] for L in LAYERS}, "lens"), None),
             (f"ablate PCA part r={r}", n1_hooks_ablate({L: Qpca[L][:, :r] for L in LAYERS}, "lens"), None),
             (f"ablate W_U part r={r}", n1_hooks_ablate({L: Qu[L][:, :r] for L in LAYERS}, "lens"), None)]
    if Qpp is not None:
        arms.append((f"ablate J++ part r={r}", n1_hooks_ablate({L: Qpp[L][:, :r] for L in LAYERS}, "lens"), None))
arms.append(("keep only J part r=256", n1_hooks_ablate({L: Qs[L][:, :256] for L in LAYERS}, "complement"), None))
# content-specific (Gurnee et al.'s token-associated J-lens vectors, rank 1): the reference word's own lens direction vs another question's word
_tok = [int(GQ.ans_ids.iloc[i][0]) for i in range(len(GQ))]
_oth = [next(_tok[(i + j) % len(_tok)] for j in range(1, len(_tok)) if _tok[(i + j) % len(_tok)] != _tok[i]) for i in range(len(_tok))]
Qtar = {L: n1_word_directions(L, _tok) for L in LAYERS}
Qoth = {L: n1_word_directions(L, _oth) for L in LAYERS}
arms += [("ablate target word's lens direction", n1_hooks_ablate_rows(Qtar), None), ("ablate another word's lens direction", n1_hooks_ablate_rows(Qoth), None)]
if os.environ.get("N1_MEAN", "1") == "1":                                        # mean-ablation arms (robustness to off-manifold zeroing; added 14 Oct 02:00, RECORD.md)
    arms += [("mean-ablate J part r=256", n1_hooks_ablate({L: Qs[L][:, :256] for L in LAYERS}, "lens", mu=MU), None),
             ("mean-ablate random part r=256", n1_hooks_ablate({L: Qr[L][:, :256] for L in LAYERS}, "lens", mu=MU), None),
             ("mean-ablate PCA part r=256", n1_hooks_ablate({L: Qpca[L][:, :256] for L in LAYERS}, "lens", mu=MU), None),
             ("mean-ablate W_U part r=256", n1_hooks_ablate({L: Qu[L][:, :256] for L in LAYERS}, "lens", mu=MU), None)]
res_rows = []
for name, hooks, mode in arms:
    tx = [n1_prompt_noimg(r.question) for r in GQ.itertuples()] if mode == "noimg" else texts
    im = None if mode == "noimg" else images
    lg, _ = mm_forward(tx, im, [], hooks=hooks)
    pr = torch.log_softmax(lg.float(), -1)
    logp = np.array([float(torch.logsumexp(pr[i, GQ.ans_ids.iloc[i]], 0)) for i in range(len(GQ))])
    gen = mm_generate(tx, im if im is not None else [None] * len(tx), 6, hooks=hooks)
    jv = JUDGE.qa_match(GQ.question.tolist(), gen, GQ.answer.tolist())
    rx = e_gen_correct(gen, GQ.answer.str.lower().tolist(), set(GQ.answer.str.lower()))
    for i, r in enumerate(GQ.itertuples()):
        res_rows.append(dict(arm=name, i=i, imageId=r.imageId, qid=r.id, bin=r.bin, plen=int(r.plen), question=r.question, reference=r.answer, generation=gen[i], correct=jv[i], correct_regex=bool(rx[i]), logp_ref=logp[i]))
    acc = np.nanmean([float(v) for v in jv if v is not None])
    print(f"{name:28s} accuracy (judge) {acc:.2f}  mean log p(ref) {logp.mean():.2f}  judge failures {sum(v is None for v in jv)}  [{time.time() - t0:.0f}s]")
N1 = pd.DataFrame(res_rows).merge(RO, left_on="i", right_index=True)
N1.to_csv(f"{RUN_DIR}/n1_gqa_necessity.csv", index=False)
JUDGE.to_label(N1[N1.arm == "clean"], "n1_clean")
# --- summary with image-grouped bootstraps
clean = N1[N1.arm == "clean"].set_index("i")
_TW, _OW = "ablate target word's lens direction", "ablate another word's lens direction"
resN1 = dict(n_images=len(pick), n_questions=int(len(GQ)), judge=JUDGE.model, by_bin={}, readout={})
for b in list(N1_BINS) + ["all"]:
    sel = (N1.bin == b) if b != "all" else np.ones(len(N1), bool)
    d = N1[sel]
    out = {}
    for name in d.arm.unique():
        x = d[d.arm == name].set_index("i")
        dlog = (x.logp_ref - clean.loc[x.index].logp_ref).values
        out[name] = dict(accuracy=b_boot_scene(x.correct.astype(float).fillna(0).values, x.imageId.values), dlogp=b_boot_scene(dlog, x.imageId.values))
    resN1["by_bin"][b] = out
    j, rnd, pca = (out.get(f"ablate J part r=256", {}).get("dlogp", (np.nan,) * 3), out.get("ablate random part r=256", {}).get("dlogp", (np.nan,) * 3), out.get("ablate PCA part r=256", {}).get("dlogp", (np.nan,) * 3))
    print(f"bin {b:3s} (n = {int(sel.sum() // len(arms))}): acc clean {out['clean']['accuracy'][0]:.2f} no-image {out['no image']['accuracy'][0]:.2f} | Δlogp: J {b_fmt(j)}  random {b_fmt(rnd)}  PCA {b_fmt(pca)}  keep-J-only {out['keep only J part r=256']['dlogp'][0]:.2f}"
          + (f"  J++ {b_fmt(out['ablate J++ part r=256']['dlogp'])}" if "ablate J++ part r=256" in out else "")
          + (f"  W_U {b_fmt(out['ablate W_U part r=256']['dlogp'])}" if "ablate W_U part r=256" in out else "")
          + (f"  | mean-ablate: J {b_fmt(out['mean-ablate J part r=256']['dlogp'])} random {out['mean-ablate random part r=256']['dlogp'][0]:.2f} PCA {out['mean-ablate PCA part r=256']['dlogp'][0]:.2f}" if "mean-ablate J part r=256" in out else "")
          + (f" W_U {out['mean-ablate W_U part r=256']['dlogp'][0]:.2f}" if "mean-ablate W_U part r=256" in out else "")
          + (f"  | word direction: target {b_fmt(out[_TW]['dlogp'])} other {out[_OW]['dlogp'][0]:.2f}" if _TW in out else ""))
for key, lab in (("img_rank_j", "J-lens on image tokens"), ("img_rank_l", "logit lens on image tokens"), ("img_rank_p", "J++ on image tokens"), ("dec_rank_j", "J-lens at the decision position"), ("dec_rank_l", "logit lens at the decision position"), ("dec_rank_p", "J++ at the decision position")):
    v = clean[key].values
    resN1["readout"][lab] = b_boot_scene((v >= 1) & (v <= B_TOP), clean.imageId.values) if (v >= 1).any() else None
print("reference word readable (top-10): " + ", ".join(f"{k} {v[0]:.2f}" for k, v in resN1["readout"].items() if v))
# the same, restricted to questions the model answers correctly with the image (competence-stratified, reported beside the unconditional)
ok_i = clean.index[clean.correct == True]
strat = {}
for name in N1.arm.unique():
    x = N1[(N1.arm == name) & (N1.i.isin(ok_i))].set_index("i")
    if len(x):
        strat[name] = dict(accuracy=b_boot_scene(x.correct.astype(float).fillna(0).values, x.imageId.values), dlogp=b_boot_scene((x.logp_ref - clean.loc[x.index].logp_ref).values, x.imageId.values), n=int(len(x)))
resN1["clean_correct_stratified"] = strat
if strat:
    print(f"clean-correct subset (n = {strat['clean']['n']}): " + "  ".join(f"{k.replace(' part', '')}: acc {v['accuracy'][0]:.2f} Δlogp {v['dlogp'][0]:.2f}" for k, v in strat.items() if k != "clean"))
a = resN1["by_bin"]["all"]
resN1["evidence_dlogp_clean_minus_noimage"] = b_boot_scene((clean.logp_ref - N1[N1.arm == "no image"].set_index("i").loc[clean.index].logp_ref).values, clean.imageId.values)
print(f"evidence: clean − no-image log p(ref) = {b_fmt(resN1['evidence_dlogp_clean_minus_noimage'])} nats")
resN1["energy_removed"] = N1_ENERGY
resN1["subspace_overlap"] = N1_OVERLAP
ev = resN1["evidence_dlogp_clean_minus_noimage"]
necessary = ev[0] >= 1.0 and ev[1] > 0                                       # amended 14 Oct 01:50: log-p evidence gate (RECORD.md)
jd, rd, pd_ = a["ablate J part r=256"]["dlogp"], a["ablate random part r=256"]["dlogp"], a["ablate PCA part r=256"]["dlogp"]
# paired J-minus-random difference with its own interval
xj = N1[N1.arm == "ablate J part r=256"].set_index("i").logp_ref; xr = N1[N1.arm == "ablate random part r=256"].set_index("i").logp_ref
dd = b_boot_scene((xj - xr).values, clean.imageId.values)
resN1["J_minus_random_r256"] = dd
resN1["verdict"] = ("evidence not necessary (no-image gate failed)" if not necessary else
                    "the J-readable part is specifically necessary" if dd[0] <= -0.5 and dd[2] < 0 and jd[0] <= pd_[0] else
                    "the J-readable part is not specifically necessary (random/PCA ablations hurt as much)" if dd[2] > -0.5 and dd[1] > -1.0 else "undetermined at this n")
resN1["seconds"] = time.time() - t0
save(resN1, "eN1_results")
print(f"\nJ − random (r = 256) {b_fmt(dd)} nats -> {resN1['verdict']}")
# --- figure
fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(wspace=0.5))
names = [n for n, _, _ in arms if n != "clean"]
cols = {"no image": "0.4", "ablate J part r=256": OKABE[0], "ablate random part r=256": OKABE[4], "ablate PCA part r=256": OKABE[3], "ablate J++ part r=256": OKABE[2],
        "ablate J part r=64": OKABE[0], "ablate random part r=64": OKABE[4], "ablate PCA part r=64": OKABE[3], "ablate J++ part r=64": OKABE[2], "keep only J part r=256": OKABE[1],
        "mean-ablate J part r=256": OKABE[0], "mean-ablate random part r=256": OKABE[4], "mean-ablate PCA part r=256": OKABE[3],
        "ablate W_U part r=256": OKABE[5], "ablate W_U part r=64": OKABE[5], "mean-ablate W_U part r=256": OKABE[5], "mean-ablate J++ part r=256": OKABE[2],
        "ablate target word's lens direction": OKABE[6], "ablate another word's lens direction": "0.55"}
bins = list(N1_BINS)
w = 0.8 / len(names)
for j, name in enumerate(names):
    v = [resN1["by_bin"][b][name]["dlogp"] for b in bins]
    xs = np.arange(len(bins)) + (j - len(names) / 2 + 0.5) * w
    solid = ("r=256" in name and "mean" not in name) or name == "no image"
    ax[0].bar(xs, [t[0] for t in v], w, color=cols.get(name, "0.6"), alpha=0.9 if solid else 0.45, hatch="" if solid else ("xx" if "mean" in name else "//"))
    ax[0].errorbar(xs, [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
    ax[0].text(0.02, 0.97 - 0.05 * j, name, color=cols.get(name, "0.6"), fontsize=7.5, transform=ax[0].transAxes, va="top")
ax[0].axhline(0, color="0.6", lw=0.8)
_lo, _hi = ax[0].get_ylim()
ax[0].set_ylim(_lo, _hi + (0.1 + 0.065 * len(names)) * (_hi - _lo))                 # headroom for the key (the bars point down)
ax[0].set(xticks=np.arange(len(bins)), xticklabels=[f"program length {b}" for b in bins], ylabel="Δ log p(reference answer) vs clean, image-grouped 95% CI", xlabel=f"GQA question complexity (n = {len(pick)} images × 1 question per bin)")
fig.subplots_adjust(bottom=0.18)
for j, name in enumerate(["clean"] + names):
    v = [resN1["by_bin"][b][name]["accuracy"] for b in bins]
    ax[1].plot(np.arange(len(bins)), [t[0] for t in v], "-o", color=cols.get(name, "0.2") if name != "clean" else "black", ms=4, lw=1.4, alpha=0.9 if "r=256" in name or name in ("clean", "no image") else 0.4)
ax[1].set(xticks=np.arange(len(bins)), xticklabels=[f"program length {b}" for b in bins], ylabel="accuracy (Gemini judge vs reference)", ylim=(0, 1.04))
ax[1].text(0.02, 0.97, "black: clean", color="black", fontsize=7.5, transform=ax[1].transAxes, va="top")
fig.suptitle(f"N1 ({MODEL_TAG}). {resN1['verdict']}: J − random at r = 256 = {b_fmt(dd)} nats; no-image accuracy {a['no image']['accuracy'][0]:.2f} vs clean {a['clean']['accuracy'][0]:.2f}", fontsize=9.5, y=1.0)
ws_finish(fig, "figN1_gqa_necessity", f"N1 · GQA testdev-balanced · {len(GQ)} questions · judge {JUDGE.model}")

# %% ---------------------------------------------------------------- T1: transport cost by modality. Is the J-readable part of the EVIDENCE tokens equally necessary for a picture, for printed words and for text?
# The E2 scenes (six numbered slots) rendered three ways: image (COCO crops in the slots), printed (the object's name printed in the slot:
# visual text), text (the lettered list); the same single-fact question, loads k in T1_KS. Arms, applied at the read layers to every
# EVIDENCE token (the image tokens; the list tokens): clean; no evidence (an empty grid / an all-empty list); ablate the J-readable r-dim
# part; ablate an equal-rank random part; ablate the top-r principal components of that rendering's evidence-token residuals; keep only the
# J part. For the two pixel renderings, four localized arms: the J part / a random part ablated on the TARGET slot's 49 tokens alone, and on
# one OTHER slot's 49 tokens (the second slot in fill order: occupied when k >= 2, empty at k = 1).
# Outcome per scene: log p(target noun) and the greedy answer (Gemini judge). Estimands per rendering: evidence = clean - no-evidence (nats);
# cost_J = random - J (paired); share = cost_J / evidence (aggregate paired effects, scene bootstrap). Rules (written before the run):
# evidence gate >= 1 nat (interval > 0) per rendering and clean accuracy >= 0.9 (perception); the J part is specifically necessary in a
# rendering when cost_J >= 0.5 nats with the interval excluding 0; between two renderings, "modality-dependent transport" when the share
# difference has |d| >= 0.2 with the scene-paired interval excluding 0, "modality-independent" when that interval lies inside [-0.2, 0.2]
# (equivalence), else undetermined. Printed words "route like language" when printed - text is inside [-0.2, 0.2] and image - printed >= 0.2
# (interval > 0); "like vision" when printed - image is inside [-0.2, 0.2] and printed - text >= 0.2; else between. Evidence-specific when
# cost_J(target slot) - cost_J(other slot) >= 0.5 with the interval > 0 and the random part on the target slot costs <= 0.2; "spread beyond
# the slot" when the target-slot J cost is under a quarter of the whole-grid J cost while the latter is >= 1 nat. The energy removed and
# the number of evidence tokens differ by construction between the renderings and are reported, not matched.
T1_KS = [int(x) for x in os.environ.get("T1_KS", "1,6").split(",")]
T1_RANKS = [256, 64]
T1_N = int(os.environ.get("T1_N", E_N_SCENES))
T1_RENDS = ["image", "printed", "text"]
T1_Q_PRINTED = "What word is written in image slot {q}? Reply with the word only."
t0 = time.time()


def t1_word_tile(word):
    """the object's name printed on a tile of the crop size (visual text on the same canvas geometry as the crops)"""
    tile = Image.new("RGB", (E_CROP, E_CROP), E_BG)
    d = ImageDraw.Draw(tile)
    font = e_font(40)
    bb = d.textbbox((0, 0), word, font=font)
    d.text(((E_CROP - (bb[2] - bb[0])) // 2 - bb[0], (E_CROP - (bb[3] - bb[1])) // 2 - bb[1]), word, fill="black", font=font)
    return tile


def t1_render(r, k, mod_):
    """(prompt, image or None, target noun) for scene r at load k under a rendering"""
    if mod_ == "image":
        p, img, _, _ = e2_render(r, k, "V0")
        return p, img, r.img_cls[r.q - 1]
    if mod_ == "text":
        p, _, _, _ = e2_render(r, k, "T0")
        return p, None, r.txt_cls[r.r - 1]
    occ = set(r.order[:k])
    img = e_canvas(E2_COLS, E2_ROWS, {s: (t1_word_tile(r.img_cls[s - 1]) if s in occ else None) for s in range(1, E2_N_SLOTS + 1)})
    return e_chat_media([E2_INTRO, "IMAGE", T1_Q_PRINTED.format(q=r.q)]), img, r.img_cls[r.q - 1]


def t1_text_span(prompt, list_text):
    """token indices (no padding) of the text list inside a text-only prompt, from character offsets"""
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    i = prompt.index(list_text)
    pos = [t for t, (a, b) in enumerate(enc["offset_mapping"]) if a < i + len(list_text) and b > i]
    assert pos, "list span not found"
    return pos


def t1_hooks_ablate(Q, part, pos_of, mu=None):
    """ablation at the read layers at the positions pos_of(unpadded_row, prompt_index) names (shifted by the row's left padding):
    part='lens' removes the Q-part, 'complement' keeps only the Q-part; with mu (per layer) the removed Q-part is replaced by the
    mean evidence-token residual's Q-part instead of zero (mean-ablation)"""
    def hooks(enc, idx):
        ids, am = enc["input_ids"], enc["attention_mask"]
        b, s = ids.shape
        rows_pos = []
        for i in range(b):
            n = int(am[i].sum())
            rows_pos.append([s - n + p for p in pos_of(ids[i, s - n:].tolist(), int(idx[i]))])
        P = max(len(r) for r in rows_pos)
        Pm = torch.as_tensor([r + [r[-1]] * (P - len(r)) for r in rows_pos], device=DEV)
        rows = torch.arange(b, device=DEV)[:, None].expand(-1, P)
        out_ = []
        for L in LAYERS:
            Qd = torch.as_tensor(Q[L], device=DEV, dtype=torch.float32)
            Md = None if mu is None else (torch.as_tensor(mu[L], device=DEV, dtype=torch.float32) @ Qd) @ Qd.T

            def fn(mod, inp, out, Qd=Qd, Md=Md):
                h = residual_of(out)
                if h.shape[1] == 1:
                    return out
                x = h.float()
                v = x[rows, Pm]
                v = (v - (v @ Qd) @ Qd.T) if part == "lens" else (v @ Qd) @ Qd.T
                if Md is not None:
                    v = v + Md
                x[rows, Pm] = v
                x = x.to(h.dtype)
                return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
            out_.append((L, fn))
        return out_
    return hooks


def t1_slot_pos(which):
    """image-token positions of one slot of the queried scene: the queried slot ('target') or the second slot in fill order ('other')"""
    def f(row, i):
        r = S2.iloc[i]
        slot = int(r.q) if which == "target" else int(r.order[1])
        ip = [j for j, t in enumerate(row) if t == B_PAD]
        return [ip[j] for j in e_slot_tokens(E2_COLS, E2_ROWS, slot)]
    return f


def t1_boot(frames, f, n=2000, seed=0):
    """scene bootstrap of f(*arrays): every frame is a Series on the same scene index"""
    rng_ = np.random.default_rng(seed)
    idx = frames[0].index.values
    vals = [f(*[fr.loc[idx[s]].values for fr in frames]) for s in rng_.integers(0, len(idx), (n, len(idx)))]
    return float(f(*[fr.values for fr in frames])), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --- reads
from judge import Judge
JUDGE = Judge(RUN_DIR)
assert JUDGE.available(), "GEMINI_API_KEY missing: the judge is required (PROTOCOL.md)"
S2 = e2_scenes().head(T1_N).reset_index(drop=True)
T1_N_IMAGE = E2_COLS * E2_ROWS * 49
Qs = {L: b_lens_subspace(L, max(T1_RANKS)) for L in LAYERS}
Qr = {L: b_random_subspace(Qs[L].shape[0], max(T1_RANKS), 100 + L) for L in LAYERS}
for L in LAYERS:
    b_assert_descending(L, Qs[L])                                                # Q[:, :r] must be the top-r (change #24)
Qu = {L: b_unembed_subspace(max(T1_RANKS)) for L in LAYERS}                       # output-sensitivity-matched control (change #30)
Qpp = {L: e_jpp_subspace(L, max(T1_RANKS)) for L in LAYERS} if JPP is not None else None   # J++ where the hub has a fit (change #31)
img_pos_of = lambda row, i: [j for j, t in enumerate(row) if t == B_PAD]
mm_contact_sheet([(f"{m} k=6", t1_render(S2.iloc[0], 6, m)[1]) for m in ("image", "printed")], 2, "figT1_stimuli")
res_rows, resT1 = [], dict(n_scenes=int(len(S2)), ks=T1_KS, renderings=T1_RENDS, judge=JUDGE.model, by_k={}, energy={}, n_evidence_tokens={})
for k in T1_KS:
    for mod_ in T1_RENDS:
        rend = [t1_render(r, k, mod_) for r in S2.itertuples()]
        texts, targets = [x[0] for x in rend], [x[2] for x in rend]
        images = [x[1] for x in rend] if mod_ != "text" else None
        tids = [E_SID[t] for t in targets]
        empty = {s: None for s in range(1, E2_N_SLOTS + 1)}
        if mod_ != "text":
            texts0, images0 = texts, [e_canvas(E2_COLS, E2_ROWS, empty)] * len(texts)
            pos_of, where, n_ev = img_pos_of, e_where_image(T1_N_IMAGE), T1_N_IMAGE
        else:
            texts0 = [chat([("user", E2_INTRO + "\n" + e2_text_list(empty) + "\n" + E2_Q["T"].format(q=r.q, rl=E2_LETTER[r.r - 1]))]) for r in S2.itertuples()]
            images0 = None
            spans = {}
            for j, r in enumerate(S2.itertuples()):
                occ = set(r.order[:k])
                lt = e2_text_list({s: (r.txt_cls[s - 1] if s in occ else None) for s in range(1, E2_N_SLOTS + 1)})
                spans[tuple(tokenizer(texts[j], add_special_tokens=False)["input_ids"])] = t1_text_span(texts[j], lt)
            P_ = max(len(v) for v in spans.values())
            n_ev = float(np.mean([len(v) for v in spans.values()]))
            pos_of = lambda row, i, _s=spans: _s[tuple(row)]

            def where(ids, _s=spans, _P=P_):
                p0 = next(j for j, t in enumerate(ids) if t != tokenizer.pad_token_id)
                span = [p0 + q for q in _s[tuple(ids[p0:])]]
                return span + [span[-1]] * (_P - len(span))
        resT1["n_evidence_tokens"][mod_] = n_ev
        # clean run with the evidence-token residuals (the PCA control is fitted per rendering)
        _, R = mm_forward(texts, images, LAYERS, where=where)
        H = {L: R[L][:, :-1].reshape(-1, R[L].shape[-1]).numpy() for L in LAYERS}
        del R
        Qpca = {L: n1_pca_subspace(H[L], max(T1_RANKS)) for L in LAYERS}

        def t1_energy(Q, r):
            out = []
            for L in LAYERS:
                X = torch.as_tensor(H[L], dtype=torch.float32, device=DEV)
                Qd = torch.as_tensor(Q[L][:, :r], dtype=torch.float32, device=DEV)
                out.append(float(((X @ Qd) ** 2).sum(1).mean() / (X ** 2).sum(1).mean()))
            return float(np.mean(out))
        resT1["energy"][f"k={k} {mod_}"] = {f"{nm} r={r}": t1_energy(Q, r) for nm, Q in (("J", Qs), ("random", Qr), ("PCA", Qpca), ("W_U", Qu)) + ((("J++", Qpp),) if Qpp is not None else ()) for r in T1_RANKS}
        MU_m = {L: H[L].mean(0) for L in LAYERS}                                  # the mean evidence-token residual of this rendering (mean-ablation arms)
        del H
        arms = [("clean", None, False), ("no evidence", None, True)]
        for r in T1_RANKS:
            arms += [(f"ablate J part r={r}", t1_hooks_ablate({L: Qs[L][:, :r] for L in LAYERS}, "lens", pos_of), False),
                     (f"ablate random part r={r}", t1_hooks_ablate({L: Qr[L][:, :r] for L in LAYERS}, "lens", pos_of), False),
                     (f"ablate PCA part r={r}", t1_hooks_ablate({L: Qpca[L][:, :r] for L in LAYERS}, "lens", pos_of), False),
                     (f"ablate W_U part r={r}", t1_hooks_ablate({L: Qu[L][:, :r] for L in LAYERS}, "lens", pos_of), False)]
            if Qpp is not None:
                arms.append((f"ablate J++ part r={r}", t1_hooks_ablate({L: Qpp[L][:, :r] for L in LAYERS}, "lens", pos_of), False))
        arms.append(("keep only J part r=256", t1_hooks_ablate({L: Qs[L][:, :256] for L in LAYERS}, "complement", pos_of), False))
        if os.environ.get("T1_MEAN", "1") == "1":                                  # mean-ablation arms (change #26; rule in RECORD.md 02:00 amendment)
            arms += [("mean-ablate J part r=256", t1_hooks_ablate({L: Qs[L][:, :256] for L in LAYERS}, "lens", pos_of, mu=MU_m), False),
                     ("mean-ablate random part r=256", t1_hooks_ablate({L: Qr[L][:, :256] for L in LAYERS}, "lens", pos_of, mu=MU_m), False),
                     ("mean-ablate PCA part r=256", t1_hooks_ablate({L: Qpca[L][:, :256] for L in LAYERS}, "lens", pos_of, mu=MU_m), False),
                     ("mean-ablate W_U part r=256", t1_hooks_ablate({L: Qu[L][:, :256] for L in LAYERS}, "lens", pos_of, mu=MU_m), False)]
            if Qpp is not None:
                arms.append(("mean-ablate J++ part r=256", t1_hooks_ablate({L: Qpp[L][:, :256] for L in LAYERS}, "lens", pos_of, mu=MU_m), False))
        if mod_ != "text":
            for which in ("target", "other"):
                arms += [(f"ablate J part r=256, {which} slot", t1_hooks_ablate({L: Qs[L][:, :256] for L in LAYERS}, "lens", t1_slot_pos(which)), False),
                         (f"ablate random part r=256, {which} slot", t1_hooks_ablate({L: Qr[L][:, :256] for L in LAYERS}, "lens", t1_slot_pos(which)), False)]
        for name, hooks, noev in arms:
            tx, im = (texts0, images0) if noev else (texts, images)
            lg, _ = mm_forward(tx, im, [], hooks=hooks)
            pr = torch.log_softmax(lg.float(), -1)
            logp = np.array([float(torch.logsumexp(pr[i, tids[i]], 0)) for i in range(len(S2))])
            gen = mm_generate(tx, im if im is not None else [None] * len(tx), 6, hooks=hooks)
            jv = JUDGE.name_match(gen, targets)
            for i, r in enumerate(S2.itertuples()):
                res_rows.append(dict(k=k, modality=mod_, arm=name, scene=int(r.scene), target=targets[i], generation=gen[i], correct=jv[i], logp=logp[i]))
            acc = np.nanmean([float(v) for v in jv if v is not None])
            print(f"k = {k} {mod_:7s} {name:36s} accuracy (judge) {acc:.2f}  mean log p(target) {logp.mean():.2f}  [{time.time() - t0:.0f}s]")
T1 = pd.DataFrame(res_rows)
T1.to_csv(f"{RUN_DIR}/t1_transport.csv", index=False)
JUDGE.to_label(T1[T1.arm == "clean"], "t1_clean")
print("energy share of the evidence-token residual removed: " + "; ".join(f"{kk}: " + ", ".join(f"{a} {v:.3f}" for a, v in d.items()) for kk, d in resT1["energy"].items()))
print("evidence tokens per prompt: " + ", ".join(f"{m} {v:.0f}" for m, v in resT1["n_evidence_tokens"].items()))


def t1_d(d, arm):
    """delta log p(target) vs clean per scene (negative = cost)"""
    c = d[d.arm == "clean"].set_index("scene").logp
    x = d[d.arm == arm].set_index("scene").logp
    return x - c.loc[x.index]


for k in T1_KS:
    out, F = {}, {}
    for mod_ in T1_RENDS:
        d = T1[(T1.k == k) & (T1.modality == mod_)]
        ev, j, rnd = -t1_d(d, "no evidence"), t1_d(d, "ablate J part r=256"), t1_d(d, "ablate random part r=256")
        F[mod_] = (j, rnd, ev)
        o = dict(accuracy={a: float(np.nanmean(d[d.arm == a].correct.astype(float).values)) for a in d.arm.unique()},
                 dlogp={a: t1_boot([t1_d(d, a)], lambda x: x.mean()) for a in d.arm.unique() if a != "clean"},
                 evidence=t1_boot([ev], lambda e: e.mean()), cost_J=t1_boot([j, rnd], lambda a, b: (b - a).mean()),
                 share=t1_boot([j, rnd, ev], lambda a, b, e: (b - a).mean() / e.mean()))
        o["seen"] = bool(o["accuracy"]["clean"] >= 0.9)
        o["gate"] = bool(o["seen"] and o["evidence"][0] >= 1 and o["evidence"][1] > 0)
        o["J_necessary"] = bool(o["cost_J"][0] >= 0.5 and o["cost_J"][1] > 0)
        if mod_ != "text":
            jt, rt, jo, ro = (t1_d(d, f"ablate {p} part r=256, {w} slot") for p, w in (("J", "target"), ("random", "target"), ("J", "other"), ("random", "other")))
            sl = dict(cost_J_target=t1_boot([jt, rt], lambda a, b: (b - a).mean()), cost_J_other=t1_boot([jo, ro], lambda a, b: (b - a).mean()),
                      target_minus_other=t1_boot([jt, rt, jo, ro], lambda a, b, c, e: (b - a).mean() - (e - c).mean()), random_target=t1_boot([rt], lambda x: -x.mean()))
            sl["verdict"] = ("evidence-specific" if sl["target_minus_other"][0] >= 0.5 and sl["target_minus_other"][1] > 0 and sl["random_target"][0] <= 0.2 else
                             "spread beyond the slot" if o["cost_J"][0] >= 1 and sl["cost_J_target"][0] < 0.25 * o["cost_J"][0] else "undetermined")
            o["slot"] = sl
        out[mod_] = o
    pairs = {}
    for a, b in (("image", "text"), ("printed", "text"), ("image", "printed")):
        (ja, ra, ea), (jb, rb, eb) = F[a], F[b]
        dsh = t1_boot([ja, ra, ea, jb, rb, eb], lambda x1, x2, x3, y1, y2, y3: (x2 - x1).mean() / x3.mean() - (y2 - y1).mean() / y3.mean())
        gate = out[a]["gate"] and out[b]["gate"]
        pairs[f"{a} - {b}"] = dict(share_diff=dsh, verdict=("gate failed" if not gate else f"modality-dependent transport ({a} > {b})" if dsh[1] > 0 and dsh[0] >= 0.2 else
                                                            f"modality-dependent transport ({b} > {a})" if dsh[2] < 0 and dsh[0] <= -0.2 else
                                                            "modality-independent (equivalence)" if dsh[1] >= -0.2 and dsh[2] <= 0.2 else "undetermined at this n"))
    pt, ip_ = pairs["printed - text"]["share_diff"], pairs["image - printed"]["share_diff"]
    printed = ("printed words route like language" if pt[1] >= -0.2 and pt[2] <= 0.2 and ip_[0] >= 0.2 and ip_[1] > 0 else
               "printed words route like vision" if ip_[1] >= -0.2 and ip_[2] <= 0.2 and pt[0] >= 0.2 and pt[1] > 0 else "printed words: between / undetermined")
    resT1["by_k"][k] = dict(**out, pairs=pairs, printed_verdict=printed, verdict=pairs["image - text"]["verdict"])
    for m in T1_RENDS:
        o = out[m]
        print(f"k = {k} {m:7s}: seen {o['seen']} evidence {b_fmt(o['evidence'])} nats | dlogp J {b_fmt(o['dlogp']['ablate J part r=256'])} random {b_fmt(o['dlogp']['ablate random part r=256'])} PCA {b_fmt(o['dlogp']['ablate PCA part r=256'])} keep-J-only {o['dlogp']['keep only J part r=256'][0]:.2f} | cost_J {b_fmt(o['cost_J'])} share {b_fmt(o['share'])}"
              + ("  J specifically necessary" if o["J_necessary"] else "  J not specifically necessary")
              + (f" | slot: J target {b_fmt(o['slot']['cost_J_target'])} other {b_fmt(o['slot']['cost_J_other'])} random target {o['slot']['random_target'][0]:.2f} -> {o['slot']['verdict']}" if "slot" in o else ""))
    print(f"k = {k}: " + "; ".join(f"share {p} {b_fmt(v['share_diff'])} -> {v['verdict']}" for p, v in pairs.items()) + f" | {printed}")
resT1["seconds"] = time.time() - t0
save(resT1, "eT1_results")
# --- figure: the arms by rendering at the largest load; the J share by rendering and load; the slot-localized arms
def t1_key(ax, labels, top=0.97, step=0.05):
    """a text key in the empty upper part of a bar panel (the bars point down); the y-limits get the headroom"""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (0.12 + step * len(labels) * 1.45) * (hi - lo))
    for j, (lab, c) in enumerate(labels):
        ax.text(0.02, top - step * j, lab, color=c, fontsize=7.5, transform=ax.transAxes, va="top")


def t1_spread(items, gap, lo, hi):
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    if ys and ys[-1] > hi:
        ys = [y - (ys[-1] - hi) for y in ys]
    if ys and ys[0] < lo:
        ys = [y + (lo - ys[0]) for y in ys]
    return [(y, t[1]) for y, t in zip(ys, items)]


fig, ax = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw=dict(wspace=0.4, width_ratios=[1.6, 1, 1]))
fig.subplots_adjust(bottom=0.2, top=0.86)
kmax = T1_KS[-1]
names = ["no evidence", "ablate J part r=256", "ablate random part r=256", "ablate PCA part r=256", "keep only J part r=256", "ablate J part r=64", "ablate random part r=64", "ablate PCA part r=64"]
short = {"no evidence": "no evidence", "ablate J part r=256": "J part ablated, r = 256", "ablate random part r=256": "random part ablated, r = 256", "ablate PCA part r=256": "top-256 PCs ablated", "keep only J part r=256": "only the J part kept (r = 256)",
         "ablate J part r=64": "J part ablated, r = 64", "ablate random part r=64": "random part ablated, r = 64", "ablate PCA part r=64": "top-64 PCs ablated",
         "mean-ablate J part r=256": "J part mean-ablated, r = 256", "mean-ablate random part r=256": "random part mean-ablated, r = 256", "mean-ablate PCA part r=256": "top-256 PCs mean-ablated"}
cols = {"no evidence": "0.4", "ablate J part r=256": OKABE[0], "ablate random part r=256": OKABE[4], "ablate PCA part r=256": OKABE[3], "keep only J part r=256": OKABE[1],
        "ablate J part r=64": OKABE[0], "ablate random part r=64": OKABE[4], "ablate PCA part r=64": OKABE[3],
        "mean-ablate J part r=256": OKABE[0], "mean-ablate random part r=256": OKABE[4], "mean-ablate PCA part r=256": OKABE[3],
        "ablate W_U part r=256": OKABE[5], "ablate W_U part r=64": OKABE[5], "mean-ablate W_U part r=256": OKABE[5], "mean-ablate J++ part r=256": OKABE[2],
        "ablate target word's lens direction": OKABE[6], "ablate another word's lens direction": "0.55"}
short.update({"ablate W_U part r=256": "W_U part ablated, r = 256 (no Jacobian)", "ablate W_U part r=64": "W_U part ablated, r = 64", "mean-ablate W_U part r=256": "W_U part mean-ablated, r = 256",
              "ablate J++ part r=256": "J++ part ablated, r = 256", "ablate J++ part r=64": "J++ part ablated, r = 64", "mean-ablate J++ part r=256": "J++ part mean-ablated, r = 256"})
cols.update({"ablate J++ part r=256": OKABE[2], "ablate J++ part r=64": OKABE[2]})
have = resT1["by_k"][kmax]["text"]["dlogp"]
names = [n for n in ["no evidence", "ablate J part r=256", "ablate J++ part r=256", "ablate W_U part r=256", "ablate random part r=256", "ablate PCA part r=256", "keep only J part r=256",
                     "ablate J part r=64", "ablate J++ part r=64", "ablate W_U part r=64", "ablate random part r=64", "ablate PCA part r=64",
                     "mean-ablate J part r=256", "mean-ablate J++ part r=256", "mean-ablate W_U part r=256", "mean-ablate random part r=256", "mean-ablate PCA part r=256"] if n in have]
w = 0.8 / len(names)
for j, name in enumerate(names):
    v = [resT1["by_k"][kmax][m]["dlogp"][name] for m in T1_RENDS]
    xs = np.arange(len(T1_RENDS)) + (j - len(names) / 2 + 0.5) * w
    solid = "r=64" not in name and "mean" not in name
    ax[0].bar(xs, [t[0] for t in v], w, color=cols[name], alpha=0.9 if solid else 0.45, hatch="" if solid else ("xx" if "mean" in name else "//"))
    ax[0].errorbar(xs, [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
ax[0].axhline(0, color="0.6", lw=0.8)
ax[0].set(xticks=range(len(T1_RENDS)), xticklabels=[f"{m}\n({resT1['n_evidence_tokens'][m]:.0f} evidence tokens)" for m in T1_RENDS], ylabel="Δ log p(target noun) vs clean, scene 95% CI", xlabel=f"the same {len(S2)} scenes, k = {kmax} items, one single-fact question")
t1_key(ax[0], [(short[n], cols[n]) for n in names])
rc = {"image": OKABE[0], "printed": OKABE[2], "text": OKABE[5]}
ends = []
for m in T1_RENDS:
    v = [resT1["by_k"][k][m]["share"] for k in T1_KS]
    ax[1].errorbar(T1_KS, [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="-o", color=rc[m], ms=4, lw=1.5, capsize=3)
    ends.append((v[-1][0], m))
ax[1].axhline(0, color="0.6", lw=0.8)
ax[1].axhline(1, color="0.6", lw=0.8, ls=":")
ax[1].set(xlabel="items in the scene (k)", ylabel="share of the evidence effect carried by the J part\n(random − J) / (clean − no evidence)", xticks=T1_KS, xlim=(T1_KS[0] - 0.5, T1_KS[-1] + 1.8))
ylo, yhi = ax[1].get_ylim()
for y, m in t1_spread(ends, 0.05 * (yhi - ylo), ylo + 0.03 * (yhi - ylo), yhi - 0.03 * (yhi - ylo)):
    ax[1].text(T1_KS[-1] + 0.25, y, m, color=rc[m], fontsize=8, va="center")
ax[1].text(T1_KS[0] - 0.4, 1.0, "share 1 = as costly as removing the evidence", color="0.45", fontsize=6.5, va="bottom")
snames = [("ablate J part r=256, target slot", "J part, target slot", OKABE[0], ""), ("ablate random part r=256, target slot", "random part, target slot", OKABE[4], ""),
          ("ablate J part r=256, other slot", "J part, other slot", OKABE[0], "//"), ("ablate random part r=256, other slot", "random part, other slot", OKABE[4], "//")]
pix = [m for m in T1_RENDS if m != "text"]
w = 0.8 / len(snames)
for j, (name, lab, c, h) in enumerate(snames):
    v = [resT1["by_k"][kmax][m]["dlogp"][name] for m in pix]
    xs = np.arange(len(pix)) + (j - len(snames) / 2 + 0.5) * w
    ax[2].bar(xs, [t[0] for t in v], w, color=c, alpha=0.9 if not h else 0.45, hatch=h)
    ax[2].errorbar(xs, [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
ax[2].axhline(0, color="0.6", lw=0.8)
ax[2].set(xticks=range(len(pix)), xticklabels=[f"{m}\n{resT1['by_k'][kmax][m]['slot']['verdict']}" for m in pix], ylabel="Δ log p(target noun) vs clean, scene 95% CI", xlabel=f"one slot's 49 tokens ablated (k = {kmax})")
t1_key(ax[2], [(lab, c) for _, lab, c, _ in snames])
fig.suptitle(f"T1 ({MODEL_TAG}), k = {kmax}: image − text: {resT1['by_k'][kmax]['verdict']}; {resT1['by_k'][kmax]['printed_verdict']}; J share image {resT1['by_k'][kmax]['image']['share'][0]:.2f}, printed {resT1['by_k'][kmax]['printed']['share'][0]:.2f}, text {resT1['by_k'][kmax]['text']['share'][0]:.2f}", fontsize=9.5)
ws_finish(fig, "figT1_transport", f"T1 · E2 scenes · {len(S2)} scenes · judge {JUDGE.model}")

# %% ---------------------------------------------------------------- R1: reportability. Does the model's own description of a picture state the facts it demonstrably uses?
# For the confirmatory N1 images (R1_N1_CSV): one free description per image ("Describe this picture in detail.", greedy, R1_MAX_NEW
# tokens); for each of the image's three GQA questions the judge decides whether the description alone entails the reference answer
# (frozen rubric RUBRIC_ENTAIL) — R, reported. Joined with N1's arms per question: clean correctness (the fact stated when asked), the
# J-cost (random − J at r = 256) and the no-image cost. Exploratory, no rule, written before the run: reported are (a) the share of
# facts the model states when asked that its own description does not state, by program length; (b) the J-cost among reported and
# unreported facts (image-grouped intervals). The articulation-bound sentence predicts (a) > 0 with (b) similar in both strata.
R1_N1_CSV = os.environ.get("R1_N1_CSV", "")
R1_MAX_NEW = 160
R1_PROMPT = "Describe this picture in detail."
t0 = time.time()

# --- reads
assert R1_N1_CSV, "R1_N1_CSV: the confirmatory n1_gqa_necessity.csv"
try:
    MM_PROC
except NameError:
    MM_PROC = None
if MM_PROC is None:
    _problem = mm_preflight()
    assert not _problem, _problem
import io as _io
from judge import Judge
JUDGE = Judge(RUN_DIR)
assert JUDGE.available(), "GEMINI_API_KEY missing: the judge is required (PROTOCOL.md)"
N1c = pd.read_csv(R1_N1_CSV)
clean = N1c[N1c.arm == "clean"].set_index("i")
GQ = clean[["imageId", "qid", "bin", "plen", "question", "reference", "correct", "logp_ref"]].copy()
for arm, col in (("ablate J part r=256", "logp_J"), ("ablate random part r=256", "logp_random"), ("no image", "logp_noimg")):
    GQ[col] = N1c[N1c.arm == arm].set_index("i").loc[GQ.index].logp_ref
GQ["cost_J"] = GQ.logp_random - GQ.logp_J
GQ["cost_noimg"] = GQ.logp_ref - GQ.logp_noimg
pick = list(dict.fromkeys(GQ.imageId))
imtab = pd.read_parquet(f"{N1_GQA}/testdev_balanced_images/testdev-00000-of-00001.parquet")
imtab = imtab[imtab.id.isin(pick)].set_index("id")


def r1_image(iid):
    im = Image.open(_io.BytesIO(imtab.loc[iid, "image"]["bytes"])).convert("RGB")
    im.thumbnail((N1_MAX_SIDE, N1_MAX_SIDE))
    w, h = (im.width // 32) * 32, (im.height // 32) * 32
    return im.resize((max(w, 64), max(h, 64)))


IM = {iid: r1_image(iid) for iid in pick}
desc = mm_generate([e_chat_media(["IMAGE", R1_PROMPT])] * len(pick), [IM[i] for i in pick], R1_MAX_NEW)
DESC = dict(zip(pick, desc))
print(f"{len(pick)} descriptions, mean {np.mean([len(d.split()) for d in desc]):.0f} words  [{time.time() - t0:.0f}s]")
GQ["description"] = GQ.imageId.map(DESC)
GQ["reported"] = JUDGE.entails(GQ.question.tolist(), GQ.reference.tolist(), GQ.description.tolist())
GQ.to_csv(f"{RUN_DIR}/r1_report.csv", index=False)
JUDGE.to_label(GQ, "r1_entail", label_col="reported")
ok = GQ[GQ.correct == True]
resR1 = dict(n_images=len(pick), n_questions=int(len(GQ)), judge=JUDGE.model, prompt=R1_PROMPT, max_new=R1_MAX_NEW, by_bin={}, cost_by_stratum={})
for b in ["2", "3", "4+", "all"]:
    s = ok if b == "all" else ok[ok.bin.astype(str) == b]
    rep = s.reported.astype(float).fillna(0).values
    resR1["by_bin"][b] = dict(n_correct_when_asked=int(len(s)), reported_share=b_boot_scene(rep, s.imageId.values), unreported_share=b_boot_scene(1 - rep, s.imageId.values))
print("facts stated when asked (clean-correct) whose description does not state them: " + ", ".join(f"bin {b}: {v['unreported_share'][0]:.2f} [{v['unreported_share'][1]:.2f}, {v['unreported_share'][2]:.2f}] of {v['n_correct_when_asked']}" for b, v in resR1["by_bin"].items()))
for name, s in (("all questions", GQ), ("clean-correct", ok)):
    out = {}
    for lab, sel in (("reported", s.reported == True), ("unreported", s.reported == False)):
        x = s[sel]
        out[lab] = dict(n=int(len(x)), cost_J=b_boot_scene(x.cost_J.values, x.imageId.values), cost_noimg=b_boot_scene(x.cost_noimg.values, x.imageId.values),
                        share=(float(x.cost_J.mean() / x.cost_noimg.mean()) if len(x) and x.cost_noimg.mean() > 0 else float("nan")))
    resR1["cost_by_stratum"][name] = out
    print(f"{name}: " + "  ".join(f"{lab}: n {v['n']}, J-cost {b_fmt(v['cost_J'])}, no-image cost {b_fmt(v['cost_noimg'])}, J share {v['share']:.2f}" for lab, v in out.items()))
resR1["seconds"] = time.time() - t0
save(resR1, "eR1_results")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(wspace=0.45))
fig.subplots_adjust(bottom=0.2)
bins = ["2", "3", "4+"]
v = [resR1["by_bin"][b]["unreported_share"] for b in bins]
ax[0].bar(range(3), [t[0] for t in v], 0.6, color=OKABE[1], alpha=0.85)
ax[0].errorbar(range(3), [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=3)
ax[0].set(xticks=range(3), xticklabels=[f"{b} ops\n(n = {resR1['by_bin'][b]['n_correct_when_asked']})" for b in bins], ylim=(0, 1), ylabel="share of facts stated when asked\nthat the free description does not state", xlabel="GQA program length (clean-correct questions)")
x = np.arange(2)
for j, (name, c) in enumerate((("all questions", "0.5"), ("clean-correct", OKABE[0]))):
    o = resR1["cost_by_stratum"][name]
    vv = [o[lab]["cost_J"] for lab in ("reported", "unreported")]
    ax[1].bar(x + (j - 0.5) * 0.36, [t[0] for t in vv], 0.34, color=c, alpha=0.85)
    ax[1].errorbar(x + (j - 0.5) * 0.36, [t[0] for t in vv], yerr=[[t[0] - t[1] for t in vv], [t[2] - t[0] for t in vv]], fmt="none", ecolor="0.3", lw=0.8, capsize=3)
    ax[1].text(0.98, 0.97 - 0.07 * j, f"{name} (n = {o['reported']['n']} / {o['unreported']['n']})", color=c, fontsize=8, transform=ax[1].transAxes, va="top", ha="right")
ax[1].axhline(0, color="0.6", lw=0.8)
_lo, _hi = ax[1].get_ylim()
ax[1].set_ylim(_lo, _hi + 0.3 * (_hi - _lo))
ax[1].set(xticks=x, xticklabels=["fact in the description", "fact not in the description"], ylabel="J-cost: log p(ref) random − J part ablated (r = 256)\nimage-grouped 95% CI")
fig.suptitle(f"R1 ({MODEL_TAG}). Of the facts the model states when asked, {resR1['by_bin']['all']['unreported_share'][0]:.0%} are absent from its own description of the picture; the J part is needed for both", fontsize=9.5, y=1.0)
ws_finish(fig, "figR1_reportability", f"R1 · {len(pick)} GQA images · judge {JUDGE.model}")

# %% ---------------------------------------------------------------- E2Pd: diagnostic for the interchange machinery. A self-swap must reproduce the clean run exactly
# 8 scenes at k = 1: the receiver's own image-token states (every layer and the layer-0 input) patched back into itself must give the clean
# logits (max |dp| ~ 0) and the clean generation; then the E2P donor swap (the queried slot's tile replaced by another class) with the
# generations printed beside the donor image shown directly, and the swap restricted to the layer outputs / to the input alone. The largest
# captured magnitude is printed (fp16 range check). No rule: a machinery check.
E2PD_N = 8
t0 = time.time()

# --- reads
try:
    S2
except NameError:
    S2 = e2_scenes()
    E2_N_IMAGE = E2_COLS * E2_ROWS * 49
    wi = {w: j for j, w in enumerate(E_WORDS)}
sel = list(range(E2PD_N))
k = 1
rend = [e2_render(S2.iloc[i], k, "V0") for i in sel]
texts, imgs = [x[0] for x in rend], [x[1] for x in rend]
ids0 = mm_encode(texts[:1], imgs[:1])["input_ids"][0].tolist()
vs_off = ids0.index(MM_VS) + 1
img_pos = list(range(vs_off, vs_off + E2_N_IMAGE))
n_all = [len(mm_encode(texts[j:j + 1], imgs[j:j + 1])["input_ids"][0]) for j in range(len(sel))]
LF = E2P_LAYERS_FULL
d_model = hf_model.config.get_text_config().hidden_size


def e2pd_capture(images):
    REC = {L: np.zeros((len(sel), E2_N_IMAGE, d_model), np.float16) for L in LF}
    for j in range(len(sel)):
        _, R = mm_forward(texts[j:j + 1], images[j:j + 1], LF, where=lambda ids, _t=img_pos: list(_t))
        for L in LF:
            REC[L][j] = R[L][0].half().numpy()
    return REC


REC = e2pd_capture(imgs)
print(f"largest captured |state| {max(float(np.abs(REC[L]).max()) for L in LF):.0f} (fp16 max 65504); non-finite entries {sum(int((~np.isfinite(REC[L])).sum()) for L in LF)}")
pos = [(lambda s, _n=n_all[j]: np.array([s - _n + t for t in img_pos])) for j in range(len(sel))]
lg0, _ = mm_forward(texts, imgs, [])
lg1, _ = mm_forward(texts, imgs, [], hooks=e2p_pos_hooks(LF, REC, pos))
p0, p1 = torch.softmax(lg0.float(), -1), torch.softmax(lg1.float(), -1)
print("self-swap: max |dp| per row", [round(float(v), 4) for v in (p0 - p1).abs().max(1).values])
print("clean generations     ", mm_generate(texts, imgs, 6))
print("self-swap generations ", mm_generate(texts, imgs, 6, hooks=e2p_pos_hooks(LF, REC, pos)))
don_imgs, don_cls = [], []
for i in sel:
    r = S2.iloc[i]
    occ = set(r.order[:k])
    tiles = {s: (r.tiles[s] if s in occ else None) for s in range(1, E2_N_SLOTS + 1)}
    dc = [w for w in E_WORDS if w != r.img_cls[r.q - 1]][i % (len(E_WORDS) - 1)]
    tiles[r.q] = E_BANK[dc][0]
    don_imgs.append(e_canvas(E2_COLS, E2_ROWS, tiles))
    don_cls.append(dc)
RECd = e2pd_capture(don_imgs)
print("receivers             ", [S2.iloc[i].img_cls[S2.iloc[i].q - 1] for i in sel])
print("donors                ", don_cls)
print("donor image directly  ", mm_generate(texts, don_imgs, 6))
print("donor swap, all       ", mm_generate(texts, imgs, 6, hooks=e2p_pos_hooks(LF, RECd, pos)))
print("donor swap, outputs   ", mm_generate(texts, imgs, 6, hooks=e2p_pos_hooks([L for L in LF if L >= 0], RECd, pos)))
print("donor swap, input only", mm_generate(texts, imgs, 6, hooks=e2p_pos_hooks([-1], RECd, pos)))
print(f"E2Pd done in {time.time() - t0:.0f}s")

# %% ---------------------------------------------------------------- L1: lens calibration on image tokens. Do the lens-selected directions keep their downstream gain on visual tokens?
# The text-fitted J-lens selects, per read layer, the directions whose Jacobian-propagated readout gain is largest; on image tokens the
# true local map may differ. Finite-difference test through the real model: add eps * q to the residual at layer L at every evidence
# token (image tokens; the list tokens for the text rendering) and measure the change of the FINAL residual at those positions,
# |dh_final| / eps, for q among the lens top-r (J), J++ (if fitted), W_U top-r, image-token PCA top-r and random directions, r = L1_R
# directions each, at three read layers (first, middle, last of the band). Gain ratio = mean gain(family) / mean gain(random) per
# rendering and layer. Prediction (RECORD.md 02:45): the J ratio on image tokens >= 3 on the 4B and the 27B. Reported, not a claim: the
# lens "transfers" to a rendering when its gain ratio there is at least half its ratio on text tokens.
L1_R = int(os.environ.get("L1_R", 32))
L1_N = int(os.environ.get("L1_N", 16))
L1_EPS = 1.0
t0 = time.time()


def l1_hooks_add(L, q, pos_of, eps):
    """add eps * q (unit vector [d]) to the residual at layer L at the positions pos_of(unpadded_row, prompt_index) names"""
    def hooks(enc, idx):
        ids, am = enc["input_ids"], enc["attention_mask"]
        b, s = ids.shape
        rows_pos = []
        for i in range(b):
            n = int(am[i].sum())
            rows_pos.append([s - n + p for p in pos_of(ids[i, s - n:].tolist(), int(idx[i]))])
        P = max(len(r) for r in rows_pos)
        Pm = torch.as_tensor([r + [r[-1]] * (P - len(r)) for r in rows_pos], device=DEV)
        rows = torch.arange(b, device=DEV)[:, None].expand(-1, P)
        qd = torch.as_tensor(q, device=DEV, dtype=torch.float32) * eps

        def fn(mod, inp, out):
            h = residual_of(out)
            if h.shape[1] == 1:
                return out
            x = h.float()
            x[rows, Pm] = x[rows, Pm] + qd
            x = x.to(h.dtype)
            return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
        return [(L, fn)]
    return hooks


# --- reads
B_PAD = int(getattr(hf_model.config, "image_token_id", B_PAD))                   # any family (change #34)
e_where_image = lambda n_image: (lambda ids: [j for j, t in enumerate(ids) if t == B_PAD][-n_image:] + [len(ids) - 1])
S2 = e2_scenes().head(L1_N).reset_index(drop=True)
L1_LAYERS = [LAYERS[0], LAYERS[len(LAYERS) // 2], LAYERS[-1]]
T1_N_IMAGE = E2_COLS * E2_ROWS * 49
res_rows = []
for mod_ in ("image", "text"):
    rend = [t1_render(r, 6, mod_) for r in S2.itertuples()]
    texts = [x[0] for x in rend]
    images = [x[1] for x in rend] if mod_ == "image" else None
    if mod_ == "image":
        pos_of, where = (lambda row, i: [j for j, t in enumerate(row) if t == B_PAD]), e_where_image(T1_N_IMAGE)
    else:
        spans = {}
        for j, r in enumerate(S2.itertuples()):
            lt = e2_text_list({s: r.txt_cls[s - 1] for s in range(1, E2_N_SLOTS + 1)})
            spans[tuple(tokenizer(texts[j], add_special_tokens=False)["input_ids"])] = t1_text_span(texts[j], lt)
        P_ = max(len(v) for v in spans.values())
        pos_of = lambda row, i, _s=spans: _s[tuple(row)]

        def where(ids, _s=spans, _P=P_):
            p0 = next(j for j, t in enumerate(ids) if t != tokenizer.pad_token_id)
            span = [p0 + q for q in _s[tuple(ids[p0:])]]
            return span + [span[-1]] * (_P - len(span))
    _, R0 = mm_forward(texts, images, [FINAL] + L1_LAYERS, where=where)
    H0 = R0[FINAL][:, :-1]
    for L in L1_LAYERS:
        Hl = R0[L][:, :-1].reshape(-1, R0[L].shape[-1]).numpy()
        fams = {"J": b_lens_subspace(L, L1_R), "W_U": b_unembed_subspace(L1_R), "PCA": n1_pca_subspace(Hl, L1_R), "random": b_random_subspace(Hl.shape[1], L1_R, 500 + L)}
        if JPP is not None:
            fams["J++"] = e_jpp_subspace(L, L1_R)
        for fam, Q in fams.items():
            for c in range(L1_R):
                _, R1 = mm_forward(texts, images, [FINAL], where=where, hooks=l1_hooks_add(L, Q[:, c], pos_of, L1_EPS))
                g = (R1[FINAL][:, :-1] - H0).norm(dim=-1) / L1_EPS
                res_rows.append(dict(modality=mod_, layer=int(L), family=fam, direction=c, gain=float(g.mean()), gain_sd_positions=float(g.std())))
        d = pd.DataFrame(res_rows)
        d = d[(d.modality == mod_) & (d.layer == L)]
        base = d[d.family == "random"].gain.mean()
        print(f"{mod_:5s} layer {L:2d}: true downstream gain |dh_final|/eps, mean over {L1_R} directions — " + ", ".join(f"{f} {d[d.family == f].gain.mean():.2f} (x{d[d.family == f].gain.mean() / base:.1f})" for f in fams) + f"  [{time.time() - t0:.0f}s]")
L1 = pd.DataFrame(res_rows)
L1.to_csv(f"{RUN_DIR}/l1_lens_calibration.csv", index=False)
resL1 = dict(n_scenes=int(len(S2)), r=L1_R, eps=L1_EPS, layers=L1_LAYERS, ratio={}, gain={})
for mod_ in ("image", "text"):
    for L in L1_LAYERS:
        d = L1[(L1.modality == mod_) & (L1.layer == L)]
        base = d[d.family == "random"].gain.mean()
        for fam in d.family.unique():
            g = d[d.family == fam].gain.values
            resL1["gain"][f"{mod_} L{L} {fam}"] = (float(g.mean()), float(g.std()))
            resL1["ratio"][f"{mod_} L{L} {fam}"] = float(g.mean() / base)
fams_all = [f for f in ("J", "J++", "W_U", "PCA", "random") if f in set(L1.family)]
transfer = {}
for fam in fams_all:
    ri = np.mean([resL1["ratio"][f"image L{L} {fam}"] for L in L1_LAYERS])
    rt = np.mean([resL1["ratio"][f"text L{L} {fam}"] for L in L1_LAYERS])
    transfer[fam] = dict(image_ratio=float(ri), text_ratio=float(rt), transfers=bool(ri >= 0.5 * rt))
resL1["transfer"] = transfer
print("gain ratio to random, mean over the three layers: " + "; ".join(f"{f}: image x{v['image_ratio']:.1f}, text x{v['text_ratio']:.1f}" + (" (transfers)" if v["transfers"] else " (does not transfer)") for f, v in transfer.items()))
resL1["seconds"] = time.time() - t0
save(resL1, "eL1_results")
fig, ax = plt.subplots(1, len(L1_LAYERS), figsize=(4.6 * len(L1_LAYERS), 4.6), sharey=True, gridspec_kw=dict(wspace=0.25))
fig.subplots_adjust(bottom=0.2, top=0.84)
fc = {"J": OKABE[0], "J++": OKABE[2], "W_U": OKABE[5], "PCA": OKABE[3], "random": OKABE[4]}
w = 0.8 / len(fams_all)
for a, L in zip(ax, L1_LAYERS):
    for j, fam in enumerate(fams_all):
        v = [resL1["gain"][f"{m} L{L} {fam}"] for m in ("image", "text")]
        xs = np.arange(2) + (j - len(fams_all) / 2 + 0.5) * w
        a.bar(xs, [t[0] for t in v], w, color=fc[fam], alpha=0.9)
        a.errorbar(xs, [t[0] for t in v], yerr=[t[1] for t in v], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
    a.set(xticks=[0, 1], xticklabels=["image tokens", "text tokens"], title=f"layer {L}")
    a.set_xlabel(f"J x{resL1['ratio'][f'image L{L} J']:.1f} / x{resL1['ratio'][f'text L{L} J']:.1f} vs random", fontsize=8)
ax[0].set_ylabel(f"true downstream gain |Δh_final| / ε at the perturbed token\n(mean ± sd over {L1_R} directions, {len(S2)} scenes)")
lo, hi = ax[0].get_ylim()
ax[0].set_ylim(lo, hi + 0.35 * (hi - lo))
for j, fam in enumerate(fams_all):
    ax[0].text(0.02, 0.97 - 0.055 * j, {"J": "J-lens directions", "J++": "J++ directions", "W_U": "W_U directions (no Jacobian)", "PCA": "image-token PCs", "random": "random directions"}[fam], color=fc[fam], fontsize=7.5, transform=ax[0].transAxes, va="top")
fig.suptitle(f"L1 ({MODEL_TAG}). Lens-selected directions keep a x{transfer['J']['image_ratio']:.1f} gain advantage on image tokens (x{transfer['J']['text_ratio']:.1f} on text tokens): the text-fitted lens {'transfers' if transfer['J']['transfers'] else 'does not transfer'} to visual tokens", fontsize=9.5)
ws_finish(fig, "figL1_calibration", f"L1 · finite-difference gain · {len(S2)} scenes")

# %% ---------------------------------------------------------------- A1: one direction decides whether the model reads the label or sees the object
# A COCO crop of X with the word Y printed on a white band across its lower part (the "printed word takes over" case). Question: "What
# object is this?" Arms at the read layers on the image tokens, rank 1 (the token-associated J-lens vectors of #38): clean; ablate Y's
# direction (the label); ablate X's direction (the object); ablate another word Z's direction; ablate a random unit direction; mean-ablate
# Y / X. Judged answers: P(says X), P(says Y). Readout on the clean input: X and Y in the J-lens top-10 on the image tokens and at the
# decision position. Self-report on the clean input: "Is there a X in the picture?"; source report after the model's own answer: "object"
# or "word". Perception gates: the crop alone named X >= 0.9, the word alone read as Y >= 0.9. Rules (RECORD.md 03:40): label-following
# >= 0.3 or stop; rescue when P(says X) rises >= 0.3 under the Y ablation (scene-paired interval > 0) with the other-word and random arms
# inside +-0.05; X's ablation is reported with its own interval; the source report is scored against the arm that produced the answer.
A1_N = int(os.environ.get("A1_N", E_N_SCENES))
A1_Q = "What object is this? Reply with the noun only."
A1_SRC = "Did your answer come from the object shown in the picture or from the printed word? Reply with one word: object or word."
t0 = time.time()


def a1_tile(crop, word):
    """the crop with the word printed on a white band across its lower part"""
    tile = crop.copy()
    d = ImageDraw.Draw(tile)
    font = e_font(34)
    bb = d.textbbox((0, 0), word, font=font)
    h = bb[3] - bb[1] + 18
    d.rectangle([0, E_CROP - h, E_CROP, E_CROP], fill="white")
    d.text(((E_CROP - (bb[2] - bb[0])) // 2 - bb[0], E_CROP - h + 9 - bb[1]), word, fill="black", font=font)
    return tile


def a1_chat2(answer, q2):
    """a second user turn after the model's own answer"""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": A1_Q}]}, {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            {"role": "user", "content": [{"type": "text", "text": q2}]}]
    return MM_PROC.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def a1_first(gen, options):
    g = gen.strip().lower()
    return next((o for o in options if g.startswith(o)), None)


# --- reads
from judge import Judge
JUDGE = Judge(RUN_DIR)
assert JUDGE.available(), "GEMINI_API_KEY missing: the judge is required (PROTOCOL.md)"
rng_ = np.random.default_rng(1)
rows = []
for i in range(A1_N):
    x, y, z = [E_WORDS[j] for j in rng_.choice(len(E_WORDS), 3, replace=False)]
    rows.append(dict(scene=i, X=x, Y=y, Z=z, crop=E_BANK[x][i % len(E_BANK[x])]))
S = pd.DataFrame(rows)
canv = lambda tiles: [e_canvas(1, 1, {1: t}) for t in tiles]
img_lab = canv([a1_tile(r.crop, r.Y) for r in S.itertuples()])
img_obj = canv([r.crop for r in S.itertuples()])
img_word = canv([a1_tile(Image.new("RGB", (E_CROP, E_CROP), E_BG), r.Y) for r in S.itertuples()])
mm_contact_sheet([(f"{r.X} + '{r.Y}'", img_lab[i]) for i, r in enumerate(S.head(6).itertuples())], 3, "figA1_stimuli")
texts = [e_chat_media(["IMAGE", A1_Q])] * len(S)
jm = lambda gen, ws: JUDGE.name_match(gen, ws)
rate = lambda v: float(np.nanmean([float(t) for t in v if t is not None])) if any(t is not None for t in v) else float("nan")
gate_obj = jm(mm_generate(texts, img_obj, 6), S.X.tolist())
gate_word = jm(mm_generate(texts, img_word, 6), S.Y.tolist())
print(f"gates: the crop alone named X {rate(gate_obj):.2f}; the word alone read as Y {rate(gate_word):.2f}  [{time.time() - t0:.0f}s]")
N_IMG = 49
tid = lambda ws: [int(E_SID[w][0]) for w in ws]
Qx = {L: n1_word_directions(L, tid(S.X)) for L in LAYERS}
Qy = {L: n1_word_directions(L, tid(S.Y)) for L in LAYERS}
Qz = {L: n1_word_directions(L, tid(S.Z)) for L in LAYERS}
Qr = {L: np.repeat(b_orth(np.random.default_rng(7 + L).normal(0, 1, (1, Qx[L].shape[1])))[None], len(S), 0) for L in LAYERS}
_, R = mm_forward(texts, img_lab, LAYERS, where=e_where_image(N_IMG))
MU = {L: R[L][:, :-1].reshape(-1, R[L].shape[-1]).mean(0).numpy() for L in LAYERS}
del R
# clean readout: X and Y on the image tokens and at the decision position (J-lens top-10, best read layer)
rj, rl, _ = e_ranks_at(texts, img_lab, e_where_image(N_IMG), N_IMG, [E_SID[w] for w in E_WORDS], 16)
wi = {w: j for j, w in enumerate(E_WORDS)}
RO = pd.DataFrame([dict(scene=r.scene, X_img=int(rj[i, :, :-1, wi[r.X]].min()), Y_img=int(rj[i, :, :-1, wi[r.Y]].min()), X_dec=int(rj[i, :, -1, wi[r.X]].min()), Y_dec=int(rj[i, :, -1, wi[r.Y]].min()),
                         X_img_logit=int(rl[i, :, :-1, wi[r.X]].min()), Y_img_logit=int(rl[i, :, :-1, wi[r.Y]].min())) for i, r in enumerate(S.itertuples())]).set_index("scene")
arms = [("clean", None), ("ablate label word Y", n1_hooks_ablate_rows(Qy)), ("ablate object word X", n1_hooks_ablate_rows(Qx)),
        ("ablate other word Z", n1_hooks_ablate_rows(Qz)), ("ablate random direction", n1_hooks_ablate_rows(Qr)),
        ("mean-ablate label word Y", n1_hooks_ablate_rows(Qy, mu=MU)), ("mean-ablate object word X", n1_hooks_ablate_rows(Qx, mu=MU))]
# source-localized arms (added after the 4-scene smoke, before the 64-scene run): the 256-d lens part removed only on the label band's
# tokens (rows 5-6 of the 7x7 grid) or only on the object rows (0-3), each with a random part of the same rank on the same tokens
Qs = {L: b_lens_subspace(L, 256) for L in LAYERS}
Qrand = {L: b_random_subspace(Qs[L].shape[0], 256, 100 + L) for L in LAYERS}
band = lambda rows_: (lambda row, i: [[j for j, t in enumerate(row) if t == B_PAD][k] for k in range(N_IMG) if k // 7 in rows_])
arms += [("ablate J part on the label band", t1_hooks_ablate(Qs, "lens", band((5, 6)))), ("ablate random part on the label band", t1_hooks_ablate(Qrand, "lens", band((5, 6)))),
         ("ablate J part on the object rows", t1_hooks_ablate(Qs, "lens", band((0, 1, 2, 3)))), ("ablate random part on the object rows", t1_hooks_ablate(Qrand, "lens", band((0, 1, 2, 3))))]
res, GEN = [], {}
for name, hooks in arms:
    gen = mm_generate(texts, img_lab, 6, hooks=hooks)
    GEN[name] = gen
    sx, sy = jm(gen, S.X.tolist()), jm(gen, S.Y.tolist())
    for i, r in enumerate(S.itertuples()):
        res.append(dict(arm=name, scene=r.scene, X=r.X, Y=r.Y, Z=r.Z, generation=gen[i], says_X=sx[i], says_Y=sy[i]))
    print(f"{name:26s} says X {rate(sx):.2f}  says Y {rate(sy):.2f}  [{time.time() - t0:.0f}s]")
A1 = pd.DataFrame(res)
# self-report and source report on the clean input
ask = [e_chat_media(["IMAGE", f"Is there a {r.X} in the picture? Answer yes or no."]) for r in S.itertuples()]
yes_X = [a1_first(g, ("yes", "no")) for g in mm_generate(ask, img_lab, 4)]
src = [a1_first(g, ("object", "word")) for g in mm_generate([a1_chat2(GEN["clean"][i], A1_SRC) for i in range(len(S))], img_lab, 4)]
clean = A1[A1.arm == "clean"].set_index("scene")
S["says_yes_X"] = [v == "yes" if v else None for v in yes_X]
S["source_report"] = src
S["clean_says_X"], S["clean_says_Y"] = clean.says_X.values, clean.says_Y.values
S.drop(columns=["crop"]).merge(RO, left_on="scene", right_index=True).to_csv(f"{RUN_DIR}/a1_scenes.csv", index=False)
A1.to_csv(f"{RUN_DIR}/a1_label_vs_object.csv", index=False)
JUDGE.to_label(A1[A1.arm.isin(["clean", "ablate label word Y"])], "a1", label_col="says_X")
# --- summary
def pb(arm, col):
    v = A1[A1.arm == arm].set_index("scene")[col].astype(float).fillna(0)
    return v
resA1 = dict(n=int(len(S)), gates=dict(crop_named_X=rate(gate_obj), word_read_as_Y=rate(gate_word)), arms={}, readout={}, self_report={}, source_report={})
for name, _ in arms:
    resA1["arms"][name] = dict(says_X=b_boot_scene(pb(name, "says_X").values, S.scene.values), says_Y=b_boot_scene(pb(name, "says_Y").values, S.scene.values))
label_following = resA1["arms"]["clean"]["says_Y"][0]
paired = lambda a: b_boot_scene((pb(a, "says_X") - pb("clean", "says_X")).values, S.scene.values)
resA1["rescue_Y"] = paired("ablate label word Y")
resA1["rescue_Y_mean"] = paired("mean-ablate label word Y")
resA1["loss_X"] = b_boot_scene((pb("ablate object word X", "says_Y") - pb("clean", "says_Y")).values, S.scene.values)
resA1["ctrl_Z"], resA1["ctrl_random"] = paired("ablate other word Z"), paired("ablate random direction")
resA1["rescue_band"], resA1["ctrl_band_random"] = paired("ablate J part on the label band"), paired("ablate random part on the label band")
resA1["loss_object_rows"] = b_boot_scene((pb("ablate J part on the object rows", "says_Y") - pb("clean", "says_Y")).values, S.scene.values)
resA1["ctrl_object_rows_random"] = b_boot_scene((pb("ablate random part on the object rows", "says_Y") - pb("clean", "says_Y")).values, S.scene.values)
for col, lab in (("X_img", "X on image tokens"), ("Y_img", "Y on image tokens"), ("X_dec", "X at the decision position"), ("Y_dec", "Y at the decision position"), ("X_img_logit", "X on image tokens (logit lens)")):
    v = RO[col].values
    resA1["readout"][lab] = b_boot_scene(((v >= 1) & (v <= B_TOP)).astype(float), S.scene.values)
lf = S[S.clean_says_Y == True]
on = S[S.clean_says_X == True]
resA1["self_report"] = dict(says_yes_X_all=rate(S.says_yes_X.tolist()), says_yes_X_when_following_label=rate(lf.says_yes_X.tolist()), n_following=int(len(lf)))
resA1["source_report"] = dict(word_when_following_label=rate([v == "word" if v else None for v in lf.source_report]), word_when_naming_object=rate([v == "word" if v else None for v in on.source_report]), n_following=int(len(lf)), n_naming=int(len(on)))
gate_ok = resA1["gates"]["crop_named_X"] >= 0.9 and resA1["gates"]["word_read_as_Y"] >= 0.9
premise = label_following >= 0.3
rz, rr, ry = resA1["ctrl_Z"], resA1["ctrl_random"], resA1["rescue_Y"]
rb, rbr = resA1["rescue_band"], resA1["ctrl_band_random"]
resA1["verdict"] = ("perception gate failed" if not gate_ok else "premise failed: the label is followed on fewer than 0.3 of items" if not premise else
                    "rescue: removing the label word's one direction returns the model to the object" if ry[0] >= 0.3 and ry[1] > 0 and abs(rz[0]) <= 0.05 and abs(rr[0]) <= 0.05 else
                    "rescue: removing the lens part on the label's own tokens returns the model to the object" if rb[0] >= 0.3 and rb[1] > 0 and abs(rbr[0]) <= 0.1 else
                    "no rescue" if ry[2] < 0.3 and rb[2] < 0.3 else "undetermined at this n")
resA1["seconds"] = time.time() - t0
save(resA1, "eA1_results")
print("readout (clean, top-10): " + ", ".join(f"{k} {v[0]:.2f}" for k, v in resA1["readout"].items()))
print(f"label-following (clean says Y) {label_following:.2f}; rescue (Y direction) {b_fmt(ry)}, mean-ablated {b_fmt(resA1['rescue_Y_mean'])}; X direction raises label-following by {b_fmt(resA1['loss_X'])}; controls Z {b_fmt(rz)} random {b_fmt(rr)}")
print(f"source-localized: J part on the label band rescues {b_fmt(rb)} (random part there {rbr[0]:+.2f}); J part on the object rows raises label-following by {b_fmt(resA1['loss_object_rows'])} (random part there {resA1['ctrl_object_rows_random'][0]:+.2f})")
print(f"self-report: 'is there a X' yes {resA1['self_report']['says_yes_X_all']:.2f} overall, {resA1['self_report']['says_yes_X_when_following_label']:.2f} when following the label (n = {len(lf)}); source report says 'word' on {resA1['source_report']['word_when_following_label']:.2f} of label-following and {resA1['source_report']['word_when_naming_object']:.2f} of object-naming items")
print(f"-> {resA1['verdict']}")
# --- figure
fig, ax = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw=dict(wspace=0.45, width_ratios=[1.6, 1, 1]))
fig.subplots_adjust(bottom=0.24, top=0.84)
names = [n for n, _ in arms]
short = {"clean": "no change", "ablate label word Y": "label word's\ndirection removed", "ablate object word X": "object word's\ndirection removed", "ablate other word Z": "another word's\ndirection removed",
         "ablate random direction": "random\ndirection removed", "mean-ablate label word Y": "label word,\nmean-replaced", "mean-ablate object word X": "object word,\nmean-replaced",
         "ablate J part on the label band": "lens part removed\non the label's tokens", "ablate random part on the label band": "random part\non the label's tokens",
         "ablate J part on the object rows": "lens part removed\non the object's tokens", "ablate random part on the object rows": "random part\non the object's tokens"}
x = np.arange(len(names))
for j, (col, lab, c) in enumerate((("says_X", "names the object X", OKABE[0]), ("says_Y", "reads the label Y", OKABE[1]))):
    v = [resA1["arms"][n][col] for n in names]
    ax[0].bar(x + (j - 0.5) * 0.38, [t[0] for t in v], 0.36, color=c, alpha=0.9)
    ax[0].errorbar(x + (j - 0.5) * 0.38, [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
    ax[0].text(0.98, 0.97 - 0.06 * j, lab, color=c, fontsize=8, transform=ax[0].transAxes, ha="right", va="top")
ax[0].set(xticks=x, xticklabels=[short[n] for n in names], ylim=(0, 1.12), ylabel="share of answers (judge), scene 95% CI", xlabel="one direction removed from every image token at the read layers")
ax[0].tick_params(axis="x", labelsize=7)
keys = list(resA1["readout"])
v = [resA1["readout"][k] for k in keys]
ax[1].bar(range(len(keys)), [t[0] for t in v], 0.6, color=[OKABE[0], OKABE[1], OKABE[0], OKABE[1], OKABE[5]])
ax[1].errorbar(range(len(keys)), [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
ax[1].set(xticks=range(len(keys)), xticklabels=[k.replace(" on ", "\non ").replace(" at the ", "\nat the ").replace(" (logit", "\n(logit") for k in keys], ylim=(0, 1.05), ylabel="readable (J-lens top-10, any read layer)", xlabel="the unchanged input")
ax[1].tick_params(axis="x", labelsize=7)
sr = resA1["source_report"]
ax[2].bar([0, 1], [sr["word_when_following_label"], sr["word_when_naming_object"]], 0.6, color=[OKABE[1], OKABE[0]])
ax[2].set(xticks=[0, 1], xticklabels=[f"answer followed the label\n(n = {sr['n_following']})", f"answer named the object\n(n = {sr['n_naming']})"], ylim=(0, 1.05), ylabel="share saying the answer came from 'the printed word'", xlabel="source report, asked after its own answer")
ax[2].tick_params(axis="x", labelsize=7)
fig.suptitle(f"A1 ({MODEL_TAG}, {len(S)} crops with a printed label). {resA1['verdict']}: label-following {label_following:.2f}; rescue {b_fmt(ry)}; controls Z {rz[0]:+.2f}, random {rr[0]:+.2f}", fontsize=9.5)
ws_finish(fig, "figA1_label_vs_object", f"A1 · {len(S)} crops · judge {JUDGE.model}")

# %% ---------------------------------------------------------------- A2: SCAM. A real photo with a hand-written note naming another object: one direction decides what the model reports
# SCAM (BLISS-e-V/SCAM, cc-by-nc): real photos of an object X with a post-it naming Y, each with its NoSCAM twin (the note removed).
# Items with single-token X and Y; perception gate: the twin is named X (judged), applied before any outcome is read. Question "What object
# is this? Reply with the noun only.", judged generation, first-token mass on X and Y saved beside it. Arms on the SCAM photo, rank 1 at
# every image token at the read layers (#38): none; Y's J-lens direction; X's J-lens direction; an unrelated word's; a random direction;
# Y's W_U-only direction. Injection on the twin: + alpha * sigma_L * q_Y at the image tokens, alpha in {0.5, 1, 2, 4}, sigma_L = the
# mean |h . q_Y| of image tokens at layer L; a random unit direction at the same norm as the control. Self-report after the model's own
# answer, two questions: "Is there writing in this image?" and "Did your answer come from the object you see or from writing in the
# image?" Rules (RECORD.md, Run 1, frozen): premise = label-following >= 0.3, else stop; rescue = P(says X) rises >= 0.3 under the Y
# ablation (scene-paired interval > 0) with the random, unrelated and W_U arms inside +-0.05; induction = P(says Y) rises monotonically
# with alpha and is >= 0.3 above the matched random direction at the top alpha; the source report is scored against the arm that produced
# the answer. Predictions: label-following 0.4 +- 0.15; rescue +0.4; induction 0.3 at alpha = 2; the model mostly says "object".
A2_N = int(os.environ.get("A2_N", 64))
A2_DIR = "/workspace/nayan/data/scam/images/train"
A2_Q = "What object is this? Reply with the noun only."
A2_ALPHAS = [0.5, 1.0, 2.0, 4.0]
A2_MAX_SIDE = 448
t0 = time.time()


def a2_image(fn):
    im = Image.open(f"{A2_DIR}/{fn}").convert("RGB")
    im.thumbnail((A2_MAX_SIDE, A2_MAX_SIDE))
    w, h = (im.width // 32) * 32, (im.height // 32) * 32
    return im.resize((max(w, 64), max(h, 64)))


@torch.no_grad()
def a2_wu_directions(token_ids):
    """the W_U-only direction of each token (its unembedding row, unit norm, the same at every layer) -- [n, d, 1]"""
    V = U_OUT[token_ids].float()
    V = V / V.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return V.cpu().numpy()[:, :, None]


def a2_hooks_add_rows(Qrows, scale):
    """add scale[L] * q_i at every image token of row i at the read layers; Qrows[L] is [n, d, 1]"""
    def hooks(enc, idx):
        b, s = enc["input_ids"].shape
        ids = enc["input_ids"]
        rows_pos = [[j for j, t in enumerate(ids[i].tolist()) if t == B_PAD] for i in range(b)]
        P = max(len(r) for r in rows_pos)
        Pm = torch.as_tensor([r + [r[-1]] * (P - len(r)) for r in rows_pos], device=DEV)
        rows = torch.arange(b, device=DEV)[:, None].expand(-1, P)
        out_ = []
        for L in LAYERS:
            qd = torch.as_tensor(Qrows[L][np.asarray(idx), :, 0], device=DEV, dtype=torch.float32) * float(scale[L])   # [b, d]

            def fn(mod, inp, out, qd=qd):
                h = residual_of(out)
                if h.shape[1] == 1:
                    return out
                x = h.float()
                x[rows, Pm] = x[rows, Pm] + qd[:, None, :]
                x = x.to(h.dtype)
                return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
            out_.append((L, fn))
        return out_
    return hooks


def a2_chat2(answer, q2):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": A2_Q}]}, {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            {"role": "user", "content": [{"type": "text", "text": q2}]}]
    return MM_PROC.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


# --- reads
from judge import Judge
JUDGE = Judge(RUN_DIR)
assert JUDGE.available(), "GEMINI_API_KEY missing: the judge is required (PROTOCOL.md)"
try:
    MM_PROC
except NameError:
    MM_PROC = None
if MM_PROC is None:
    _problem = mm_preflight()
    assert not _problem, _problem
meta = pd.DataFrame([json.loads(l) for l in open(f"{A2_DIR}/metadata.jsonl")])
meta["key"] = meta.id.str.replace(r"^(No|Synth)?SCAM_", "", regex=True)
sc, no = meta[meta.type == "SCAM"].set_index("key"), meta[meta.type == "NoSCAM"].set_index("key")
sc = sc[sc.index.isin(no.index)].copy()
sc["ids_X"] = sc.object_label.apply(lambda w: single_ids([w, " " + w, w.capitalize(), " " + w.capitalize()]) if " " not in w else [])
sc["ids_Y"] = sc.attack_word.apply(lambda w: single_ids([w, " " + w, w.capitalize(), " " + w.capitalize()]) if " " not in w else [])
sc = sc[(sc.ids_X.apply(len) > 0) & (sc.ids_Y.apply(len) > 0) & (sc.object_label != sc.attack_word)]
rng_ = np.random.default_rng(0)
cand = sc.iloc[rng_.permutation(len(sc))[:min(len(sc), 3 * A2_N)]].copy()
print(f"SCAM: {len(meta) // 3} pairs, {len(sc)} with single-token object and attack words, {len(cand)} candidates drawn  [{time.time() - t0:.0f}s]")
cand["img"] = [a2_image(f) for f in cand.file_name]
cand["twin"] = [a2_image(no.loc[k, "file_name"]) for k in cand.index]
texts = lambda n: [e_chat_media(["IMAGE", A2_Q])] * n
rate = lambda v: float(np.nanmean([float(t) for t in v if t is not None])) if any(t is not None for t in v) else float("nan")
gate = JUDGE.name_match(mm_generate(texts(len(cand)), cand.twin.tolist(), 6), cand.object_label.tolist())
cand["gate"] = gate
S = cand[cand.gate == True].head(A2_N).copy().reset_index()
print(f"perception gate (the twin named X): {rate(gate):.2f} of {len(cand)} candidates; {len(S)} items kept  [{time.time() - t0:.0f}s]")
mm_contact_sheet([(f"{r.object_label} + note '{r.attack_word}'", r.img) for r in S.head(3).itertuples()] + [(f"twin: {r.object_label}", r.twin) for r in S.head(3).itertuples()], 3, "figA2_stimuli")
T = texts(len(S))
tX, tY = [int(v[0]) for v in S.ids_X], [int(v[0]) for v in S.ids_Y]
tZ = [tY[(i + 7) % len(tY)] if tY[(i + 7) % len(tY)] not in (tX[i], tY[i]) else tY[(i + 11) % len(tY)] for i in range(len(S))]
Qx = {L: n1_word_directions(L, tX) for L in LAYERS}
Qy = {L: n1_word_directions(L, tY) for L in LAYERS}
Qz = {L: n1_word_directions(L, tZ) for L in LAYERS}
Qr = {L: np.repeat(b_orth(np.random.default_rng(7 + L).normal(0, 1, (1, Qx[L].shape[1])))[None], len(S), 0) for L in LAYERS}
Qwu = {L: a2_wu_directions(tY) for L in LAYERS}
# readout on the SCAM photo and sigma on the twin (one image at a time: the token counts differ)
RO, SIG = [], {L: [] for L in LAYERS}
for i, r in enumerate(S.itertuples()):
    n_img = len([t for t in mm_encode(T[i:i + 1], [r.img])["input_ids"][0].tolist() if t == B_PAD])
    rj, rl, _ = e_ranks_at(T[i:i + 1], [r.img], e_where_image(n_img), n_img, [S.ids_X.iloc[i], S.ids_Y.iloc[i]], 1)
    RO.append(dict(X_img=int(rj[0, :, :-1, 0].min()), Y_img=int(rj[0, :, :-1, 1].min()), X_dec=int(rj[0, :, -1, 0].min()), Y_dec=int(rj[0, :, -1, 1].min()), X_img_logit=int(rl[0, :, :-1, 0].min()), n_image=n_img))
    n_tw = len([t for t in mm_encode(T[i:i + 1], [r.twin])["input_ids"][0].tolist() if t == B_PAD])
    _, R = mm_forward(T[i:i + 1], [r.twin], LAYERS, where=e_where_image(n_tw))
    for L in LAYERS:
        SIG[L].append(float((R[L][0, :-1].float() @ torch.as_tensor(Qy[L][i, :, 0])).abs().mean()))
RO = pd.DataFrame(RO)
sigma = {L: float(np.mean(SIG[L])) for L in LAYERS}
print(f"readout on the SCAM photo (top-10): X on image tokens {((RO.X_img >= 1) & (RO.X_img <= B_TOP)).mean():.2f}, Y {((RO.Y_img >= 1) & (RO.Y_img <= B_TOP)).mean():.2f}; at the decision position X {((RO.X_dec >= 1) & (RO.X_dec <= B_TOP)).mean():.2f}, Y {((RO.Y_dec >= 1) & (RO.Y_dec <= B_TOP)).mean():.2f}; image tokens {RO.n_image.mean():.0f} on average  [{time.time() - t0:.0f}s]")
# arms on the SCAM photo
arms = [("none", None), ("ablate note word Y", n1_hooks_ablate_rows(Qy)), ("ablate object word X", n1_hooks_ablate_rows(Qx)), ("ablate unrelated word", n1_hooks_ablate_rows(Qz)),
        ("ablate random direction", n1_hooks_ablate_rows(Qr)), ("ablate note word Y, W_U only", n1_hooks_ablate_rows(Qwu))]
res, GEN = [], {}


def a2_mass(lg, ids_list):
    pr = torch.softmax(lg.float(), -1)
    return np.array([float(pr[i, ids].sum()) for i, ids in enumerate(ids_list)])


for name, hooks in arms:
    gen = mm_generate(T, S.img.tolist(), 6, hooks=hooks)
    GEN[name] = gen
    lg, _ = mm_forward(T, S.img.tolist(), [], hooks=hooks)
    mX, mY = a2_mass(lg, S.ids_X.tolist()), a2_mass(lg, S.ids_Y.tolist())
    sx, sy = JUDGE.name_match(gen, S.object_label.tolist()), JUDGE.name_match(gen, S.attack_word.tolist())
    for i, r in enumerate(S.itertuples()):
        res.append(dict(kind="ablation", arm=name, alpha=0.0, item=r.key, X=r.object_label, Y=r.attack_word, generation=gen[i], says_X=sx[i], says_Y=sy[i], mass_X=mX[i], mass_Y=mY[i]))
    print(f"{name:30s} says X {rate(sx):.2f}  says Y {rate(sy):.2f}  first-token mass X {mX.mean():.2f} Y {mY.mean():.2f}  [{time.time() - t0:.0f}s]")
# injection on the twin
for alpha in [0.0] + A2_ALPHAS:
    for fam, Q in (("note word Y", Qy), ("random, matched norm", Qr)):
        if alpha == 0.0 and fam != "note word Y":
            continue
        hooks = None if alpha == 0.0 else a2_hooks_add_rows(Q, {L: alpha * sigma[L] for L in LAYERS})
        gen = mm_generate(T, S.twin.tolist(), 6, hooks=hooks)
        lg, _ = mm_forward(T, S.twin.tolist(), [], hooks=hooks)
        mX, mY = a2_mass(lg, S.ids_X.tolist()), a2_mass(lg, S.ids_Y.tolist())
        sx, sy = JUDGE.name_match(gen, S.object_label.tolist()), JUDGE.name_match(gen, S.attack_word.tolist())
        for i, r in enumerate(S.itertuples()):
            res.append(dict(kind="injection", arm=fam if alpha else "none", alpha=alpha, item=r.key, X=r.object_label, Y=r.attack_word, generation=gen[i], says_X=sx[i], says_Y=sy[i], mass_X=mX[i], mass_Y=mY[i]))
        print(f"twin + {alpha:.1f} sigma along {fam:22s} says X {rate(sx):.2f}  says Y {rate(sy):.2f}  [{time.time() - t0:.0f}s]")
A2 = pd.DataFrame(res)
A2.to_csv(f"{RUN_DIR}/a2_scam.csv", index=False)
JUDGE.to_label(A2[(A2.kind == "ablation") & (A2.arm.isin(["none", "ablate note word Y"]))], "a2", label_col="says_X")
# self-report after the model's own answer on the SCAM photo
wr = [(g.strip().lower().startswith("yes")) if g.strip().lower()[:3] in ("yes", "no", "no.") else None for g in mm_generate([a2_chat2(GEN["none"][i], "Is there writing in this image? Answer yes or no.") for i in range(len(S))], S.img.tolist(), 4)]
src = [("object" if g.strip().lower().startswith("object") else "writing" if g.strip().lower().startswith("writ") else None) for g in mm_generate([a2_chat2(GEN["none"][i], "Did your answer come from the object you see or from writing in the image? Answer object or writing.") for i in range(len(S))], S.img.tolist(), 4)]
base = A2[(A2.kind == "ablation") & (A2.arm == "none")].set_index("item")
S["says_writing_present"], S["source_report"], S["clean_says_X"], S["clean_says_Y"] = wr, src, base.loc[S.key].says_X.values, base.loc[S.key].says_Y.values
S.drop(columns=["img", "twin", "ids_X", "ids_Y"]).merge(RO, left_index=True, right_index=True).to_csv(f"{RUN_DIR}/a2_items.csv", index=False)
# --- summary
abl = A2[A2.kind == "ablation"]
pb = lambda arm, col: abl[abl.arm == arm].set_index("item")[col].astype(float).fillna(0).loc[S.key]
paired = lambda arm, col="says_X": b_boot_scene((pb(arm, col) - pb("none", col)).values, S.key.values)
resA2 = dict(n=int(len(S)), n_candidates=int(len(cand)), gate_rate=rate(gate), arms={}, injection={}, readout={}, self_report={})
for name, _ in arms:
    resA2["arms"][name] = dict(says_X=b_boot_scene(pb(name, "says_X").values, S.key.values), says_Y=b_boot_scene(pb(name, "says_Y").values, S.key.values), mass_X=float(pb(name, "mass_X").mean()), mass_Y=float(pb(name, "mass_Y").mean()))
label_following = resA2["arms"]["none"]["says_Y"][0]
ry, rx = paired("ablate note word Y"), b_boot_scene((pb("ablate object word X", "says_Y") - pb("none", "says_Y")).values, S.key.values)
rz, rr, rw = paired("ablate unrelated word"), paired("ablate random direction"), paired("ablate note word Y, W_U only")
resA2.update(rescue_Y=ry, loss_X=rx, ctrl_unrelated=rz, ctrl_random=rr, ctrl_wu=rw)
inj = A2[A2.kind == "injection"]
for fam in ("note word Y", "random, matched norm"):
    resA2["injection"][fam] = {str(a): b_boot_scene(inj[(inj.arm == fam) & (inj.alpha == a)].set_index("item").says_Y.astype(float).fillna(0).loc[S.key].values, S.key.values) for a in A2_ALPHAS}
resA2["injection"]["baseline_says_Y"] = b_boot_scene(inj[inj.alpha == 0].set_index("item").says_Y.astype(float).fillna(0).loc[S.key].values, S.key.values)
top = A2_ALPHAS[-1]
dose = [resA2["injection"]["note word Y"][str(a)][0] for a in A2_ALPHAS]
ind_gap = b_boot_scene((inj[(inj.arm == "note word Y") & (inj.alpha == top)].set_index("item").says_Y.astype(float).fillna(0).loc[S.key] - inj[(inj.arm == "random, matched norm") & (inj.alpha == top)].set_index("item").says_Y.astype(float).fillna(0).loc[S.key]).values, S.key.values)
resA2["induction_gap_top_alpha"] = ind_gap
resA2["induction_monotone"] = bool(all(dose[i] <= dose[i + 1] + 1e-9 for i in range(len(dose) - 1)))
for col, lab in (("X_img", "X on image tokens"), ("Y_img", "Y on image tokens"), ("X_dec", "X at the decision position"), ("Y_dec", "Y at the decision position"), ("X_img_logit", "X on image tokens (logit lens)")):
    v = RO[col].values
    resA2["readout"][lab] = b_boot_scene(((v >= 1) & (v <= B_TOP)).astype(float), S.key.values)
lf, on = S[S.clean_says_Y == True], S[S.clean_says_X == True]
resA2["self_report"] = dict(writing_present_yes=rate(S.says_writing_present.tolist()), source_writing_when_following=rate([v == "writing" if v else None for v in lf.source_report]), source_writing_when_naming_object=rate([v == "writing" if v else None for v in on.source_report]), n_following=int(len(lf)), n_naming=int(len(on)))
premise = label_following >= 0.3
ctrl_ok = all(abs(c[0]) <= 0.05 for c in (rz, rr, rw))
resA2["verdict"] = ("premise failed: label-following below 0.3" if not premise else
                    "rescue: removing the note word's one direction returns the model to the object" if ry[0] >= 0.3 and ry[1] > 0 and ctrl_ok else
                    "no rescue" if ry[2] < 0.3 else "undetermined at this n")
resA2["induction_verdict"] = "induction: the note word's direction, added to the clean photo, makes the model say it" if resA2["induction_monotone"] and ind_gap[0] >= 0.3 and ind_gap[1] > 0 else "no induction" if ind_gap[2] < 0.3 else "undetermined"
resA2["seconds"] = time.time() - t0
save(resA2, "eA2_results")
print("readout: " + ", ".join(f"{k} {v[0]:.2f}" for k, v in resA2["readout"].items()))
print(f"label-following {label_following:.2f}; rescue {b_fmt(ry)}; X direction raises label-following by {b_fmt(rx)}; controls unrelated {rz[0]:+.2f} random {rr[0]:+.2f} W_U-only {rw[0]:+.2f}")
print(f"injection on the twin, says Y by alpha {A2_ALPHAS}: note direction {[round(d, 2) for d in dose]}, random {[round(resA2['injection']['random, matched norm'][str(a)][0], 2) for a in A2_ALPHAS]}; gap at the top alpha {b_fmt(ind_gap)}; baseline {resA2['injection']['baseline_says_Y'][0]:.2f}")
print(f"self-report: writing present {resA2['self_report']['writing_present_yes']:.2f}; source 'writing' when following the note {resA2['self_report']['source_writing_when_following']:.2f} (n = {len(lf)}), when naming the object {resA2['self_report']['source_writing_when_naming_object']:.2f} (n = {len(on)})")
print(f"-> {resA2['verdict']} | {resA2['induction_verdict']}")
# --- figure
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5), gridspec_kw=dict(wspace=0.42, width_ratios=[1.5, 1, 1]))
fig.subplots_adjust(bottom=0.24, top=0.84)
names = [n for n, _ in arms]
short = {"none": "no change", "ablate note word Y": "note word's\ndirection removed", "ablate object word X": "object word's\ndirection removed", "ablate unrelated word": "unrelated word's\ndirection removed", "ablate random direction": "random\ndirection removed", "ablate note word Y, W_U only": "note word,\nW_U-only direction"}
x = np.arange(len(names))
for j, (col, lab, c) in enumerate((("says_X", "names the object", OKABE[0]), ("says_Y", "reads the note", OKABE[1]))):
    v = [resA2["arms"][n][col] for n in names]
    ax[0].bar(x + (j - 0.5) * 0.38, [t[0] for t in v], 0.36, color=c, alpha=0.9)
    ax[0].errorbar(x + (j - 0.5) * 0.38, [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
    ax[0].text(0.98, 0.97 - 0.06 * j, lab, color=c, fontsize=8, transform=ax[0].transAxes, ha="right", va="top")
ax[0].set(xticks=x, xticklabels=[short[n] for n in names], ylim=(0, 1.12), ylabel="share of answers (judge), item 95% CI", xlabel=f"SCAM photos with a note ({len(S)} items); one direction removed at every image token, read layers")
ax[0].tick_params(axis="x", labelsize=7)
for fam, c, ls in (("note word Y", OKABE[1], "-"), ("random, matched norm", OKABE[4], ":")):
    v = [resA2["injection"][fam][str(a)] for a in A2_ALPHAS]
    ax[1].errorbar(A2_ALPHAS, [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="-o", ls=ls, color=c, ms=4, lw=1.6, capsize=3)
    ax[1].text(A2_ALPHAS[-1] * 1.08, v[-1][0], "the note word's\ndirection" if fam == "note word Y" else "random direction,\nsame norm", color=c, fontsize=7.5, va="center")
ax[1].axhline(resA2["injection"]["baseline_says_Y"][0], color="0.5", lw=0.8, ls="--")
ax[1].set(xscale="log", xticks=A2_ALPHAS, xticklabels=[str(a) for a in A2_ALPHAS], xlim=(0.35, 7.5), ylim=(0, 1.05), xlabel="alpha (units of the direction's mean projection)", ylabel="share saying the note word (judge)\non the twin photo with no note")
ax[1].minorticks_off()
sr = resA2["self_report"]
ax[2].bar([0, 1], [sr["source_writing_when_following"], sr["source_writing_when_naming_object"]], 0.6, color=[OKABE[1], OKABE[0]])
ax[2].set(xticks=[0, 1], xticklabels=[f"answer followed the note\n(n = {sr['n_following']})", f"answer named the object\n(n = {sr['n_naming']})"], ylim=(0, 1.05), ylabel="share saying the answer came from 'writing'", xlabel=f"source report after its own answer; says writing is present on {sr['writing_present_yes']:.2f}")
import textwrap as _tw
fig.suptitle("\n".join(_tw.wrap(f"A2 ({MODEL_TAG}, SCAM, {len(S)} items). {resA2['verdict']}; {resA2['induction_verdict']}. Label-following {label_following:.2f}; rescue {b_fmt(ry)}; induction gap at alpha = {top}: {b_fmt(ind_gap)}", 170)), fontsize=9.5)
ws_finish(fig, "figA2_scam", f"A2 · SCAM · {len(S)} items · judge {JUDGE.model}")

# %% ---------------------------------------------------------------- F1: a fact seen in a picture, used by a computation that never saw the picture
# Donor: an E2 canvas with one object X in slot q, read with a neutral prompt ("Study the scene.") and no question. Receiver: a blank canvas,
# the same intro, and a rule the donor never saw: "If the object in image slot q is an animal, answer 7; otherwise answer 3." and "What
# is the first letter of the name of the object in image slot q?" Transplants into the receiver's image tokens: the donor's lens part
# (r = 256, read layers), its J++ part (where fitted), a random part, the full state at every layer plus the layer-0 input (positive
# control); a non-target donor (X in another slot; the correct answer for slot q is then "3"; the letter rule is not scored there); the
# ceiling: the fact typed as text with no image. Answers are a digit or a letter, scored by the first character. Rules (RECORD.md 03:40):
# positive control >= 0.8; the J part carries a reusable fact when its accuracy >= random + 0.3 with the scene-paired interval > 0; the
# gap to the typed ceiling is reported. Predictions: full >= 0.85; J part 0.6-0.8; random <= 0.35; typed >= 0.9.
F1_N = int(os.environ.get("F1_N", E_N_SCENES))
F1_ANIMALS = {"dog", "cat", "bird", "horse"}
F1_INTRO = "Study the scene. It has an image grid with numbered slots."
F1_RULES = {"animal": "Rule: if the object in image slot {q} is an animal, answer 7; otherwise answer 3. Reply with the number only.",
            "letter": "What is the first letter of the name of the object in image slot {q}? Reply with the letter only."}
t0 = time.time()


def f1_truth(rule, x):
    return ("7" if x in F1_ANIMALS else "3") if rule == "animal" else x[0].upper()


def f1_score(gens, truths):
    return [((g.strip().strip("*").upper()[:1] == t) if g.strip() else None) for g, t in zip(gens, truths)]


# --- reads
rng_ = np.random.default_rng(3)
rows = []
for i in range(F1_N):
    pool = [w for w in E_WORDS if (w in F1_ANIMALS) == (i % 2 == 0)]
    x = pool[int(rng_.integers(len(pool)))]
    q = int(rng_.integers(1, E2_N_SLOTS + 1))
    q2 = int(rng_.choice([s for s in range(1, E2_N_SLOTS + 1) if s != q]))
    rows.append(dict(scene=i, X=x, q=q, q2=q2, tile=E_BANK[x][i % len(E_BANK[x])]))
S = pd.DataFrame(rows)
empty = {s: None for s in range(1, E2_N_SLOTS + 1)}
don_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q: r.tile}) for r in S.itertuples()]
don2_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q2: r.tile}) for r in S.itertuples()]
blank = e_canvas(E2_COLS, E2_ROWS, empty)
don_txt = [e_chat_media([F1_INTRO, "IMAGE"])] * len(S)
N_IMG = E2_COLS * E2_ROWS * 49
ids0 = mm_encode(don_txt[:1], don_img[:1])["input_ids"][0].tolist()
vs_off = ids0.index(MM_VS) + 1
img_pos = list(range(vs_off, vs_off + N_IMG))
LF = [-1] + list(range(FINAL + 1))
Qs = {L: b_lens_subspace(L, 256) for L in LAYERS}
Qr = {L: b_random_subspace(Qs[L].shape[0], 256, 100 + L) for L in LAYERS}
Qpp = {L: e_jpp_subspace(L, 256) for L in LAYERS} if JPP is not None else None
d_model = hf_model.config.get_text_config().hidden_size


def f1_capture(images):
    REC = {L: np.zeros((len(S), N_IMG, d_model), np.float16) for L in LF}
    for j in range(len(S)):
        _, R = mm_forward(don_txt[j:j + 1], images[j:j + 1], LF, where=lambda ids, _t=img_pos: list(_t))
        for L in LF:
            REC[L][j] = R[L][0].half().numpy()
    return REC


REC, REC2 = f1_capture(don_img), f1_capture(don2_img)
print(f"donor states captured for {len(S)} scenes ({len(LF)} layers)  [{time.time() - t0:.0f}s]")
mm_contact_sheet([(f"donor: {r.X} in slot {r.q}", don_img[i]) for i, r in enumerate(S.head(3).itertuples())] + [("receiver: blank", blank)], 4, "figF1_stimuli")
res = []
for rule, tmpl in F1_RULES.items():
    rec_txt = [e_chat_media([F1_INTRO, "IMAGE", tmpl.format(q=r.q)]) for r in S.itertuples()]
    typed = [chat([("user", F1_INTRO + f" Image slot {r.q} contains a {r.X}. " + tmpl.format(q=r.q))]) for r in S.itertuples()]
    truths = [f1_truth(rule, r.X) for r in S.itertuples()]
    n_all = [len(mm_encode(rec_txt[j:j + 1], [blank])["input_ids"][0]) for j in range(len(S))]
    assert mm_encode(rec_txt[:1], [blank])["input_ids"][0].tolist().index(MM_VS) + 1 == vs_off, "the image does not sit at the donor's positions"
    pos = [(lambda s, _n=n_all[j]: np.array([s - _n + t for t in img_pos])) for j in range(len(S))]
    _lo = lens if hasattr(lens, "jacobians") else JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION)
    for L in LF:                                                                 # JAC holds the read layers only; the every-layer arm needs every fitted J
        if 0 <= L < FINAL and L not in JAC:
            JAC[L] = _lo.jacobians[L].to(DEV)
    QsA = {L: (b_lens_subspace(L, 256) if 0 <= L < FINAL else Qs[LAYERS[0] if L < 0 else LAYERS[-1]]) for L in LF}   # the lens basis at every layer that has a J (the input and the last layer borrow the nearest)
    QrA = {L: b_random_subspace(Qs[LAYERS[0]].shape[0], 256, 100 + L) for L in LF}
    arms = [("blank, no transplant", None, None), ("full state, every layer", e2p_pos_hooks(LF, REC, pos), None), ("full state, read layers", e2p_pos_hooks(LAYERS, REC, pos), None),
            ("J part, read layers", e2p_pos_hooks(LAYERS, REC, pos, Qs, "lens"), None), ("J part, every layer", e2p_pos_hooks(LF, REC, pos, QsA, "lens"), None),
            ("random part, read layers", e2p_pos_hooks(LAYERS, REC, pos, Qr, "lens"), None), ("random part, every layer", e2p_pos_hooks(LF, REC, pos, QrA, "lens"), None)]
    if Qpp is not None:
        arms.insert(3, ("J++ part, read layers", e2p_pos_hooks(LAYERS, REC, pos, Qpp, "lens"), None))
    if rule == "animal":
        arms.append(("non-target donor, every layer", e2p_pos_hooks(LF, REC2, pos), None))
    arms.append(("the fact typed as text", None, "typed"))
    for name, hooks, mode in arms:
        tx, im = (typed, [None] * len(S)) if mode == "typed" else (rec_txt, [blank] * len(S))
        gen = mm_generate(tx, im, 4, hooks=hooks)
        tr = ["3"] * len(S) if name.startswith("non-target") else truths
        ok = f1_score(gen, tr)
        for i, r in enumerate(S.itertuples()):
            res.append(dict(rule=rule, arm=name, scene=r.scene, X=r.X, q=r.q, truth=tr[i], generation=gen[i], correct=ok[i]))
        print(f"{rule:7s} {name:30s} accuracy {np.nanmean([float(v) for v in ok if v is not None]):.2f}  [{time.time() - t0:.0f}s]")
F1 = pd.DataFrame(res)
F1.to_csv(f"{RUN_DIR}/f1_fact_reuse.csv", index=False)
resF1 = dict(n=int(len(S)), rules={})
for rule in F1_RULES:
    d = F1[F1.rule == rule]
    acc = lambda arm: d[d.arm == arm].set_index("scene").correct.astype(float).fillna(0).loc[S.scene]
    out = {arm: b_boot_scene(acc(arm).values, S.scene.values) for arm in d.arm.unique()}
    jr = b_boot_scene((acc("J part, read layers") - acc("random part, read layers")).values, S.scene.values)
    ok_ctrl = out["full state, every layer"][0] >= 0.8
    verdict = ("positive control failed" if not ok_ctrl else "the lens part carries a reusable fact" if jr[0] >= 0.3 and jr[1] > 0 else "the lens part does not carry the fact into the new rule" if jr[2] < 0.3 else "undetermined at this n")
    resF1["rules"][rule] = dict(accuracy=out, J_minus_random=jr, control_ok=bool(ok_ctrl), verdict=verdict)
    print(f"{rule}: " + "  ".join(f"{a} {v[0]:.2f}" for a, v in out.items()) + f" | J − random {b_fmt(jr)} -> {verdict}")
resF1["seconds"] = time.time() - t0
save(resF1, "eF1_results")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(wspace=0.9))
fig.subplots_adjust(left=0.2, bottom=0.16, top=0.82)
cols = {"blank, no transplant": "0.4", "full state, every layer": OKABE[6], "full state, read layers": "0.25", "J part, read layers": OKABE[0], "J part, every layer": OKABE[0], "J++ part, read layers": OKABE[2],
        "random part, read layers": OKABE[4], "random part, every layer": OKABE[4], "non-target donor, every layer": OKABE[3], "the fact typed as text": OKABE[5]}
for a_, rule in zip(ax, F1_RULES):
    out = resF1["rules"][rule]["accuracy"]
    names = list(out)
    yy = np.arange(len(names))[::-1]
    a_.barh(yy, [out[n][0] for n in names], 0.6, color=[cols.get(n, "0.6") for n in names])
    a_.errorbar([out[n][0] for n in names], yy, xerr=[[out[n][0] - out[n][1] for n in names], [out[n][2] - out[n][0] for n in names]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
    a_.axvline(0.5 if rule == "animal" else 1 / 26, color="0.6", lw=0.8, ls=":")
    a_.set(yticks=yy, yticklabels=names, xlim=(0, 1.02), xlabel=f"accuracy on the new rule ({rule}), 95% CI over {len(S)} scenes", title=f"{resF1['rules'][rule]['verdict']}")
    a_.tick_params(axis="y", labelsize=8)
import textwrap as _tw
fig.suptitle("\n".join(_tw.wrap(f"F1 ({MODEL_TAG}). A receiver that never saw the picture gets a blank canvas, the donor's transplanted part, and a rule the donor never saw. Animal rule: J − random {b_fmt(resF1['rules']['animal']['J_minus_random'])}; letter rule: {b_fmt(resF1['rules']['letter']['J_minus_random'])}.", 150)), fontsize=9.5)
ws_finish(fig, "figF1_fact_reuse", f"F1 · {len(S)} scenes · first-character scoring")

# %% ---------------------------------------------------------------- F2: two facts seen in a picture, used by a computation chosen after the picture was encoded
# Donor: an E2 canvas with two objects, in slots q and r (half the scenes one animal and one non-animal, half the same kind), two sources
# (COCO crops; the object names printed in the slots), read under a neutral prompt with no question. Receiver: a blank canvas, a
# transplanted part of the donor's image tokens, and a rule chosen after encoding: (1) the first letter of the object in slot q; (2) slot q
# animal, 7 / 3; (3) "Are the objects in slots q and r the same kind, both animals or both not? yes / no"; (4) on mixed scenes, "Which
# slot holds the animal, q or r?" Arms: blank; the full state at every layer and the layer-0 input; the lens part at every fitted layer;
# a random part at every layer; the facts typed as text. First-character scoring. Rules (RECORD.md 04:00): full >= 0.8 per rule and
# source; the lens part carries a fact when >= random + 0.3 with the scene-paired interval > 0; the saying-using dissociation holds when
# the lens part passes rule (1) and fails (3) and (4) (lens - random <= 0.1 with full >= 0.8) for a source.
F2_N = int(os.environ.get("F2_N", E_N_SCENES))
F2_ANIMALS = {"dog", "cat", "bird", "horse"}
F2_INTRO = "Study the scene. It has an image grid with numbered slots."
F2_RULES = {"letter": "What is the first letter of the name of the object in image slot {q}? Reply with the letter only.",
            "animal": "Rule: if the object in image slot {q} is an animal, answer 7; otherwise answer 3. Reply with the number only.",
            "same kind": "Are the objects in image slots {q} and {r} the same kind, both animals or both not animals? Answer yes or no.",
            "which slot": "Which of image slots {q} and {r} holds an animal? Reply with the slot number only."}
t0 = time.time()


def f2_truth(rule, x, y, q, r):
    if rule == "letter":
        return x[0].upper()
    if rule == "animal":
        return "7" if x in F2_ANIMALS else "3"
    if rule == "same kind":
        return "Y" if (x in F2_ANIMALS) == (y in F2_ANIMALS) else "N"
    return str(q) if x in F2_ANIMALS else str(r)


def f2_score(gens, truths):
    return [((g.strip().strip("*").upper()[:1] == t) if g.strip() else None) for g, t in zip(gens, truths)]


# --- reads
rng_ = np.random.default_rng(5)
rows = []
animals = [w for w in E_WORDS if w in F2_ANIMALS]
others = [w for w in E_WORDS if w not in F2_ANIMALS]
for i in range(F2_N):
    mixed = i % 2 == 0
    if mixed:
        x, y = animals[int(rng_.integers(len(animals)))], others[int(rng_.integers(len(others)))]
        if rng_.random() < 0.5:
            x, y = y, x
    else:
        pool = animals if rng_.random() < 0.5 else others
        x, y = [pool[j] for j in rng_.choice(len(pool), 2, replace=False)]
    q, r = [int(s) for s in rng_.choice(np.arange(1, E2_N_SLOTS + 1), 2, replace=False)]
    rows.append(dict(scene=i, X=x, Y=y, q=q, r=r, mixed=mixed, tx=E_BANK[x][i % len(E_BANK[x])], ty=E_BANK[y][(i + 1) % len(E_BANK[y])]))
S = pd.DataFrame(rows)
empty = {s: None for s in range(1, E2_N_SLOTS + 1)}
sources = {"image": [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q: r.tx, r.r: r.ty}) for r in S.itertuples()],
           "printed": [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q: t1_word_tile(r.X), r.r: t1_word_tile(r.Y)}) for r in S.itertuples()]}
blank = e_canvas(E2_COLS, E2_ROWS, empty)
don_txt = [e_chat_media([F2_INTRO, "IMAGE"])] * len(S)
N_IMG = E2_COLS * E2_ROWS * 49
ids0 = mm_encode(don_txt[:1], sources["image"][:1])["input_ids"][0].tolist()
vs_off = ids0.index(MM_VS) + 1
img_pos = list(range(vs_off, vs_off + N_IMG))
LF = [-1] + list(range(FINAL + 1))
_lo = lens if hasattr(lens, "jacobians") else JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION)
for L in LF:
    if 0 <= L < FINAL and L not in JAC:
        JAC[L] = _lo.jacobians[L].to(DEV)
Qs = {L: b_lens_subspace(L, 256) for L in LAYERS}
QsA = {L: (b_lens_subspace(L, 256) if 0 <= L < FINAL else Qs[LAYERS[0] if L < 0 else LAYERS[-1]]) for L in LF}
QrA = {L: b_random_subspace(Qs[LAYERS[0]].shape[0], 256, 100 + L) for L in LF}
d_model = hf_model.config.get_text_config().hidden_size


def f2_capture(images):
    REC = {L: np.zeros((len(S), N_IMG, d_model), np.float16) for L in LF}
    for j in range(len(S)):
        _, R = mm_forward(don_txt[j:j + 1], images[j:j + 1], LF, where=lambda ids, _t=img_pos: list(_t))
        for L in LF:
            REC[L][j] = R[L][0].half().numpy()
    return REC


mm_contact_sheet([(f"image: {r.X} in {r.q}, {r.Y} in {r.r}", sources["image"][i]) for i, r in enumerate(S.head(2).itertuples())] + [(f"printed: {r.X} in {r.q}, {r.Y} in {r.r}", sources["printed"][i]) for i, r in enumerate(S.head(2).itertuples())], 4, "figF2_stimuli")
res = []
for src, imgs in sources.items():
    REC = f2_capture(imgs)
    print(f"{src}: donor states captured  [{time.time() - t0:.0f}s]")
    for rule, tmpl in F2_RULES.items():
        sel = S if rule != "which slot" else S[S.mixed]
        rec_txt = [e_chat_media([F2_INTRO, "IMAGE", tmpl.format(q=r.q, r=r.r)]) for r in sel.itertuples()]
        typed = [chat([("user", F2_INTRO + f" Image slot {r.q} contains a {r.X} and image slot {r.r} contains a {r.Y}. " + tmpl.format(q=r.q, r=r.r))]) for r in sel.itertuples()]
        truths = [f2_truth(rule, r.X, r.Y, r.q, r.r) for r in sel.itertuples()]
        idx = sel.index.values
        n_all = [len(mm_encode(rec_txt[j:j + 1], [blank])["input_ids"][0]) for j in range(len(sel))]
        pos = [(lambda s, _n=n_all[j]: np.array([s - _n + t for t in img_pos])) for j in range(len(sel))]
        RECs = {L: REC[L][idx] for L in LF}
        arms = [("blank, no transplant", None, None), ("full state, every layer", e2p_pos_hooks(LF, RECs, pos), None), ("lens part, every layer", e2p_pos_hooks(LF, RECs, pos, QsA, "lens"), None),
                ("random part, every layer", e2p_pos_hooks(LF, RECs, pos, QrA, "lens"), None), ("full state, read layers", e2p_pos_hooks(LAYERS, RECs, pos), None),
                ("lens part, read layers", e2p_pos_hooks(LAYERS, RECs, pos, Qs, "lens"), None), ("random part, read layers", e2p_pos_hooks(LAYERS, RECs, pos, {L: QrA[L] for L in LAYERS}, "lens"), None),
                ("the facts typed as text", None, "typed")]
        for name, hooks, mode in arms:
            tx, im = (typed, [None] * len(sel)) if mode == "typed" else (rec_txt, [blank] * len(sel))
            gen = mm_generate(tx, im, 4, hooks=hooks)
            ok = f2_score(gen, truths)
            for j, r in enumerate(sel.itertuples()):
                res.append(dict(source=src, rule=rule, arm=name, scene=r.scene, X=r.X, Y=r.Y, q=r.q, r=r.r, truth=truths[j], generation=gen[j], correct=ok[j]))
            print(f"{src:7s} {rule:10s} {name:26s} accuracy {np.nanmean([float(v) for v in ok if v is not None]):.2f}  [{time.time() - t0:.0f}s]")
    del REC
F2 = pd.DataFrame(res)
F2.to_csv(f"{RUN_DIR}/f2_two_facts.csv", index=False)
resF2 = dict(n=int(len(S)), cells={}, dissociation={})
for src in sources:
    for rule in F2_RULES:
        d = F2[(F2.source == src) & (F2.rule == rule)]
        scenes = sorted(d.scene.unique())
        acc = lambda arm: d[d.arm == arm].set_index("scene").correct.astype(float).fillna(0).loc[scenes]
        out = {arm: b_boot_scene(acc(arm).values, np.array(scenes)) for arm in d.arm.unique()}
        lr = b_boot_scene((acc("lens part, read layers") - acc("random part, read layers")).values, np.array(scenes))
        gain = {arm: b_boot_scene((acc(arm) - acc("blank, no transplant")).values, np.array(scenes)) for arm in ("lens part, read layers", "random part, read layers", "lens part, every layer")}
        ok_ctrl = out["full state, read layers"][0] >= 0.8
        sl = out["lens part, read layers"][0]
        verdict = ("positive control failed" if not ok_ctrl else "the mid-layer slice carries it" if sl >= out["full state, read layers"][0] - 0.15 and gain["lens part, read layers"][1] > 0
                   else "the mid-layer slice does not carry it" if gain["lens part, read layers"][2] <= 0.15 else "undetermined at this n")
        resF2["cells"][f"{src} / {rule}"] = dict(accuracy=out, lens_minus_random=lr, gain_over_blank=gain, n=len(scenes), control_ok=bool(ok_ctrl), verdict=verdict)
        print(f"{src:7s} {rule:10s} (n = {len(scenes)}): " + "  ".join(f"{a} {v[0]:.2f}" for a, v in out.items()) + f" | lens − random (read layers) {b_fmt(lr)} -> {verdict}")
    c = resF2["cells"]
    says = c[f"{src} / letter"]["verdict"] == "the mid-layer slice carries it"
    uses = [c[f"{src} / {ru}"]["verdict"] for ru in ("same kind", "which slot")]
    resF2["dissociation"][src] = ("saying without using: a mid-layer slice carries the name but not the two-fact inference" if says and all(u == "the mid-layer slice does not carry it" for u in uses) else
                                  "the mid-layer slice carries both" if says and all(u == "the mid-layer slice carries it" for u in uses) else "no dissociation established at this n")
    print(f"{src}: {resF2['dissociation'][src]}")
resF2["seconds"] = time.time() - t0
save(resF2, "eF2_results")
fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), gridspec_kw=dict(wspace=0.95, hspace=0.5))
fig.subplots_adjust(left=0.14, bottom=0.08, top=0.86)
cols = {"blank, no transplant": "0.4", "full state, every layer": OKABE[6], "lens part, every layer": OKABE[0], "random part, every layer": OKABE[4], "full state, read layers": "0.25", "lens part, read layers": OKABE[0], "random part, read layers": OKABE[4], "the facts typed as text": OKABE[5]}
for i, src in enumerate(sources):
    for j, rule in enumerate(F2_RULES):
        a_ = axes[i, j]
        cell = resF2["cells"][f"{src} / {rule}"]
        names = list(cell["accuracy"])
        yy = np.arange(len(names))[::-1]
        a_.barh(yy, [cell["accuracy"][n][0] for n in names], 0.6, color=[cols.get(n, "0.6") for n in names])
        a_.errorbar([cell["accuracy"][n][0] for n in names], yy, xerr=[[cell["accuracy"][n][0] - cell["accuracy"][n][1] for n in names], [cell["accuracy"][n][2] - cell["accuracy"][n][0] for n in names]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
        a_.axvline(1 / 26 if rule == "letter" else 0.5, color="0.6", lw=0.8, ls=":")
        a_.set(yticks=yy, yticklabels=names if j == 0 else [""] * len(names), xlim=(0, 1.02), xlabel=f"accuracy (n = {cell['n']})", title=f"{src} donor, rule: {rule}\n{cell['verdict']}")
        a_.tick_params(axis="y", labelsize=7.5)
import textwrap as _tw
fig.suptitle("\n".join(_tw.wrap(f"F2 ({MODEL_TAG}). A receiver that never saw the picture gets a blank canvas, a transplanted part of the donor's image tokens, and a rule chosen after encoding. Picture donor: {resF2['dissociation']['image']}. Printed-word donor: {resF2['dissociation']['printed']}.", 160)), fontsize=9.5)
ws_finish(fig, "figF2_two_facts", f"F2 · {len(S)} scenes · first-character scoring")


# %% ---------------------------------------------------------------- G1: the name as the handle. Does the model use what it sees through the direction that lets it say it?
# One COCO crop X in slot q of a six-slot canvas. Four questions about slot q, asked of the model's own picture: say (the noun), letter
# (its first letter), animal (7 / 3), ride (7 / 3: something people ride or drive). Removal on the slot's 49 image tokens at the read
# layers, mean-ablated: the own name's lens direction (rank 1), the other name Y's, a random direction, the own name's W_U-only
# direction, J++ where fitted, the word "animal"'s direction, the own name orthogonal to "animal", the slot's top principal direction
# (the energy control), the 256-d lens part, a random 256-d part. The printed rendering (the word X in the slot) clean and with the own
# name removed. Injection of alpha * sigma_L along Y (the other kind), Z (the same kind), a random direction and Y's W_U-only direction
# at two sites: the slot's image tokens, and the decision position (the last prompt token) alone; alpha in {1, 2, 4, 8}; alpha* per site
# picked on 16 calibration scenes, verdicts read on the other 48. Injection along X on the blank canvas. Rules R0-R6: RECORD.md 03:55 / 04:00.
G1_N = int(os.environ.get("G1_N", E_N_SCENES))
G1_CAL = int(os.environ.get("G1_CAL", 16))
G1_ANIMALS = {"dog", "cat", "bird", "horse"}
G1_RIDE = {"horse", "car", "bus", "train", "bicycle", "airplane", "boat"}
G1_INTRO = "Study the scene. It has an image grid with numbered slots."
G1_TASKS = {"say": "What is the object in image slot {q}? Reply with the noun only.",
            "letter": "What is the first letter of the name of the object in image slot {q}? Reply with the letter only.",
            "animal": "Rule: if the object in image slot {q} is an animal, answer 7; otherwise answer 3. Reply with the number only.",
            "ride": "Rule: if the object in image slot {q} is something people ride or drive, answer 7; otherwise answer 3. Reply with the number only."}
G1_RULES = ["animal", "ride"]
G1_ALPHAS = [1.0, 2.0, 4.0, 8.0]
B_PAD = int(getattr(hf_model.config, "image_token_id", B_PAD))
t0 = time.time()


def g1_truth(task, x):
    if task == "say":
        return x
    if task == "letter":
        return x[0].upper()
    return "7" if x in (G1_ANIMALS if task == "animal" else G1_RIDE) else "3"


def g1_score(task, gens, truths):
    if task == "say":
        return [bool(v) for v in e_gen_correct(gens, truths, set(E_WORDS))]
    return [((g.strip().strip("*").upper()[:1] == t) if g.strip() else False) for g, t in zip(gens, truths)]


def g1_hooks(Qrows, pos_of, mode="ablate", mu=None, scale=None):
    """at the read layers, at the positions pos_of(unpadded_row_ids, prompt_index) names: 'ablate' removes each row's Q part (Qrows[L]
    is [n, d, r] per row or [d, r] shared), replaced by mu's part when mu is given (mean-ablation); 'add' adds scale[L] * Qrows[L][i, :, 0]"""
    def hooks(enc, idx):
        ids, am = enc["input_ids"], enc["attention_mask"]
        b, s = ids.shape
        rows_pos = []
        for i in range(b):
            n = int(am[i].sum())
            rows_pos.append([s - n + p for p in pos_of(ids[i, s - n:].tolist(), int(idx[i]))])
        P = max(len(r) for r in rows_pos)
        Pm = torch.as_tensor([r + [r[-1]] * (P - len(r)) for r in rows_pos], device=DEV)
        rows = torch.arange(b, device=DEV)[:, None].expand(-1, P)
        out_ = []
        for L in LAYERS:
            Qn = np.asarray(Qrows[L])
            Qd = torch.as_tensor(Qn[np.asarray(idx)] if Qn.ndim == 3 else Qn, device=DEV, dtype=torch.float32)
            if Qd.ndim == 2:
                Qd = Qd[None].expand(b, -1, -1)
            if mode == "add":
                qd = Qd[:, :, 0] * float(scale[L])

                def fn(mod, inp, out, qd=qd):
                    h = residual_of(out)
                    if h.shape[1] == 1:
                        return out
                    x = h.float()
                    x[rows, Pm] = x[rows, Pm] + qd[:, None, :]
                    x = x.to(h.dtype)
                    return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
            else:
                Md = None if mu is None else torch.einsum("br,bdr->bd", torch.einsum("d,bdr->br", torch.as_tensor(mu[L], device=DEV, dtype=torch.float32), Qd), Qd)

                def fn(mod, inp, out, Qd=Qd, Md=Md):
                    h = residual_of(out)
                    if h.shape[1] == 1:
                        return out
                    x = h.float()
                    v = x[rows, Pm]
                    v = v - torch.einsum("bpr,bdr->bpd", torch.einsum("bpd,bdr->bpr", v, Qd), Qd)
                    if Md is not None:
                        v = v + Md[:, None, :]
                    x[rows, Pm] = v
                    x = x.to(h.dtype)
                    return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
            out_.append((L, fn))
        return out_
    return hooks


@torch.no_grad()
def g1_word_directions(L, token_ids, kind):
    """unit token directions at layer L: 'J' J_L^T W_U[t], 'J++' the J++ analogue, 'W_U' the unembedding row alone -- [n, d, 1]"""
    W = U_OUT[token_ids].float()
    V = W if kind == "W_U" else (JAC[L].float().T @ W.T).T if kind == "J" else (JPP[L].float().T @ W.T).T
    V = V / V.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return V.cpu().numpy()[:, :, None]


def g1_unit_rows(seed):
    V = np.random.default_rng(seed).normal(0, 1, (len(S), d_model)).astype(np.float32)
    return (V / np.linalg.norm(V, axis=1, keepdims=True))[:, :, None]


# --- reads
rng_ = np.random.default_rng(11)
rows = []
for i in range(G1_N):
    pool = [w for w in E_WORDS if (w in G1_ANIMALS) == (i % 2 == 0)]
    x = pool[int(rng_.integers(len(pool)))]
    ys = [w for w in E_WORDS if (w in G1_ANIMALS) != (x in G1_ANIMALS) and (w in G1_RIDE) != (x in G1_RIDE) and w[0] != x[0]]
    zs = [w for w in E_WORDS if w != x and (w in G1_ANIMALS) == (x in G1_ANIMALS) and (w in G1_RIDE) == (x in G1_RIDE) and w[0] != x[0]]
    zs = zs or [w for w in E_WORDS if w != x and (w in G1_ANIMALS) == (x in G1_ANIMALS) and w[0] != x[0]]
    assert ys and zs, x
    rows.append(dict(scene=i, X=x, Y=ys[int(rng_.integers(len(ys)))], Z=zs[int(rng_.integers(len(zs)))], q=int(rng_.integers(1, E2_N_SLOTS + 1)), tile=E_BANK[x][i % len(E_BANK[x])]))
S = pd.DataFrame(rows)
S["calibration"] = S.scene < G1_CAL
empty = {s: None for s in range(1, E2_N_SLOTS + 1)}
own_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q: r.tile}) for r in S.itertuples()]
prt_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q: t1_word_tile(r.X)}) for r in S.itertuples()]
blank = e_canvas(E2_COLS, E2_ROWS, empty)
N_IMG = E2_COLS * E2_ROWS * 49
d_model = hf_model.config.get_text_config().hidden_size
slot_idx = {s: e_slot_tokens(E2_COLS, E2_ROWS, s) for s in range(1, E2_N_SLOTS + 1)}
slot_pos = lambda row, i: [[j for j, t in enumerate(row) if t == B_PAD][k] for k in slot_idx[int(S.q.iloc[i])]]
dec_pos = lambda row, i: [len(row) - 1]
all_pos = lambda row, i: [j for j, t in enumerate(row) if t == B_PAD]
mm_contact_sheet([(f"photo: {r.X} in slot {r.q}", own_img[i]) for i, r in enumerate(S.head(2).itertuples())] + [(f"printed: {r.X} in slot {r.q}", prt_img[i]) for i, r in enumerate(S.head(2).itertuples())], 4, "figG1_stimuli")

# the clean image-token residuals at the read layers (the image tokens precede every question, so one capture serves all four)
cap_txt = [e_chat_media([G1_INTRO, "IMAGE", G1_TASKS["say"].format(q=r.q)]) for r in S.itertuples()]
_, R = mm_forward(cap_txt, own_img, LAYERS, where=lambda ids: [j for j, t in enumerate(ids) if t == B_PAD])
assert R[LAYERS[0]].shape[1] == N_IMG, R[LAYERS[0]].shape
H = {L: R[L].half().numpy() for L in LAYERS}
del R
MU = {L: H[L].reshape(-1, d_model).astype(np.float32).mean(0) for L in LAYERS}
ids_x, ids_y, ids_z = [[E_SID[w][0] for w in S[c]] for c in ("X", "Y", "Z")]
ids_animal, _ok = e_single("animal")
assert _ok, "no single-token form of 'animal'"
U = {}
U["own name"] = {L: g1_word_directions(L, ids_x, "J") for L in LAYERS}
U["other name"] = {L: g1_word_directions(L, ids_y, "J") for L in LAYERS}
U["random direction"] = {L: g1_unit_rows(200 + L) for L in LAYERS}
U["own name, W_U only"] = {L: g1_word_directions(L, ids_x, "W_U") for L in LAYERS}
if JPP is not None:
    U["own name, J++"] = {L: g1_word_directions(L, ids_x, "J++") for L in LAYERS}
U["the word 'animal'"] = {L: g1_word_directions(L, [ids_animal[0]] * len(S), "J") for L in LAYERS}
U["own name, orthogonal to 'animal'"] = {}
for L in LAYERS:
    a, c = U["own name"][L][:, :, 0], U["the word 'animal'"][L][:, :, 0]
    v = a - (a * c).sum(1, keepdims=True) * c
    U["own name, orthogonal to 'animal'"][L] = (v / np.linalg.norm(v, axis=1, keepdims=True))[:, :, None]
U["top principal direction of the slot"] = {}
for L in LAYERS:
    V = []
    for i, r in enumerate(S.itertuples()):
        Xs = torch.as_tensor(H[L][i, slot_idx[r.q]], dtype=torch.float32, device=DEV)
        Xs = Xs - Xs.mean(0, keepdim=True)
        V.append(torch.linalg.svd(Xs, full_matrices=False)[2][0].cpu().numpy())
    U["top principal direction of the slot"][L] = np.stack(V)[:, :, None]
U["lens part, 256 directions"] = {L: b_lens_subspace(L, 256) for L in LAYERS}
U["random part, 256 directions"] = {L: b_random_subspace(d_model, 256, 100 + L) for L in LAYERS}
REMOVALS = list(U)
U["same-kind name"] = {L: g1_word_directions(L, ids_z, "J") for L in LAYERS}
U["other name, W_U only"] = {L: g1_word_directions(L, ids_y, "W_U") for L in LAYERS}
U["random direction, blank"] = {L: g1_unit_rows(300 + L) for L in LAYERS}
sigma = {L: float(np.mean([np.abs(H[L][i, slot_idx[r.q]].astype(np.float32) @ U["own name"][L][i, :, 0]).mean() for i, r in enumerate(S.itertuples())])) for L in LAYERS}
energy = {}                                                                         # share of the centred slot tokens' energy each removed part holds
for name in REMOVALS:
    e = []
    for L in LAYERS:
        Qn = np.asarray(U[name][L])
        for i, r in enumerate(S.itertuples()):
            Xs = H[L][i, slot_idx[r.q]].astype(np.float32) - MU[L]
            e.append(float(((Xs @ (Qn[i] if Qn.ndim == 3 else Qn)) ** 2).sum() / (Xs ** 2).sum()))
    energy[name] = float(np.mean(e))
print("sigma_L (mean size of a slot token's own-name component): " + ", ".join(f"L{L} {sigma[L]:.1f}" for L in LAYERS[::max(1, len(LAYERS) // 4)]))
print("energy share removed: " + ", ".join(f"{k} {v:.3f}" for k, v in energy.items()) + f"  [{time.time() - t0:.0f}s]")

# --- the arms
res = []
TX = {task: [g1_truth(task, r.X) for r in S.itertuples()] for task in G1_TASKS}
TY = {task: [g1_truth(task, r.Y) for r in S.itertuples()] for task in G1_TASKS}
TZ = {task: [g1_truth(task, r.Z) for r in S.itertuples()] for task in G1_TASKS}


def g1_run(block, arm, task, images, hooks, follow, alpha=0.0):
    """one arm of one question; 'correct' scores against X's truth, 'follow' against the injected word's truth (Y, Z) when given"""
    tx = [e_chat_media([G1_INTRO, "IMAGE", G1_TASKS[task].format(q=r.q)]) for r in S.itertuples()]
    gen = mm_generate(tx, images, 5, hooks=hooks)
    ok, fo = g1_score(task, gen, TX[task]), (g1_score(task, gen, follow) if follow is not None else [None] * len(S))
    for i, r in enumerate(S.itertuples()):
        res.append(dict(block=block, arm=arm, task=task, scene=r.scene, X=r.X, Y=r.Y, Z=r.Z, q=r.q, alpha=alpha, truth=TX[task][i], generation=gen[i], correct=ok[i], follow=fo[i]))
    return float(np.mean(ok)), (float(np.mean(fo)) if follow is not None else float("nan"))


ablate = {name: g1_hooks(U[name], slot_pos, "ablate", mu=MU) for name in REMOVALS}
for arm in ["clean"] + REMOVALS:
    acc = {task: g1_run("own picture", arm, task, own_img, None if arm == "clean" else ablate[arm], TY[task]) for task in G1_TASKS}
    print(f"own picture   {arm:36s} " + "  ".join(f"{t} {acc[t][0]:.2f}" for t in G1_TASKS) + f"  [{time.time() - t0:.0f}s]")
for arm, hooks in [("clean", None), ("own name", ablate["own name"])]:
    acc = {task: g1_run("printed word", arm, task, prt_img, hooks, None) for task in G1_TASKS}
    print(f"printed word  {arm:36s} " + "  ".join(f"{t} {acc[t][0]:.2f}" for t in G1_TASKS) + f"  [{time.time() - t0:.0f}s]")

def g1_first_ids(task, truth):
    return E_SID[truth] if task == "say" else single_ids([truth, " " + truth])


def g1_logp(task, images, hooks):
    """log p of the correct first token (summed over its single-token forms) at the decision position, per scene"""
    tx = [e_chat_media([G1_INTRO, "IMAGE", G1_TASKS[task].format(q=r.q)]) for r in S.itertuples()]
    lg, _ = mm_forward(tx, images, hooks=hooks)
    lp = torch.log_softmax(lg.float(), -1)
    return np.array([float(torch.logsumexp(lp[i, g1_first_ids(task, TX[task][i])], 0)) for i in range(len(S))])


LP = {}
for block, images in (("own picture", own_img), ("printed word", prt_img)):
    for arm in (["clean"] + REMOVALS if block == "own picture" else ["clean", "own name"]):
        for task in G1_TASKS:
            LP[(block, arm, task)] = g1_logp(task, images, None if arm == "clean" else ablate[arm])
    print(f"{block}: log p captured for {len(LP)} arm-task pairs  [{time.time() - t0:.0f}s]")
pd.DataFrame([dict(block=b, arm=a, task=t, scene=int(S.scene.iloc[i]), logp=float(v[i])) for (b, a, t), v in LP.items() for i in range(len(S))]).to_csv(f"{RUN_DIR}/g1_logp.csv", index=False)
SITES = {"image tokens": slot_pos, "all image tokens": all_pos, "decision position": dec_pos}
INJ = [("other name", TY), ("random direction", TY), ("same-kind name", TZ), ("other name, W_U only", TY)]
for site, pos_of in SITES.items():
    for arm, follow in INJ:
        for a in G1_ALPHAS:
            hooks = g1_hooks(U[arm], pos_of, "add", scale={L: a * sigma[L] for L in LAYERS})
            acc = {task: g1_run(f"inject at the {site}", arm, task, own_img, hooks, follow[task], alpha=a) for task in G1_TASKS}
            print(f"{site:17s} x{a:.0f} {arm:24s} follow " + "  ".join(f"{t} {acc[t][1]:.2f}" for t in G1_TASKS) + " | keep " + "  ".join(f"{t} {acc[t][0]:.2f}" for t in G1_TASKS) + f"  [{time.time() - t0:.0f}s]")
blank_imgs = [blank] * len(S)
for arm, a in [("none", 0.0)] + [("own name", a) for a in G1_ALPHAS] + [("random direction, blank", 8.0)]:
    hooks = None if a == 0 else g1_hooks(U[arm], slot_pos, "add", scale={L: a * sigma[L] for L in LAYERS})
    acc = {task: g1_run("blank canvas", arm, task, blank_imgs, hooks, None, alpha=a) for task in G1_TASKS}
    print(f"blank X x{a:.0f}   {arm:36s} " + "  ".join(f"{t} {acc[t][0]:.2f}" for t in G1_TASKS) + f"  [{time.time() - t0:.0f}s]")
G1 = pd.DataFrame(res)
G1.to_csv(f"{RUN_DIR}/g1_handle.csv", index=False)

# --- the rules (RECORD.md 03:55 / 04:00)
HELD = S.scene[~S.calibration].values if G1_N > G1_CAL else S.scene.values


def g1_vals(block, arm, task, alpha=0.0, col="correct", scenes=None):
    d = G1[(G1.block == block) & (G1.arm == arm) & (G1.task == task) & (G1.alpha == alpha)]
    v = d.set_index("scene")[col].astype(float).reindex(S.scene).fillna(0).values
    return v if scenes is None else v[np.isin(S.scene.values, scenes)]


def g1_boot(v, scenes=None):
    return b_boot_scene(v, S.scene.values if scenes is None else S.scene.values[np.isin(S.scene.values, scenes)])


resG1 = dict(n=int(len(S)), n_held=int(len(HELD)), sigma=sigma, energy=energy, accuracy={}, follow={}, D={}, verdict={})
for (block, arm, a), _ in G1.groupby(["block", "arm", "alpha"]):
    key = f"{block} | {arm} | {a:g}"
    resG1["accuracy"][key] = {task: g1_boot(g1_vals(block, arm, task, a)) for task in G1_TASKS}
    if block.startswith("inject") or block == "own picture":
        resG1["follow"][key] = {task: g1_boot(g1_vals(block, arm, task, a, "follow")) for task in G1_TASKS}
clean = {task: resG1["accuracy"]["own picture | clean | 0"][task][0] for task in G1_TASKS}
valid = {task: clean[task] >= (0.9 if task == "say" else 0.85) for task in G1_TASKS}
rules_ok = [t for t in G1_RULES if valid[t]]
D = {task: g1_boot(g1_vals("own picture", "other name", task) - g1_vals("own picture", "own name", task)) for task in G1_TASKS}
resG1["D"] = D


def r1(task):
    m, lo, hi = D[task]
    return "necessary" if m >= 0.25 and lo > 0 else "not needed" if m <= 0.10 and lo > -0.15 and hi < 0.15 else "undetermined"


resG1["verdict"]["R0"] = {task: bool(valid[task]) for task in G1_TASKS}
resG1["verdict"]["R1"] = {task: r1(task) for task in G1_TASKS}
resG1["dlogp"] = {f"{b} | {a}": {t: g1_boot(LP[(b, a, t)] - LP[(b, "clean", t)]) for t in G1_TASKS} for (b, a, _) in LP if a != "clean"}
DN = {task: g1_boot((LP[("own picture", "own name", task)] - LP[("own picture", "clean", task)]) - (LP[("own picture", "other name", task)] - LP[("own picture", "clean", task)])) for task in G1_TASKS}
resG1["D_nats"] = DN


def r1n(task):
    m, lo, hi = DN[task]
    return "necessary" if m <= -0.5 and hi < 0 else "not needed" if m >= -0.2 and lo > -0.3 and hi < 0.3 else "undetermined"


resG1["verdict"]["R1_nats"] = {task: r1n(task) for task in G1_TASKS}
gn = g1_boot((LP[("own picture", "own name", "say")] - LP[("own picture", "other name", "say")]) - (np.mean([LP[("own picture", "own name", t)] - LP[("own picture", "other name", t)] for t in rules_ok], 0) if rules_ok else 0))
resG1["G_nats"] = gn
r2n = ("positive control failed" if not valid["say"] or not rules_ok else "use runs through the handle" if all(r1n(t) == "necessary" for t in ["say"] + rules_ok) and gn[1] > -0.3 and gn[2] < 0.3
       else "the handle is for saying only" if r1n("say") == "necessary" and all(r1n(t) == "not needed" for t in rules_ok) else "saying needs it more" if gn[0] <= -0.3 and gn[2] < 0 else "undetermined at this n")
resG1["verdict"]["R2_nats"] = r2n
g_vals = g1_vals("own picture", "other name", "say") - g1_vals("own picture", "own name", "say") - (np.mean([g1_vals("own picture", "other name", t) - g1_vals("own picture", "own name", t) for t in rules_ok], 0) if rules_ok else 0)
G = g1_boot(g_vals)
resG1["G"] = G
if not valid["say"] or not rules_ok:
    r2 = "positive control failed"
elif all(r1(t) == "necessary" for t in ["say"] + rules_ok) and G[1] > -0.15 and G[2] < 0.15:
    r2 = "use runs through the handle"
elif r1("say") == "necessary" and all(r1(t) == "not needed" for t in rules_ok):
    r2 = "the handle is for saying only"
elif G[0] >= 0.15 and G[1] > 0:
    r2 = "saying needs it more"
else:
    r2 = "undetermined at this n"
resG1["verdict"]["R2"] = r2
ctrl = {arm: max(abs(clean[t] - resG1["accuracy"][f"own picture | {arm} | 0"][t][0]) for t in G1_TASKS) for arm in ["random direction", "own name, W_U only", "other name"]}
own_drop = {t: clean[t] - resG1["accuracy"]["own picture | own name | 0"][t][0] for t in G1_TASKS}
pc_drop = {t: clean[t] - resG1["accuracy"]["own picture | top principal direction of the slot | 0"][t][0] for t in G1_TASKS}
resG1["verdict"]["R3"] = dict(controls_within_0_10=bool(max(ctrl.values()) <= 0.10), control_max_shift=ctrl, energy_explains=bool(any(pc_drop[t] >= own_drop[t] - 0.10 and own_drop[t] >= 0.25 for t in G1_TASKS)))

# R4-R6: the two injection sites. alpha* per site = the smallest alpha at which say follows Y on >= 0.5 of the calibration scenes; verdicts on the held-out scenes
CAL = S.scene[S.calibration].values if G1_N > G1_CAL else S.scene.values
site_out = {}
for site in SITES:
    blk = f"inject at the {site}"
    a_star = next((a for a in G1_ALPHAS if g1_vals(blk, "other name", "say", a, "follow", CAL).mean() >= 0.5), None)
    o = dict(alpha_star=a_star)
    if a_star is not None:
        rulesY = np.mean([g1_vals(blk, "other name", t, a_star, "follow", HELD) for t in G1_RULES], 0)
        rulesR = np.mean([g1_vals(blk, "random direction", t, a_star, "follow", HELD) for t in G1_RULES], 0)
        o["say_follows_Y"] = g1_boot(g1_vals(blk, "other name", "say", a_star, "follow", HELD), HELD)
        o["rules_follow_Y"] = g1_boot(rulesY, HELD)
        o["rules_follow_random"] = g1_boot(rulesR, HELD)
        o["rules_Y_minus_random"] = g1_boot(rulesY - rulesR, HELD)
        o["Z_say_follows_Z"] = g1_boot(g1_vals(blk, "same-kind name", "say", a_star, "follow", HELD), HELD)
        o["Z_rules_flip"] = g1_boot(1 - np.mean([g1_vals(blk, "same-kind name", t, a_star, "correct", HELD) for t in G1_RULES], 0), HELD)
        o["WU_say_follows_Y"] = g1_boot(g1_vals(blk, "other name, W_U only", "say", a_star, "follow", HELD), HELD)
        o["random_say_keeps_X"] = g1_boot(g1_vals(blk, "random direction", "say", a_star, "correct", HELD), HELD)
        o["letter_follows_Y"] = g1_boot(g1_vals(blk, "other name", "letter", a_star, "follow", HELD), HELD)
        clean_keep = g1_vals("own picture", "clean", "say", 0, "correct", HELD).mean()
        o["collateral_ok"] = bool(o["random_say_keeps_X"][0] >= clean_keep - 0.10)
        o["J_specific"] = bool(o["WU_say_follows_Y"][0] <= 0.10)
        d = o["rules_Y_minus_random"]
        o["verdict"] = ("uninterpretable (the norm alone breaks the picture)" if not o["collateral_ok"] else "seen" if d[0] >= 0.30 and d[1] > 0 and o["Z_rules_flip"][0] <= 0.10
                        else "only said" if o["rules_follow_Y"][0] <= o["rules_follow_random"][0] + 0.10 else "undetermined")
        o["rules_follow_Y_vals"] = rulesY.tolist()
    else:
        o["verdict"] = "naming never follows the name at any dose"
    site_out[site] = o
resG1["dissociation"], r6 = {}, {}
for isite in ["image tokens", "all image tokens"]:
    if site_out[isite].get("alpha_star") is not None and site_out["decision position"].get("alpha_star") is not None:
        dis = np.asarray(site_out[isite]["rules_follow_Y_vals"]) - np.asarray(site_out["decision position"]["rules_follow_Y_vals"])
        resG1["dissociation"][isite] = g1_boot(dis, HELD)
        r6[isite] = "the same direction, written on the picture, is seen; written at the answer, it is only said" if resG1["dissociation"][isite][0] >= 0.30 and resG1["dissociation"][isite][1] > 0 else "no site dissociation at this n"
r6_txt = "; ".join(f"{s_}: {t_} ({b_fmt(resG1['dissociation'][s_])})" for s_, t_ in r6.items()) or "not testable (no alpha* at one site)"
resG1["verdict"]["R6"] = r6_txt
for s in SITES:
    site_out[s].pop("rules_follow_Y_vals", None)
resG1["sites"] = site_out
resG1["verdict"]["R4"] = {s: site_out[s]["verdict"] for s in SITES}
resG1["verdict"]["R5"] = {task: next((a for a in G1_ALPHAS if resG1["accuracy"][f"blank canvas | own name | {a:g}"][task][0] >= 0.5 and resG1["accuracy"]["blank canvas | random direction, blank | 8"][task][0] <= 0.1), None) for task in G1_TASKS}
print("D (other name removed minus own name removed): " + "  ".join(f"{t} {b_fmt(D[t])} {r1(t)}" for t in G1_TASKS))
print("D in nats (own name minus other name, log p of the correct answer): " + "  ".join(f"{t} {b_fmt(DN[t])} {r1n(t)}" for t in G1_TASKS) + f" | G {b_fmt(gn)} -> R2 (nats): {r2n}")
print("Δ log p under each removal (own picture): " + " | ".join(f"{a}: " + " ".join(f"{t} {resG1['dlogp'][f'own picture | {a}'][t][0]:+.2f}" for t in G1_TASKS) for a in REMOVALS))
print(f"G (D_say minus the rules' D) {b_fmt(G)} -> R2: {r2}; R3 {resG1['verdict']['R3']}")
for s in SITES:
    o = site_out[s]
    print(f"{s}: alpha* {o['alpha_star']} -> {o['verdict']}" + ("" if o["alpha_star"] is None else f"; say follows Y {b_fmt(o['say_follows_Y'])}, rules follow Y {b_fmt(o['rules_follow_Y'])} vs random {o['rules_follow_random'][0]:.2f}, Z flips rules {o['Z_rules_flip'][0]:.2f}, W_U says Y {o['WU_say_follows_Y'][0]:.2f}, letter follows Y {o['letter_follows_Y'][0]:.2f}"))
print(f"R6: {r6_txt}; R5 blank first alpha {resG1['verdict']['R5']}")
resG1["seconds"] = time.time() - t0
save(resG1, "eG1_results")

# --- the figure: removal (left, full height), the dose of the other name at both sites (top right), what follows at alpha* (bottom right)
TCOL = {"say": OKABE[0], "letter": OKABE[5], "animal": OKABE[1], "ride": OKABE[3]}
TMK = {"say": "o", "letter": "^", "animal": "s", "ride": "D"}
fig = plt.figure(figsize=(15, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1], wspace=0.5, hspace=0.75, left=0.21, right=0.98, bottom=0.14, top=0.8)
axA, axB, axC = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
labels = REMOVALS + ["printed word: own name"]
keys = [f"own picture | {a}" for a in REMOVALS] + ["printed word | own name"]
yy = np.arange(len(labels))[::-1].astype(float)
yy[-1] -= 0.8
for k, task in enumerate(G1_TASKS):
    off = (1.5 - k) * 0.19
    pts = [resG1["dlogp"][key][task] for key in keys]
    axA.errorbar([p[0] for p in pts], yy + off, xerr=[[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]], fmt=TMK[task], ms=4.5, color=TCOL[task], ecolor=TCOL[task], elinewidth=0.7, capsize=0, label=task, lw=0)
axA.axvline(0, color="0.75", lw=0.7, ls=":")
axA.set(yticks=yy, yticklabels=["removed: " + l for l in labels], xlabel=f"change in log p of the correct answer, nats (95% CI, {len(S)} scenes)")
axA.tick_params(axis="y", labelsize=8)
axA.legend(loc="lower left", fontsize=8, frameon=False, ncol=2, title="question about the slot", title_fontsize=8)
axA.set_title("one part removed from the slot's 49 image tokens (mean-ablated)", fontsize=9.5, loc="left")
xs = [0.0] + G1_ALPHAS
for site, ls in (("image tokens", "-"), ("all image tokens", "-."), ("decision position", "--")):
    blk = f"inject at the {site}"
    for task in ["say", "animal"]:
        ys_ = [resG1["follow"][f"own picture | clean | 0"][task]] + [resG1["follow"][f"{blk} | other name | {a:g}"][task] for a in G1_ALPHAS]
        axB.plot(xs, [y[0] for y in ys_], ls=ls, marker=TMK[task], ms=4, color=TCOL[task], label=f"{task}, {site}")
    yr = [resG1["follow"][f"own picture | clean | 0"]["say"][0]] + [resG1["follow"][f"{blk} | random direction | {a:g}"]["say"][0] for a in G1_ALPHAS]
    axB.plot(xs, yr, ls=ls, color="0.6", lw=0.8)
axB.set(xticks=xs, xlabel="dose along the other name's direction (units of a token's own-name component)", ylim=(-0.02, 1.02), ylabel="share following the injected name")
axB.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
axB.set_title("the other name injected on the slot (solid), on every image token (dash-dot) or at the answer (dashed); grey, random", fontsize=8.5, loc="left")
bars, cols_, labs = [], [], []
for site in SITES:
    o = site_out[site]
    if o["alpha_star"] is None:
        continue
    for key, lab, c in (("rules_follow_Y", f"other name, {site}", TCOL["animal"]), ("rules_follow_random", f"random, {site}", OKABE[4]), ("Z_rules_flip", f"same-kind name, {site}", OKABE[2])):
        bars.append(o[key]); labs.append(lab + f" (α* {o['alpha_star']:g})"); cols_.append(c)
if bars:
    yb = np.arange(len(bars))[::-1]
    axC.barh(yb, [b[0] for b in bars], 0.6, color=cols_)
    axC.errorbar([b[0] for b in bars], yb, xerr=[[b[0] - b[1] for b in bars], [b[2] - b[0] for b in bars]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
    axC.set(yticks=yb, yticklabels=labs, xlim=(0, 1.02), xlabel=f"share of rule answers that flip to the injected name's kind ({len(HELD)} held-out scenes)")
    axC.tick_params(axis="y", labelsize=8)
axC.set_title("does the rule follow the injected name at the matched dose?", fontsize=9.5, loc="left")
inj_txt = "; ".join(f"{s_} {site_out[s_]['verdict']}" for s_ in SITES)
dl = resG1["dlogp"]["own picture | own name"]
fig.suptitle(f"G1 ({MODEL_TAG}). The own name direction removed from the slot: log p of the answer falls by {-dl['say'][0]:.2f} nats (say), {-dl['letter'][0]:.2f} (letter), {-dl['animal'][0]:.2f} (animal), {-dl['ride'][0]:.2f} (ride); another name's direction {-resG1['dlogp']['own picture | other name']['say'][0]:.2f} (say). R2 {r2n}. Injection: {inj_txt}. {r6_txt}.", fontsize=9.5)
ws_finish(fig, "figG1_handle", f"G1 · {len(S)} scenes · mean-ablation and injection at the read layers")


# %% ---------------------------------------------------------------- F3: F1 with a competing receiver (the Run 2 fix). The receiver's slot holds its own object Z of the other kind; the donor's part must displace it
# Donor: X in slot q under the neutral prompt. Receiver: Z (the other animal and ride classes, another initial) in the same slot q, and one
# of four questions the donor never saw: say, letter, animal, ride. Transplants into the receiver's image tokens at the read layers: the
# full state (positive control), the lens part (256), J++ where fitted, three independent random parts, the complement of the lens part;
# at every layer (plus the layer-0 input): full, lens part, random part; a non-target donor (X in another slot); the fact typed as text.
# Scored against X's answer (carried) and Z's (kept). Rules and predictions: RECORD.md 04:10.
F3_N = int(os.environ.get("F3_N", E_N_SCENES))
F3_ANIMALS = {"dog", "cat", "bird", "horse"}
F3_RIDE = {"horse", "car", "bus", "train", "bicycle", "airplane", "boat"}
F3_INTRO = "Study the scene. It has an image grid with numbered slots."
F3_TASKS = {"say": "What is the object in image slot {q}? Reply with the noun only.",
            "letter": "What is the first letter of the name of the object in image slot {q}? Reply with the letter only.",
            "animal": "Rule: if the object in image slot {q} is an animal, answer 7; otherwise answer 3. Reply with the number only.",
            "ride": "Rule: if the object in image slot {q} is something people ride or drive, answer 7; otherwise answer 3. Reply with the number only."}
F3_RULES = ["animal", "ride"]
F3_SEEDS = [100, 400, 500]
t0 = time.time()


def f3_truth(task, x):
    if task == "say":
        return x
    if task == "letter":
        return x[0].upper()
    return "7" if x in (F3_ANIMALS if task == "animal" else F3_RIDE) else "3"


def f3_score(task, gens, truths):
    if task == "say":
        return [bool(v) for v in e_gen_correct(gens, truths, set(E_WORDS))]
    return [((g.strip().strip("*").upper()[:1] == t) if g.strip() else False) for g, t in zip(gens, truths)]


# --- reads
rng_ = np.random.default_rng(13)
rows = []
for i in range(F3_N):
    pool = [w for w in E_WORDS if (w in F3_ANIMALS) == (i % 2 == 0)]
    x = pool[int(rng_.integers(len(pool)))]
    zs = [w for w in E_WORDS if (w in F3_ANIMALS) != (x in F3_ANIMALS) and (w in F3_RIDE) != (x in F3_RIDE) and w[0] != x[0]]
    assert zs, x
    z = zs[int(rng_.integers(len(zs)))]
    q = int(rng_.integers(1, E2_N_SLOTS + 1))
    q2 = int(rng_.choice([s for s in range(1, E2_N_SLOTS + 1) if s != q]))
    rows.append(dict(scene=i, X=x, Z=z, q=q, q2=q2, tx=E_BANK[x][i % len(E_BANK[x])], tz=E_BANK[z][(i + 1) % len(E_BANK[z])]))
S = pd.DataFrame(rows)
empty = {s: None for s in range(1, E2_N_SLOTS + 1)}
don_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q: r.tx}) for r in S.itertuples()]
don2_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q2: r.tx}) for r in S.itertuples()]
rec_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q: r.tz}) for r in S.itertuples()]
don_txt = [e_chat_media([F3_INTRO, "IMAGE"])] * len(S)
N_IMG = E2_COLS * E2_ROWS * 49
ids0 = mm_encode(don_txt[:1], don_img[:1])["input_ids"][0].tolist()
vs_off = ids0.index(MM_VS) + 1
img_pos = list(range(vs_off, vs_off + N_IMG))
LF = [-1] + list(range(FINAL + 1))
_lo = lens if hasattr(lens, "jacobians") else JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION)
for L in LF:
    if 0 <= L < FINAL and L not in JAC:
        JAC[L] = _lo.jacobians[L].to(DEV)
d_model = hf_model.config.get_text_config().hidden_size
Qs = {L: b_lens_subspace(L, 256) for L in LAYERS}
QsA = {L: (b_lens_subspace(L, 256) if 0 <= L < FINAL else Qs[LAYERS[0] if L < 0 else LAYERS[-1]]) for L in LF}
QrA = {L: b_random_subspace(d_model, 256, 100 + L) for L in LF}
Qr3 = {seed: {L: b_random_subspace(d_model, 256, seed + L) for L in LAYERS} for seed in F3_SEEDS}
Qpp = {L: e_jpp_subspace(L, 256) for L in LAYERS} if JPP is not None else None


def f3_capture(images):
    REC = {L: np.zeros((len(S), N_IMG, d_model), np.float16) for L in LF}
    for j in range(len(S)):
        _, R = mm_forward(don_txt[j:j + 1], images[j:j + 1], LF, where=lambda ids, _t=img_pos: list(_t))
        for L in LF:
            REC[L][j] = R[L][0].half().numpy()
    return REC


REC, REC2 = f3_capture(don_img), f3_capture(don2_img)
print(f"donor states captured for {len(S)} scenes ({len(LF)} layers)  [{time.time() - t0:.0f}s]")
mm_contact_sheet([(f"donor: {r.X} in slot {r.q}", don_img[i]) for i, r in enumerate(S.head(2).itertuples())] + [(f"receiver: {r.Z} in slot {r.q}", rec_img[i]) for i, r in enumerate(S.head(2).itertuples())], 4, "figF3_stimuli")
res = []
TX = {task: [f3_truth(task, r.X) for r in S.itertuples()] for task in F3_TASKS}
TZ = {task: [f3_truth(task, r.Z) for r in S.itertuples()] for task in F3_TASKS}
for task, tmpl in F3_TASKS.items():
    rec_txt = [e_chat_media([F3_INTRO, "IMAGE", tmpl.format(q=r.q)]) for r in S.itertuples()]
    typed = [chat([("user", F3_INTRO + f" Image slot {r.q} contains a {r.X}. " + tmpl.format(q=r.q))]) for r in S.itertuples()]
    n_all = [len(mm_encode(rec_txt[j:j + 1], [rec_img[j]])["input_ids"][0]) for j in range(len(S))]
    assert mm_encode(rec_txt[:1], rec_img[:1])["input_ids"][0].tolist().index(MM_VS) + 1 == vs_off, "the image does not sit at the donor's positions"
    pos = [(lambda s, _n=n_all[j]: np.array([s - _n + t for t in img_pos])) for j in range(len(S))]
    arms = [("receiver alone", None), ("full state, read layers", e2p_pos_hooks(LAYERS, REC, pos)), ("lens part, read layers", e2p_pos_hooks(LAYERS, REC, pos, Qs, "lens"))]
    if Qpp is not None:
        arms.append(("J++ part, read layers", e2p_pos_hooks(LAYERS, REC, pos, Qpp, "lens")))
    arms += [(f"random part, read layers, draw {k + 1}", e2p_pos_hooks(LAYERS, REC, pos, Qr3[seed], "lens")) for k, seed in enumerate(F3_SEEDS)]
    arms += [("complement of the lens part, read layers", e2p_pos_hooks(LAYERS, REC, pos, Qs, "complement")), ("full state, every layer", e2p_pos_hooks(LF, REC, pos)),
             ("lens part, every layer", e2p_pos_hooks(LF, REC, pos, QsA, "lens")), ("random part, every layer", e2p_pos_hooks(LF, REC, pos, QrA, "lens")),
             ("non-target donor, full state, every layer", e2p_pos_hooks(LF, REC2, pos)), ("the fact typed as text", "typed")]
    for name, hooks in arms:
        tx, im, hk = (typed, [None] * len(S), None) if hooks == "typed" else (rec_txt, rec_img, hooks)
        gen = mm_generate(tx, im, 5, hk)
        okX, okZ = f3_score(task, gen, TX[task]), f3_score(task, gen, TZ[task])
        for i, r in enumerate(S.itertuples()):
            res.append(dict(task=task, arm=name, scene=r.scene, X=r.X, Z=r.Z, q=r.q, generation=gen[i], carried=okX[i], kept=okZ[i]))
        print(f"{task:7s} {name:44s} carried {np.mean(okX):.2f}  kept {np.mean(okZ):.2f}  [{time.time() - t0:.0f}s]")
F3 = pd.DataFrame(res)
F3.to_csv(f"{RUN_DIR}/f3_competing_receiver.csv", index=False)


def f3_vals(arm, task, col="carried"):
    return F3[(F3.arm == arm) & (F3.task == task)].set_index("scene")[col].astype(float).reindex(S.scene).fillna(0).values


resF3 = dict(n=int(len(S)), carried={}, kept={}, J_minus_random={}, verdict={})
ARMS = list(F3.arm.unique())
for task in F3_TASKS:
    resF3["carried"][task] = {arm: b_boot_scene(f3_vals(arm, task), S.scene.values) for arm in ARMS}
    resF3["kept"][task] = {arm: b_boot_scene(f3_vals(arm, task, "kept"), S.scene.values) for arm in ARMS}
    rnd = np.mean([f3_vals(f"random part, read layers, draw {k + 1}", task) for k in range(len(F3_SEEDS))], 0)
    jr = b_boot_scene(f3_vals("lens part, read layers", task) - rnd, S.scene.values)
    resF3["J_minus_random"][task] = jr
    ctrl_ok = resF3["carried"][task]["full state, read layers"][0] >= 0.8 and resF3["carried"][task]["receiver alone"][0] <= 0.2
    resF3["verdict"][task] = ("positive control failed" if not ctrl_ok else "carried by the lens part" if jr[0] >= 0.3 and jr[1] > 0 else "not beyond a random part" if jr[0] <= 0.1 and jr[1] > -0.15 and jr[2] < 0.15 else "undetermined at this n")
    print(f"{task}: " + "  ".join(f"{a} {v[0]:.2f}" for a, v in resF3["carried"][task].items()) + f" | lens − random {b_fmt(jr)} -> {resF3['verdict'][task]}")
v = resF3["verdict"]
rules_v = [v[t] for t in F3_RULES]
head = ("positive control failed" if any(x == "positive control failed" for x in v.values()) else "the lens part carries the name and the fact" if v["say"] == "carried by the lens part" and all(x == "carried by the lens part" for x in rules_v)
        else "the lens part carries the name, not the fact" if v["say"] == "carried by the lens part" and all(x == "not beyond a random part" for x in rules_v)
        else "no privileged interface in a 256-direction lens part" if all(x == "not beyond a random part" for x in [v["say"]] + rules_v) else "mixed or undetermined at this n")
resF3["headline"] = head
comp = {t: resF3["carried"][t]["complement of the lens part, read layers"][0] for t in F3_TASKS}
print(f"F3 headline: {head}; complement carries say {comp['say']:.2f}, animal {comp['animal']:.2f}, ride {comp['ride']:.2f}")
resF3["seconds"] = time.time() - t0
save(resF3, "eF3_results")
TCOL = {"say": OKABE[0], "letter": OKABE[5], "animal": OKABE[1], "ride": OKABE[3]}
TMK = {"say": "o", "letter": "^", "animal": "s", "ride": "D"}
fig, ax = plt.subplots(figsize=(9.5, 6.2))
fig.subplots_adjust(left=0.36, right=0.97, bottom=0.12, top=0.84)
yy = np.arange(len(ARMS))[::-1].astype(float)
for k, task in enumerate(F3_TASKS):
    off = (1.5 - k) * 0.19
    pts = [resF3["carried"][task][a] for a in ARMS]
    ax.errorbar([p[0] for p in pts], yy + off, xerr=[[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]], fmt=TMK[task], ms=4.5, color=TCOL[task], ecolor=TCOL[task], elinewidth=0.7, capsize=0, label=task, lw=0)
ax.set(yticks=yy, yticklabels=ARMS, xlim=(-0.02, 1.02), xlabel=f"share of answers that follow the donor's object rather than the receiver's own (95% CI, {len(S)} scenes)")
ax.tick_params(axis="y", labelsize=8)
ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=2, title="question the donor never saw", title_fontsize=8)
fig.suptitle(f"F3 ({MODEL_TAG}). A receiver that sees its own object gets a part of a donor that saw another. Lens − random at the read layers: say {b_fmt(resF3['J_minus_random']['say'])}, animal {b_fmt(resF3['J_minus_random']['animal'])}, ride {b_fmt(resF3['J_minus_random']['ride'])}. {head}.", fontsize=9.5)
ws_finish(fig, "figF3_competing_receiver", f"F3 · {len(S)} scenes · first-character and first-word scoring")


# %% ---------------------------------------------------------------- G1b (exploratory, after the G1 printed-word result): is the printed word's rule failure a literal reading of "object"?
# The same 64 scenes as G1, the printed rendering only. The animal and ride rules phrased for an object (as in G1) and for a word
# ("if the word in image slot q names an animal"), and a reading check ("What does the word in image slot q say?"). No rule frozen:
# this separates "the model cannot use a printed word" from "the model reads 'object' literally and a word is not an animal".
G1B_N = int(os.environ.get("G1B_N", E_N_SCENES))
G1B_ANIMALS = {"dog", "cat", "bird", "horse"}
G1B_RIDE = {"horse", "car", "bus", "train", "bicycle", "airplane", "boat"}
G1B_INTRO = "Study the scene. It has an image grid with numbered slots."
G1B_Q = {"animal, object wording": "Rule: if the object in image slot {q} is an animal, answer 7; otherwise answer 3. Reply with the number only.",
         "animal, word wording": "Rule: if the word in image slot {q} names an animal, answer 7; otherwise answer 3. Reply with the number only.",
         "ride, object wording": "Rule: if the object in image slot {q} is something people ride or drive, answer 7; otherwise answer 3. Reply with the number only.",
         "ride, word wording": "Rule: if the word in image slot {q} names something people ride or drive, answer 7; otherwise answer 3. Reply with the number only.",
         "read the word": "What does the word in image slot {q} say? Reply with the word only."}
t0 = time.time()
rng_ = np.random.default_rng(11)
rows = []
for i in range(G1B_N):
    pool = [w for w in E_WORDS if (w in G1B_ANIMALS) == (i % 2 == 0)]
    x = pool[int(rng_.integers(len(pool)))]
    ys = [w for w in E_WORDS if (w in G1B_ANIMALS) != (x in G1B_ANIMALS) and (w in G1B_RIDE) != (x in G1B_RIDE) and w[0] != x[0]]
    zs = [w for w in E_WORDS if w != x and (w in G1B_ANIMALS) == (x in G1B_ANIMALS) and (w in G1B_RIDE) == (x in G1B_RIDE) and w[0] != x[0]]
    zs = zs or [w for w in E_WORDS if w != x and (w in G1B_ANIMALS) == (x in G1B_ANIMALS) and w[0] != x[0]]
    rows.append(dict(scene=i, X=x, Y=ys[int(rng_.integers(len(ys)))], Z=zs[int(rng_.integers(len(zs)))], q=int(rng_.integers(1, E2_N_SLOTS + 1))))
S = pd.DataFrame(rows)
empty = {s: None for s in range(1, E2_N_SLOTS + 1)}
prt_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q: t1_word_tile(r.X)}) for r in S.itertuples()]
res = []
for key, tmpl in G1B_Q.items():
    tx = [e_chat_media([G1B_INTRO, "IMAGE", tmpl.format(q=r.q)]) for r in S.itertuples()]
    gen = mm_generate(tx, prt_img, 5)
    if key == "read the word":
        ok = [bool(v) for v in e_gen_correct(gen, list(S.X), set(E_WORDS))]
    else:
        truth = ["7" if (r.X in (G1B_ANIMALS if key.startswith("animal") else G1B_RIDE)) else "3" for r in S.itertuples()]
        ok = [(g.strip().strip("*")[:1] == t) for g, t in zip(gen, truth)]
    for i, r in enumerate(S.itertuples()):
        res.append(dict(question=key, scene=r.scene, X=r.X, q=r.q, generation=gen[i], correct=ok[i]))
    print(f"printed word  {key:24s} accuracy {np.mean(ok):.2f}  [{time.time() - t0:.0f}s]")
G1B = pd.DataFrame(res)
G1B.to_csv(f"{RUN_DIR}/g1b_printed_wording.csv", index=False)
resG1B = {key: b_boot_scene(G1B[G1B.question == key].correct.astype(float).values, G1B[G1B.question == key].scene.values) for key in G1B_Q}
d = b_boot_scene(G1B[G1B.question == "animal, word wording"].correct.astype(float).values - G1B[G1B.question == "animal, object wording"].correct.astype(float).values, S.scene.values)
resG1B["word_minus_object_wording_animal"] = d
print("word wording minus object wording (animal rule): " + b_fmt(d))
save(resG1B, "eG1b_results")


# %% ---------------------------------------------------------------- F4: the name without the fact, confirmed on fresh scenes (S1 of the 14 Oct night plan)
# F3's design on crops never used before (E_CROP_SET=1: the next 12 crops per class of the same shuffle, ids saved), with a placebo donor
# (the donor saw the empty canvas), a joint question that asks for the name and the animal digit in one answer (both orders), the rank
# r = 0.1 d, and the positive control for "the fact is movable" read at every layer. Primary estimand D(arm) = P(names X and the animal
# answer is Z's), scene-paired. Rules and predictions: RECORD.md 04:45. Optional exploratory rank sweep (F4_RANKS="16,64,256,1024").
F4_N = int(os.environ.get("F4_N", E_N_SCENES))
F4_RANKS = [int(v) for v in os.environ.get("F4_RANKS", "").split(",") if v.strip()]
assert int(os.environ.get("E_CROP_SET", 0)) == 1 or os.environ.get("F4_ALLOW_OLD_CROPS") == "1", "F4 runs on the fresh crop set: launch with E_CROP_SET=1"
F4_ANIMALS = {"dog", "cat", "bird", "horse"}
F4_RIDE = {"horse", "car", "bus", "train", "bicycle", "airplane", "boat"}
F4_INTRO = "Study the scene. It has an image grid with numbered slots."
F4_Q = {"say": "What is the object in image slot {q}? Reply with the noun only.",
        "letter": "What is the first letter of the name of the object in image slot {q}? Reply with the letter only.",
        "animal": "Rule: if the object in image slot {q} is an animal, answer 7; otherwise answer 3. Reply with the number only.",
        "ride": "Rule: if the object in image slot {q} is something people ride or drive, answer 7; otherwise answer 3. Reply with the number only.",
        "name then animal": "What is the object in image slot {q}, and is it an animal? Reply with the noun, then 7 if it is an animal or 3 if it is not.",
        "animal then name": "Is the object in image slot {q} an animal, and what is it? Reply with 7 if it is an animal or 3 if it is not, then the noun."}
F4_SEEDS = [100, 400, 500]
t0 = time.time()


def f4_truth(task, x):
    if task == "say":
        return x
    if task == "letter":
        return x[0].upper()
    return "7" if x in (F4_ANIMALS if task == "animal" else F4_RIDE) else "3"


def f4_parse(task, g):
    """(noun or None, digit or None) from a generation; single questions fill one field"""
    import re
    toks = re.findall(r"[a-z]+", g.lower())
    noun = next((w if w in E_WORDS else w.rstrip("s") for w in toks if w in E_WORDS or w.rstrip("s") in E_WORDS), None)
    m = re.search(r"[73]", g)
    digit = m.group(0) if m else None
    if task == "say":
        return noun, None
    if task == "letter":
        return (g.strip().strip("*").upper()[:1] or None), None
    if task in ("animal", "ride"):
        return None, (g.strip().strip("*")[:1] if g.strip() else None)
    return noun, digit


# --- reads
rng_ = np.random.default_rng(17)
rows = []
for i in range(F4_N):
    pool = [w for w in E_WORDS if (w in F4_ANIMALS) == (i % 2 == 0)]
    x = pool[int(rng_.integers(len(pool)))]
    zs = [w for w in E_WORDS if (w in F4_ANIMALS) != (x in F4_ANIMALS) and (w in F4_RIDE) != (x in F4_RIDE) and w[0] != x[0]]
    assert zs, x
    z = zs[int(rng_.integers(len(zs)))]
    q = int(rng_.integers(1, E2_N_SLOTS + 1))
    q2 = int(rng_.choice([s for s in range(1, E2_N_SLOTS + 1) if s != q]))
    jx, jz = i % len(E_BANK[x]), (i + 1) % len(E_BANK[z])
    rows.append(dict(scene=i, X=x, Z=z, q=q, q2=q2, tx=E_BANK[x][jx], tz=E_BANK[z][jz], crop_x=E_BANK_META.get(x, [None] * 99)[jx], crop_z=E_BANK_META.get(z, [None] * 99)[jz]))
S = pd.DataFrame(rows)
empty = {s: None for s in range(1, E2_N_SLOTS + 1)}
don_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q: r.tx}) for r in S.itertuples()]
don2_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q2: r.tx}) for r in S.itertuples()]
rec_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q: r.tz}) for r in S.itertuples()]
blank_img = [e_canvas(E2_COLS, E2_ROWS, empty)] * len(S)
don_txt = [e_chat_media([F4_INTRO, "IMAGE"])] * len(S)
B_PAD = int(getattr(hf_model.config, "image_token_id", B_PAD))
ids0 = mm_encode(don_txt[:1], don_img[:1])["input_ids"][0].tolist()
img_pos = [j for j, t in enumerate(ids0) if t == B_PAD]                            # every image token, whatever the family's grid (Gemma 4 has no 7 x 7 slots)
N_IMG = len(img_pos)
vs_off = img_pos[0]
LF = [-1] + list(range(FINAL + 1))
_lo = lens if hasattr(lens, "jacobians") else JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION)
for L in LF:
    if 0 <= L < FINAL and L not in JAC:
        JAC[L] = _lo.jacobians[L].to(DEV)
d_model = hf_model.config.get_text_config().hidden_size
R = int(0.1 * d_model)
Qs = {L: b_lens_subspace(L, R) for L in LAYERS}
QsA = {L: (b_lens_subspace(L, R) if 0 <= L < FINAL else Qs[LAYERS[0] if L < 0 else LAYERS[-1]]) for L in LF}
QrA = {L: b_random_subspace(d_model, R, 100 + L) for L in LF}
Qr3 = {seed: {L: b_random_subspace(d_model, R, seed + L) for L in LAYERS} for seed in F4_SEEDS}
Qpp = {L: e_jpp_subspace(L, R) for L in LAYERS} if JPP is not None else None
print(f"rank r = {R} (0.1 d); crops from set {os.environ.get('E_CROP_SET', 0)}; {len(S)} scenes  [{time.time() - t0:.0f}s]")


def f4_capture(images):
    REC = {L: np.zeros((len(S), N_IMG, d_model), np.float16) for L in LF}
    for j in range(len(S)):
        _, Rr = mm_forward(don_txt[j:j + 1], images[j:j + 1], LF, where=lambda ids, _t=img_pos: list(_t))
        for L in LF:
            REC[L][j] = Rr[L][0].half().numpy()
    return REC


REC, REC2, RECB = f4_capture(don_img), f4_capture(don2_img), f4_capture(blank_img)
print(f"donor states captured ({len(LF)} layers)  [{time.time() - t0:.0f}s]")
mm_contact_sheet([(f"donor: {r.X} in slot {r.q}", don_img[i]) for i, r in enumerate(S.head(2).itertuples())] + [(f"receiver: {r.Z} in slot {r.q}", rec_img[i]) for i, r in enumerate(S.head(2).itertuples())], 4, "figF4_stimuli")
res = []
for task, tmpl in F4_Q.items():
    rec_txt = [e_chat_media([F4_INTRO, "IMAGE", tmpl.format(q=r.q)]) for r in S.itertuples()]
    typed = [chat([("user", F4_INTRO + f" Image slot {r.q} contains a {r.X}. " + tmpl.format(q=r.q))]) for r in S.itertuples()]
    n_all = [len(mm_encode(rec_txt[j:j + 1], [rec_img[j]])["input_ids"][0]) for j in range(len(S))]
    assert [j for j, t in enumerate(mm_encode(rec_txt[:1], rec_img[:1])["input_ids"][0].tolist()) if t == B_PAD][0] == vs_off, "the image does not sit at the donor's positions"
    pos = [(lambda s, _n=n_all[j]: np.array([s - _n + t for t in img_pos])) for j in range(len(S))]
    arms = [("receiver alone", None), ("full state, read layers", e2p_pos_hooks(LAYERS, REC, pos)), ("lens part, read layers", e2p_pos_hooks(LAYERS, REC, pos, Qs, "lens"))]
    if Qpp is not None:
        arms.append(("J++ part, read layers", e2p_pos_hooks(LAYERS, REC, pos, Qpp, "lens")))
    arms += [(f"random part, read layers, draw {k + 1}", e2p_pos_hooks(LAYERS, REC, pos, Qr3[seed], "lens")) for k, seed in enumerate(F4_SEEDS)]
    arms += [("complement of the lens part, read layers", e2p_pos_hooks(LAYERS, REC, pos, Qs, "complement")), ("placebo donor (empty canvas), read layers", e2p_pos_hooks(LAYERS, RECB, pos)),
             ("full state, every layer", e2p_pos_hooks(LF, REC, pos)), ("lens part, every layer", e2p_pos_hooks(LF, REC, pos, QsA, "lens")), ("random part, every layer", e2p_pos_hooks(LF, REC, pos, QrA, "lens")),
             ("non-target donor, full state, every layer", e2p_pos_hooks(LF, REC2, pos)), ("the fact typed as text", "typed")]
    if F4_RANKS and task in ("say", "animal"):
        for rk in F4_RANKS:
            if rk == R:
                continue
            arms += [(f"lens part r={rk}, read layers", e2p_pos_hooks(LAYERS, REC, pos, {L: b_lens_subspace(L, rk) for L in LAYERS}, "lens")),
                     (f"random part r={rk}, read layers", e2p_pos_hooks(LAYERS, REC, pos, {L: b_random_subspace(d_model, rk, 700 + L) for L in LAYERS}, "lens"))]
    for name, hooks in arms:
        tx, im, hk = (typed, [None] * len(S), None) if hooks == "typed" else (rec_txt, rec_img, hooks)
        gen = mm_generate(tx, im, 8, hk)
        for i, r in enumerate(S.itertuples()):
            noun, digit = f4_parse(task, gen[i])
            tX, tZ = f4_truth(task, r.X), f4_truth(task, r.Z)
            if task in ("say",):
                carried, kept = noun == r.X, noun == r.Z
            elif task == "letter":
                carried, kept = noun == tX, noun == tZ
            elif task in ("animal", "ride"):
                carried, kept = digit == tX, digit == tZ
            else:
                carried, kept = (noun == r.X and digit == f4_truth("animal", r.X)), (noun == r.Z and digit == f4_truth("animal", r.Z))
            res.append(dict(task=task, arm=name, scene=r.scene, X=r.X, Z=r.Z, q=r.q, generation=gen[i], noun=noun, digit=digit, carried=bool(carried), kept=bool(kept),
                            name_X=(noun == r.X), digit_Z=(digit == f4_truth("animal", r.Z)) if task.startswith(("name", "animal then")) else None))
        d_ = [x for x in res if x["task"] == task and x["arm"] == name]
        print(f"{task:16s} {name:44s} carried {np.mean([x['carried'] for x in d_]):.2f}  kept {np.mean([x['kept'] for x in d_]):.2f}" + (f"  name X & digit Z {np.mean([bool(x['name_X'] and x['digit_Z']) for x in d_]):.2f}" if task.startswith(("name", "animal then")) else "") + f"  [{time.time() - t0:.0f}s]")
F4 = pd.DataFrame(res)
F4.to_csv(f"{RUN_DIR}/f4_name_without_fact.csv", index=False)


def f4_vals(arm, task, col="carried"):
    return F4[(F4.arm == arm) & (F4.task == task)].set_index("scene")[col].astype(float).reindex(S.scene).fillna(0).values


ARMS = [a for a in F4.arm.unique() if "r=" not in a]
boot = lambda v: b_boot_scene(v, S.scene.values)
resF4 = dict(n=int(len(S)), rank=R, crop_set=int(os.environ.get("E_CROP_SET", 0)), crops=S[["scene", "X", "Z", "q", "crop_x", "crop_z"]].to_dict("records"), carried={}, kept={}, D={}, D_ride={}, joint={}, verdict={})
for task in F4_Q:
    resF4["carried"][task] = {arm: boot(f4_vals(arm, task)) for arm in ARMS}
    resF4["kept"][task] = {arm: boot(f4_vals(arm, task, "kept")) for arm in ARMS}
for arm in ARMS:
    resF4["D"][arm] = boot(f4_vals(arm, "say") * f4_vals(arm, "animal", "kept"))
    resF4["D_ride"][arm] = boot(f4_vals(arm, "say") * f4_vals(arm, "ride", "kept"))
    resF4["joint"][arm] = {task: boot((F4[(F4.arm == arm) & (F4.task == task)].set_index("scene").name_X.astype(bool) & F4[(F4.arm == arm) & (F4.task == task)].set_index("scene").digit_Z.astype(bool)).astype(float).reindex(S.scene).fillna(0).values) for task in ("name then animal", "animal then name")}
rnd_say = np.mean([f4_vals(f"random part, read layers, draw {k + 1}", "say") for k in range(3)], 0)
rnd_D = np.mean([f4_vals(f"random part, read layers, draw {k + 1}", "say") * f4_vals(f"random part, read layers, draw {k + 1}", "animal", "kept") for k in range(3)], 0)
resF4["D"]["random part, read layers (mean of 3)"] = boot(rnd_D)
gate = {t: resF4["kept"][t]["receiver alone"][0] >= 0.85 for t in ("animal", "ride")}
pc = dict(full_read_names_X=resF4["carried"]["say"]["full state, read layers"][0] >= 0.8, full_every_flips_animal=resF4["carried"]["animal"]["full state, every layer"][0] >= 0.8,
          placebo_inert=(resF4["carried"]["say"]["placebo donor (empty canvas), read layers"][0] <= 0.1 and resF4["kept"]["animal"]["placebo donor (empty canvas), read layers"][0] >= 0.85))
dl = boot(f4_vals("lens part, read layers", "say") * f4_vals("lens part, read layers", "animal", "kept") - f4_vals("full state, read layers", "say") * f4_vals("full state, read layers", "animal", "kept"))
ls = boot(f4_vals("lens part, read layers", "say") - rnd_say)
resF4["lens_minus_full_D"], resF4["lens_minus_random_say"], resF4["gate"], resF4["positive_controls"] = dl, ls, gate, pc
if not (gate["animal"] and pc["full_read_names_X"] and pc["full_every_flips_animal"] and pc["placebo_inert"]):
    head = "positive control or gate failed: " + ", ".join(k for k, v in {**gate, **pc}.items() if not v)
elif dl[0] >= 0.4 and dl[1] > 0 and ls[0] >= 0.3 and ls[1] > 0:
    head = "name without fact: the lens part installs the donor's name and leaves the receiver's fact in place"
else:
    head = "no dissociation at this n"
resF4["verdict"]["headline"] = head
Dd = resF4["D"]
print("D = P(names X and the animal answer is Z's): " + "  ".join(f"{a}: {v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" for a, v in Dd.items() if "draw" not in a))
print("joint question, name X and digit Z in one answer: " + "  ".join(f"{a[:28]} {resF4['joint'][a]['name then animal'][0]:.2f} / {resF4['joint'][a]['animal then name'][0]:.2f}" for a in ("full state, read layers", "lens part, read layers", "complement of the lens part, read layers", "random part, read layers, draw 1")))
print(f"gate {gate}; positive controls {pc}; lens − full on D {b_fmt(dl)}; lens − random on say {b_fmt(ls)} -> {head}")
if F4_RANKS:
    resF4["rank_sweep"] = {arm: {t: boot(f4_vals(arm, t)) for t in ("say", "animal")} for arm in F4.arm.unique() if "r=" in arm}
    print("rank sweep (say-carry / animal-carry): " + "  ".join(f"{a}: {v['say'][0]:.2f} / {v['animal'][0]:.2f}" for a, v in resF4["rank_sweep"].items()))
resF4["seconds"] = time.time() - t0
save(resF4, "eF4_results")
TCOL = {"say": OKABE[0], "animal": OKABE[1]}
fig, ax = plt.subplots(figsize=(9.5, 5.6))
fig.subplots_adjust(left=0.4, right=0.97, bottom=0.14, top=0.84)
show = [a for a in ARMS if "draw 2" not in a and "draw 3" not in a]
y = np.arange(len(show))[::-1]
v = [Dd[a] for a in show]
ax.barh(y, [t[0] for t in v], 0.6, color=OKABE[2])
ax.errorbar([t[0] for t in v], y, xerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
ax.set(yticks=y, yticklabels=show, xlim=(0, 1.02), xlabel=f"share of scenes where the receiver names the donor's object and answers the animal rule for its own (95% CI, {len(S)} scenes)")
ax.tick_params(axis="y", labelsize=8)
fig.suptitle(f"F4 ({MODEL_TAG}, fresh crops, r = {R}). {head}. D: lens {Dd['lens part, read layers'][0]:.2f}, complement {Dd['complement of the lens part, read layers'][0]:.2f}, full {Dd['full state, read layers'][0]:.2f}, random {Dd['random part, read layers (mean of 3)'][0]:.2f}.", fontsize=9.5)
ws_finish(fig, "figF4_name_without_fact", f"F4 · {len(S)} scenes · first-word and first-character scoring")


# %% ---------------------------------------------------------------- D1: the depth law. In which layers can a name still be said, and in which still used? (plus C1: the word "animal")
# F4's fresh scenes. Two sliding-window interventions on every image token, one window of 4 consecutive layers at a time, from the input
# (layer -1) to the last layer: (a) the donor's full state moved into the receiver in that window only; (b) Y's J-lens direction of each
# layer (nearest fitted J where none) added at 4 sigma_L. Questions: say and the animal rule. Read-outs per model: L_say / L_use = the last
# window start at which say / the animal rule follows on >= 0.5 / >= random + 0.3. C1: the word "animal"'s direction at the read layers
# (dose 4 and 8) against random: does the animal rule read verbal category content? Rules: RECORD.md 04:55.
D1_N = int(os.environ.get("D1_N", E_N_SCENES))
D1_W = 4
D1_ANIMALS = {"dog", "cat", "bird", "horse"}
D1_RIDE = {"horse", "car", "bus", "train", "bicycle", "airplane", "boat"}
D1_INTRO = "Study the scene. It has an image grid with numbered slots."
D1_Q = {"say": "What is the object in image slot {q}? Reply with the noun only.", "animal": "Rule: if the object in image slot {q} is an animal, answer 7; otherwise answer 3. Reply with the number only."}
if os.environ.get("D1_SPECTRUM") == "1":                                        # the horizon spectrum (amendment 05:30): a lexical, a symbolic and a functional property beside say and animal
    D1_Q.update({"letter": "What is the first letter of the name of the object in image slot {q}? Reply with the letter only.",
                 "set": "Rule: if the object in image slot {q} is a {a}, a {b} or a {c}, answer 7; otherwise answer 3. Reply with the number only.",
                 "ride": "Rule: if the object in image slot {q} is something people ride or drive, answer 7; otherwise answer 3. Reply with the number only."})
t0 = time.time()
rng_ = np.random.default_rng(17)
rows = []
for i in range(D1_N):
    pool = [w for w in E_WORDS if (w in D1_ANIMALS) == (i % 2 == 0)]
    x = pool[int(rng_.integers(len(pool)))]
    zs = [w for w in E_WORDS if (w in D1_ANIMALS) != (x in D1_ANIMALS) and (w in D1_RIDE) != (x in D1_RIDE) and w[0] != x[0]]
    z = zs[int(rng_.integers(len(zs)))]
    q = int(rng_.integers(1, E2_N_SLOTS + 1)); q2 = int(rng_.choice([s for s in range(1, E2_N_SLOTS + 1) if s != q]))
    others = [w for w in E_WORDS if w not in (x, z)]
    fill = [others[j] for j in rng_.choice(len(others), 2, replace=False)]
    sset = [x if i % 4 < 2 else z] + fill
    rng_.shuffle(sset)
    rows.append(dict(scene=i, X=x, Z=z, q=q, set=sset, tx=E_BANK[x][i % len(E_BANK[x])], tz=E_BANK[z][(i + 1) % len(E_BANK[z])]))
S = pd.DataFrame(rows)
empty = {s: None for s in range(1, E2_N_SLOTS + 1)}
don_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q: r.tx}) for r in S.itertuples()]
prt_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q: t1_word_tile(r.X)}) for r in S.itertuples()]      # the same fact as a printed word (the second data type)
rec_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q: r.tz}) for r in S.itertuples()]
don_txt = [e_chat_media([D1_INTRO, "IMAGE"])] * len(S)
don_typed = [chat([("user", D1_INTRO + f" Image slot {r.q} contains a {r.X}.")]) for r in S.itertuples()]
pos_typed = [e_word_positions(t, r.X, t.index("contains a "))[:1] for t, r in zip(don_typed, S.itertuples())]
B_PAD = int(getattr(hf_model.config, "image_token_id", B_PAD))
ids0 = mm_encode(don_txt[:1], don_img[:1])["input_ids"][0].tolist()
img_pos = [j for j, t in enumerate(ids0) if t == B_PAD]; N_IMG = len(img_pos); vs_off = img_pos[0]
LF = [-1] + list(range(FINAL + 1))
_lo = lens if hasattr(lens, "jacobians") else JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION)
for L in LF:
    if 0 <= L < FINAL and L not in JAC:
        JAC[L] = _lo.jacobians[L].to(DEV)
d_model = hf_model.config.get_text_config().hidden_size
REC = {L: np.zeros((len(S), N_IMG, d_model), np.float16) for L in LF}
RECZ = {L: np.zeros((len(S), N_IMG, d_model), np.float16) for L in LF}
RECP = {L: np.zeros((len(S), N_IMG, d_model), np.float16) for L in LF}
RECT = {L: np.zeros((len(S), 1, d_model), np.float16) for L in LF}
for j in range(len(S)):
    _, Rt = mm_forward(don_typed[j:j + 1], None, LF, where=lambda ids, _t=pos_typed[j]: list(_t))
    for L in LF:
        RECT[L][j] = Rt[L][0].half().numpy()
    _, Rr = mm_forward(don_txt[j:j + 1], don_img[j:j + 1], LF, where=lambda ids, _t=img_pos: list(_t))
    _, Rz = mm_forward(don_txt[j:j + 1], rec_img[j:j + 1], LF, where=lambda ids, _t=img_pos: list(_t))
    _, Rp = mm_forward(don_txt[j:j + 1], prt_img[j:j + 1], LF, where=lambda ids, _t=img_pos: list(_t))
    for L in LF:
        REC[L][j] = Rr[L][0].half().numpy(); RECZ[L][j] = Rz[L][0].half().numpy(); RECP[L][j] = Rp[L][0].half().numpy()
print(f"states captured ({len(LF)} layers)  [{time.time() - t0:.0f}s]")
WINDOWS = [list(range(a, min(a + D1_W, FINAL + 1))) for a in range(-1, FINAL + 1, D1_W)]
ids_x = [E_SID[w][0] for w in S.X]; ids_y = [E_SID[w][0] for w in S.Z]
ids_animal = e_single("animal")[0]
nearest = lambda L: L if (0 <= L < FINAL and L in JAC) else (0 if L < 0 else FINAL - 1)
UY = {L: g1_word_directions(nearest(L), ids_y, "J") for L in LF}
UX = {L: g1_word_directions(nearest(L), ids_x, "J") for L in LF}
UA = {L: g1_word_directions(nearest(L), [ids_animal[0]] * len(S), "J") for L in LF}
UR = {L: g1_unit_rows(900 + L) for L in LF}
slot_idx = {s: e_slot_tokens(E2_COLS, E2_ROWS, s) for s in range(1, E2_N_SLOTS + 1)} if N_IMG == E2_COLS * E2_ROWS * 49 else None
sigma = {L: float(np.mean([np.abs(RECZ[L][i][slot_idx[r.q] if slot_idx else slice(None)].astype(np.float32) @ UX[L][i, :, 0]).mean() for i, r in enumerate(S.itertuples())])) for L in LF}
all_pos = lambda row, i: [j for j, t in enumerate(row) if t == B_PAD]


def d1_hooks_add(U, layers, scale):
    """add scale[L] * U[L][i] at every image token in the given layers (any layers, not only the read layers)"""
    def hooks(enc, idx):
        ids, am = enc["input_ids"], enc["attention_mask"]; b, s = ids.shape
        rows_pos = [[s - int(am[i].sum()) + p for p in all_pos(ids[i, s - int(am[i].sum()):].tolist(), int(idx[i]))] for i in range(b)]
        P = max(len(r) for r in rows_pos); Pm = torch.as_tensor([r + [r[-1]] * (P - len(r)) for r in rows_pos], device=DEV); rows = torch.arange(b, device=DEV)[:, None].expand(-1, P)
        out_ = []
        for L in layers:
            qd = torch.as_tensor(np.asarray(U[L])[np.asarray(idx), :, 0], device=DEV, dtype=torch.float32) * float(scale[L])
            def fn(mod, inp, out, qd=qd):
                h = residual_of(out)
                if h.shape[1] == 1: return out
                x = h.float(); x[rows, Pm] = x[rows, Pm] + qd[:, None, :]; x = x.to(h.dtype)
                return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
            out_.append((L, fn))
        return out_
    return hooks


res = []
d1_truth = lambda t, r, w: (w if t == "say" else w[0].upper() if t == "letter" else ("7" if w in r.set else "3") if t == "set" else ("7" if w in D1_RIDE else "3") if t == "ride" else ("7" if w in D1_ANIMALS else "3"))
TX = {t: [d1_truth(t, r, r.X) for r in S.itertuples()] for t in D1_Q}
TZ = {t: [d1_truth(t, r, r.Z) for r in S.itertuples()] for t in D1_Q}
for task, tmpl in D1_Q.items():
    rec_txt = [e_chat_media([D1_INTRO, "IMAGE", tmpl.format(q=r.q, a=r.set[0], b=r.set[1], c=r.set[2])]) for r in S.itertuples()]
    n_all = [len(mm_encode(rec_txt[j:j + 1], [rec_img[j]])["input_ids"][0]) for j in range(len(S))]
    pos = [(lambda s, _n=n_all[j]: np.array([s - _n + t for t in img_pos])) for j in range(len(S))]
    arms = [("clean", None, "none", 0)]
    for w in WINDOWS:
        arms += [(f"move the donor's state, layers {w[0]} to {w[-1]}", e2p_pos_hooks(w, REC, pos), "transplant", w[0]),
                 (f"move the printed-word donor's state, layers {w[0]} to {w[-1]}", e2p_pos_hooks(w, RECP, pos), "transplant printed", w[0]),
                 (f"inject Y, layers {w[0]} to {w[-1]}", d1_hooks_add(UY, w, {L: 4 * sigma[L] for L in w}), "inject", w[0]),
                 (f"inject random, layers {w[0]} to {w[-1]}", d1_hooks_add(UR, w, {L: 4 * sigma[L] for L in w}), "random", w[0])]
    arms += [(f"inject the word 'animal' x{a}, read layers", d1_hooks_add(UA, LAYERS, {L: a * sigma[L] for L in LAYERS}), "C1 animal", a) for a in (4, 8)]
    arms += [(f"inject random x{a}, read layers", d1_hooks_add(UR, LAYERS, {L: a * sigma[L] for L in LAYERS}), "C1 random", a) for a in (4, 8)]
    rec_typed = [chat([("user", D1_INTRO + f" Image slot {r.q} contains a {r.Z}. " + tmpl.format(q=r.q, a=r.set[0], b=r.set[1], c=r.set[2]))]) for r in S.itertuples()]
    n_typed = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in rec_typed]
    pos_t = [(lambda s, _n=n_typed[j], _p=pos_typed[j]: np.array([s - _n + t for t in _p])) for j in range(len(S))]
    arms.append(("typed: receiver alone", None, "typed none", 0))
    arms += [(f"move the typed donor's state, layers {w[0]} to {w[-1]}", e2p_pos_hooks(w, RECT, pos_t), "transplant typed", w[0]) for w in WINDOWS]
    for name, hooks, kind, w0 in arms:
        typed_arm = kind.startswith("typed") or kind == "transplant typed"
        gen = mm_generate(rec_typed if typed_arm else rec_txt, [None] * len(S) if typed_arm else rec_img, 5, hooks)
        for i, r in enumerate(S.itertuples()):
            ok_x = (bool(e_gen_correct([gen[i]], [TX[task][i]], set(E_WORDS))[0]) if task == "say" else gen[i].strip().strip("*").upper()[:1] == TX[task][i])
            ok_z = (bool(e_gen_correct([gen[i]], [TZ[task][i]], set(E_WORDS))[0]) if task == "say" else gen[i].strip().strip("*").upper()[:1] == TZ[task][i])
            res.append(dict(task=task, arm=name, kind=kind, w0=w0, scene=r.scene, X=r.X, Z=r.Z, generation=gen[i], follows_X=ok_x, keeps_Z=ok_z))
        d_ = [x for x in res if x["task"] == task and x["arm"] == name]
        print(f"{task:6s} {name:46s} follows X {np.mean([x['follows_X'] for x in d_]):.2f}  keeps Z {np.mean([x['keeps_Z'] for x in d_]):.2f}  [{time.time() - t0:.0f}s]")
D1 = pd.DataFrame(res); D1.to_csv(f"{RUN_DIR}/d1_depth.csv", index=False)
boot = lambda v: b_boot_scene(v, S.scene.values)
val = lambda arm, task: D1[(D1.arm == arm) & (D1.task == task)].set_index("scene").follows_X.astype(float).reindex(S.scene).fillna(0).values
resD1 = dict(n=int(len(S)), n_layers=int(FINAL + 1), windows=[w[0] for w in WINDOWS], sigma=sigma, curves={}, C1={}, cut={})
for kind in ("transplant", "transplant printed", "transplant typed", "inject", "random"):
    resD1["curves"][kind] = {task: [boot(val(a, task)) for a, k in [(x["arm"], x["kind"]) for x in D1[["arm", "kind", "w0"]].drop_duplicates().sort_values("w0").to_dict("records")] if k == kind] for task in D1_Q}
resD1["horizon"] = {}                                                            # the last window start at which each property follows the donor on >= 0.5 (picture donor), per data type
for kind in ("transplant", "transplant printed", "transplant typed"):
    resD1["horizon"][kind] = {task: max([w for w, v in zip(resD1["windows"], resD1["curves"][kind][task]) if v[0] >= 0.5], default=None) for task in D1_Q}
print("horizons (last window start with >= 0.5): " + "; ".join(f"{k}: " + ", ".join(f"{t} {v}" for t, v in h.items()) for k, h in resD1["horizon"].items()))
for kind in ("transplant", "transplant printed", "transplant typed", "inject"):
    cs, ca = resD1["curves"][kind]["say"], resD1["curves"][kind]["animal"]
    rnd = resD1["curves"]["random"]["animal"]
    say_ok = [w for w, v in zip(resD1["windows"], cs) if v[0] >= 0.5]
    use_ok = [w for w, v, rv in zip(resD1["windows"], ca, rnd) if v[0] >= rv[0] + 0.3]
    resD1["cut"][kind] = dict(L_say=(max(say_ok) if say_ok else None), L_use=(max(use_ok) if use_ok else None), say_windows=say_ok, use_windows=use_ok,
                              verdict=("never" if say_ok and not use_ok else "too late" if say_ok and use_ok and max(use_ok) <= max(say_ok) - 2 * D1_W else "together" if say_ok and use_ok else "nothing moves"))
for a in (4, 8):
    fa = val(f"inject the word 'animal' x{a}, read layers", "animal"); fr = val(f"inject random x{a}, read layers", "animal")
    resD1["C1"][a] = dict(animal_follows=boot(fa), random=boot(fr), diff=boot(fa - fr), say_keeps_Z=boot(D1[(D1.arm == f"inject the word 'animal' x{a}, read layers") & (D1.task == "say")].set_index("scene").keeps_Z.astype(float).reindex(S.scene).fillna(0).values))
    resD1["C1"][a]["verdict"] = "the rule reads verbal category content" if resD1["C1"][a]["diff"][0] >= 0.3 and resD1["C1"][a]["diff"][1] > 0 else "the rule reads non-verbal features" if resD1["C1"][a]["diff"][0] <= 0.1 else "undetermined"
print("depth: " + "; ".join(f"{k}: L_say {v['L_say']} L_use {v['L_use']} -> {v['verdict']}" for k, v in resD1["cut"].items()))
print("C1: " + "; ".join(f"x{a}: animal rule follows 'animal' {v['animal_follows'][0]:.2f} vs random {v['random'][0]:.2f} -> {v['verdict']}" for a, v in resD1["C1"].items()))
resD1["seconds"] = time.time() - t0
save(resD1, "eD1_results")


# %% ---------------------------------------------------------------- K1: the load law. Does the object's name direction become necessary as the scene fills?
# G1's removal (the own name's lens direction, mean-ablated at the slot's tokens, read layers; other name and random as controls), on
# canvases holding k in {1, 2, 4, 6} objects: the target X in slot q and k - 1 distractors. Measured as log p of the correct first token
# for say and the animal rule. Rule (RECORD.md 05:05): the necessity on say grows with load when the own-name cost at k = 6 minus k = 1
# is <= -0.4 nats with the interval below 0. Prediction: -0.09 at k = 1 (G1), <= -0.5 at k = 6; the animal rule moves less.
K1_N = int(os.environ.get("K1_N", E_N_SCENES))
K1_KS = [1, 2, 4, 6]
K1_ANIMALS = {"dog", "cat", "bird", "horse"}
K1_INTRO = "Study the scene. It has an image grid with numbered slots."
K1_Q = {"say": "What is the object in image slot {q}? Reply with the noun only.", "animal": "Rule: if the object in image slot {q} is an animal, answer 7; otherwise answer 3. Reply with the number only."}
t0 = time.time()
B_PAD = int(getattr(hf_model.config, "image_token_id", B_PAD))
rng_ = np.random.default_rng(23)
rows = []
for i in range(K1_N):
    pool = [w for w in E_WORDS if (w in K1_ANIMALS) == (i % 2 == 0)]
    x = pool[int(rng_.integers(len(pool)))]
    others = [w for w in E_WORDS if w != x and w[0] != x[0]]
    ds = [others[j] for j in rng_.choice(len(others), 5, replace=False)]
    slots = [int(v) for v in rng_.permutation(np.arange(1, E2_N_SLOTS + 1))]
    rows.append(dict(scene=i, X=x, Y=ds[0], q=slots[0], distractors=ds, dslots=slots[1:], tile=E_BANK[x][i % len(E_BANK[x])]))
S = pd.DataFrame(rows)
d_model = hf_model.config.get_text_config().hidden_size
slot_idx = {s_: e_slot_tokens(E2_COLS, E2_ROWS, s_) for s_ in range(1, E2_N_SLOTS + 1)}
slot_pos = lambda row, i: [[j for j, t in enumerate(row) if t == B_PAD][k] for k in slot_idx[int(S.q.iloc[i])]]
ids_x = [E_SID[w][0] for w in S.X]; ids_y = [E_SID[w][0] for w in S.Y]
U = {"own name": {L: g1_word_directions(L, ids_x, "J") for L in LAYERS}, "other name": {L: g1_word_directions(L, ids_y, "J") for L in LAYERS}, "random direction": {L: g1_unit_rows(500 + L) for L in LAYERS}}
TX = {t: [(r.X if t == "say" else ("7" if r.X in K1_ANIMALS else "3")) for r in S.itertuples()] for t in K1_Q}
first_ids = lambda task, truth: E_SID[truth] if task == "say" else single_ids([truth, " " + truth])
res = {}
for k in K1_KS:
    imgs = []
    for r in S.itertuples():
        slots_ = {s_: None for s_ in range(1, E2_N_SLOTS + 1)}; slots_[r.q] = r.tile
        for d, s_ in zip(r.distractors[:k - 1], r.dslots[:k - 1]):
            slots_[s_] = E_BANK[d][r.scene % len(E_BANK[d])]
        imgs.append(e_canvas(E2_COLS, E2_ROWS, slots_))
    cap = [e_chat_media([K1_INTRO, "IMAGE", K1_Q["say"].format(q=r.q)]) for r in S.itertuples()]
    _, R = mm_forward(cap, imgs, LAYERS, where=lambda ids: [j for j, t in enumerate(ids) if t == B_PAD])
    MU = {L: R[L].float().reshape(-1, d_model).mean(0).numpy() for L in LAYERS}
    del R
    for task in K1_Q:
        tx = [e_chat_media([K1_INTRO, "IMAGE", K1_Q[task].format(q=r.q)]) for r in S.itertuples()]
        for arm in ["clean"] + list(U):
            hooks = None if arm == "clean" else g1_hooks(U[arm], slot_pos, "ablate", mu=MU)
            lg, _ = mm_forward(tx, imgs, hooks=hooks)
            lp = torch.log_softmax(lg.float(), -1)
            res[(k, task, arm)] = np.array([float(torch.logsumexp(lp[i, first_ids(task, TX[task][i])], 0)) for i in range(len(S))])
        print(f"k = {k} {task:6s} clean log p {res[(k, task, 'clean')].mean():.2f} | " + "  ".join(f"{a} {(res[(k, task, a)] - res[(k, task, 'clean')]).mean():+.2f}" for a in U) + f"  [{time.time() - t0:.0f}s]")
boot = lambda v: b_boot_scene(v, S.scene.values)
resK1 = dict(n=int(len(S)), loads=K1_KS, cost={f"k={k} | {t} | {a}": boot(res[(k, t, a)] - res[(k, t, "clean")]) for k in K1_KS for t in K1_Q for a in U}, clean_logp={f"k={k} | {t}": boot(res[(k, t, "clean")]) for k in K1_KS for t in K1_Q})
d = boot((res[(6, "say", "own name")] - res[(6, "say", "clean")]) - (res[(1, "say", "own name")] - res[(1, "say", "clean")]))
resK1["k6_minus_k1_say"] = d
resK1["verdict"] = "the name becomes necessary under load" if d[0] <= -0.4 and d[2] < 0 else "no load effect at this n" if d[0] > -0.2 and d[1] > -0.4 else "undetermined"
print("own name removed, cost in nats by load (say / animal): " + "  ".join(f"k={k}: {resK1['cost'][f'k={k} | say | own name'][0]:+.2f} / {resK1['cost'][f'k={k} | animal | own name'][0]:+.2f}" for k in K1_KS) + f" | k6 - k1 on say {b_fmt(d)} -> {resK1['verdict']}")
resK1["seconds"] = time.time() - t0
save(resK1, "eK1_results")


# %% ---------------------------------------------------------------- S1: does a computation follow the name when the name alone suffices for it?
# F4's fresh scenes (donor X, receiver Z of the other kind). Three questions revealed after encoding: say; the animal rule (defined over
# the referent); a set rule defined over names, "if the object in slot q is a {a}, a {b} or a {c}, answer 7; otherwise 3", with a random
# set of three names per scene that contains exactly one of X and Z. Interventions: the lens part of the donor's state at the read layers;
# the full state; a random part; and X's name direction injected on every image token of the receiver's picture at 4 sigma (G1's dose).
# Rule (RECORD.md 05:20): "a computation follows the name when the name suffices for it" when, for the lens part and for the injection,
# the set rule follows X on >= the animal rule + 0.3 with the scene-paired interval above 0. Predictions: set rule 0.6 (lens), 0.5
# (injection); animal 0.1 for both; the full state carries both (>= 0.8).
S1_N = int(os.environ.get("S1_N", E_N_SCENES))
S1_ANIMALS = {"dog", "cat", "bird", "horse"}
S1_RIDE = {"horse", "car", "bus", "train", "bicycle", "airplane", "boat"}
S1_INTRO = "Study the scene. It has an image grid with numbered slots."
S1_Q = {"say": "What is the object in image slot {q}? Reply with the noun only.",
        "animal": "Rule: if the object in image slot {q} is an animal, answer 7; otherwise answer 3. Reply with the number only.",
        "set": "Rule: if the object in image slot {q} is a {a}, a {b} or a {c}, answer 7; otherwise answer 3. Reply with the number only."}
t0 = time.time()
rng_ = np.random.default_rng(17)
rows = []
for i in range(S1_N):
    pool = [w for w in E_WORDS if (w in S1_ANIMALS) == (i % 2 == 0)]
    x = pool[int(rng_.integers(len(pool)))]
    zs = [w for w in E_WORDS if (w in S1_ANIMALS) != (x in S1_ANIMALS) and (w in S1_RIDE) != (x in S1_RIDE) and w[0] != x[0]]
    z = zs[int(rng_.integers(len(zs)))]
    q = int(rng_.integers(1, E2_N_SLOTS + 1)); q2 = int(rng_.choice([s for s in range(1, E2_N_SLOTS + 1) if s != q]))
    others = [w for w in E_WORDS if w not in (x, z)]
    fill = [others[j] for j in rng_.choice(len(others), 2, replace=False)]
    member = x if i % 4 < 2 else z
    sset = [member] + fill
    rng_.shuffle(sset)
    rows.append(dict(scene=i, X=x, Z=z, q=q, set=sset, tx=E_BANK[x][i % len(E_BANK[x])], tz=E_BANK[z][(i + 1) % len(E_BANK[z])]))
S = pd.DataFrame(rows)
empty = {s: None for s in range(1, E2_N_SLOTS + 1)}
don_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q: r.tx}) for r in S.itertuples()]
rec_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q: r.tz}) for r in S.itertuples()]
don_txt = [e_chat_media([S1_INTRO, "IMAGE"])] * len(S)
B_PAD = int(getattr(hf_model.config, "image_token_id", B_PAD))
ids0 = mm_encode(don_txt[:1], don_img[:1])["input_ids"][0].tolist()
img_pos = [j for j, t in enumerate(ids0) if t == B_PAD]; N_IMG = len(img_pos)
d_model = hf_model.config.get_text_config().hidden_size
R = int(0.1 * d_model)
Qs = {L: b_lens_subspace(L, R) for L in LAYERS}
Qr = {L: b_random_subspace(d_model, R, 100 + L) for L in LAYERS}
REC = {L: np.zeros((len(S), N_IMG, d_model), np.float16) for L in LAYERS}
RECZ = {L: np.zeros((len(S), N_IMG, d_model), np.float16) for L in LAYERS}
for j in range(len(S)):
    _, Rr = mm_forward(don_txt[j:j + 1], don_img[j:j + 1], LAYERS, where=lambda ids, _t=img_pos: list(_t))
    _, Rz = mm_forward(don_txt[j:j + 1], rec_img[j:j + 1], LAYERS, where=lambda ids, _t=img_pos: list(_t))
    for L in LAYERS:
        REC[L][j] = Rr[L][0].half().numpy(); RECZ[L][j] = Rz[L][0].half().numpy()
ids_x = [E_SID[w][0] for w in S.X]
UX = {L: g1_word_directions(L, ids_x, "J") for L in LAYERS}
UR = {L: g1_unit_rows(600 + L) for L in LAYERS}
slot_idx = {s_: e_slot_tokens(E2_COLS, E2_ROWS, s_) for s_ in range(1, E2_N_SLOTS + 1)} if N_IMG == E2_COLS * E2_ROWS * 49 else None
sigma = {L: float(np.mean([np.abs(RECZ[L][i][slot_idx[r.q] if slot_idx else slice(None)].astype(np.float32) @ UX[L][i, :, 0]).mean() for i, r in enumerate(S.itertuples())])) for L in LAYERS}
all_pos = lambda row, i: [j for j, t in enumerate(row) if t == B_PAD]
truth = lambda task, r, w: (w if task == "say" else ("7" if w in S1_ANIMALS else "3") if task == "animal" else ("7" if w in r.set else "3"))
res = []
for task, tmpl in S1_Q.items():
    rec_txt = [e_chat_media([S1_INTRO, "IMAGE", tmpl.format(q=r.q, a=r.set[0], b=r.set[1], c=r.set[2])]) for r in S.itertuples()]
    n_all = [len(mm_encode(rec_txt[j:j + 1], [rec_img[j]])["input_ids"][0]) for j in range(len(S))]
    pos = [(lambda s, _n=n_all[j]: np.array([s - _n + t for t in img_pos])) for j in range(len(S))]
    arms = [("receiver alone", None), ("full state, read layers", e2p_pos_hooks(LAYERS, REC, pos)), ("lens part, read layers", e2p_pos_hooks(LAYERS, REC, pos, Qs, "lens")),
            ("random part, read layers", e2p_pos_hooks(LAYERS, REC, pos, Qr, "lens")), ("X's name direction injected, every image token, dose 4", g1_hooks(UX, all_pos, "add", scale={L: 4 * sigma[L] for L in LAYERS})),
            ("a random direction injected, dose 4", g1_hooks(UR, all_pos, "add", scale={L: 4 * sigma[L] for L in LAYERS}))]
    for name, hooks in arms:
        gen = mm_generate(rec_txt, rec_img, 5, hooks)
        for i, r in enumerate(S.itertuples()):
            tX, tZ = truth(task, r, r.X), truth(task, r, r.Z)
            fx = bool(e_gen_correct([gen[i]], [tX], set(E_WORDS))[0]) if task == "say" else gen[i].strip().strip("*")[:1] == tX
            kz = bool(e_gen_correct([gen[i]], [tZ], set(E_WORDS))[0]) if task == "say" else gen[i].strip().strip("*")[:1] == tZ
            res.append(dict(task=task, arm=name, scene=r.scene, X=r.X, Z=r.Z, set=" ".join(r.set), generation=gen[i], follows_X=fx, keeps_Z=kz))
        d_ = [x for x in res if x["task"] == task and x["arm"] == name]
        print(f"{task:7s} {name:52s} follows X {np.mean([x['follows_X'] for x in d_]):.2f}  keeps Z {np.mean([x['keeps_Z'] for x in d_]):.2f}  [{time.time() - t0:.0f}s]")
S1 = pd.DataFrame(res); S1.to_csv(f"{RUN_DIR}/s1_name_rule.csv", index=False)
boot = lambda v: b_boot_scene(v, S.scene.values)
val = lambda arm, task: S1[(S1.arm == arm) & (S1.task == task)].set_index("scene").follows_X.astype(float).reindex(S.scene).fillna(0).values
ARMS = list(S1.arm.unique())
resS1 = dict(n=int(len(S)), rank=R, follows={arm: {t: boot(val(arm, t)) for t in S1_Q} for arm in ARMS}, verdict={})
for arm in ("lens part, read layers", "X's name direction injected, every image token, dose 4"):
    d = boot(val(arm, "set") - val(arm, "animal"))
    resS1["verdict"][arm] = dict(set_minus_animal=d, reading="a computation follows the name when the name suffices for it" if d[0] >= 0.3 and d[1] > 0 else "no name-rule advantage at this n" if d[0] <= 0.1 else "undetermined")
print("follows X (say / animal / set): " + "  ".join(f"{a}: {resS1['follows'][a]['say'][0]:.2f} / {resS1['follows'][a]['animal'][0]:.2f} / {resS1['follows'][a]['set'][0]:.2f}" for a in ARMS))
print("set minus animal: " + "; ".join(f"{a[:22]} {b_fmt(v['set_minus_animal'])} -> {v['reading']}" for a, v in resS1["verdict"].items()))
resS1["seconds"] = time.time() - t0
save(resS1, "eS1_results")


# %% ---------------------------------------------------------------- D2: the reverse by timing alone. Donor before the use horizon, the receiver's own state restored in the band the lens reads
# D1's fresh scenes and its two horizons per model (from runs/run4/<model>/eD1_results.json; fallback 0.33 and 0.70 of depth). Arms, on
# the image tokens: the donor's state in the early layers only (up to the use horizon); the same with the receiver's own state re-imposed
# in the band between the horizons (the restore arm); the donor's state in the band only; early plus band; every layer; and the same
# for a four-frame video donor (the canvas shown as four frames) when the processor takes it. Questions: say, animal. Rule (RECORD.md
# 05:45): "two horizons, in both directions" when the restore arm names the receiver's own object on >= 0.6 and applies the animal rule to
# the donor's on >= 0.6, while the band-only arm names the donor's on >= 0.6 and applies the rule to the receiver's own on >= 0.6.
D2_N = int(os.environ.get("D2_N", E_N_SCENES))
D2_ANIMALS = {"dog", "cat", "bird", "horse"}
D2_RIDE = {"horse", "car", "bus", "train", "bicycle", "airplane", "boat"}
D2_INTRO = "Study the scene. It has an image grid with numbered slots."
D2_Q = {"say": "What is the object in image slot {q}? Reply with the noun only.", "animal": "Rule: if the object in image slot {q} is an animal, answer 7; otherwise answer 3. Reply with the number only."}
D2_FRAMES = int(os.environ.get("D2_FRAMES", 4))
t0 = time.time()
rng_ = np.random.default_rng(17)
rows = []
for i in range(D2_N):
    pool = [w for w in E_WORDS if (w in D2_ANIMALS) == (i % 2 == 0)]
    x = pool[int(rng_.integers(len(pool)))]
    zs = [w for w in E_WORDS if (w in D2_ANIMALS) != (x in D2_ANIMALS) and (w in D2_RIDE) != (x in D2_RIDE) and w[0] != x[0]]
    z = zs[int(rng_.integers(len(zs)))]
    q = int(rng_.integers(1, E2_N_SLOTS + 1)); q2 = int(rng_.choice([s for s in range(1, E2_N_SLOTS + 1) if s != q]))
    rows.append(dict(scene=i, X=x, Z=z, q=q, tx=E_BANK[x][i % len(E_BANK[x])], tz=E_BANK[z][(i + 1) % len(E_BANK[z])]))
S = pd.DataFrame(rows)
empty = {s: None for s in range(1, E2_N_SLOTS + 1)}
don_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q: r.tx}) for r in S.itertuples()]
rec_img = [e_canvas(E2_COLS, E2_ROWS, {**empty, r.q: r.tz}) for r in S.itertuples()]
B_PAD = int(getattr(hf_model.config, "image_token_id", B_PAD))
d_model = hf_model.config.get_text_config().hidden_size
_h = f"{RUN_ROOT}/run4/{MODEL_TAG}/eD1_results.json" if "RUN_ROOT" in globals() else f"/workspace/nayan/runs/run4/{MODEL_TAG}/eD1_results.json"
try:
    _d1 = json.load(open(_h)); L_use, L_say = _d1["cut"]["transplant"]["L_use"], _d1["cut"]["transplant"]["L_say"]
    assert L_use is not None and L_say is not None
except Exception:
    L_use, L_say = int(0.33 * (FINAL + 1)) - 4, int(0.70 * (FINAL + 1)) - 4
EARLY = [L for L in range(-1, FINAL + 1) if L <= L_use + 3]
BAND = [L for L in range(-1, FINAL + 1) if L_use + 3 < L <= L_say + 3]
LATE = [L for L in range(-1, FINAL + 1) if L > L_say + 3]
print(f"horizons from D1: L_use {L_use}, L_say {L_say}; early {EARLY[0]}..{EARLY[-1]}, band {BAND[0]}..{BAND[-1]}, late {LATE[0] if LATE else None}..{FINAL}")
LF = [-1] + list(range(FINAL + 1))
_lo = lens if hasattr(lens, "jacobians") else JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION)
for L in LF:
    if 0 <= L < FINAL and L not in JAC:
        JAC[L] = _lo.jacobians[L].to(DEV)
SOURCES = {"picture": (don_img, rec_img, [e_chat_media([D2_INTRO, "IMAGE"])] * len(S), lambda task, r: e_chat_media([D2_INTRO, "IMAGE", D2_Q[task].format(q=r.q)]))}
if D2_FRAMES > 1:
    SOURCES["video"] = ([[im] * D2_FRAMES for im in don_img], [[im] * D2_FRAMES for im in rec_img], [e_chat_media([D2_INTRO] + ["IMAGE"] * D2_FRAMES)] * len(S),
                        lambda task, r: e_chat_media([D2_INTRO] + ["IMAGE"] * D2_FRAMES + [D2_Q[task].format(q=r.q)]))
res = []
for src, (dimg, rimg, dtxt, rprompt) in list(SOURCES.items()):
    try:
        ids0 = mm_encode(dtxt[:1], dimg[:1])["input_ids"][0].tolist()
    except Exception as e:
        print(f"{src}: the processor does not take this input ({type(e).__name__}: {str(e)[:120]}); skipped"); continue
    img_pos = [j for j, t in enumerate(ids0) if t == B_PAD]; N_IMG = len(img_pos)
    REC = {L: np.zeros((len(S), N_IMG, d_model), np.float16) for L in LF}
    RECZ = {L: np.zeros((len(S), N_IMG, d_model), np.float16) for L in LF}
    for j in range(len(S)):
        _, Rr = mm_forward(dtxt[j:j + 1], dimg[j:j + 1], LF, where=lambda ids, _t=img_pos: list(_t))
        _, Rz = mm_forward(dtxt[j:j + 1], rimg[j:j + 1], LF, where=lambda ids, _t=img_pos: list(_t))
        for L in LF:
            REC[L][j] = Rr[L][0].half().numpy(); RECZ[L][j] = Rz[L][0].half().numpy()
    print(f"{src}: {N_IMG} image tokens, states captured  [{time.time() - t0:.0f}s]")
    # the paradox: is the category decodable from the donor's state at every depth? leave-one-out nearest-centroid on the mean-pooled state
    slot_idx = {s_: e_slot_tokens(E2_COLS, E2_ROWS, s_) for s_ in range(1, E2_N_SLOTS + 1)} if N_IMG == E2_COLS * E2_ROWS * 49 else None
    lab = np.array([r.X in D2_ANIMALS for r in S.itertuples()])
    dec = {}
    for L in LF:
        F = np.stack([REC[L][i][slot_idx[r.q] if slot_idx else slice(None)].astype(np.float32).mean(0) for i, r in enumerate(S.itertuples())])
        F = (F - F.mean(0)) / (F.std(0) + 1e-6)
        ok = []
        for i in range(len(S)):
            m = np.ones(len(S), bool); m[i] = False
            c1, c0 = F[m & lab].mean(0), F[m & ~lab].mean(0)
            ok.append((np.linalg.norm(F[i] - c1) < np.linalg.norm(F[i] - c0)) == lab[i])
        dec[int(L)] = float(np.mean(ok))
    resD2_dec = dec
    print(f"{src}: category decodable from the donor's state (leave-one-out, by layer): " + " ".join(f"{L}:{v:.2f}" for L, v in dec.items()))
    # the mechanism: the same late state delivered early, and the early state delivered late
    E4, B4 = EARLY[-4:], BAND[:4]
    REC_late_early = {Lt: REC[Ls] for Lt, Ls in zip(E4, B4)}
    REC_early_late = {Lt: REC[Ls] for Lt, Ls in zip(B4, E4)}
    for task in D2_Q:
        rec_txt = [rprompt(task, r) for r in S.itertuples()]
        n_all = [len(mm_encode(rec_txt[j:j + 1], [rimg[j]])["input_ids"][0]) for j in range(len(S))]
        pos = [(lambda s, _n=n_all[j]: np.array([s - _n + t for t in img_pos])) for j in range(len(S))]
        both = lambda h1, h2: (lambda enc, idx: h1(enc, idx) + h2(enc, idx))
        arms = [("receiver alone", None), ("donor early only", e2p_pos_hooks(EARLY, REC, pos)),
                ("donor early, receiver restored in the band", both(e2p_pos_hooks(EARLY, REC, pos), e2p_pos_hooks(BAND, RECZ, pos))),
                ("donor in the band only", e2p_pos_hooks(BAND, REC, pos)), ("donor early and band", e2p_pos_hooks(EARLY + BAND, REC, pos)), ("donor every layer", e2p_pos_hooks(LF, REC, pos)),
                ("donor's band state delivered in the early window", e2p_pos_hooks(E4, REC_late_early, pos)), ("donor's early state delivered in the band", e2p_pos_hooks(B4, REC_early_late, pos)),
                ("donor's early-window state in place (control)", e2p_pos_hooks(E4, REC, pos))]
        for name, hooks in arms:
            gen = mm_generate(rec_txt, rimg, 5, hooks)
            for i, r in enumerate(S.itertuples()):
                tX = r.X if task == "say" else ("7" if r.X in D2_ANIMALS else "3"); tZ = r.Z if task == "say" else ("7" if r.Z in D2_ANIMALS else "3")
                fx = bool(e_gen_correct([gen[i]], [tX], set(E_WORDS))[0]) if task == "say" else gen[i].strip().strip("*")[:1] == tX
                kz = bool(e_gen_correct([gen[i]], [tZ], set(E_WORDS))[0]) if task == "say" else gen[i].strip().strip("*")[:1] == tZ
                res.append(dict(source=src, task=task, arm=name, scene=r.scene, X=r.X, Z=r.Z, generation=gen[i], follows_X=fx, keeps_Z=kz))
            d_ = [x for x in res if x["source"] == src and x["task"] == task and x["arm"] == name]
            print(f"{src:7s} {task:6s} {name:44s} follows X {np.mean([x['follows_X'] for x in d_]):.2f}  keeps Z {np.mean([x['keeps_Z'] for x in d_]):.2f}  [{time.time() - t0:.0f}s]")
    del REC, RECZ
D2 = pd.DataFrame(res); D2.to_csv(f"{RUN_DIR}/d2_reverse.csv", index=False)
boot = lambda v: b_boot_scene(v, S.scene.values)
resD2 = dict(n=int(len(S)), L_use=L_use, L_say=L_say, early=[EARLY[0], EARLY[-1]], band=[BAND[0], BAND[-1]], decodable_by_layer=resD2_dec, arms={}, verdict={})
for src in D2.source.unique():
    resD2["arms"][src] = {}
    for arm in D2[D2.source == src].arm.unique():
        g = lambda task, col: D2[(D2.source == src) & (D2.task == task) & (D2.arm == arm)].set_index("scene")[col].astype(float).reindex(S.scene).fillna(0).values
        resD2["arms"][src][arm] = dict(say_X=boot(g("say", "follows_X")), say_Z=boot(g("say", "keeps_Z")), animal_X=boot(g("animal", "follows_X")), animal_Z=boot(g("animal", "keeps_Z")),
                                       says_Z_and_classifies_X=boot(g("say", "keeps_Z") * g("animal", "follows_X")), says_X_and_classifies_Z=boot(g("say", "follows_X") * g("animal", "keeps_Z")))
    r_ = resD2["arms"][src]["donor early, receiver restored in the band"]; b_ = resD2["arms"][src]["donor in the band only"]
    resD2["verdict"][src] = ("two horizons, in both directions" if r_["say_Z"][0] >= 0.6 and r_["animal_X"][0] >= 0.6 and b_["say_X"][0] >= 0.6 and b_["animal_Z"][0] >= 0.6 else "one direction only" if b_["say_X"][0] >= 0.6 and b_["animal_Z"][0] >= 0.6 else "not shown")
    print(f"{src}: restore arm says Z {r_['say_Z'][0]:.2f}, classifies X {r_['animal_X'][0]:.2f} (both on {r_['says_Z_and_classifies_X'][0]:.2f} of scenes); band-only arm says X {b_['say_X'][0]:.2f}, classifies Z {b_['animal_Z'][0]:.2f} (both {b_['says_X_and_classifies_Z'][0]:.2f}) -> {resD2['verdict'][src]}")
    t_ = resD2["arms"][src].get("donor's band state delivered in the early window"); u_ = resD2["arms"][src].get("donor's early-window state in place (control)"); v_ = resD2["arms"][src].get("donor's early state delivered in the band")
    if t_ and u_:
        resD2["verdict"][f"{src} retimed"] = "the late state still holds the fact: delivered early it is used" if t_["animal_X"][0] >= 0.6 else "the late state has lost the fact" if t_["animal_X"][0] <= 0.2 else "undetermined"
        print(f"{src}: the band state delivered early classifies X {t_['animal_X'][0]:.2f} and says X {t_['say_X'][0]:.2f} (the early state in place: {u_['animal_X'][0]:.2f}, {u_['say_X'][0]:.2f}); the early state delivered in the band says X {v_['say_X'][0]:.2f}, classifies X {v_['animal_X'][0]:.2f} -> {resD2['verdict'][f'{src} retimed']}")
resD2["seconds"] = time.time() - t0
save(resD2, "eD2_results")


# %% ---------------------------------------------------------------- O1: where does the lens first read the object's name, against where the model stops reading the fact?
# One COCO crop on a one-object canvas, neutral prompt, 64 fresh crops. The image tokens' state is mean-pooled at every layer; the J-lens
# and the logit lens rank the object's name (best single-token form) against another object's name. Onset = the first layer at which
# the name is in the J-lens top 10 on >= 0.5 of scenes. Compared with the model's fact and name horizons from D1. Rules: RECORD.md 07:00.
O1_N = int(os.environ.get("O1_N", E_N_SCENES))
t0 = time.time()
rng_ = np.random.default_rng(29)
o1x = [E_WORDS[int(rng_.integers(len(E_WORDS)))] for _ in range(O1_N)]
o1o = [str(rng_.choice([w for w in E_WORDS if w != x])) for x in o1x]
o1img = [e_canvas(1, 1, {1: E_BANK[x][i % len(E_BANK[x])]}) for i, x in enumerate(o1x)]
o1txt = [e_chat_media(["Study the scene.", "IMAGE"])] * O1_N
B_PAD = int(getattr(hf_model.config, "image_token_id", B_PAD))
_lo = lens if hasattr(lens, "jacobians") else JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION)
LJ = list(range(FINAL))
for L in LJ:
    if L not in JAC:
        JAC[L] = _lo.jacobians[L].to(DEV)
pooled = {L: [] for L in LJ + [FINAL]}
for j in range(O1_N):
    _, R = mm_forward(o1txt[j:j + 1], o1img[j:j + 1], LJ + [FINAL], where=lambda ids: [k for k, t in enumerate(ids) if t == B_PAD])
    for L in pooled:
        pooled[L].append(R[L][0].float().mean(0, keepdim=True))
pooled = {L: torch.stack(v) for L, v in pooled.items()}                          # [n, 1, d]
tgt, nul = [E_SID[x] for x in o1x], [E_SID[o] for o in o1o]
rj, nj = mm_ranks(pooled, tgt, nul, jac=True, layers=LJ)
rl, nl = mm_ranks(pooled, tgt, nul, jac=False, layers=LJ + [FINAL])
top = lambda r: (np.asarray(r)[:, :, 0] <= 10).mean(0)
curve = dict(J=top(rj).tolist(), J_null=top(nj).tolist(), logit=top(rl).tolist(), logit_null=top(nl).tolist(), J_median=np.median(np.asarray(rj)[:, :, 0], 0).tolist())
onset = lambda c, Ls: next((L for L, v in zip(Ls, c) if v >= 0.5), None)
oJ, oL = onset(curve["J"], LJ), onset(curve["logit"], LJ + [FINAL])
hz = {}
try:
    hz = json.load(open(f"/workspace/nayan/runs/run4/{MODEL_TAG}/horizons.json"))
except Exception:
    pass
c_use = hz.get("transplant | animal", {}).get("c"); c_say = hz.get("transplant | say", {}).get("c")
n = FINAL + 1
resO1 = dict(n=O1_N, n_layers=n, layers=LJ, curve=curve, onset_J=oJ, onset_logit=oL, onset_J_frac=(None if oJ is None else (oJ + 1) / n), onset_logit_frac=(None if oL is None else (oL + 1) / n), c_use=c_use, c_say=c_say)
print("J-lens top-10 share by layer: " + " ".join(f"{L}:{v:.2f}" for L, v in zip(LJ, curve["J"])))
print("logit-lens top-10 share by layer: " + " ".join(f"{L}:{v:.2f}" for L, v in zip(LJ + [FINAL], curve["logit"])))
print(f"onset: J-lens layer {oJ} ({resO1['onset_J_frac']}), logit lens layer {oL} ({resO1['onset_logit_frac']}); fact horizon {c_use}, name horizon {c_say}  [{time.time() - t0:.0f}s]")
save(resO1, "eO1_results")
