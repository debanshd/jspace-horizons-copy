# J-space capacity by modality and the articulation bound. Seven cells on Qwen3.5 (0.8B to 27B), Qwen3.6-27B and Gemma 4 (E2B, E4B, 31B) with the Neuronpedia J-lenses.
# Paste after the notebook's model, lens and setup cells. M0 is the picture helper cell from the workspace suite (skip it if M0 already ran in this
# session). B1 to B7 are new. Nothing a running cell uses is changed: every new name starts with b_, b4_, b7_ or B_. Run order: B1, B2, B3, B4, B5, B6 on each
# Qwen3.5 size, B7 on Gemma 4 E4B (it needs no M0 and no other B cell).

# Psychophysics of the workspace: two experiments on Qwen3.5-4B with the J-lens, pictures as the stimulus
#
# Paste after the notebook's model and lens cells and the suite's setup cell (uses tokenizer, hf_model, model, MODEL_NAME, LAYERS, JAC, U_OUT,
# DEV, BATCH, BATCH_TOKENS, FINAL, TEXT, _dtype, chat, n_tokens, single_ids, residual_of, LETTER_IDS, YES_IDS, NO_IDS, ANSWER_PREFIX, INSTR,
# boot_ci, f3, auroc_ci, save, finish, OKABE, FIGDIR). Independent of every experiment cell. Nothing a running cell uses is changed: every
# new name starts with mm_, ws_, MM_ or WS_.
#
#   M0  picture helpers        the multimodal forward, the processor preflight, the lens ranks (J-lens and logit lens)
#   W1  subliminal             a word at ten contrasts over noise, one of two options in a choice the text asks for. The choice shift, a
#                              detection d-prime, naming, the self-report, and the word's lens rank after the picture and at the answer,
#                              all against noise-only pictures. The order of the thresholds is the result. The pictures where the choice
#                              moves and the model reports no word are the subliminal trials, read by the lens
#   W2  the plan               n circles of distinct colours. Asked to list the colours, how many of them are in the workspace before
#                              the first one is said, as a function of n. Asked to count instead, the same readout: the relevance rule
#   W3  the triangle under load  a triangle among 2 to 24 circles to count, the same 98 tokens every time. Seen when asked, reported to an
#                              open question, read by the lens at the report: three curves against load, the load at which each halves,
#                              and a dial on the coordinate of "triangle" at the report
#   W4  imagine                a sentence before the picture asks the model to imagine a green triangle below the blue circle. The J-lens at
#                              every image token: is "triangle" written into the visual tokens at the imagined place, everywhere, or nowhere
#   W5  where it came from     the same concept by picture or by note, the residual at the answer split into the lens span and the rest, a
#                              probe for the route on each: is the workspace amodal, and does the source report read it
#   W6  the stream             eight frames of a ball rolling to the edge of a platform, shown t at a time. The J-lens rank of "fall" with no
#                              question asked, and p(yes) to "will it fall", both aligned to the frame it leaves: how far ahead of its own
#                              report the workspace runs
#
# Rubric these cells were built against
#   speed         answer distributions only, no generation except W2's twenty tokens, no judge anywhere, one batched forward per prompt set,
#                 pictures drawn from a seed. W1 about 10 min on a T4, W2 about 5
#   correctness   every behavioural number is a d-prime or an accuracy against a null drawn from the same pictures (noise only, absent
#                 colours), every lens number against the same null. Bootstrap intervals over pictures. The full-contrast level is the
#                 positive control for every readout, and a cell says at the top of its output when it fails. The J-lens is read next to the logit lens so the
#                 method's contribution is a number. Yes/no mass printed. Predictions written before the run and both outcomes read
#   completeness  behaviour at three levels (detect, use, name), the workspace at every read layer, both lenses, the null readout
#                 (another digit, an absent colour) in every panel
#   finesse       one classic paradigm per cell, one figure per cell, the stimuli shown, the known critique of the human result
#                 (a criterion artefact, Phillips 2021) ruled out by design because every measure is bias-free

# %% ---------------------------------------------------------------- M0: picture helpers
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor
MM_W, MM_H = 448, 224                                  # 32 by 32 pixels become one token after the vision merge
MM_IMG_TOKENS = (MM_W // 32) * (MM_H // 32) + 8        # allowance per picture in the batch planner
MM_IM_END, MM_IM_START = tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|im_start|>")
MM_VS, MM_VE = tokenizer.convert_tokens_to_ids("<|vision_start|>"), tokenizer.convert_tokens_to_ids("<|vision_end|>")
t0 = time.time()


def mm_font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def mm_text_picture(text, size=56):
    img = Image.new("RGB", (MM_W, MM_H), "white")
    d, font = ImageDraw.Draw(img), mm_font(size)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
    d.text(((MM_W - (x1 - x0)) / 2 - x0, (MM_H - (y1 - y0)) / 2 - y0), text, fill="black", font=font)
    return img


def mm_contact_sheet(frames, cols, name):
    """the pictures the model saw, as one figure file"""
    w, h = frames[0][1].size
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (w + 12), rows * (h + 34)), "white")
    d, font = ImageDraw.Draw(sheet), mm_font(16)
    for j, (label, im) in enumerate(frames):
        x, y = (j % cols) * (w + 12), (j // cols) * (h + 34)
        sheet.paste(im, (x, y + 28))
        d.text((x + 2, y + 6), label, fill="black", font=font)
    path = f"{FIGDIR}/{name}.png"
    sheet.save(path)
    print("saved", path)


# --- the multimodal forward
def mm_chat(msgs):
    """chat template with one picture in the first user turn. msgs are (role, text) pairs"""
    content = [[{"type": "image"}, {"type": "text", "text": msgs[0][1]}]] + [m[1] for m in msgs[1:]]
    return MM_PROC.apply_chat_template([{"role": r, "content": c} for (r, _), c in zip(msgs, content)], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def mm_chat_frames(k, text, later=()):
    """one user turn with k pictures in order, then the text, then any later (role, text) turns"""
    content = [{"type": "image"}] * k + [{"type": "text", "text": text}]
    msgs = [{"role": "user", "content": content}] + [{"role": r, "content": t} for r, t in later]
    return MM_PROC.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def mm_encode(texts, images):
    """images: one picture per prompt, or a list of pictures per prompt (frames), or None. The processor takes them flat, in prompt order"""
    if images is None or all(im is None for im in images):
        return tokenizer(list(texts), return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    flat = [f for im in images for f in (im if isinstance(im, (list, tuple)) else [im])]
    if type(MM_PROC).__name__.startswith("Gemma4"):                                 # Gemma 4's processor wants one list of pictures per prompt (change #35)
        flat = [list(im) if isinstance(im, (list, tuple)) else [im] for im in images]
    enc = MM_PROC(text=list(texts), images=flat, return_tensors="pt", padding=True)
    if "pixel_values" in enc:
        enc["pixel_values"] = enc["pixel_values"].to(_dtype)
    return enc.to(DEV)


def mm_run(enc):
    """the forward without building logits at every position: the decoder stack for text, the multimodal entry point for pictures"""
    kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
    if "pixel_values" not in kw:
        return TEXT(input_ids=kw["input_ids"], attention_mask=kw["attention_mask"], use_cache=False)
    if MM_VISION_PATH == "base":
        return hf_model.model(**kw, use_cache=False)
    return hf_model(**kw, use_cache=False, logits_to_keep=1)


def mm_run_batches(texts, images, fn):
    """fn(idx) -> one result per prompt. Length-sorted batches under BATCH and BATCH_TOKENS with an allowance per picture. A batch that runs out of memory is halved"""
    images = images if images is not None else [None] * len(texts)
    lens = np.array([n_tokens(t) + MM_IMG_TOKENS * (len(im) if isinstance(im, (list, tuple)) else 1 if im is not None else 0) for t, im in zip(texts, images)])
    plan, cur = [], []
    for i in np.argsort(lens, kind="stable"):
        if cur and (len(cur) == BATCH or (len(cur) + 1) * lens[i] > BATCH_TOKENS):
            plan.append(np.array(cur))
            cur = []
        cur.append(int(i))
    if cur:
        plan.append(np.array(cur))
    out = [None] * len(texts)
    stack = plan[::-1]
    bar = tqdm(total=len(stack), desc=f"{len(texts)} prompts", unit="batch", leave=False, dynamic_ncols=True, file=sys.__stderr__, disable=len(stack) < 3)
    while stack:
        idx = stack.pop()
        try:
            for i, r in zip(idx, fn(idx)):
                out[i] = r
            bar.update(1)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(idx) == 1:
                raise
            stack += [idx[len(idx) // 2:], idx[:len(idx) // 2]]
            bar.total += 1
    bar.close()
    torch.cuda.empty_cache()
    return out


def mm_where(names, n_image=0):
    """positions in a padded row: last, q_end (the token before the user turn's end), ve (the vision end token), after (the token after it),
    image (the n_image tokens before ve). A name that is not in the row falls back to the last token."""
    def f(ids):
        s = len(ids)
        ends = [j for j, t in enumerate(ids) if t == MM_IM_END]
        ve = next((j for j, t in enumerate(ids) if t == MM_VE), None)
        at = dict(last=s - 1, q_end=(ends[-1] - 1) if ends else s - 1, ve=ve if ve is not None else s - 1, after=(ve + 1) if ve is not None else s - 1)
        out = []
        for nm in names:
            out += (list(range(ve - n_image, ve)) if ve is not None else [s - 1] * n_image) if nm == "image" else [at[nm]]
        return out
    return f


def mm_register(L, f):
    """forward hook on the output of decoder layer L. L == -1 is the input to layer 0 (the embeddings, image embeddings scattered in):
    a pre-hook that presents that tensor to f as (mod, inp, out) and installs f's return (None = unchanged), so one hook shape fits both."""
    if L >= 0:
        return model.layers[L].register_forward_hook(f)

    def pre(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        new = f(mod, args, h)
        if new is None:
            return None
        new = new[0] if isinstance(new, tuple) else new
        if args:
            return (new,) + tuple(args[1:]), kwargs
        return args, dict(kwargs, hidden_states=new)
    return model.layers[0].register_forward_pre_hook(pre, with_kwargs=True)


@torch.no_grad()
def mm_forward(texts, images=None, layers=(), where=None, hooks=None):
    """batched forward for prompts with or without a picture: the model's logits at the last token [n, vocab] (float32, cpu) and
    {layer: residual [n, P, d]} (float32, cpu) at the positions where(ids_row) names, the same P for every row, or the last token alone when
    where is None. hooks(enc, idx) -> [(layer, fn)] is registered before the capture hooks, so the capture sees the edited residual."""
    keep = set(layers) | {FINAL}
    images = images if images is not None else [None] * len(texts)

    def fn(idx):
        enc = mm_encode([texts[i] for i in idx], [images[i] for i in idx])
        ids = enc["input_ids"]
        b, s = ids.shape
        pos = torch.as_tensor([(where(ids[j].tolist()) if where else []) + [s - 1] for j in range(b)], device=DEV)
        rows_ = torch.arange(b, device=DEV)[:, None]
        store = {}

        def capture(L):
            def f(mod, inp, out):
                store[L] = residual_of(out)[rows_, pos].float()
            return f

        handles = [mm_register(L, f) for L, f in (hooks(enc, idx) if hooks else [])] + \
                  [mm_register(L, capture(L)) for L in keep]
        try:
            mm_run(enc)
        finally:
            for h in handles:
                h.remove()
        logits = model.unembed(store[FINAL][:, -1]).float().cpu()
        return [(logits[j], {L: (store[L][j, :-1] if where else store[L][j, -1:]).cpu() for L in layers}) for j in range(b)]

    rows = mm_run_batches(texts, images, fn)
    return torch.stack([r[0] for r in rows]), {L: torch.stack([r[1][L] for r in rows]) for L in layers}


def mm_letters(texts, images=None, hooks=None):
    """[n, 4] distribution over A-D at the last token and the raw mass on the four letters"""
    pr = torch.softmax(mm_forward(texts, images, hooks=hooks)[0], -1)
    P = np.stack([sum(pr[:, t].numpy() for t in LETTER_IDS[l]) for l in "ABCD"], 1)
    return P / np.maximum(P.sum(1, keepdims=True), 1e-9), P.sum(1)


def mm_yes(texts, images=None, hooks=None):
    pr = torch.softmax(mm_forward(texts, images, hooks=hooks)[0], -1)
    y = sum(pr[:, t].numpy() for t in YES_IDS)
    n = sum(pr[:, t].numpy() for t in NO_IDS)
    return y / np.maximum(y + n, 1e-9), y + n


@torch.no_grad()
def mm_generate(texts, images, n_new, hooks=None):
    """greedy continuations. hooks(enc, idx) -> [(layer, fn)] stays registered through the generation, so a clamp holds on every new token"""
    def fn(idx):
        enc = mm_encode([texts[i] for i in idx], [images[i] for i in idx])
        handles = [mm_register(L, f) for L, f in (hooks(enc, idx) if hooks else [])]
        try:
            g = hf_model.generate(**enc, max_new_tokens=n_new, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        finally:
            for h in handles:
                h.remove()
        return [tokenizer.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for row in g]
    return mm_run_batches(texts, images, fn)


def mm_preflight():
    """does the loaded model have a vision path, and does it read a picture: three questions whose answer exists only in the picture.
    Sets MM_PROC, MM_VISION_PATH and MM_N_IMAGE (tokens per family picture)."""
    global MM_PROC, MM_VISION_PATH, MM_N_IMAGE
    try:
        MM_PROC = AutoProcessor.from_pretrained(MODEL_NAME)
        MM_PROC.tokenizer.padding_side = "left"
    except Exception as e:
        return f"no processor for {MODEL_NAME}: {e}"
    qs = ["Which letter is written in the picture?\n(A) A\n(B) B\n(C) C\n(D) D"] * 3
    imgs = [mm_text_picture(l) for l in "ACD"]
    texts = [mm_chat([("user", q + INSTR)]) + ANSWER_PREFIX for q in qs]
    err = ""
    for MM_VISION_PATH in ("base", "logits_to_keep"):
        try:
            P, mass = mm_letters(texts, imgs)
            break
        except Exception as e:
            err = f"{MM_VISION_PATH}: {type(e).__name__}: {str(e)[:160]}"
    else:
        return f"the multimodal forward failed on both entry points ({err}). The loaded hf_model may be text only"
    picks = "".join("ABCD"[j] for j in P.argmax(1))
    n_ok = sum(a == b for a, b in zip(picks, "ACD"))
    if mass.mean() < 0.5 or n_ok < 2:
        return f"the picture is not being read: picks {picks} for A, C, D with letter mass {mass.mean():.2f}"
    if n_ok < 3:
        print(f"vision probe: picks {picks} for A, C, D (one glyph confused; the path reads pictures)")      # the 27B reads the D card as B
    ids = mm_encode(texts[:1], imgs[:1])["input_ids"][0].tolist()
    MM_N_IMAGE = (ids.index(MM_VE) - ids.index(MM_VS) - 1) if MM_VE in ids and MM_VS in ids else MM_IMG_TOKENS - 8
    return None


@torch.no_grad()
def mm_ranks(resid, targets, null_targets, jac=True, layers=None):
    """rank (1 = top) of the best variant of each row's targets at every read layer and captured position, [n, len(layers), P], and the same
    for the null targets from the same logits. jac=True reads through the J-lens (W_U J_L), jac=False through the plain logit lens (W_U)."""
    layers = list(LAYERS if layers is None else layers)
    n, P = resid[layers[0]].shape[:2]
    m = max(len(t) for t in list(targets) + list(null_targets))
    T = torch.as_tensor([list(t) + [t[0]] * (m - len(t)) for t in targets], device=DEV)
    N = torch.as_tensor([list(t) + [t[0]] * (m - len(t)) for t in null_targets], device=DEV)
    out, nul = np.zeros((n, len(layers), P), int), np.zeros((n, len(layers), P), int)
    for li, L in enumerate(layers):
        for p in range(P):
            for s in range(0, n, BATCH):
                h = resid[L][s:s + BATCH, p].to(DEV)
                x = model.unembed(h @ JAC[L].T if jac else h).float()
                for ids, dst in ((T, out), (N, nul)):
                    v = x.gather(1, ids[s:s + BATCH])
                    dst[s:s + BATCH, li, p] = ((x[:, None, :] > v[:, :, None]).sum(-1) + 1).min(1).values.cpu().numpy()
    return out, nul


@torch.no_grad()
def mm_split_span(H, L, K, random_tokens=False, seed=0):
    """the 5b split for a residual matrix H [n, d]: the part inside the span of the lens vectors of the top-K lens tokens at that position (or K random
    tokens), the remainder, and the share of squared norm inside"""
    H = torch.as_tensor(np.asarray(H), device=DEV, dtype=torch.float32)
    if random_tokens:
        tok = torch.as_tensor(np.random.default_rng(seed).integers(0, U_OUT.shape[0], (len(H), K)), device=DEV)
    else:
        tok = torch.cat([model.unembed(H[s:s + BATCH] @ JAC[L].T).float().topk(K, dim=1).indices for s in range(0, len(H), BATCH)])
    Vv = U_OUT[tok].float().to(DEV) @ JAC[L]
    Vv = Vv / Vv.norm(dim=2, keepdim=True)
    Gm = (Vv @ Vv.transpose(1, 2)).double().cpu() + 1e-4 * torch.eye(K, dtype=torch.float64)
    b = torch.einsum("nkd,nd->nk", Vv, H).double().cpu()
    c = torch.linalg.solve(Gm, b[..., None])[..., 0].float().to(DEV)
    inside = torch.einsum("nk,nkd->nd", c, Vv)
    share = float(((inside.norm(dim=1) / H.norm(dim=1)) ** 2).mean())
    return inside.cpu().numpy(), (H - inside).cpu().numpy(), share


def ws_probs(texts, images, id_sets, hooks=None):
    """next-token probability mass on each set of token ids, [n, len(id_sets)], and the total mass on all of them"""
    pr = torch.softmax(mm_forward(texts, images, hooks=hooks)[0], -1)
    P = np.stack([sum(pr[:, t].numpy() for t in ids) for ids in id_sets], 1)
    return P, P.sum(1)


def ws_probs_pairs(texts, images, sets_a, sets_b, hooks=None):
    """next-token probability mass on two id sets that differ by row, [n, 2], and their total"""
    pr = torch.softmax(mm_forward(texts, images, hooks=hooks)[0], -1).numpy()
    P = np.stack([[pr[j, a].sum(), pr[j, b].sum()] for j, (a, b) in enumerate(zip(sets_a, sets_b))])
    return P, P.sum(1)


def ws_forms(word):
    return [word, " " + word, word.capitalize(), " " + word.capitalize()]


def ws_ids(word):
    """single-token ids of a word's forms, or the first token of ' word'"""
    return single_ids(ws_forms(word)) or [tokenizer.encode(" " + word, add_special_tokens=False)[0]]


def ws_z(p):
    from scipy.stats import norm
    return float(norm.ppf(np.clip(p, 0.01, 0.99)))


def ws_dprime(hit, fa, n=2000, seed=0):
    """d-prime from per-picture response probabilities: z(mean hit) - z(mean false alarm), with a bootstrap over pictures"""
    hit, fa = np.asarray(hit, float), np.asarray(fa, float)
    if len(hit) == 0 or len(fa) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    d = [ws_z(rng.choice(hit, len(hit)).mean()) - ws_z(rng.choice(fa, len(fa)).mean()) for _ in range(n)]
    return ws_z(hit.mean()) - ws_z(fa.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# --- headline figures: one axis, one claim in the title computed from the numbers, bands for intervals, labels at the line ends, no legend box
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def ws_finish(fig, name, text):
    """save without the judge in the footer, these cells use none. The title is wrapped to two lines at most"""
    import textwrap
    sns.despine(fig=fig)
    if fig._suptitle is not None:
        t = fig._suptitle
        t.set_text("\n".join(textwrap.wrap(t.get_text(), max(80, int(fig.get_size_inches()[0] * 11)))))
        t.set_y(1.02 if "\n" in t.get_text() else 1.0)
    fig.text(0.005, 0.005, f"{text} · {MODEL_TAG} · {TAG}", fontsize=6.5, color="0.5")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{FIGDIR}/{name}.{ext}")
    plt.show()
    print(f"saved {FIGDIR}/{name}.png .pdf .svg")


def ws_band(ax, xs, vals, color, label=None, ls="-", lw=2.0):
    """a line with its interval as a band. vals are (mean, lo, hi) per x"""
    m, lo, hi = [v[0] for v in vals], [v[1] for v in vals], [v[2] for v in vals]
    ax.fill_between(xs, lo, hi, color=color, alpha=0.16, lw=0)
    ax.plot(xs, m, ls, color=color, lw=lw, label=label, solid_capstyle="round")
    return m[-1]


def ws_end_labels(ax, items, gap):
    """labels just right of the axis at each line's last value, pushed apart by gap (data units) so none overlap"""
    import textwrap
    items = sorted(items, key=lambda t: t[0])
    texts = ["\n".join(textwrap.wrap(t[1], 24)) for t in items]
    lines = [t.count("\n") + 1 for t in texts]
    ys = [t[0] for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap * (0.5 + 0.5 * max(lines[i - 1], lines[i])))
    top = ax.get_ylim()[1]
    if ys and ys[-1] > top:
        ys = [y - (ys[-1] - top) for y in ys]
    for (y0, _, color), text, y in zip(items, texts, ys):
        ax.text(1.015, y, text, transform=ax.get_yaxis_transform(), va="center", ha="left", color=color, fontsize=8, linespacing=0.95, clip_on=False)


def ws_thumbnails(ax, xs, images, zoom, dy=-30):
    """the stimuli under the axis at their x positions"""
    for x, im in zip(xs, images):
        ab = AnnotationBbox(OffsetImage(np.asarray(im), zoom=zoom), (x, 0), xycoords=("data", "axes fraction"), xybox=(0, dy), boxcoords="offset points",
                            box_alignment=(0.5, 1.0), frameon=False, annotation_clip=False, pad=0)
        ax.add_artist(ab)


# --- reads
MM_PROC, MM_VISION_PATH, MM_N_IMAGE = None, "base", MM_IMG_TOKENS - 8
problem = mm_preflight()
if problem:
    print(f"picture cells skipped: {problem}")
else:
    print(f"vision path '{MM_VISION_PATH}' reads the picture. {MM_N_IMAGE} image tokens per {MM_W}x{MM_H} picture  [{time.time() - t0:.0f}s]")

# %% ---------------------------------------------------------------- B1: the bound. How many items J-space holds, by modality
# Run after M0 in the same session. Nothing a running cell uses is changed: every new name starts with b_ or B_.
#
# Hypothesis. J-space has a capacity per modality. Show k items and ask for all of them: the number of item words the J-lens reads at the
# report position rises with k and then stops at K, and K depends on how the items came in. Text K above picture K is a bound set by the
# modality. The same K is an amodal workspace. The Chinese list with an English readout is the control for copying: if text K is high
# only because the words sit in the context, the Chinese list falls to the picture's K.
#
# Stimuli. 12 shapes in 8 colours, k in {1, 2, 3, 4, 6, 8}, 24 scenes per k, the same 24 scenes rendered as an English list, as a Chinese
# list, and as coloured icons on one 448 by 224 card (98 vision tokens at every k). One question everywhere.
# Readout. The bench's: a present shape is in J-space when its word is in the lens top-10 at any read layer at the report position. The
# absent shapes under the same rule are the hallucination control. Use is a second turn, "Is there a {shape}?", for four present and four
# absent shapes per scene, as hits, false alarms and d', and it gates every lens number (hits >= 0.9 at the load).
# Rules, written before the run. K is the largest mean count over k, with a bootstrap over scenes. Bound by modality: K_text - K_picture
# with a 95% interval above 1. Amodal: the interval inside [-1, 1]. Copying: K_zh within 1 of K_picture and K_text above both by 2.
B_SHAPES = ["circle", "square", "triangle", "star", "heart", "diamond", "cross", "hexagon", "arrow", "ring", "moon", "oval"]
B_ZH_SHAPE = {"circle": "圆形", "square": "正方形", "triangle": "三角形", "star": "星形", "heart": "心形", "diamond": "菱形", "cross": "十字",
              "hexagon": "六边形", "arrow": "箭头", "ring": "圆环", "moon": "月牙", "oval": "椭圆"}
B_COLOURS = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (40, 160, 40), "yellow": (240, 200, 20), "orange": (250, 140, 20),
             "purple": (130, 30, 160), "pink": (250, 120, 180), "brown": (130, 70, 20)}
B_ZH_COLOUR = {"red": "红色", "blue": "蓝色", "green": "绿色", "yellow": "黄色", "orange": "橙色", "purple": "紫色", "pink": "粉色", "brown": "棕色"}
B_RENDERINGS = ["text", "text zh", "picture"]
B_COLOR = {"text": OKABE[0], "text zh": OKABE[5], "picture": OKABE[1]}
B_KS, B_PER_K = [1, 2, 3, 4, 6, 8], 24
B_TOP = 10
B_ASK = "Which shapes are present? Answer with the shape names in English, separated by commas."
B_ASK_ALT = "Name the shapes that are present, in English, separated by commas."       # the second wording: the lens count is read under both, the rest of the program uses B_ASK
B_ASK_COUNT = "How many shapes are present? Answer with a number."     # B1c, the relevance control: a question no shape word answers (CHANGES.md)
B_OK = "Okay."
B_YESNO = "Is there a {shape}? Answer yes or no."
B_RADIUS = 24
B_RNG = np.random.default_rng(11)
import collections
t0 = time.time()
save(dict(shapes=B_SHAPES, colours=list(B_COLOURS), renderings=B_RENDERINGS, ks=B_KS, per_k=B_PER_K, top=B_TOP, ask=B_ASK, yesno=B_YESNO, layers=list(LAYERS), model=MODEL_NAME,
          prediction="the count of present shapes in the lens top-10 at the report rises with k and stops. Picture stops below text (a bound by modality) or at the same "
                     "place (amodal). The Chinese list says whether the text count is copying"), "eB1_config")


def b_draw(d, shape, cx, cy, r, fill, bg):
    import math
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif shape == "square":
        d.rectangle([cx - 0.85 * r, cy - 0.85 * r, cx + 0.85 * r, cy + 0.85 * r], fill=fill)
    elif shape == "triangle":
        d.polygon([(cx, cy - r), (cx - 0.95 * r, cy + 0.7 * r), (cx + 0.95 * r, cy + 0.7 * r)], fill=fill)
    elif shape == "star":
        d.polygon([(cx + (r if i % 2 == 0 else 0.45 * r) * math.cos(math.radians(-90 + 36 * i)), cy + (r if i % 2 == 0 else 0.45 * r) * math.sin(math.radians(-90 + 36 * i))) for i in range(10)], fill=fill)
    elif shape == "heart":
        d.ellipse([cx - r, cy - 0.85 * r, cx, cy + 0.15 * r], fill=fill)
        d.ellipse([cx, cy - 0.85 * r, cx + r, cy + 0.15 * r], fill=fill)
        d.polygon([(cx - 0.98 * r, cy - 0.1 * r), (cx + 0.98 * r, cy - 0.1 * r), (cx, cy + r)], fill=fill)
    elif shape == "diamond":
        d.polygon([(cx, cy - r), (cx + 0.65 * r, cy), (cx, cy + r), (cx - 0.65 * r, cy)], fill=fill)
    elif shape == "cross":
        d.rectangle([cx - 0.3 * r, cy - r, cx + 0.3 * r, cy + r], fill=fill)
        d.rectangle([cx - r, cy - 0.3 * r, cx + r, cy + 0.3 * r], fill=fill)
    elif shape == "hexagon":
        d.polygon([(cx + r * math.cos(math.radians(60 * i)), cy + r * math.sin(math.radians(60 * i))) for i in range(6)], fill=fill)
    elif shape == "arrow":
        d.polygon([(cx - r, cy - 0.3 * r), (cx + 0.1 * r, cy - 0.3 * r), (cx + 0.1 * r, cy - 0.8 * r), (cx + r, cy), (cx + 0.1 * r, cy + 0.8 * r), (cx + 0.1 * r, cy + 0.3 * r), (cx - r, cy + 0.3 * r)], fill=fill)
    elif shape == "ring":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
        d.ellipse([cx - 0.5 * r, cy - 0.5 * r, cx + 0.5 * r, cy + 0.5 * r], fill=bg)
    elif shape == "moon":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
        d.ellipse([cx - 0.45 * r, cy - 0.95 * r, cx + 1.45 * r, cy + 0.95 * r], fill=bg)
    elif shape == "oval":
        d.ellipse([cx - 1.2 * r, cy - 0.62 * r, cx + 1.2 * r, cy + 0.62 * r], fill=fill)


def b_card(items, rng):
    """items are (shape, colour, slot). Returns the card and, per item, the vision-token indices (row-major, 14 by 7) its icon covers"""
    bg = (238, 238, 238)
    img = Image.new("RGB", (MM_W, MM_H), bg)
    d = ImageDraw.Draw(img)
    slots = [(56 + 112 * i + rng.uniform(-12, 12), 40 + 72 * j + rng.uniform(-8, 8)) for i in range(4) for j in range(3)]
    patches = []
    for shape, colour, slot in items:
        cx, cy = slots[slot]
        b_draw(d, shape, cx, cy, B_RADIUS, B_COLOURS[colour], bg)
        cols = range(max(0, int((cx - B_RADIUS) // 32)), min(MM_W // 32 - 1, int((cx + B_RADIUS) // 32)) + 1)
        rows_ = range(max(0, int((cy - B_RADIUS) // 32)), min(MM_H // 32 - 1, int((cy + B_RADIUS) // 32)) + 1)
        patches.append([r * (MM_W // 32) + c for r in rows_ for c in cols])
    return img, patches


def b_list(items, zh=False, ask=None):
    ask = B_ASK if ask is None else ask
    if zh:
        return "这是一个列表：" + "，".join(f"一个{B_ZH_COLOUR[c]}的{B_ZH_SHAPE[s]}" for s, c, _ in items) + f"。\n{ask}"
    return "Here is a list: " + ", ".join(f"{'an' if c[0] in 'aeiou' else 'a'} {c} {s}" for s, c, _ in items) + f".\n{ask}"


def b_scenes():
    """the scene specifications, shared by every rendering. k = 1 covers every shape twice"""
    singles = [B_SHAPES[i] for i in B_RNG.permutation(len(B_SHAPES) * 2) % len(B_SHAPES)]
    rows = []
    for k in B_KS:
        for seed in range(B_PER_K):
            sh = [singles[seed]] if k == 1 else [B_SHAPES[i] for i in B_RNG.choice(len(B_SHAPES), k, replace=False)]
            co = [list(B_COLOURS)[i] for i in B_RNG.choice(len(B_COLOURS), k, replace=False)]
            sl = [int(i) for i in B_RNG.choice(12, k, replace=False)]
            absent = [s for s in B_SHAPES if s not in sh]
            ask = [sh[i] for i in B_RNG.choice(k, min(4, k), replace=False)] + [absent[i] for i in B_RNG.choice(len(absent), 4, replace=False)]   # present shapes asked about are drawn at random, so the serial position of a shape never decides whether it is asked
            rng = np.random.default_rng(1000 * k + seed)
            img, patches = b_card(list(zip(sh, co, sl)), rng)
            it = list(zip(sh, co, sl))
            rows.append(dict(k=k, seed=seed, shapes=sh, ask=ask, img=img, patches=patches, slots=sl,
                             prompt={"text": chat([("user", b_list(it))]), "text zh": chat([("user", b_list(it, zh=True))]), "picture": mm_chat([("user", B_ASK)])},
                             prompt_alt={"text": chat([("user", b_list(it, ask=B_ASK_ALT))]), "text zh": chat([("user", b_list(it, zh=True, ask=B_ASK_ALT))]), "picture": mm_chat([("user", B_ASK_ALT)])},
                             prompt_cnt={"text": chat([("user", b_list(it, ask=B_ASK_COUNT))]), "text zh": chat([("user", b_list(it, zh=True, ask=B_ASK_COUNT))]), "picture": mm_chat([("user", B_ASK_COUNT)])}))
    return pd.DataFrame(rows)


def b_turn2(prompt, shape):
    """the first prompt continued by hand with a fixed assistant turn and the yes or no question"""
    return prompt + B_OK + "<|im_end|>\n<|im_start|>user\n" + B_YESNO.format(shape=shape) + "<|im_end|>\n" + B_ASSIST


@torch.no_grad()
def b_ranks(Hd, id_sets, jac=True):
    """rank (1 = top) of the best variant of each id set at every read layer, [n, len(LAYERS), len(id_sets)], from {layer: [n, d]}.
    jac=True reads through the J-lens, "jpp" through the J++ lens (when loaded), False through the plain logit lens"""
    M = JPP if jac == "jpp" else JAC
    n = np.asarray(Hd[LAYERS[0]]).shape[0]
    m = max(len(t) for t in id_sets)
    Sx = torch.as_tensor([list(t) + [t[0]] * (m - len(t)) for t in id_sets], device=DEV)
    out = np.zeros((n, len(LAYERS), len(id_sets)), int)
    for li, L in enumerate(LAYERS):
        for s in range(0, n, BATCH):
            h = torch.as_tensor(np.asarray(Hd[L][s:s + BATCH]), device=DEV, dtype=torch.float32)
            x = model.unembed(h @ M[L].T if jac else h).float()
            for q in range(len(id_sets)):
                v = x[:, Sx[q]]
                out[s:s + BATCH, li, q] = ((x[:, None, :] > v[:, :, None]).sum(-1) + 1).min(1).values.cpu().numpy()
    return out


def b_capacity(ks, counts_by_k, n=2000, seed=0):
    """K = the largest mean over k, the knee the smallest k with mean >= 0.9 K, a bootstrap over scenes within k, and the samples"""
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(counts_by_k[k]) for k in ks])
    bs = np.array([max(rng.choice(counts_by_k[k], len(counts_by_k[k])).mean() for k in ks) for _ in range(n)])
    return dict(K=(float(means.max()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))), knee=int(ks[int(np.argmax(means >= 0.9 * means.max()))]), curve=[float(m) for m in means]), bs


def b_fmt(t):
    return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"


# --- reads
if MM_PROC is None:
    print("B1 skipped: no vision path (see M0)")
    B_OK_RUN, resB1 = False, {}
else:
    B_SID = {s: ws_ids(s) for s in B_SHAPES}
    B_SID_ZH = {s: single_ids([B_ZH_SHAPE[s]]) or [tokenizer.encode(B_ZH_SHAPE[s], add_special_tokens=False)[0]] for s in B_SHAPES}     # the Chinese name, first token
    print("shape words that are not single tokens (read as the first token of ' word'):", [s for s in B_SHAPES if not single_ids(ws_forms(s))] or "none")
    SB = b_scenes()
    B_ASSIST = SB.prompt.iloc[0]["text"][SB.prompt.iloc[0]["text"].rindex("<|im_start|>assistant"):]
    assert all(SB.prompt.iloc[0][r].endswith(B_ASSIST) for r in B_RENDERINGS)
    ids1, ids2 = tokenizer.encode(SB.prompt.iloc[0]["text"], add_special_tokens=False), tokenizer.encode(b_turn2(SB.prompt.iloc[0]["text"], "star"), add_special_tokens=False)
    assert ids2[:len(ids1)] == ids1, "the continued prompt does not tokenize as prompt + continuation"
    mm_contact_sheet([(f"k = {k}", SB[SB.k == k].img.iloc[0]) for k in B_KS], 3, "figB1_stimuli")
    present = np.array([[s in sh for s in B_SHAPES] for sh in SB.shapes])
    B_REP, B_BEST, B_USE, resB1, B_SAMPLES = {}, {}, {}, dict(by_rendering={}, n_scenes=int(len(SB))), {}
    for rend in B_RENDERINGS:
        texts = [p[rend] for p in SB.prompt]
        imgs = SB.img.tolist() if rend == "picture" else None
        _, R = mm_forward(texts, imgs, LAYERS)
        B_REP[rend] = {L: R[L][:, 0].numpy() for L in LAYERS}
        del R
        rj = b_ranks(B_REP[rend], [B_SID[s] for s in B_SHAPES], jac=True).min(1)         # best rank over read layers, [scenes, shapes]
        rl = b_ranks(B_REP[rend], [B_SID[s] for s in B_SHAPES], jac=False).min(1)
        rz = b_ranks(B_REP[rend], [B_SID_ZH[s] for s in B_SHAPES], jac=True).min(1)      # the Chinese names through the same lens
        B_BEST[rend] = rj
        inside, inside_l, inside_either = rj <= B_TOP, rl <= B_TOP, np.minimum(rj, rz) <= B_TOP
        # the second wording, lens only
        _, Ra = mm_forward([p[rend] for p in SB.prompt_alt], imgs, LAYERS)
        ra = b_ranks({L: Ra[L][:, 0].numpy() for L in LAYERS}, [B_SID[s] for s in B_SHAPES], jac=True).min(1)
        del Ra
        inside_alt = ra <= B_TOP
        # B1c, the relevance control: the same scenes asked "how many", at the same report position (CHANGES.md)
        _, Rc = mm_forward([p[rend] for p in SB.prompt_cnt], imgs, LAYERS)
        rc = b_ranks({L: Rc[L][:, 0].numpy() for L in LAYERS}, [B_SID[s] for s in B_SHAPES], jac=True).min(1)
        del Rc
        inside_cnt = rc <= B_TOP
        # use: the second turn on the full scene
        rows = [(i, s, s in SB.shapes.iloc[i]) for i in range(len(SB)) for s in SB.ask.iloc[i]]
        P, mass = ws_probs([b_turn2(texts[i], s) for i, s, _ in rows], [imgs[i] for i, _, _ in rows] if imgs else None, [YES_IDS, NO_IDS])
        p_yes = P[:, 0] / np.maximum(P.sum(1), 1e-9)
        B_USE[rend] = p_yes                                                                   # rows in the order (scene, asked shape), reused by B3
        qk, qp = np.array([SB.k.iloc[i] for i, _, _ in rows]), np.array([p for _, _, p in rows])
        row = dict(by_k={}, yes_no_mass=float(mass.mean()))
        for k in B_KS:
            sel = (SB.k == k).values
            sq = qk == k
            hit, fa = p_yes[sq & qp], p_yes[sq & ~qp]
            row["by_k"][k] = dict(present=boot_ci((inside & present)[sel].sum(1)), absent=boot_ci((inside & ~present)[sel].sum(1)),
                                  present_logit=boot_ci((inside_l & present)[sel].sum(1)), present_either_language=boot_ci((inside_either & present)[sel].sum(1)),
                                  absent_either_language=boot_ci((inside_either & ~present)[sel].sum(1)), present_alt_wording=boot_ci((inside_alt & present)[sel].sum(1)),
                                  present_count_question=boot_ci((inside_cnt & present)[sel].sum(1)), absent_count_question=boot_ci((inside_cnt & ~present)[sel].sum(1)),
                                  present_top5=boot_ci(((rj <= 5) & present)[sel].sum(1)), absent_top5=boot_ci(((rj <= 5) & ~present)[sel].sum(1)),
                                  present_top20=boot_ci(((rj <= 20) & present)[sel].sum(1)), absent_top20=boot_ci(((rj <= 20) & ~present)[sel].sum(1)),
                                  present_null_corrected=boot_ci((inside & present)[sel].sum(1) - (inside & ~present)[sel].sum(1) * k / (len(B_SHAPES) - k)),
                                  auroc=auroc_ci(-np.log10(rj[sel][present[sel]]), -np.log10(rj[sel][~present[sel]])),
                                  hit=boot_ci((hit >= 0.5).astype(float)), fa=boot_ci((fa >= 0.5).astype(float)), dprime=ws_dprime(hit, fa))
        cap, B_SAMPLES[rend] = b_capacity(B_KS, {k: (inside & present)[(SB.k == k).values].sum(1) for k in B_KS})
        cap_alt, _ = b_capacity(B_KS, {k: (inside_alt & present)[(SB.k == k).values].sum(1) for k in B_KS})
        cap_either, _ = b_capacity(B_KS, {k: (inside_either & present)[(SB.k == k).values].sum(1) for k in B_KS})
        cap_cnt, _ = b_capacity(B_KS, {k: (inside_cnt & present)[(SB.k == k).values].sum(1) for k in B_KS})
        # serial position: the share of present shapes read at the report by their place in the list (text) or reading order on the card (picture), k = 8
        pos_read = collections.defaultdict(list)
        for i in np.flatnonzero((SB.k == B_KS[-1]).values):
            order_ = np.argsort(SB.slots.iloc[i]) if rend == "picture" else np.arange(len(SB.shapes.iloc[i]))
            for place, q in enumerate(order_):
                pos_read[int(place) + 1].append(bool(rj[i, B_SHAPES.index(SB.shapes.iloc[i][q])] <= B_TOP))
        row.update(capacity=cap, capacity_alt_wording=cap_alt, capacity_either_language=cap_either, capacity_count_question=cap_cnt, seen=bool(row["by_k"][1]["hit"][0] >= 0.95 and row["by_k"][1]["fa"][0] <= 0.10),
                   hold={k: bool(row["by_k"][k]["hit"][0] >= 0.9) for k in B_KS}, serial_position_k8={p_: boot_ci(np.array(v, float)) for p_, v in sorted(pos_read.items())})
        resB1["by_rendering"][rend] = row
        print(f"{rend:9s} K {b_fmt(cap['K'])} knee {cap['knee']} (second wording K {cap_alt['K'][0]:.1f}, either language K {cap_either['K'][0]:.1f}, count question K {cap_cnt['K'][0]:.1f})   in J-space by k: " + " ".join("%d:%.1f" % (k, row["by_k"][k]["present"][0]) for k in B_KS) +
              "   absent (null): " + " ".join("%.1f" % row["by_k"][k]["absent"][0] for k in B_KS) + "   use: " + " ".join("%.2f" % row["by_k"][k]["hit"][0] for k in B_KS) +
              f"   seen {'ok' if row['seen'] else 'NO'}   read by position at k = {B_KS[-1]}: " + " ".join("%d:%.2f" % (p_, v[0]) for p_, v in row["serial_position_k8"].items()) + f"  [{time.time() - t0:.0f}s]")
    d_tp = np.percentile(B_SAMPLES["text"] - B_SAMPLES["picture"], [2.5, 97.5])
    Kt, Kz, Kp = (resB1["by_rendering"][r]["capacity"]["K"][0] for r in B_RENDERINGS)
    Kz_either = resB1["by_rendering"]["text zh"]["capacity_either_language"]["K"][0]
    Kz = max(Kz, Kz_either)                                                                 # the Chinese list counts a shape as held in either language, so a readout-language artefact cannot lower it
    verdict = ("bound by modality: the picture holds fewer items in J-space than the list" if d_tp[0] > 1 else "amodal: text and picture hold the same number" if -1 <= d_tp[0] and d_tp[1] <= 1 else "undetermined at this n")
    copying = "copying: the Chinese list falls to the picture's count, so the English count is the context" if abs(Kz - Kp) <= 1 and Kt >= max(Kz, Kp) + 2 else "no copying: the Chinese list holds as many English words as the English list" if abs(Kz - Kt) <= 1 else "copying undetermined"
    resB1.update(verdict=verdict, copying=copying, K=dict(text=Kt, text_zh=Kz, text_zh_english_only=resB1["by_rendering"]["text zh"]["capacity"]["K"][0], picture=Kp),
                 K_alt_wording={r: resB1["by_rendering"][r]["capacity_alt_wording"]["K"][0] for r in B_RENDERINGS}, text_minus_picture=[float(d_tp[0]), float(d_tp[1])], seconds=time.time() - t0)
    relevance = {}
    for r in B_RENDERINGS:
        Kl, Kc = resB1["by_rendering"][r]["capacity"]["K"][0], resB1["by_rendering"][r]["capacity_count_question"]["K"][0]
        relevance[r] = ("capacity-like: the report holds the items whether or not they are about to be said" if abs(Kc - Kl) <= 1 else
                        "planning horizon: the report holds what is about to be said" if Kc <= Kl - 2 else "between")
    resB1["relevance"] = relevance
    resB1["K_count_question"] = {r: resB1["by_rendering"][r]["capacity_count_question"]["K"][0] for r in B_RENDERINGS}
    B_OK_RUN = True
    save(resB1, "eB1_results")
    print(f"\nK text {Kt:.1f}, text zh {Kz:.1f} (English readout alone {resB1['K']['text_zh_english_only']:.1f}), picture {Kp:.1f}. text minus picture in [{d_tp[0]:.1f}, {d_tp[1]:.1f}] -> {verdict}. {copying}")
    print("second wording K: " + ", ".join(f"{r} {resB1['K_alt_wording'][r]:.1f}" for r in B_RENDERINGS) + " (a plateau that moves by more than 1 between wordings is reported as wording-dependent)")
    print("count question K (B1c, the relevance control): " + ", ".join(f"{r} {resB1['K_count_question'][r]:.1f} -> {relevance[r].split(':')[0]}" for r in B_RENDERINGS))
    # --- figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(wspace=0.85, width_ratios=[1.15, 1]))
    ends_a, ends_b = [], []
    _kc = [resB1["by_rendering"][r]["capacity_count_question"]["K"][0] for r in B_RENDERINGS]
    _kc_same = max(_kc) - min(_kc) <= 0.5
    for rend in B_RENDERINGS:
        row, c = resB1["by_rendering"][rend], B_COLOR[rend]
        ls = "-" if row["seen"] else ":"
        ya = ws_band(ax[0], B_KS, [row["by_k"][k]["present"] for k in B_KS], c, ls=ls)
        ax[0].plot(B_KS, [row["by_k"][k]["absent"][0] for k in B_KS], "--", color=c, lw=1.0, alpha=0.7)
        ax[0].plot(B_KS, [row["by_k"][k]["present_count_question"][0] for k in B_KS], ":", color=c, lw=1.4, alpha=0.9)
        ends_a.append((ya, f"{rend}, K = {row['capacity']['K'][0]:.1f}" + ("" if row["seen"] else ", not seen"), c))
        if not _kc_same:
            ends_a.append((row["by_k"][B_KS[-1]]["present_count_question"][0], f"{rend}, asked how many: K = {row['capacity_count_question']['K'][0]:.1f}", c))
    if _kc_same:
        ends_a.append((float(np.mean(_kc)), f"asked how many (dotted): K = {np.mean(_kc):.1f} for all three", "0.35"))
        yb = ws_band(ax[1], B_KS, [row["by_k"][k]["hit"] for k in B_KS], c, ls=ls)
        ax[1].plot(B_KS, [row["by_k"][k]["fa"][0] for k in B_KS], "--", color=c, lw=1.0, alpha=0.7)
        ends_b.append((yb, rend, c))
    ax[0].plot(B_KS, B_KS, color="0.75", lw=1, ls="-.")
    ax[0].text(B_KS[-1], B_KS[-1] + 0.2, "every item", color="0.5", fontsize=7.5, ha="right")
    ax[0].set(xlabel=f"items in the scene, k (n = {B_PER_K} scenes per k)", ylabel=f"present shapes in the J-lens top-{B_TOP} at the report\n(dashed: absent shapes, the null; dotted: asked how many)", xticks=B_KS, ylim=(0, B_KS[-1] + 0.6))
    ws_end_labels(ax[0], ends_a, 0.55)
    ax[1].set(xlabel="items in the scene, k", ylabel="use: p(yes) >= 0.5 for a present shape\n(dashed: for an absent shape)", xticks=B_KS, ylim=(0, 1.04))
    ax[1].axhline(0.9, color="0.8", lw=1, ls="-.")
    ws_end_labels(ax[1], ends_b, 0.06)
    fig.suptitle(f"J-space holds {Kt:.1f} of the list's items and {Kp:.1f} of the picture's at the plateau (Chinese list {Kz:.1f}), "
                 f"while the model answers about {resB1['by_rendering']['picture']['by_k'][8]['hit'][0]:.0%} of the picture's items at k = 8: {verdict.split(':')[0]}", fontsize=9.5, y=1.0)
    ws_finish(fig, "figB1_bound", f"B1 · {len(SB)} scenes per rendering · {copying.split(':')[0]}")

# %% ---------------------------------------------------------------- B2: the collapse. Where the dropped items are, and whether the count is geometry
# Run after B1. Two readouts on B1's scenes. (a) The J-lens at every vision token of the card: for each present shape, whether its word is
# readable (top-10, any read layer) on its own patches, anywhere on the card, and at the report. Items on their patches and missing at the
# report are a gathering limit, items readable nowhere an entry limit. (b) The additive ceiling: the report state of each k = 1 scene minus a
# blank card's gives one vector per shape, the k vectors of every k-scene are summed onto the blank state and read by the same lens. If the
# sum predicts the measured count, the limit is interference in the readout and nothing collapses. A measured count below the sum is the
# model dropping items. Rules: gathering if anywhere minus report >= 0.25 at k = 8 with absent shapes readable somewhere under 0.3. Entry if
# the difference is under 0.1. Geometric if |measured - sum| <= 0.5 at every k. Drops if measured is below the sum by 1 at k = 8.
B_PAD = tokenizer.convert_tokens_to_ids("<|image_pad|>")
B_CHUNK = 12
t0 = time.time()


def b_where_pads(ids):
    return [j for j, t in enumerate(ids) if t == B_PAD]


# --- reads
if not B_OK_RUN or not resB1["by_rendering"]["picture"]["seen"]:
    print("B2 skipped")
    resB2 = {}
else:
    # (a) the lens on the patches
    POS = {}
    for k in B_KS:
        dk = SB[SB.k == k]
        parts = []
        for s in range(0, len(dk), B_CHUNK):
            sub = dk.iloc[s:s + B_CHUNK]
            _, R = mm_forward([p["picture"] for p in sub.prompt], sub.img.tolist(), LAYERS, where=b_where_pads)
            n, P = R[LAYERS[0]].shape[:2]
            parts.append(b_ranks({L: R[L].reshape(n * P, -1).numpy() for L in LAYERS}, [B_SID[s_] for s_ in B_SHAPES], jac=True).min(1).reshape(n, P, len(B_SHAPES)))
            del R
        POS[k] = np.concatenate(parts)                                                        # [scenes, patches, shapes], best rank over read layers
    where = {}
    example = None
    for k in B_KS:
        dk = SB[SB.k == k]
        own, anyw, rep, abs_any = [], [], [], []
        for j, (i, r) in enumerate(dk.iterrows()):
            for q, s in enumerate(r.shapes):
                si = B_SHAPES.index(s)
                own.append(POS[k][j, r.patches[q], si].min() <= B_TOP)
                anyw.append(POS[k][j, :, si].min() <= B_TOP)
                rep.append(B_BEST["picture"][i, si] <= B_TOP)
            for s in B_SHAPES:
                if s not in r.shapes:
                    abs_any.append(POS[k][j, :, B_SHAPES.index(s)].min() <= B_TOP)
        where[k] = dict(own=boot_ci(np.array(own, float)), anywhere=boot_ci(np.array(anyw, float)), report=boot_ci(np.array(rep, float)), absent_anywhere=boot_ci(np.array(abs_any, float)), n_items=len(own))
        if k == B_KS[-1]:
            r0 = dk.iloc[0]
            heat = np.full(POS[k].shape[1], -np.inf)
            for s in r0.shapes:
                heat = np.maximum(heat, -np.log10(POS[k][0, :, B_SHAPES.index(s)].astype(float)))
            example = dict(img=r0.img, heat=heat.reshape(MM_H // 32, MM_W // 32))
    w8 = where[B_KS[-1]]
    where_v = ("a gathering limit: the words sit on the patches and fail to reach the report" if w8["anywhere"][0] - w8["report"][0] >= 0.25 and w8["absent_anywhere"][0] < 0.3
               else "an entry limit: the items missing at the report are readable nowhere on the card" if w8["anywhere"][0] - w8["report"][0] < 0.1 else "between")
    print("picture, share of present shapes readable: " + "  ".join(f"k={k}: own {where[k]['own'][0]:.2f} anywhere {where[k]['anywhere'][0]:.2f} report {where[k]['report'][0]:.2f} (absent somewhere {where[k]['absent_anywhere'][0]:.2f})" for k in B_KS) + f"\n-> {where_v}  [{time.time() - t0:.0f}s]")
    # (b) the additive ceiling, text and picture
    blank = {}
    _, Rb = mm_forward([SB.prompt.iloc[0]["picture"]], [Image.new("RGB", (MM_W, MM_H), (238, 238, 238))], LAYERS)
    blank["picture"] = {L: Rb[L][0, 0].numpy() for L in LAYERS}
    _, Rb = mm_forward([chat([("user", "Here is a list: an item.\n" + B_ASK)])], None, LAYERS)
    blank["text"] = {L: Rb[L][0, 0].numpy() for L in LAYERS}
    del Rb
    additive = {}
    for rend in ("text", "picture"):
        one = SB[SB.k == 1]
        vec = {s: {L: B_REP[rend][L][one.index[[sh[0] == s for sh in one.shapes]]].mean(0) - blank[rend][L] for L in LAYERS} for s in B_SHAPES if any(sh[0] == s for sh in one.shapes)}
        additive[rend] = {}
        for k in B_KS:
            dk = SB[(SB.k == k).values & np.array([all(s in vec for s in sh) for sh in SB.shapes])]        # scenes whose shapes all have a single-item vector
            assert len(dk) >= 6, f"too few k = {k} scenes with single-item vectors for every shape ({len(dk)})"
            pres = present[dk.index.values]
            H = {L: np.stack([blank[rend][L] + sum(vec[s][L] for s in sh) for sh in dk.shapes]) for L in LAYERS}
            rs = b_ranks(H, [B_SID[s] for s in B_SHAPES], jac=True).min(1) <= B_TOP
            rm = B_BEST[rend][dk.index.values] <= B_TOP
            additive[rend][k] = dict(sum=boot_ci((rs & pres).sum(1)), sum_absent=boot_ci((rs & ~pres).sum(1)), measured=boot_ci((rm & pres).sum(1)), n=int(len(dk)))
        gaps = [additive[rend][k]["measured"][0] - additive[rend][k]["sum"][0] for k in B_KS]
        additive[rend]["verdict"] = ("geometric: the sum of single-item states predicts the count" if all(abs(g) <= 0.5 for g in gaps) else "the model drops items: the count is below the additive ceiling" if gaps[-1] <= -1
                                     else "a code: the count is above the additive ceiling" if gaps[-1] >= 1 else "between")
        print(f"{rend:9s} measured vs sum: " + "  ".join(f"k={k}: {additive[rend][k]['measured'][0]:.1f} vs {additive[rend][k]['sum'][0]:.1f}" for k in B_KS) + f"\n-> {additive[rend]['verdict']}")
    resB2 = dict(where={int(k): v for k, v in where.items()}, where_verdict=where_v, additive={r: {str(k): v for k, v in additive[r].items()} for r in additive}, seconds=time.time() - t0)
    save(resB2, "eB2_results")
    # --- figure
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(wspace=0.7, width_ratios=[1, 1.5, 1]))
    c = B_COLOR["picture"]
    y1 = ws_band(ax[0], B_KS, [where[k]["anywhere"] for k in B_KS], c)
    y2 = ws_band(ax[0], B_KS, [where[k]["own"] for k in B_KS], c, ls=":")
    y3 = ws_band(ax[0], B_KS, [where[k]["report"] for k in B_KS], c, ls="--")
    ax[0].plot(B_KS, [where[k]["absent_anywhere"][0] for k in B_KS], "-", color="0.6", lw=0.9)
    ws_end_labels(ax[0], [(y1, "anywhere on the card", c), (y2, "on its own patches", c), (y3, "at the report", c), (where[B_KS[-1]]["absent_anywhere"][0], "absent shape, somewhere", "0.5")], 0.075)
    ax[0].set(xlabel=f"items in the picture, k (n = {where[B_KS[-1]]['n_items']} items at k = {B_KS[-1]})", ylabel=f"share of present shapes readable (J-lens top-{B_TOP}, any read layer)", xticks=B_KS, ylim=(0, 1.04))
    ax[1].imshow(np.asarray(example["img"]), extent=(0, MM_W, MM_H, 0))
    hm = ax[1].imshow(example["heat"], cmap="magma", alpha=0.55, extent=(0, MM_W, MM_H, 0), vmin=-np.log10(5000), vmax=0)
    cb = fig.colorbar(hm, ax=ax[1], orientation="horizontal", fraction=0.06, pad=0.06, shrink=0.8)
    cb.set_label("best present shape at the token, -log10 rank (0 = top)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    for x in range(0, MM_W + 1, 32):
        ax[1].axvline(x, color="1", lw=0.3, alpha=0.5)
    for y in range(0, MM_H + 1, 32):
        ax[1].axhline(y, color="1", lw=0.3, alpha=0.5)
    ax[1].set(xticks=[], yticks=[], title=f"one k = {B_KS[-1]} card, the lens at each vision token")
    ax[1].title.set_fontsize(9)
    ends = []
    for rend in ("text", "picture"):
        cc = B_COLOR[rend]
        ym = ws_band(ax[2], B_KS, [additive[rend][k]["measured"] for k in B_KS], cc)
        ax[2].plot(B_KS, [additive[rend][k]["sum"][0] for k in B_KS], ":", color=cc, lw=2)
        ends.append((ym, f"{rend}, measured", cc))
        ends.append((additive[rend][B_KS[-1]]["sum"][0], f"{rend}, sum of singles", cc))
    ax[2].plot(B_KS, B_KS, color="0.8", lw=1, ls="-.")
    ws_end_labels(ax[2], ends, 0.45)
    ax[2].set(xlabel="items in the scene, k", ylabel=f"present shapes in the J-lens top-{B_TOP} at the report\n(solid: the real state, dotted: the k single-item states summed)", xticks=B_KS, ylim=(0, B_KS[-1] + 0.6))
    fig.suptitle(f"Picture at k = {B_KS[-1]}: present shapes readable somewhere on the card {w8['anywhere'][0]:.0%}, on their own patches {w8['own'][0]:.0%}, at the report {w8['report'][0]:.0%} ({where_v.split(':')[0]}). "
                 f"Additive ceiling: text {additive['text']['verdict'].split(':')[0]}, picture {additive['picture']['verdict'].split(':')[0]}", fontsize=9.5, y=1.0)
    ws_finish(fig, "figB2_collapse", "B2 · the lens at every vision token, and the sum of single-item states")

# %% ---------------------------------------------------------------- B3: carried. What the model uses from a scene it can no longer see, against what the lens read
# Run after B1. The scene is replaced by a blank frame of the same rendering (a blank card, a list of "an item") and the model's state at the
# receipt positions (the question tokens and the report position, every layer) is patched in from the real scene. The second turn then asks
# "Is there a {shape}?" from that state alone. Items answered correctly were carried. Per present item, the lens's read at the report (B1)
# and the carry cross into four shares: carried and read, carried without a word (the bound), a word without carry, neither. Arms: the
# blank frame alone (the floor), the receipt at every layer (carried), the receipt at the report position alone, and at the read layers for
# k in {4, 8}: the receipt's lens-readable part (top-256 right singular directions of W_U J_L), a random 256-dimensional part (the
# dimension-matched control), and the complement. Rules: the k = 1 receipt carries its item on at least 0.8 of scenes and the blank frame
# alone has |d'| under 0.3, else the cell is uninterpretable. A bound exists when the silent share at k = 8 has a 95% lower bound at or
# above 0.10. The lens-rank AUROC for carry is reported against a permutation null. Carried content sits outside the lens's span when
# the complement arm's d' is at least 0.2 above both the lens-part arm's and the random part's, inside it when the lens part leads by 0.2.
B_ALL = list(range(FINAL + 1))
B_R, B_SPLIT_KS, B_SPLIT_RENDS = 256, [4, 8], ["text", "picture"]
B_CARRY, B_FLOOR, B_SILENT, B_PART = 0.80, 0.30, 0.10, 0.20
_B_WUTWU = {}
t0 = time.time()


def b_orth(V):
    Qm, _ = torch.linalg.qr(torch.as_tensor(np.asarray(V), dtype=torch.float32).T)
    err = float((Qm.T @ Qm - torch.eye(Qm.shape[1])).abs().max())
    assert err < 1e-3, f"projector is not orthogonal: {err}"
    return Qm.numpy()


@torch.no_grad()
def b_lens_subspace(L, r):
    """Q [d, r], the top-r right singular directions of W_U J_L (largest readout gain), from the Gram of W_U in chunks"""
    if "G" not in _B_WUTWU:
        Gm = torch.zeros(U_OUT.shape[1], U_OUT.shape[1], device=DEV)
        for s in range(0, U_OUT.shape[0], 8192):
            w = U_OUT[s:s + 8192].float().to(DEV)
            Gm += w.T @ w
        _B_WUTWU["G"] = Gm
    if L not in _B_WUTWU:
        M = JAC[L].float().T @ _B_WUTWU["G"] @ JAC[L].float()
        _B_WUTWU[L] = torch.linalg.eigh(M)[1].cpu().numpy()
    return b_orth(np.ascontiguousarray(_B_WUTWU[L][:, -r:][:, ::-1].T))   # strongest direction first, so Q[:, :k] is the top-k for every k <= r (14 Oct, change #24)


@torch.no_grad()
def b_assert_descending(L, Q, G=None):
    """the readout gain q^T (J^T W_U^T W_U J) q must not rise along the columns of Q (guards the slicing Q[:, :k] of a larger basis)"""
    Jm = (JAC if G is None else G)[L].float()
    Qd = torch.as_tensor(Q, dtype=torch.float32, device=DEV)
    g = ((_B_WUTWU["G"] @ (Jm @ Qd)) * (Jm @ Qd)).sum(0)
    assert g[0] >= g[-1] and g[: len(g) // 2].mean() >= g[len(g) // 2:].mean(), f"subspace columns are not in descending readout gain at layer {L}: {g[:3].tolist()} ... {g[-3:].tolist()}"


def b_random_subspace(d, r, seed):
    return b_orth(np.random.default_rng(seed).normal(0, 1, (r, d)))


@torch.no_grad()
def b_unembed_subspace(r):
    """Q [d, r], the top-r right singular directions of W_U alone (no Jacobian): the output-sensitivity-matched control for the J-lens
    subspace — what the unembedding amplifies most, the same at every layer. Strongest first (change #30)"""
    if "G" not in _B_WUTWU:
        b_lens_subspace(LAYERS[0], r)
    if "U" not in _B_WUTWU:
        _B_WUTWU["U"] = torch.linalg.eigh(_B_WUTWU["G"])[1].cpu().numpy()
    return b_orth(np.ascontiguousarray(_B_WUTWU["U"][:, -r:][:, ::-1].T))


def b_receipt_hooks(layers, REC, src, n_after, n_receipt, Q=None, part="all"):
    """hooks(enc, idx) -> [(L, fn)]. Row i of a batch receives REC[L][src[i]][-n_receipt:] on the n_receipt positions ending n_after[i] before the
    end of its row. With Q (per layer), part="lens" replaces only Q Q^T h, part="complement" only h - Q Q^T h. Rows are left padded"""
    src, n_after = np.asarray(src), np.asarray(n_after)

    def hooks(enc, idx):
        b, s = enc["input_ids"].shape
        idx = np.asarray(idx)
        end = torch.full((b,), s - 1, device=DEV) - torch.as_tensor(n_after[idx], device=DEV)
        P = end[:, None] - torch.arange(n_receipt - 1, -1, -1, device=DEV)[None]
        rows = torch.arange(b, device=DEV)[:, None].expand(-1, n_receipt)
        out_ = []
        for L in layers:
            D = torch.as_tensor(np.asarray(REC[L])[src[idx]][:, -n_receipt:], device=DEV, dtype=torch.float32)
            Qd = None if Q is None else torch.as_tensor(Q[L], device=DEV, dtype=torch.float32)

            def fn(mod, inp, out, D=D, Qd=Qd):
                h = residual_of(out)
                if h.shape[1] == 1:
                    return out
                x = h.float()
                v = x[rows, P]
                assert v.shape == D.shape, (v.shape, D.shape)
                if part == "all":
                    v = D
                elif part == "lens":
                    v = v - (v @ Qd) @ Qd.T + (D @ Qd) @ Qd.T
                else:
                    v = (v @ Qd) @ Qd.T + (D - (D @ Qd) @ Qd.T)
                x[rows, P] = v
                x = x.to(h.dtype)
                return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
            out_.append((L, fn))
        return out_
    hooks.part, hooks.n_receipt, hooks.layers, hooks.src = part, n_receipt, list(layers), src
    return hooks


def b_receipt_len(prompt):
    """tokens from the start of the question to the end of the prompt, on a token boundary (the newline before the question may join the receipt)"""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    q = prompt.rindex(B_ASK)
    for cut in (q, q - 1):
        pre = tokenizer.encode(prompt[:cut], add_special_tokens=False)
        if ids[:len(pre)] == pre:
            return len(ids) - len(pre)
    raise AssertionError("the question does not start on a token boundary")


def b_where_last(k):
    return lambda ids: list(range(len(ids) - k, len(ids)))


def b_boot_scene(values, scenes, n=2000, seed=0):
    """mean with a bootstrap over scenes (rows sharing a scene are resampled together)"""
    values, scenes = np.asarray(values, float), np.asarray(scenes)
    u = np.unique(scenes)
    by = {g: values[scenes == g] for g in u}
    rng = np.random.default_rng(seed)
    bs = [np.concatenate([by[g] for g in rng.choice(u, len(u))]).mean() for _ in range(n)]
    return float(values.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def b_score(Q, p):
    """hit rate, false-alarm rate (scene bootstrap) and d' over query rows"""
    pr = Q.present.values
    return dict(hit=b_boot_scene((p[pr] >= 0.5), Q.scene.values[pr]), fa=b_boot_scene((p[~pr] >= 0.5), Q.scene.values[~pr]), dprime=ws_dprime(p[pr], p[~pr]), n=int(len(Q)))


def b_blank_prompt(rend, k):
    if rend == "text":
        return chat([("user", "Here is a list: " + ", ".join(["an item"] * k) + f".\n{B_ASK}")])
    if rend == "text zh":
        return chat([("user", "这是一个列表：" + "，".join(["一个物品"] * k) + f"。\n{B_ASK}")])
    return mm_chat([("user", B_ASK)])


# --- reads
if not B_OK_RUN:
    print("B3 skipped")
    resB3 = {}
else:
    resB3 = dict(by_rendering={}, split={})
    B_BLANK_IMG = Image.new("RGB", (MM_W, MM_H), (238, 238, 238))
    B_REC, B_PY = {}, {}
    for rend in B_RENDERINGS:
        if not resB1["by_rendering"][rend]["seen"]:
            print(f"{rend}: not seen at k = 1, skipped")
            continue
        texts = [p[rend] for p in SB.prompt]
        imgs = SB.img.tolist() if rend == "picture" else None
        nrec = b_receipt_len(texts[0])
        assert all(b_receipt_len(t) == nrec for t in texts[::13]), "the receipt length differs across scenes"
        blanks = [b_blank_prompt(rend, int(k)) for k in SB.k]
        assert all(b_receipt_len(t) == nrec for t in blanks[::13]), "the blank frame's receipt differs in length"
        assert tokenizer.encode(blanks[0], add_special_tokens=False)[-nrec:] == tokenizer.encode(texts[0], add_special_tokens=False)[-nrec:], "the receipt tokens differ between the scene and the blank frame"
        _, RR = mm_forward(texts, imgs, B_ALL, where=b_where_last(nrec))
        B_REC[rend] = {L: RR[L].half().numpy() for L in B_ALL}
        assert B_REC[rend][B_ALL[0]].shape[:2] == (len(SB), nrec)
        del RR
        Q = pd.DataFrame([dict(scene=i, k=int(SB.k.iloc[i]), shape=s, present=s in SB.shapes.iloc[i]) for i in range(len(SB)) for s in SB.ask.iloc[i]])
        t2 = [b_turn2(blanks[i], s) for i, s in zip(Q.scene, Q["shape"])]
        na = np.array([n_tokens(t) - n_tokens(blanks[i]) for t, i in zip(t2, Q.scene)])
        assert (na > 0).all()
        bimgs = [B_BLANK_IMG] * len(Q) if rend == "picture" else None
        P = {"full scene": B_USE[rend]}                                                       # B1's full-scene p(yes), the same rows in the same order
        assert len(P["full scene"]) == len(Q)
        keyed = Q.assign(key=[f"{k}|{s}" for k, s in zip(Q.k, Q["shape"])]).drop_duplicates("key")
        pb, _ = ws_probs([b_turn2(blanks[i], s) for i, s in zip(keyed.scene, keyed["shape"])], [B_BLANK_IMG] * len(keyed) if rend == "picture" else None, [YES_IDS, NO_IDS])
        look = dict(zip(keyed.key, pb[:, 0] / np.maximum(pb.sum(1), 1e-9)))
        P["blank frame"] = np.array([look[f"{k}|{s}"] for k, s in zip(Q.k, Q["shape"])])
        for name, nr in (("receipt, every layer", nrec), ("receipt at the report position", 1)):
            pp, mass = ws_probs(t2, bimgs, [YES_IDS, NO_IDS], hooks=b_receipt_hooks(B_ALL, B_REC[rend], Q.scene.values, na, nr))
            P[name] = pp[:, 0] / np.maximum(pp.sum(1), 1e-9)
        B_PY[rend] = (Q, P, t2, na, bimgs, nrec)
        pr = Q.present.values
        best = np.array([B_BEST[rend][i, B_SHAPES.index(s)] for i, s in zip(Q.scene, Q["shape"])])
        # the lens at every receipt position, every read layer: a word anywhere in the patched state counts as read
        n_sc = len(SB)
        rec_all = b_ranks({L: B_REC[rend][L][:, :, :].reshape(n_sc * nrec, -1).astype(np.float32) for L in LAYERS}, [B_SID[s] for s in B_SHAPES], jac=True).min(1).reshape(n_sc, nrec, len(B_SHAPES)).min(1)
        best_any = np.array([rec_all[i, B_SHAPES.index(s)] for i, s in zip(Q.scene, Q["shape"])])
        read, read_any, carried = best <= B_TOP, best_any <= B_TOP, P["receipt, every layer"] >= 0.5
        row = dict(by_k={}, n_receipt=int(nrec))
        for k in B_KS:
            sel = (Q.k == k).values
            qk = Q[sel]
            r = {name: b_score(qk, P[name][sel]) for name in P}
            m = sel & pr
            rd, ca, sc = read[m], carried[m], -np.log10(best[m])
            r["shares"] = {nm: b_boot_scene(v, Q.scene.values[m]) for nm, v in (("carried_read", ca & rd), ("carried_unread", ca & ~rd), ("read_uncarried", ~ca & rd), ("neither", ~ca & ~rd))}
            ra_ = read_any[m]
            r["silent_anywhere_in_receipt"] = b_boot_scene(ca & ~ra_, Q.scene.values[m])          # carried, and no word at any receipt position
            r["carried_items"] = b_boot_scene(k * carried[m], Q.scene.values[m])
            r["lens_predicts_carry"] = auroc_ci(sc[ca], sc[~ca]) if ca.any() and (~ca).any() else (np.nan, np.nan, np.nan)
            row["by_k"][k] = r
        c1 = row["by_k"][1]
        ok = c1["receipt, every layer"]["hit"][0] >= B_CARRY and abs(c1["blank frame"]["dprime"][0]) < B_FLOOR
        pooled = pr & (Q.k.values >= 2)
        sc, ca = -np.log10(best[pooled]), carried[pooled]
        au = auroc_ci(sc[ca], sc[~ca]) if ca.any() and (~ca).any() else (np.nan, np.nan, np.nan)
        rng_p = np.random.default_rng(1)
        null = [auroc(sc[perm], sc[~perm]) for perm in (rng_p.permutation(ca) for _ in range(500))]
        row.update(control_ok=bool(ok), lens_predicts_carry_pooled=au, permutation_95=float(np.percentile(null, 95)), K_carried=b_capacity(B_KS, {k: k * carried[(Q.k == k).values & pr] for k in B_KS})[0])
        resB3["by_rendering"][rend] = row
        b8 = row["by_k"][B_KS[-1]]
        print(f"{rend:9s} control {'ok' if ok else 'FAILED'} (k = 1 receipt hits {c1['receipt, every layer']['hit'][0]:.2f}, blank d' {c1['blank frame']['dprime'][0]:.2f})  at k = {B_KS[-1]}: "
              f"full {b8['full scene']['hit'][0]:.2f} receipt {b8['receipt, every layer']['hit'][0]:.2f} report pos {b8['receipt at the report position']['hit'][0]:.2f} blank {b8['blank frame']['hit'][0]:.2f}  "
              f"carried {b8['carried_items'][0]:.1f} items, read {resB1['by_rendering'][rend]['by_k'][B_KS[-1]]['present'][0]:.1f}, silent share {b_fmt(b8['shares']['carried_unread'])} (no word at any receipt position: {b8['silent_anywhere_in_receipt'][0]:.2f}), "
              f"lens->carry AUROC {au[0]:.2f} (permutation 95th {row['permutation_95']:.2f})  [{time.time() - t0:.0f}s]")
        if not ok:
            print(f"POSITIVE CONTROL FAILED for {rend}: the one-item receipt does not carry its item, or the blank frame answers on its own. This rendering's carried numbers are uninterpretable")
    # the part split at the read layers, with the dimension-matched random control
    Qs = {L: b_lens_subspace(L, B_R) for L in LAYERS}
    Qr = {L: b_random_subspace(Qs[L].shape[0], B_R, 100 + L) for L in LAYERS}
    for rend in B_SPLIT_RENDS:
        if rend not in B_PY or not resB3["by_rendering"][rend]["control_ok"]:
            continue
        Q, P, t2, na, bimgs, nrec = B_PY[rend]
        sel = Q.k.isin(B_SPLIT_KS).values
        Qsub, tsub, nsub = Q[sel], [t for t, s_ in zip(t2, sel) if s_], na[sel]
        bsub = [b for b, s_ in zip(bimgs, sel) if s_] if bimgs else None
        out = {}
        for name, part, Qd in (("receipt, read layers", "all", None), ("lens part, read layers", "lens", Qs), ("random part, read layers", "lens", Qr), ("complement, read layers", "complement", Qs)):
            pp, _ = ws_probs(tsub, bsub, [YES_IDS, NO_IDS], hooks=b_receipt_hooks(LAYERS, B_REC[rend], Qsub.scene.values, nsub, nrec, Qd, part))
            pp = pp[:, 0] / np.maximum(pp.sum(1), 1e-9)
            out[name] = {k: b_score(Qsub[Qsub.k == k], pp[(Qsub.k == k).values]) for k in B_SPLIT_KS}
        for name in ("full scene", "blank frame", "receipt, every layer", "receipt at the report position"):
            out[name] = {k: b_score(Qsub[Qsub.k == k], P[name][sel][(Qsub.k == k).values]) for k in B_SPLIT_KS}
        resB3["split"][rend] = out
        k8 = B_SPLIT_KS[-1]
        lp, rp, cp, ap = (out[n][k8]["dprime"][0] for n in ("lens part, read layers", "random part, read layers", "complement, read layers", "receipt, read layers"))
        where = ("outside the lens's span" if cp >= max(lp, rp) + B_PART else "inside the lens's span" if lp >= max(cp, rp) + B_PART else "in both parts or in neither alone")
        out["verdict"] = where
        print(f"{rend:9s} at k = {k8}: read-layer receipt d' {ap:.2f}, lens part {lp:.2f}, random part {rp:.2f}, complement {cp:.2f} -> carried content {where}")
    k8 = B_KS[-1]
    if "picture" in resB3["by_rendering"]:
        vp = resB3["by_rendering"]["picture"]["by_k"][k8]
        silent = vp["shares"]["carried_unread"]
        silent_any = vp["silent_anywhere_in_receipt"]
        bound = ("uninterpretable, the positive control failed" if not resB3["by_rendering"]["picture"]["control_ok"] else
                 "a bound: items are carried into the answer with no readable word at any receipt position" if silent_any[1] >= B_SILENT else
                 "a bound at the report only: the carried items have a word at an earlier receipt position" if silent[1] >= B_SILENT else "no bound at this load: what is carried is read")
        resB3["verdict"] = dict(picture_silent_k8=silent, picture_silent_anywhere_k8=silent_any, bound=bound)
        print(f"\npicture at k = {k8}: carried {vp['carried_items'][0]:.1f}, read {resB1['by_rendering']['picture']['by_k'][k8]['present'][0]:.1f}, silent share {b_fmt(silent)} -> {bound}")
    resB3["seconds"] = time.time() - t0
    save(resB3, "eB3_results")
    # --- figure
    rends = [r for r in B_RENDERINGS if r in resB3["by_rendering"]]
    fig, ax = plt.subplots(1, 3, figsize=(18.5, 5.0), gridspec_kw=dict(wspace=0.75, width_ratios=[1, 1.55, 0.9]))
    ends = []
    for rend in rends:
        row, c = resB3["by_rendering"][rend], B_COLOR[rend]
        ls = "-" if row["control_ok"] else ":"
        yc = ws_band(ax[0], B_KS, [row["by_k"][k]["carried_items"] for k in B_KS], c, ls=ls)
        ax[0].plot(B_KS, [resB1["by_rendering"][rend]["by_k"][k]["present"][0] for k in B_KS], "--", color=c, lw=1.2)
        ends.append((yc, f"{rend}, carried" + ("" if row["control_ok"] else " (control failed)"), c))
    ax[0].plot(B_KS, B_KS, color="0.75", lw=1, ls="-.")
    ax[0].set(xlabel=f"items in the scene, k (n = {B_PER_K} scenes per k)", ylabel="items carried into the answer from the receipt alone\n(dashed: items the lens reads at the report)", xticks=B_KS, ylim=(0, B_KS[-1] + 0.6))
    ws_end_labels(ax[0], ends, 0.45)
    names = ["full scene", "blank frame", "receipt, every layer", "receipt at the report position", "receipt, read layers", "lens part, read layers", "random part, read layers", "complement, read layers"]
    short = ["full\nscene", "blank\nframe", "receipt,\nall layers", "receipt,\nreport\nposition", "receipt,\nread\nlayers", "lens\npart", "random\npart", "comple-\nment"]
    w = 0.38
    srends = [r for r in B_SPLIT_RENDS if r in resB3["split"]]
    for j, rend in enumerate(srends):
        sp = resB3["split"][rend]
        vals = [sp[n][k8]["dprime"] for n in names]
        xs = np.arange(len(names)) + (j - 0.5) * w
        ax[1].bar(xs, [v[0] for v in vals], w, color=B_COLOR[rend], alpha=0.85)
        ax[1].errorbar(xs, [v[0] for v in vals], yerr=[[max(v[0] - v[1], 0) for v in vals], [max(v[2] - v[0], 0) for v in vals]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
        ax[1].text(0.98, 0.97 - 0.07 * j, rend, color=B_COLOR[rend], fontsize=8.5, transform=ax[1].transAxes, va="top", ha="right")
    if not srends:
        ax[1].text(0.5, 0.5, "split arms not run:\nthe carry control failed", ha="center", va="center", color="0.5", fontsize=9, transform=ax[1].transAxes)
        ax[1].set(xticks=[], yticks=[])
    if srends:
        ax[1].set(xticks=np.arange(len(names)), xticklabels=short, ylabel=f"d' for \"is there a {{shape}}\" at k = {k8}\n(n = {resB3['split'][srends[0]]['full scene'][k8]['n']} queries per bar, parts of {B_R} dimensions)")
        ax[1].tick_params(axis="x", labelsize=7)
        ax[1].axhline(0, color="0.6", lw=0.8)
    if "picture" in resB3["by_rendering"]:
        rowp = resB3["by_rendering"]["picture"]
        cols4, labels4 = [OKABE[1], OKABE[6], OKABE[5], "0.85"], ["carried and read", "carried, no word (the bound)", "word, not carried", "neither"]
        bottom = np.zeros(len(B_KS))
        items = []
        for key, cc, lab in zip(("carried_read", "carried_unread", "read_uncarried", "neither"), cols4, labels4):
            v = np.array([rowp["by_k"][k]["shares"][key][0] for k in B_KS])
            ax[2].bar(np.arange(len(B_KS)), v, 0.7, bottom=bottom, color=cc, alpha=0.9 if key != "neither" else 0.6)
            items.append((bottom[-1] + v[-1] / 2, lab, cc if key != "neither" else "0.5"))
            bottom += v
        ax[2].set(xticks=np.arange(len(B_KS)), xticklabels=[str(k) for k in B_KS], xlabel="items in the picture, k", ylabel="share of present items (picture, n = 4 per scene)", ylim=(0, 1.0), xlim=(-0.6, len(B_KS) - 0.4))
        ws_end_labels(ax[2], items, 0.07)
        fig.suptitle(f"At k = {k8} a picture's state carries {vp['carried_items'][0]:.1f} items into the answer and the lens reads {resB1['by_rendering']['picture']['by_k'][k8]['present'][0]:.1f} at the report: "
                     f"{silent[0]:.0%} of items carried without a word. Lens rank predicts carry: AUROC {rowp['lens_predicts_carry_pooled'][0]:.2f} (permutation 95th {rowp['permutation_95']:.2f})", fontsize=9.5, y=1.0)
    ws_finish(fig, "figB3_carried", "B3 · " + " ".join(r + ":" + ("ok" if resB3["by_rendering"][r]["control_ok"] else "failed") for r in rends) + " · " + resB3.get("verdict", {}).get("bound", ""))

# %% ---------------------------------------------------------------- B3s: carried from the scene state. The carry test moved to where the content is
# Run after B1 and B3's definitions. On Qwen3.5-4B the state at the question positions carries nothing into a later answer while the
# scene positions carry everything (CHANGES.md, 13 Oct: patching every position of the prompt reproduces the full scene, the scene
# positions alone do too, the receipt alone does not), so the carry test is run on the scene positions' state: the list tokens (text)
# or the 98 image tokens (picture), patched from the real run into a blank frame of the same length (a list of filler words, a card
# with nothing on it), the second turn asked from that state. Arms: full scene, blank frame, the scene state at every layer (the
# machinery control and the carried set), the scene state at the read layers alone, and at the read layers for k in B_SPLIT_KS the
# state's lens-readable part (top-256 right singular directions of W_U J_L), a random 256-dimensional part and the complement. Rules
# as B3: the k = 1 scene state carries on >= 0.8 of scenes and the blank frame has |d'| < 0.3, else uninterpretable; the silent share
# (carried and unread at the report) at k = 8 with a 95% lower bound >= 0.10 is a bound at the report; carried content sits outside
# the lens's span when the complement's d' leads the lens part and the random part by 0.2, inside when the lens part leads by 0.2.
B3S_RENDS = ["text", "picture"]
B3S_FILLER = " item"
t0 = time.time()


def b3s_text_blank(prompt):
    """the real list replaced by filler tokens: the blank tokenizes to the same length with identical tokens before and after the list.
    Returns the blank, and the token span [lo, hi) of the list in the real prompt"""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    head = prompt[:prompt.index("Here is a list:") + len("Here is a list:")]
    cut = prompt.index(".\n" + B_ASK)
    lo, hi = len(tokenizer.encode(head, add_special_tokens=False)), len(tokenizer.encode(prompt[:cut], add_special_tokens=False))
    assert ids[:hi] == tokenizer.encode(prompt[:cut], add_special_tokens=False), "the list does not end on a token boundary"
    for m in range(1, hi - lo + 4):
        blank = head + B3S_FILLER * m + prompt[cut:]
        bids = tokenizer.encode(blank, add_special_tokens=False)
        if len(bids) == len(ids) and bids[:lo] == ids[:lo] and bids[hi:] == ids[hi:]:
            return blank, lo, hi
    raise AssertionError("no filler count reproduces the real prompt's length")


# --- reads
if not B_OK_RUN:
    print("B3s skipped")
    resB3s = {}
else:
    resB3s = dict(by_rendering={}, split={})
    B_BLANK_IMG = Image.new("RGB", (MM_W, MM_H), (238, 238, 238))
    Qs = {L: b_lens_subspace(L, B_R) for L in LAYERS}
    Qr = {L: b_random_subspace(Qs[L].shape[0], B_R, 100 + L) for L in LAYERS}
    B3S_ARMS = [("scene state, every layer", B_ALL, None, "all", B_KS), ("scene state, read layers", LAYERS, None, "all", B_KS),
                ("lens part, read layers", LAYERS, Qs, "lens", B_SPLIT_KS), ("random part, read layers", LAYERS, Qr, "lens", B_SPLIT_KS), ("complement, read layers", LAYERS, Qs, "complement", B_SPLIT_KS)]
    for rend in B3S_RENDS:
        if not resB1["by_rendering"][rend]["seen"]:
            print(f"{rend}: not seen at k = 1, skipped")
            continue
        texts = [p[rend] for p in SB.prompt]
        imgs = SB.img.tolist() if rend == "picture" else None
        if rend == "text":
            blanks, spans = [], []
            for t in texts:
                b, lo, hi = b3s_text_blank(t)
                blanks.append(b)
                spans.append((lo, hi, n_tokens(t)))
        else:
            blanks, spans = list(texts), []
            for t, im in zip(texts, imgs):
                row_ids = mm_encode([t], [im])["input_ids"][0].tolist()
                spans.append((row_ids.index(MM_VS) + 1, row_ids.index(MM_VE), len(row_ids)))
        off_set = {n_all - hi for lo, hi, n_all in spans}
        assert len(off_set) == 1, f"the scene span ends at different offsets from the prompt's end: {off_set}"
        off_end = off_set.pop()
        Q = pd.DataFrame([dict(scene=i, k=int(SB.k.iloc[i]), shape=s, present=s in SB.shapes.iloc[i]) for i in range(len(SB)) for s in SB.ask.iloc[i]])
        t2 = [b_turn2(blanks[i], s) for i, s in zip(Q.scene, Q["shape"])]
        na = np.array([n_tokens(t) - n_tokens(blanks[i]) for t, i in zip(t2, Q.scene)])
        assert (na > 0).all()
        bimgs = [B_BLANK_IMG] * len(Q) if rend == "picture" else None
        P = {"full scene": B_USE[rend]}                                                       # B1's full-scene p(yes), the same rows in the same order
        assert len(P["full scene"]) == len(Q)
        pb, _ = ws_probs(t2, bimgs, [YES_IDS, NO_IDS])
        P["blank frame"] = pb[:, 0] / np.maximum(pb.sum(1), 1e-9)
        for name, *_ in B3S_ARMS:
            P[name] = np.full(len(Q), np.nan)
        for k in B_KS:
            sc = np.flatnonzero((SB.k == k).values)
            n_max = max(spans[i][1] - spans[i][0] for i in sc)
            _, RR = mm_forward([texts[i] for i in sc], [imgs[i] for i in sc] if imgs else None, B_ALL, where=b_where_last(off_end + n_max))
            REC = {L: RR[L][:, :n_max].half().numpy() for L in B_ALL}           # the scene span, right-aligned to its end; a shorter span reaches into the identical prefix
            del RR
            qsel = np.flatnonzero((Q.k == k).values)
            local = {int(s_): j for j, s_ in enumerate(sc)}
            src = np.array([local[int(Q.scene.iloc[q])] for q in qsel])
            for name, layers, Qd, part, ks in B3S_ARMS:
                if k not in ks:
                    continue
                pp, _ = ws_probs([t2[q] for q in qsel], [bimgs[q] for q in qsel] if bimgs else None, [YES_IDS, NO_IDS], hooks=b_receipt_hooks(layers, REC, src, na[qsel] + off_end, n_max, Qd, part))
                P[name][qsel] = pp[:, 0] / np.maximum(pp.sum(1), 1e-9)
            del REC
        pr = Q.present.values
        best = np.array([B_BEST[rend][i, B_SHAPES.index(s)] for i, s in zip(Q.scene, Q["shape"])])
        read, carried = best <= B_TOP, P["scene state, every layer"] >= 0.5
        row = dict(by_k={}, off_end=int(off_end))
        for k in B_KS:
            sel = (Q.k == k).values
            qk = Q[sel]
            r = {name: b_score(qk, P[name][sel]) for name in P if not np.isnan(P[name][sel]).all()}
            m = sel & pr
            rd, ca, sc_ = read[m], carried[m], -np.log10(best[m])
            r["shares"] = {nm: b_boot_scene(v, Q.scene.values[m]) for nm, v in (("carried_read", ca & rd), ("carried_unread", ca & ~rd), ("read_uncarried", ~ca & rd), ("neither", ~ca & ~rd))}
            r["carried_items"] = b_boot_scene(k * carried[m], Q.scene.values[m])
            r["carried_items_read_layers"] = b_boot_scene(k * (P["scene state, read layers"][m] >= 0.5), Q.scene.values[m])
            r["lens_predicts_carry"] = auroc_ci(sc_[ca], sc_[~ca]) if ca.any() and (~ca).any() else (np.nan, np.nan, np.nan)
            row["by_k"][k] = r
        c1 = row["by_k"][1]
        ok = c1["scene state, every layer"]["hit"][0] >= B_CARRY and abs(c1["blank frame"]["dprime"][0]) < B_FLOOR
        pooled = pr & (Q.k.values >= 2)
        sc_, ca = -np.log10(best[pooled]), carried[pooled]
        au = auroc_ci(sc_[ca], sc_[~ca]) if ca.any() and (~ca).any() else (np.nan, np.nan, np.nan)
        rng_p = np.random.default_rng(1)
        null = [auroc(sc_[perm], sc_[~perm]) for perm in (rng_p.permutation(ca) for _ in range(500))] if ca.any() and (~ca).any() else [np.nan]
        row.update(control_ok=bool(ok), lens_predicts_carry_pooled=au, permutation_95=float(np.nanpercentile(null, 95)), K_carried=b_capacity(B_KS, {k: k * carried[(Q.k == k).values & pr] for k in B_KS})[0])
        k8 = B_SPLIT_KS[-1]
        names_split = ("scene state, read layers", "lens part, read layers", "random part, read layers", "complement, read layers")
        ap, lp, rp, cp = (row["by_k"][k8][n]["dprime"][0] for n in names_split)
        where = ("uninterpretable, the control failed" if not ok else "outside the lens's span" if cp >= max(lp, rp) + B_PART else "inside the lens's span" if lp >= max(cp, rp) + B_PART else "in both parts or in neither alone")
        row["split_verdict"] = where
        resB3s["by_rendering"][rend] = row
        resB3s["split"][rend] = dict({n: {k: row["by_k"][k][n] for k in B_SPLIT_KS} for n in ("full scene", "blank frame", "scene state, every layer") + names_split}, verdict=where)
        b8 = row["by_k"][B_KS[-1]]
        print(f"{rend:9s} control {'ok' if ok else 'FAILED'} (k = 1 scene state hits {c1['scene state, every layer']['hit'][0]:.2f}, blank d' {c1['blank frame']['dprime'][0]:.2f})  at k = {B_KS[-1]}: "
              f"full {b8['full scene']['hit'][0]:.2f} scene state {b8['scene state, every layer']['hit'][0]:.2f} read layers {b8['scene state, read layers']['hit'][0]:.2f} blank {b8['blank frame']['hit'][0]:.2f}  "
              f"carried {b8['carried_items'][0]:.1f} items, read at the report {resB1['by_rendering'][rend]['by_k'][B_KS[-1]]['present'][0]:.1f}, silent share {b_fmt(b8['shares']['carried_unread'])}, "
              f"lens->carry AUROC {au[0]:.2f} (permutation 95th {row['permutation_95']:.2f})  [{time.time() - t0:.0f}s]")
        print(f"{rend:9s} at k = {k8}: read-layer scene state d' {ap:.2f}, lens part {lp:.2f}, random part {rp:.2f}, complement {cp:.2f} -> carried content {where}")
        if not ok:
            print(f"POSITIVE CONTROL FAILED for {rend}: the one-item scene state does not carry its item, or the blank frame answers on its own. This rendering's carried numbers are uninterpretable")
    k8 = B_KS[-1]
    if "picture" in resB3s["by_rendering"]:
        vp = resB3s["by_rendering"]["picture"]["by_k"][k8]
        silent = vp["shares"]["carried_unread"]
        bound = ("uninterpretable, the positive control failed" if not resB3s["by_rendering"]["picture"]["control_ok"] else
                 "a bound at the report: items carried by the scene state have no readable word at the report" if silent[1] >= B_SILENT else "no bound at this load: what is carried is read at the report")
        resB3s["verdict"] = dict(picture_silent_k8=silent, bound=bound, split={r: resB3s["split"][r]["verdict"] for r in resB3s["split"]})
        print(f"\npicture at k = {k8}: carried {vp['carried_items'][0]:.1f}, read at the report {resB1['by_rendering']['picture']['by_k'][k8]['present'][0]:.1f}, silent share {b_fmt(silent)} -> {bound}")
    resB3s["seconds"] = time.time() - t0
    save(resB3s, "eB3s_results")
    # --- figure
    rends = [r for r in B3S_RENDS if r in resB3s["by_rendering"]]
    fig, ax = plt.subplots(1, 3, figsize=(18.5, 5.0), gridspec_kw=dict(wspace=0.75, width_ratios=[1, 1.55, 0.9]))
    ends = []
    for rend in rends:
        row, c = resB3s["by_rendering"][rend], B_COLOR[rend]
        ls = "-" if row["control_ok"] else ":"
        yc = ws_band(ax[0], B_KS, [row["by_k"][k]["carried_items"] for k in B_KS], c, ls=ls)
        ax[0].plot(B_KS, [row["by_k"][k]["carried_items_read_layers"][0] for k in B_KS], ":", color=c, lw=1.2, alpha=0.8)
        ax[0].plot(B_KS, [resB1["by_rendering"][rend]["by_k"][k]["present"][0] for k in B_KS], "--", color=c, lw=1.2)
        ends.append((yc, f"{rend}, carried by the scene state" + ("" if row["control_ok"] else " (control failed)"), c))
        ends.append((resB1["by_rendering"][rend]["by_k"][B_KS[-1]]["present"][0], f"{rend}, read at the report", c))
    ax[0].plot(B_KS, B_KS, color="0.75", lw=1, ls="-.")
    ax[0].set(xlabel=f"items in the scene, k (n = {B_PER_K} scenes per k)", ylabel="items carried into the answer by the scene state alone\n(dotted: read layers only; dashed: items the lens reads at the report)", xticks=B_KS, ylim=(0, B_KS[-1] + 0.6))
    ws_end_labels(ax[0], ends, 0.45)
    names = ["full scene", "blank frame", "scene state, every layer", "scene state, read layers", "lens part, read layers", "random part, read layers", "complement, read layers"]
    short = ["full\nscene", "blank\nframe", "scene state,\nall layers", "scene state,\nread layers", "lens\npart", "random\npart", "comple-\nment"]
    w = 0.38
    k8 = B_SPLIT_KS[-1]
    for j, rend in enumerate(rends):
        sp = resB3s["split"][rend]
        vals = [sp[n][k8]["dprime"] for n in names]
        xs = np.arange(len(names)) + (j - 0.5) * w
        ax[1].bar(xs, [v[0] for v in vals], w, color=B_COLOR[rend], alpha=0.85)
        ax[1].errorbar(xs, [v[0] for v in vals], yerr=[[max(v[0] - v[1], 0) for v in vals], [max(v[2] - v[0], 0) for v in vals]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
        ax[1].text(0.98, 0.97 - 0.07 * j, f"{rend}: {sp['verdict']}", color=B_COLOR[rend], fontsize=8.5, transform=ax[1].transAxes, va="top", ha="right")
    if rends:
        ax[1].set(xticks=np.arange(len(names)), xticklabels=short, ylabel=f"d' for \"is there a {{shape}}\" at k = {k8}\n(n = {resB3s['split'][rends[0]]['full scene'][k8]['n']} queries per bar, parts of {B_R} dimensions at the read layers)")
        ax[1].tick_params(axis="x", labelsize=7)
        ax[1].axhline(0, color="0.6", lw=0.8)
    if "picture" in resB3s["by_rendering"]:
        rowp = resB3s["by_rendering"]["picture"]
        cols4, labels4 = [OKABE[1], "#3b3b3b", OKABE[5], "0.85"], ["carried and read at the report", "carried, unread at the report", "read, not carried", "neither"]
        bottom = np.zeros(len(B_KS))
        items = []
        for key, cc, lab in zip(("carried_read", "carried_unread", "read_uncarried", "neither"), cols4, labels4):
            v = np.array([rowp["by_k"][k]["shares"][key][0] for k in B_KS])
            ax[2].bar(np.arange(len(B_KS)), v, 0.7, bottom=bottom, color=cc, alpha=0.9 if key != "neither" else 0.6)
            items.append((bottom[-1] + v[-1] / 2, lab, cc if key != "neither" else "0.5"))
            bottom += v
        ax[2].set(xticks=np.arange(len(B_KS)), xticklabels=[str(k) for k in B_KS], xlabel="items in the picture, k", ylabel="share of present items (picture, n = 4 per scene)", ylim=(0, 1.0), xlim=(-0.6, len(B_KS) - 0.4))
        ws_end_labels(ax[2], items, 0.07)
        fig.suptitle(f"At k = {B_KS[-1]} the picture's own scene state carries {vp['carried_items'][0]:.1f} items into the answer and the lens reads {resB1['by_rendering']['picture']['by_k'][B_KS[-1]]['present'][0]:.1f} at the report: "
                     f"{silent[0]:.0%} carried and unread at the report. Carried content: " + ", ".join(f"{r} {resB3s['split'][r]['verdict']}" for r in rends), fontsize=9.5, y=1.0)
    ws_finish(fig, "figB3s_carried_scene", "B3s · " + " ".join(r + ":" + ("ok" if resB3s["by_rendering"][r]["control_ok"] else "failed") for r in rends) + " · " + resB3s.get("verdict", {}).get("bound", ""))

# %% ---------------------------------------------------------------- B4: natural pictures. The same curve and the same bound on COCO under the POPE protocol
# Run after B3. Photographs instead of icons: COCO val2017 images with 1 to 8 distinct annotated categories from the single-token subset of
# the 80 names, 20 images per k. The question "Which objects are in the picture?", the lens at the report (top-10, any read layer, absent
# categories as the null), the POPE yes or no questions with its three negative types (random, popular, adversarial), and the receipt
# patched into a grey frame of the same size for the carried set. POPE is diagnostic here because the objects have no word anywhere in the
# input, the negatives are adversarial by construction, and Qwen numbers on it are published. Downloads val2017 (about 1 GB) and the
# annotations (about 240 MB) into B4_DIR when absent. Rules as in B1 and B3, with the perception gate at hits >= 0.85 and false alarms <=
# 0.20 at k = 1 (photographs are harder than icons) and the carried control at hits >= 0.7.
import urllib.request, zipfile
B4_DIR = "/workspace/nayan/data/coco2017"
B4_N_PER_K, B4_KS = 20, [1, 2, 3, 4, 6, 8]
B4_SIZE = (448, 320)
B4_ASK = "Which objects are in the picture? Answer with the object names separated by commas."
B4_YESNO = "Is there a {obj} in the picture? Answer yes or no."
B4_HIT, B4_FA, B4_CARRY = 0.85, 0.20, 0.70
t0 = time.time()


def b4_fetch():
    os.makedirs(B4_DIR, exist_ok=True)
    for name, url in (("annotations_trainval2017.zip", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"), ("val2017.zip", "http://images.cocodataset.org/zips/val2017.zip")):
        path = f"{B4_DIR}/{name}"
        if not os.path.exists(path):
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, path)
    if not os.path.exists(f"{B4_DIR}/annotations/instances_val2017.json"):
        zipfile.ZipFile(f"{B4_DIR}/annotations_trainval2017.zip").extract("annotations/instances_val2017.json", B4_DIR)
    return json.load(open(f"{B4_DIR}/annotations/instances_val2017.json"))


def b4_letterbox(img):
    img = img.convert("RGB")
    w, h = img.size
    sc = min(B4_SIZE[0] / w, B4_SIZE[1] / h)
    im = img.resize((max(1, int(w * sc)), max(1, int(h * sc))))
    out = Image.new("RGB", B4_SIZE, (128, 128, 128))
    out.paste(im, ((B4_SIZE[0] - im.size[0]) // 2, (B4_SIZE[1] - im.size[1]) // 2))
    return out


def b4_turn2(prompt, obj):
    return prompt + B_OK + "<|im_end|>\n<|im_start|>user\n" + B4_YESNO.format(obj=obj) + "<|im_end|>\n" + B_ASSIST


def b4_receipt_len(prompt):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    q = prompt.rindex(B4_ASK)
    for cut in (q, q - 1):
        pre = tokenizer.encode(prompt[:cut], add_special_tokens=False)
        if ids[:len(pre)] == pre:
            return len(ids) - len(pre)
    raise AssertionError("the question does not start on a token boundary")


def b4_items(ann, rng):
    """images by k (distinct single-token categories present), the present names, and the three POPE negatives per image"""
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    vocab = [n for n in cats.values() if " " not in n and single_ids(ws_forms(n))]
    by_img = collections.defaultdict(set)
    for a in ann["annotations"]:
        if not a.get("iscrowd", 0) and cats[a["category_id"]] in vocab:
            by_img[a["image_id"]].add(cats[a["category_id"]])
    freq = collections.Counter(n for s in by_img.values() for n in s)
    co = collections.defaultdict(collections.Counter)
    for s in by_img.values():
        for a_ in s:
            for b_ in s:
                if a_ != b_:
                    co[a_][b_] += 1
    files = {im["id"]: im["file_name"] for im in ann["images"]}
    rows = []
    for k in B4_KS:
        pool = [i for i, s in by_img.items() if len(s) == k]
        print(f"k = {k}: {len(pool)} images with exactly {k} single-token categories, {min(B4_N_PER_K, len(pool))} used")
        for i in rng.choice(pool, min(B4_N_PER_K, len(pool)), replace=False):
            present = sorted(by_img[i])
            absent = [n for n in vocab if n not in present]
            random_neg = [absent[j] for j in rng.choice(len(absent), 2, replace=False)]
            popular = [n for n, _ in freq.most_common() if n in absent][:2]
            adv_counts = collections.Counter()
            for p_ in present:
                adv_counts.update({n: c for n, c in co[p_].items() if n in absent})
            adversarial = [n for n, _ in adv_counts.most_common(2)] or random_neg
            ask_present = [present[j] for j in rng.choice(k, min(4, k), replace=False)]
            rows.append(dict(k=k, image_id=int(i), file=files[int(i)], present=present, ask_present=ask_present, negatives={"random": random_neg, "popular": popular, "adversarial": adversarial}))
    return pd.DataFrame(rows), vocab


# --- reads
if not B_OK_RUN or not (globals().get("resB3") or globals().get("resB3s")):
    print("B4 skipped")
    resB4 = {}
else:
    ann = b4_fetch()
    rng4 = np.random.default_rng(4)
    N4, B4_VOCAB = b4_items(ann, rng4)
    B4_VID = {n: ws_ids(n) for n in B4_VOCAB}
    zf = zipfile.ZipFile(f"{B4_DIR}/val2017.zip")
    N4["img"] = [b4_letterbox(Image.open(zf.open(f"val2017/{f}"))) for f in N4.file]
    N4["prompt"] = [mm_chat([("user", B4_ASK)])] * len(N4)
    nrec = b4_receipt_len(N4.prompt.iloc[0])
    ids_a, ids_b = tokenizer.encode(N4.prompt.iloc[0], add_special_tokens=False), tokenizer.encode(b4_turn2(N4.prompt.iloc[0], "dog"), add_special_tokens=False)
    assert ids_b[:len(ids_a)] == ids_a
    mm_contact_sheet([(f"k = {k}", N4[N4.k == k].img.iloc[0]) for k in B4_KS], 3, "figB4_stimuli")
    print(f"{len(N4)} photographs, {len(B4_VOCAB)} single-token categories, receipt of {nrec} tokens  [{time.time() - t0:.0f}s]")
    # entry at the report, and the receipt at every layer
    _, RR = mm_forward(N4.prompt.tolist(), N4.img.tolist(), B_ALL, where=b_where_last(nrec))
    REC4 = {L: RR[L].half().numpy() for L in B_ALL}
    report = {L: REC4[L][:, -1].astype(np.float32) for L in LAYERS}
    del RR
    best4 = b_ranks(report, [B4_VID[n] for n in B4_VOCAB], jac=True).min(1)
    present4 = np.array([[n in p for n in B4_VOCAB] for p in N4.present])
    inside4 = best4 <= B_TOP
    # the queries: present, and the three negative types
    Q4 = pd.DataFrame([dict(scene=i, k=int(r.k), obj=o, present=True, kind="present") for i, r in N4.iterrows() for o in r.ask_present] +
                      [dict(scene=i, k=int(r.k), obj=o, present=False, kind=kind) for i, r in N4.iterrows() for kind, objs in r.negatives.items() for o in objs])
    blank4 = Image.new("RGB", B4_SIZE, (128, 128, 128))
    t_full = [b4_turn2(N4.prompt.iloc[i], o) for i, o in zip(Q4.scene, Q4.obj)]
    na4 = np.array([n_tokens(t) - n_tokens(N4.prompt.iloc[i]) for t, i in zip(t_full, Q4.scene)])
    P4 = {}
    pp, mass4 = ws_probs(t_full, [N4.img.iloc[i] for i in Q4.scene], [YES_IDS, NO_IDS])
    P4["full scene"] = pp[:, 0] / np.maximum(pp.sum(1), 1e-9)
    keyed = Q4.drop_duplicates("obj")
    pb, _ = ws_probs([b4_turn2(N4.prompt.iloc[0], o) for o in keyed.obj], [blank4] * len(keyed), [YES_IDS, NO_IDS])
    look = dict(zip(keyed.obj, pb[:, 0] / np.maximum(pb.sum(1), 1e-9)))
    P4["blank frame"] = np.array([look[o] for o in Q4.obj])
    pp, _ = ws_probs(t_full, [blank4] * len(Q4), [YES_IDS, NO_IDS], hooks=b_receipt_hooks(B_ALL, REC4, Q4.scene.values, na4, nrec))
    P4["receipt, every layer"] = pp[:, 0] / np.maximum(pp.sum(1), 1e-9)
    pr4 = Q4.present.values
    bestq = np.array([best4[i, B4_VOCAB.index(o)] for i, o in zip(Q4.scene, Q4.obj)])
    read4, carried4 = bestq <= B_TOP, P4["receipt, every layer"] >= 0.5
    resB4 = dict(n_images=int(len(N4)), vocab=B4_VOCAB, yes_no_mass=float(mass4.mean()), by_k={}, by_negative={})
    for k in B4_KS:
        sel = (N4.k == k).values
        sq = (Q4.k == k).values
        r = dict(entry=dict(present=boot_ci((inside4 & present4)[sel].sum(1)), absent=boot_ci((inside4 & ~present4)[sel].sum(1))), n=int(sel.sum()))
        for name in P4:
            r[name] = b_score(Q4[sq], P4[name][sq])
        m = sq & pr4
        rd, ca, sc = read4[m], carried4[m], -np.log10(bestq[m])
        r["shares"] = {nm: b_boot_scene(v, Q4.scene.values[m]) for nm, v in (("carried_read", ca & rd), ("carried_unread", ca & ~rd), ("read_uncarried", ~ca & rd), ("neither", ~ca & ~rd))}
        r["carried_items"] = b_boot_scene(k * carried4[m], Q4.scene.values[m])
        resB4["by_k"][k] = r
    for kind in ("random", "popular", "adversarial"):
        sel = (Q4.kind == kind).values
        resB4["by_negative"][kind] = {name: dict(fa=b_boot_scene(P4[name][sel] >= 0.5, Q4.scene.values[sel])) for name in P4}
    cap4, _ = b_capacity(B4_KS, {k: (inside4 & present4)[(N4.k == k).values].sum(1) for k in B4_KS})
    c1 = resB4["by_k"][1]
    seen = c1["full scene"]["hit"][0] >= B4_HIT and c1["full scene"]["fa"][0] <= B4_FA
    control_ok = c1["receipt, every layer"]["hit"][0] >= B4_CARRY and abs(c1["blank frame"]["dprime"][0]) < B_FLOOR
    pooled = pr4 & (Q4.k.values >= 2)
    sc, ca = -np.log10(bestq[pooled]), carried4[pooled]
    au4 = auroc_ci(sc[ca], sc[~ca]) if ca.any() and (~ca).any() else (np.nan, np.nan, np.nan)
    k8 = B4_KS[-1]
    silent4 = resB4["by_k"][k8]["shares"]["carried_unread"]
    verdict4 = ("not seen: photographs fail the perception gate at k = 1" if not seen else "uninterpretable: the carried control failed" if not control_ok else
                "the bound replicates on photographs" if silent4[1] >= B_SILENT else "no bound on photographs at this load")
    resB4.update(capacity=cap4, seen=bool(seen), control_ok=bool(control_ok), lens_predicts_carry_pooled=au4, verdict=verdict4, seconds=time.time() - t0)
    save(resB4, "eB4_results")
    print(f"photographs: seen {'ok' if seen else 'NO'} (k = 1 hits {c1['full scene']['hit'][0]:.2f}, fa {c1['full scene']['fa'][0]:.2f}), carried control {'ok' if control_ok else 'FAILED'}, K_lens {b_fmt(cap4['K'])}, "
          f"at k = {k8}: in J-space {resB4['by_k'][k8]['entry']['present'][0]:.1f} of {k8}, carried {resB4['by_k'][k8]['carried_items'][0]:.1f}, silent share {b_fmt(silent4)}, lens->carry AUROC {au4[0]:.2f}  -> {verdict4}")
    print("false alarms by negative type (full scene): " + "  ".join(f"{kind} {resB4['by_negative'][kind]['full scene']['fa'][0]:.2f}" for kind in resB4["by_negative"]))
    # --- figure
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.0), gridspec_kw=dict(wspace=0.8))
    c = OKABE[3]
    y1 = ws_band(ax[0], B4_KS, [resB4["by_k"][k]["entry"]["present"] for k in B4_KS], c)
    ax[0].plot(B4_KS, [resB4["by_k"][k]["entry"]["absent"][0] for k in B4_KS], "--", color=c, lw=1.0, alpha=0.7)
    y2 = ws_band(ax[0], B4_KS, [resB4["by_k"][k]["carried_items"] for k in B4_KS], c, ls=":")
    ax[0].plot(B_KS, [resB1["by_rendering"]["picture"]["by_k"][k]["present"][0] for k in B_KS], "-", color=B_COLOR["picture"], lw=1.2, alpha=0.8)
    ax[0].plot(B4_KS, B4_KS, color="0.75", lw=1, ls="-.")
    ws_end_labels(ax[0], [(y1, "photos, in J-space", c), (y2, "photos, carried", c), (resB1["by_rendering"]["picture"]["by_k"][B_KS[-1]]["present"][0], "icons (B1)", B_COLOR["picture"])], 0.45)
    ax[0].set(xlabel=f"distinct objects in the photograph, k (n = {B4_N_PER_K} per k)", ylabel=f"present objects in the J-lens top-{B_TOP} at the report\n(dashed: absent objects, the null, dotted: carried)", xticks=B4_KS, ylim=(0, B4_KS[-1] + 0.6))
    kinds = ["random", "popular", "adversarial"]
    for j, name in enumerate(("full scene", "receipt, every layer", "blank frame")):
        vals = [resB4["by_negative"][kd][name]["fa"] for kd in kinds]
        xs = np.arange(len(kinds)) + (j - 1) * 0.27
        ax[1].bar(xs, [v[0] for v in vals], 0.27, color=[c, OKABE[6], "0.7"][j], alpha=0.85)
        ax[1].errorbar(xs, [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
        ax[1].text(0.98, 0.97 - 0.07 * j, name, color=[c, OKABE[6], "0.5"][j], fontsize=8.5, transform=ax[1].transAxes, va="top", ha="right")
    ax[1].set(xticks=np.arange(len(kinds)), xticklabels=[f"{kd}\nnegatives" for kd in kinds], ylabel="false alarms, p(yes) >= 0.5 for an absent object", ylim=(0, 1.0))
    ax[1].tick_params(axis="x", labelsize=8)
    cols4, labels4 = [c, OKABE[6], OKABE[5], "0.85"], ["carried and read", "carried, no word (the bound)", "word, not carried", "neither"]
    bottom = np.zeros(len(B4_KS))
    items = []
    for key, cc, lab in zip(("carried_read", "carried_unread", "read_uncarried", "neither"), cols4, labels4):
        v = np.array([resB4["by_k"][k]["shares"][key][0] for k in B4_KS])
        ax[2].bar(np.arange(len(B4_KS)), v, 0.7, bottom=bottom, color=cc, alpha=0.9 if key != "neither" else 0.6)
        items.append((bottom[-1] + v[-1] / 2, lab, cc if key != "neither" else "0.5"))
        bottom += v
    ax[2].set(xticks=np.arange(len(B4_KS)), xticklabels=[str(k) for k in B4_KS], xlabel="distinct objects in the photograph, k", ylabel="share of present objects asked about", ylim=(0, 1.0), xlim=(-0.6, len(B4_KS) - 0.4))
    ws_end_labels(ax[2], items, 0.07)
    fig.suptitle(f"Photographs: J-space holds {cap4['K'][0]:.1f} objects at the plateau, at k = {k8} the receipt carries {resB4['by_k'][k8]['carried_items'][0]:.1f} and the lens reads {resB4['by_k'][k8]['entry']['present'][0]:.1f}, "
                 f"{silent4[0]:.0%} carried without a word: {verdict4}", fontsize=9.5, y=1.0)
    ws_finish(fig, "figB4_photographs", f"B4 · COCO val2017 under POPE · {len(N4)} images · yes/no mass {mass4.mean():.2f}")

# %% ---------------------------------------------------------------- B5: the slope. The same numbers from the 4B and the 27B side by side
# Run after B4 on each model. Writes this model's summary and reads every summary in the run directory (and any paths in B_OTHER), then draws
# K_lens, K_carried and the silent share per rendering against model size. Two models is two points and is called a slope.
import glob as _glob
B_OTHER = list(globals().get("B_OTHER_PATHS", []))          # paths to eB5_summary_*.json files from the other models (run.py --other-summaries)
t0 = time.time()


def b_params(tag):
    for piece in str(tag).replace("_", "-").split("-"):
        if piece.endswith("B") and piece[:-1].replace(".", "").isdigit():
            return float(piece[:-1])
    return None


# --- reads
if not B_OK_RUN or not (globals().get("resB3") or globals().get("resB3s")):
    print("B5 skipped")
    resB5 = {}
else:
    k8 = B_KS[-1]
    summary = dict(model=MODEL_TAG, params=b_params(MODEL_TAG), renderings={})
    for rend in B_RENDERINGS:
        r1 = resB1["by_rendering"][rend]
        r3 = (resB3s if globals().get("resB3s") else resB3)["by_rendering"].get(rend)         # the scene-state carry (B3s) when it ran, else the receipt carry
        summary["renderings"][rend] = dict(seen=r1["seen"], K_lens=r1["capacity"]["K"], m_curve=r1["capacity"]["curve"], use_curve=[r1["by_k"][k]["hit"][0] for k in B_KS],
                                           K_carried=r3["K_carried"]["K"] if r3 else None, silent_k8=r3["by_k"][k8]["shares"]["carried_unread"] if r3 else None, control_ok=r3["control_ok"] if r3 else None,
                                           lens_predicts_carry=r3["lens_predicts_carry_pooled"] if r3 else None)
    if resB4:
        summary["renderings"]["photos"] = dict(seen=resB4["seen"], K_lens=resB4["capacity"]["K"], K_carried=(resB4["by_k"][k8]["carried_items"]), silent_k8=resB4["by_k"][k8]["shares"]["carried_unread"], control_ok=resB4["control_ok"])
    summary.update(verdict_B1=resB1["verdict"], copying=resB1["copying"], relevance=resB1.get("relevance"), where=resB2.get("where_verdict"), additive={r: resB2["additive"][r]["verdict"] for r in resB2.get("additive", {})}, bound=(resB3s if globals().get("resB3s") else resB3).get("verdict", {}).get("bound"), carry_source="scene state" if globals().get("resB3s") else "receipt")
    save(summary, f"eB5_summary_{MODEL_TAG}")
    models = {}
    for p in sorted(set(_glob.glob(f"{RUN_DIR}/eB5_summary_*.json") + list(B_OTHER))):
        try:
            d = json.load(open(p))
            models[d["model"]] = d
        except Exception as e:
            print(f"could not read {p}: {e}")
    models[MODEL_TAG] = summary
    order = sorted(models, key=lambda m: (models[m]["params"] is None, models[m]["params"] or 0))
    resB5 = dict(models=order)
    for rend in list(B_RENDERINGS) + ["photos"]:
        print(f"{rend:9s} " + "  ".join(f"{m}: lens {models[m]['renderings'][rend]['K_lens'][0]:.1f}" + (f", carried {models[m]['renderings'][rend]['K_carried'][0]:.1f}, silent {models[m]['renderings'][rend]['silent_k8'][0]:.2f}" if models[m]["renderings"][rend].get("K_carried") else "")
                                        for m in order if rend in models[m]["renderings"]))
    if len(order) >= 2:
        a, b = order[0], order[-1]
        gap = {m: models[m]["renderings"]["text"]["K_lens"][0] - models[m]["renderings"]["picture"]["K_lens"][0] for m in (a, b)}
        resB5["verdict"] = "scale closes the gap between text and picture" if gap[b] <= gap[a] - 1 else "scale widens the gap" if gap[b] >= gap[a] + 1 else "the gap is unchanged by scale"
        print(f"\ntext minus picture K_lens: {a} {gap[a]:.1f}, {b} {gap[b]:.1f} -> {resB5['verdict']}")
    else:
        print("\none model so far. Copy the other machine's eB5_summary_*.json into the run directory (or list it in B_OTHER) and rerun this cell for the slope")
    save(resB5, "eB5_results")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw=dict(wspace=0.5))
    xs = np.arange(len(order))
    colors = dict(B_COLOR, photos=OKABE[3])
    for rend in list(B_RENDERINGS) + ["photos"]:
        ok = [m for m in order if rend in models[m]["renderings"] and models[m]["renderings"][rend]["seen"]]
        if not ok:
            continue
        xi = [order.index(m) for m in ok]
        for a_, key in zip(ax, ("K_lens", "K_carried", "silent_k8")):
            v = [models[m]["renderings"][rend][key] for m in ok]
            if any(t is None for t in v):
                continue
            a_.errorbar(xi, [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="-o", color=colors[rend], lw=1.6, ms=5, capsize=2)
            a_.text(xi[-1] + 0.06, v[-1][0], rend, color=colors[rend], fontsize=8, va="center")
    for a_, lab in zip(ax, (f"K_lens: items in the lens top-{B_TOP} at the plateau", "K_carried: items carried into the answer at the plateau", f"silent share at k = {k8}: carried, no word")):
        a_.set(xticks=xs, xticklabels=[f"{m}\n({models[m]['params']:.0f}B)" if models[m]["params"] else m for m in order], ylabel=lab, xlim=(-0.4, len(order) - 0.3 + 0.9))
        a_.tick_params(axis="x", labelsize=8)
    ax[0].set_ylim(0, B_KS[-1] + 0.6)
    ax[1].set_ylim(0, B_KS[-1] + 0.6)
    ax[2].set_ylim(0, 1)
    fig.suptitle((f"{len(order)} model" + ("s. " if len(order) > 1 else ". ") + resB5.get("verdict", "")).strip(), fontsize=10, y=1.0)
    ws_finish(fig, "figB5_slope", f"B5 · {', '.join(order)}")

# %% ---------------------------------------------------------------- B6: the articulation frontier and the law in k
# Run after B5. No forwards: a figure and two fits from the saved results. For every rendering (and the photographs) and every model in the run
# directory, the curve of items said (in the lens at the report, B1) against items used (carried into the answer from the receipt, B3) as k
# grows, with the diagonal "says what it uses". Each curve is fitted with a saturating form n(k) = K (1 - exp(-k / kappa)) for said and for used,
# and the articulation ratio of a modality is K_said / K_used, the share of what the workspace uses that it can say. The law across size is
# the ratio per modality against parameters. Two models is two points and is called a slope.
from scipy.optimize import curve_fit
t0 = time.time()


def b_sat(k, K, kappa):
    return K * (1 - np.exp(-np.asarray(k, float) / max(kappa, 1e-3)))


def b_fit(ks, ys):
    try:
        (K, kappa), _ = curve_fit(b_sat, np.asarray(ks, float), np.asarray(ys, float), p0=(max(ys), 2.0), bounds=([0, 0.1], [20, 50]), maxfev=5000)
        return float(K), float(kappa)
    except Exception:
        return float(np.nan), float(np.nan)


# --- reads
if not B_OK_RUN or not (globals().get("resB3") or globals().get("resB3s")):
    print("B6 skipped")
    resB6 = {}
else:
    models = {}
    for p in sorted(set(_glob.glob(f"{RUN_DIR}/eB5_summary_*.json") + list(B_OTHER))):
        try:
            d = json.load(open(p))
            models[d["model"]] = d
        except Exception as e:
            print(f"could not read {p}: {e}")
    # this model's curves from the live results, so the cell works before B5 has been copied across machines
    live = dict(model=MODEL_TAG, params=b_params(MODEL_TAG), curves={})
    for rend in B_RENDERINGS:
        r3 = (resB3s if globals().get("resB3s") else resB3)["by_rendering"].get(rend)
        if r3 and resB1["by_rendering"][rend]["seen"]:
            live["curves"][rend] = dict(ks=B_KS, said=[resB1["by_rendering"][rend]["by_k"][k]["present"][0] for k in B_KS], used=[r3["by_k"][k]["carried_items"][0] for k in B_KS],
                                        said_ci=[resB1["by_rendering"][rend]["by_k"][k]["present"][1:] for k in B_KS], used_ci=[r3["by_k"][k]["carried_items"][1:] for k in B_KS])
    if resB4 and resB4["seen"] and resB4["control_ok"]:
        live["curves"]["photos"] = dict(ks=B4_KS, said=[resB4["by_k"][k]["entry"]["present"][0] for k in B4_KS], used=[resB4["by_k"][k]["carried_items"][0] for k in B4_KS],
                                        said_ci=[resB4["by_k"][k]["entry"]["present"][1:] for k in B4_KS], used_ci=[resB4["by_k"][k]["carried_items"][1:] for k in B4_KS])
    for m in models:
        models[m].setdefault("curves", {})
    models[MODEL_TAG] = dict(models.get(MODEL_TAG, {}), **live)
    order = sorted(models, key=lambda m: (models[m]["params"] is None, models[m]["params"] or 0))
    resB6 = dict(models=order, fits={})
    for m in order:
        resB6["fits"][m] = {}
        for rend, cv in models[m].get("curves", {}).items():
            Ks, ks_ = b_fit(cv["ks"], cv["said"])
            Ku, ku_ = b_fit(cv["ks"], cv["used"])
            resB6["fits"][m][rend] = dict(K_said=Ks, kappa_said=ks_, K_used=Ku, kappa_used=ku_, ratio=float(Ks / Ku) if Ku and Ku >= 0.5 else float("nan"))     # no ratio over a used-capacity under half an item
        print(f"{m}: " + "  ".join(f"{rend}: said K {f['K_said']:.1f}{' (no plateau within the loads)' if f['K_said'] > B_KS[-1] else ''} (kappa {f['kappa_said']:.1f}), used K {f['K_used']:.1f}{' (no plateau within the loads)' if f['K_used'] > B_KS[-1] else ''} (kappa {f['kappa_used']:.1f}), ratio {f['ratio']:.2f}" for rend, f in resB6["fits"][m].items()))
    # the summary for the other machine carries the curves too
    summ_path = f"{RUN_DIR}/eB5_summary_{MODEL_TAG}.json"
    try:
        d = json.load(open(summ_path))
        d["curves"] = live["curves"]
        json.dump(d, open(summ_path, "w"))
    except Exception as e:
        print(f"could not add the curves to {summ_path}: {e}")
    resB6["seconds"] = time.time() - t0
    save(resB6, "eB6_results")
    # --- figure
    colors = dict(B_COLOR, photos=OKABE[3])
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw=dict(wspace=0.6, width_ratios=[1.3, 1]))
    lim = B_KS[-1] + 0.6
    ax[0].plot([0, lim], [0, lim], color="0.75", lw=1, ls="-.")
    ax[0].text(lim - 0.2, lim - 0.55, "says what it uses", color="0.5", fontsize=7.5, ha="right", rotation=45)
    ends = []
    for mi, m in enumerate(order):
        for rend, cv in models[m].get("curves", {}).items():
            c = colors.get(rend, "0.3")
            xs, ys = cv["said"], cv["used"]
            ax[0].plot(xs, ys, "-" if m == order[-1] else "--", color=c, lw=1.8 if m == order[-1] else 1.2, alpha=0.95 if m == order[-1] else 0.6)
            ax[0].errorbar(xs, ys, xerr=[[x - ci[0] for x, ci in zip(xs, cv["said_ci"])], [ci[1] - x for x, ci in zip(xs, cv["said_ci"])]],
                           yerr=[[y - ci[0] for y, ci in zip(ys, cv["used_ci"])], [ci[1] - y for y, ci in zip(ys, cv["used_ci"])]], fmt="o", color=c, ms=3.5 if m == order[-1] else 2.5, lw=0.6, alpha=0.8 if m == order[-1] else 0.45)
            ends.append((ys[-1], xs[-1], f"{rend}, {m}" if len(order) > 1 else rend, c))
    ends = sorted(ends)
    ys_ = [e[0] for e in ends]
    for i in range(1, len(ys_)):
        ys_[i] = max(ys_[i], ys_[i - 1] + 0.32)
    for (y0, x0, text, c), y in zip(ends, ys_):
        ax[0].text(x0 + 0.1, y, text, color=c, fontsize=7.5, va="center")
    ax[0].set(xlabel="items said: present items in the lens top-10 at the report", ylabel="items used: present items carried into the answer from the receipt alone", xlim=(0, lim), ylim=(0, lim))
    ax[0].text(0.3, lim - 0.4, "each curve runs k = 1 to 8 along its length" + (" (dashed: the smaller model)" if len(order) > 1 else ""), color="0.4", fontsize=7.5)
    xs_m = np.arange(len(order))
    for rend in list(B_RENDERINGS) + ["photos"]:
        pts = [(order.index(m), resB6["fits"][m][rend]["ratio"]) for m in order if rend in resB6["fits"][m] and np.isfinite(resB6["fits"][m][rend]["ratio"])]
        if not pts:
            continue
        ax[1].plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=colors.get(rend, "0.3"), lw=1.6, ms=5)
        ax[1].text(pts[-1][0] + 0.06, min(pts[-1][1], 1.25), rend, color=colors.get(rend, "0.3"), fontsize=8, va="center")
    ax[1].axhline(1.0, color="0.75", lw=1, ls="-.")
    ax[1].set(xticks=xs_m, xticklabels=[f"{m}\n({models[m]['params']:.0f}B)" if models[m].get("params") else m for m in order], ylabel="articulation ratio, K_said / K_used (fitted plateaus)", xlim=(-0.4, len(order) - 0.3 + 0.9), ylim=(0, 1.3))
    ax[1].tick_params(axis="x", labelsize=8)
    fm = resB6["fits"][order[-1]]
    fig.suptitle("The articulation frontier. " + ", ".join(f"{rend}: says {f['K_said']:.1f} of the {f['K_used']:.1f} items it uses (ratio {f['ratio']:.2f})" for rend, f in fm.items()) + f" on {order[-1]}", fontsize=9.5, y=1.0)
    ws_finish(fig, "figB6_frontier", f"B6 · {', '.join(order)} · n(k) = K (1 - exp(-k / kappa)) fitted to the six loads")

# %% ---------------------------------------------------------------- B7: omni. One vocabulary through four channels on a model with a lens that hears
# Run on Gemma 4 E4B (or E2B): text, image, video and audio in, with the Neuronpedia J-lens for it loaded by the notebook's model and lens cells
# (model, tokenizer, hf_model, JAC, LAYERS, FINAL, DEV, BATCH, U_OUT, residual_of, YES_IDS, NO_IDS, chat). Independent of M0: this cell encodes its
# own media through the model's processor. Sixteen things that have a word, an emoji and a sound (ESC-50): a scene of k of them is rendered as
# an English list (text), emoji on one card (picture), one emoji per frame (video), the k sounds in sequence (sound), and the list read aloud
# (spoken, when espeak-ng is present). Three channels, and a lexical and a non-lexical rendering in two of them, so the question the cell
# decides is whether the articulation bound follows words or follows senses. The measures are B1's and B3's: items in the lens at the
# report, use, and items carried from the receipt into a blank frame (a blank card, blank frames, silence, a list of "an item").
# Rules. Perception and machinery as in B1 and B3 (hits >= 0.9 and false alarms <= 0.15 at k = 1 for the natural channels, carried control
# at 0.7). The bound follows words when the silent share at k = 8 has a 95% lower bound at or above 0.10 for picture and sound and an
# upper bound under 0.10 for text and spoken. The channel matters when sound and picture differ by 0.15 with intervals apart. Both can fail.
import io as _io, subprocess, zipfile as _zipfile, urllib.request as _urllib
B7_ITEMS = {"dog": ("🐕", "dog"), "cat": ("🐈", "cat"), "cow": ("🐄", "cow"), "pig": ("🐖", "pig"), "sheep": ("🐑", "sheep"), "frog": ("🐸", "frog"), "rooster": ("🐓", "rooster"),
            "rain": ("🌧", "rain"), "baby": ("👶", "crying_baby"), "train": ("🚆", "train"), "helicopter": ("🚁", "helicopter"), "siren": ("🚨", "siren"),
            "bell": ("🔔", "church_bells"), "airplane": ("✈", "airplane"), "fireworks": ("🎆", "fireworks"), "clock": ("⏰", "clock_alarm")}
B7_RENDERINGS = ["text", "spoken", "picture", "video", "sound"]
B7_LEXICAL = {"text": True, "spoken": True, "picture": False, "video": False, "sound": False}
B7_COLOR = {"text": OKABE[0], "spoken": OKABE[5], "picture": OKABE[1], "video": OKABE[2], "sound": OKABE[3]}
B7_KS, B7_PER_K = [1, 2, 3, 4, 6, 8], 16
B7_ASK = "Which things are present? Answer with their names in English, separated by commas."
B7_YESNO = "Is there a {item}? Answer yes or no."
B7_OK = "Okay."
B7_ESC_DIR, B7_FONT_PATH = "/workspace/nayan/data/esc50", "/workspace/nayan/data/NotoColorEmoji.ttf"
B7_SR, B7_CLIP_S, B7_GAP_S = 16000, 2.5, 0.3
B7_W, B7_H = 448, 224
B7_BATCH = 1 if BATCH == 1 else 8          # unbatched when the setup's padding check fell back to BATCH = 1 (Gemma 4: left padding moved letter probabilities by 0.03)
B7_RNG = np.random.default_rng(17)
t0 = time.time()


def b7_fetch_esc50():
    """ESC-50 from GitHub (about 600 MB): {category: [wav paths]} for the categories used"""
    os.makedirs(B7_ESC_DIR, exist_ok=True)
    z = f"{B7_ESC_DIR}/ESC-50-master.zip"
    if not os.path.exists(z):
        print("downloading ESC-50 ...")
        _urllib.urlretrieve("https://github.com/karolpiczak/ESC-50/archive/master.zip", z)
    zf = _zipfile.ZipFile(z)
    meta = [l.decode().strip().split(",") for l in zf.open("ESC-50-master/meta/esc50.csv").readlines()[1:]]
    want = {cat for _, cat in B7_ITEMS.values()}
    out = {cat: [] for cat in want}
    for row in meta:
        if row[3] in want:
            out[row[3]].append("ESC-50-master/audio/" + row[0])
    assert all(out[c] for c in want), f"ESC-50 categories missing: {[c for c in want if not out[c]]}"
    return zf, out


def b7_wav(zf, path):
    """mono float32 at B7_SR, the middle B7_CLIP_S seconds"""
    import soundfile as sf
    data, sr = sf.read(_io.BytesIO(zf.read(path)), dtype="float32")
    if data.ndim > 1:
        data = data.mean(1)
    if sr != B7_SR:
        idx = np.linspace(0, len(data) - 1, int(len(data) * B7_SR / sr))
        data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
    n = int(B7_CLIP_S * B7_SR)
    s = max(0, (len(data) - n) // 2)
    return data[s:s + n]


class B7Audio:
    """an audio scene: samples at B7_SR, with the items it holds (for the record only)"""
    def __init__(self, samples, items):
        self.samples, self.items = np.asarray(samples, np.float32), list(items)


def b7_sound(zf, esc, items, rng):
    parts = []
    for w in items:
        clip = b7_wav(zf, esc[B7_ITEMS[w][1]][int(rng.integers(len(esc[B7_ITEMS[w][1]])))])
        parts += [clip / max(np.abs(clip).max(), 1e-6) * 0.5, np.zeros(int(B7_GAP_S * B7_SR), np.float32)]
    return B7Audio(np.concatenate(parts), items)


def b7_silence(k):
    return B7Audio(np.zeros(int(k * (B7_CLIP_S + B7_GAP_S) * B7_SR), np.float32), [])


def b7_spoken(items):
    """the English list read by espeak-ng, or None when it is not installed"""
    text = "Here is a list: " + ", ".join(f"{'an' if w[0] in 'aeiou' else 'a'} {w}" for w in items) + "."
    try:
        import soundfile as sf
        wav = subprocess.run(["espeak-ng", "-v", "en-us", "-s", "150", "--stdout", text], capture_output=True, check=True).stdout
        data, sr = sf.read(_io.BytesIO(wav), dtype="float32")
        if data.ndim > 1:
            data = data.mean(1)
        idx = np.linspace(0, len(data) - 1, int(len(data) * B7_SR / sr))
        return B7Audio(np.interp(idx, np.arange(len(data)), data).astype(np.float32), items)
    except Exception:
        return None


def b7_font():
    if not os.path.exists(B7_FONT_PATH):
        try:
            _urllib.urlretrieve("https://github.com/googlefonts/noto-emoji/raw/main/fonts/NotoColorEmoji.ttf", B7_FONT_PATH)
        except Exception as e:
            print(f"could not fetch the emoji font: {e}")
    return ImageFont.truetype(B7_FONT_PATH, 109)          # the bitmap strike size of Noto Color Emoji


def b7_card(items, slots, rng, font):
    """emoji at jittered slots on a card, scaled to about 60 px"""
    img = Image.new("RGB", (B7_W, B7_H), (238, 238, 238))
    pos = [(56 + 112 * i + rng.uniform(-10, 10), 40 + 72 * j + rng.uniform(-6, 6)) for i in range(4) for j in range(3)]
    for w, sl in zip(items, slots):
        glyph = Image.new("RGBA", (136, 136), (0, 0, 0, 0))
        ImageDraw.Draw(glyph).text((0, 0), B7_ITEMS[w][0], font=font, embedded_color=True)
        glyph = glyph.resize((60, 60))
        x, y = pos[sl]
        img.paste(glyph, (int(x - 30), int(y - 30)), glyph)
    return img


def b7_chat(kind, k, text):
    """one user turn with k media entries of one kind (or none) then the text, through the model's processor template"""
    content = ([{"type": kind}] * k if kind else []) + [{"type": "text", "text": text}]
    return MM_PROC.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)


def b7_encode(texts, media):
    """processor call for one kind of media per batch: PIL images, lists of PIL frames (video), B7Audio, or None"""
    kw = dict(text=list(texts), return_tensors="pt", padding=True)
    m0 = media[0]
    if m0 is None:
        enc = MM_PROC.tokenizer(list(texts), return_tensors="pt", padding=True, add_special_tokens=False)
        return enc.to(DEV)
    if isinstance(m0, B7Audio):
        kw["audio"] = [m.samples for m in media]
        assert all(len(m.samples) <= 30 * B7_SR for m in media), "an audio scene is longer than the model's 30 s limit"
    elif isinstance(m0, (list, tuple)):
        kw["images"] = [f for m in media for f in m]        # the "video" prompt holds k image placeholders (b7_chat("image", k, ...)), so its k frames go in as k images, flat in prompt order
    else:
        kw["images"] = list(media)
    enc = MM_PROC(**kw)
    for key in ("pixel_values", "input_features", "pixel_values_videos"):
        if key in enc:
            enc[key] = enc[key].to(_dtype)
    return enc.to(DEV)


@torch.no_grad()
def b7_forward(texts, media, layers=(), where=None, hooks=None):
    """as mm_forward, for this cell's media. Batches of B7_BATCH in the given order (the caller groups by media kind and length)"""
    keep = set(layers) | {FINAL}
    out_logits, out_res = [], {L: [] for L in layers}
    for s in tqdm(range(0, len(texts), B7_BATCH), desc=f"{len(texts)} prompts", unit="batch", leave=False, dynamic_ncols=True, file=sys.__stderr__, disable=len(texts) <= 2 * B7_BATCH):
        idx = np.arange(s, min(s + B7_BATCH, len(texts)))
        enc = b7_encode([texts[i] for i in idx], [media[i] for i in idx])
        ids = enc["input_ids"]
        b, n = ids.shape
        pos = torch.as_tensor([(where(ids[j].tolist()) if where else []) + [n - 1] for j in range(b)], device=DEV)
        rows_ = torch.arange(b, device=DEV)[:, None]
        store = {}

        def capture(L):
            def f(mod, inp, out):
                store[L] = residual_of(out)[rows_, pos].float()
            return f
        handles = [model.layers[L].register_forward_hook(f) for L, f in (hooks(enc, idx) if hooks else [])] + [model.layers[L].register_forward_hook(capture(L)) for L in keep]
        try:
            kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
            hf_model(**kw, use_cache=False, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        out_logits.append(model.unembed(store[FINAL][:, -1]).float().cpu())
        for L in layers:
            out_res[L].append((store[L][:, :-1] if where else store[L][:, -1:]).cpu())
        torch.cuda.empty_cache()
    return torch.cat(out_logits), {L: torch.cat(out_res[L]) for L in layers}


def b7_probs(texts, media, hooks=None):
    pr = torch.softmax(b7_forward(texts, media, hooks=hooks)[0], -1)
    P = np.stack([sum(pr[:, t].numpy() for t in ids) for ids in (YES_IDS, NO_IDS)], 1)
    return P[:, 0] / np.maximum(P.sum(1), 1e-9), float(P.sum(1).mean())


def b7_turn2(prompt, item):
    """the first prompt continued with the template's own pieces: the fixed assistant turn, then the yes or no question, then the assistant prefix"""
    assert prompt.endswith(B7_ASSIST), "the prompt does not end with the template's assistant prefix"
    return prompt + B7_SUFFIX.replace(B7_YESNO.format(item="dog"), B7_YESNO.format(item=item))


def b7_receipt_len(prompt):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    q = prompt.rindex(B7_ASK)
    for cut in (q, q - 1):
        pre = tokenizer.encode(prompt[:cut], add_special_tokens=False)
        if ids[:len(pre)] == pre:
            return len(ids) - len(pre)
    raise AssertionError("the question does not start on a token boundary")


# --- reads
B7_OK_RUN = False
try:
    MM_PROC
except NameError:
    from transformers import AutoProcessor
    MM_PROC = AutoProcessor.from_pretrained(MODEL_NAME)
    MM_PROC.tokenizer.padding_side = "left"
if MM_PROC is None:
    print("B7 skipped: no processor")
    resB7 = {}
else:
    B7_WORDS = list(B7_ITEMS)
    _fe_sr = getattr(getattr(MM_PROC, "feature_extractor", None), "sampling_rate", None)
    assert _fe_sr in (None, B7_SR), f"the processor's audio rate is {_fe_sr}, the cell renders at {B7_SR}"
    B7_SID = {w: ws_ids(w) if "ws_ids" in globals() else (single_ids([w, " " + w, w.capitalize(), " " + w.capitalize()]) or [tokenizer.encode(" " + w, add_special_tokens=False)[0]]) for w in B7_WORDS}
    print("item words that are not single tokens (read as the first token):", [w for w in B7_WORDS if len(tokenizer.encode(" " + w, add_special_tokens=False)) > 1] or "none")
    # the template's turn pieces, from a two-turn render, so the continued prompt is a token prefix of the first
    one = b7_chat(None, 0, B7_ASK)
    two = MM_PROC.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": B7_ASK}]}, {"role": "assistant", "content": [{"type": "text", "text": B7_OK}]},
                                       {"role": "user", "content": [{"type": "text", "text": B7_YESNO.format(item="dog")}]}], tokenize=False, add_generation_prompt=True)
    B7_ASSIST = one[one.rindex(B7_ASK) + len(B7_ASK):]                       # the end of the user turn and the assistant prefix, the same in every prompt of this cell
    yes_dog = B7_YESNO.format(item="dog")
    if two.startswith(one):
        B7_SUFFIX = two[len(one):]                                           # the fixed assistant turn, the second user turn with "dog", the assistant prefix
    else:                                                                    # the template renders a middle assistant turn differently (Qwen drops the think block): build the continuation by hand
        B7_SUFFIX = B7_OK + two[two.index(B7_OK) + len(B7_OK):two.index(yes_dog)] + yes_dog + B7_ASSIST
        print("the template is not a prefix of its own continuation, the second turn is appended by hand from the template's pieces")
    ids1, ids2 = tokenizer.encode(one, add_special_tokens=False), tokenizer.encode(b7_turn2(one, "dog"), add_special_tokens=False)
    assert ids2[:len(ids1)] == ids1, "the continued prompt does not tokenize as prompt + continuation"
    # scenes
    specs = []
    singles = [B7_WORDS[i] for i in B7_RNG.permutation(len(B7_WORDS))]
    for k in B7_KS:
        for seed in range(B7_PER_K):
            items = [singles[seed % len(singles)]] if k == 1 else [B7_WORDS[i] for i in B7_RNG.choice(len(B7_WORDS), k, replace=False)]
            absent = [w for w in B7_WORDS if w not in items]
            ask = [items[i] for i in B7_RNG.choice(k, min(4, k), replace=False)] + [absent[i] for i in B7_RNG.choice(len(absent), 4, replace=False)]
            specs.append(dict(k=k, seed=seed, items=items, ask=ask, slots=[int(i) for i in B7_RNG.choice(12, k, replace=False)]))
    S7 = pd.DataFrame(specs)
    zf7, esc = b7_fetch_esc50()
    font = b7_font()
    media, prompts, blanks, bmedia = {}, {}, {}, {}
    for rend in B7_RENDERINGS:
        if rend == "spoken" and b7_spoken(["dog"]) is None:
            print("spoken rendering skipped: espeak-ng is not installed (apt-get install espeak-ng)")
            continue
        md, pr, bl, bm = [], [], [], []
        for i, r in S7.iterrows():
            rng = np.random.default_rng(100 * r.k + r.seed)
            blank_card = Image.new("RGB", (B7_W, B7_H), (238, 238, 238))
            if rend == "text":
                m_, b_ = None, None
                p_ = chat([("user", "Here is a list: " + ", ".join(f"{'an' if w[0] in 'aeiou' else 'a'} {w}" for w in r["items"]) + f".\n{B7_ASK}")])
                bp_ = chat([("user", "Here is a list: " + ", ".join(["an item"] * r.k) + f".\n{B7_ASK}")])
            elif rend == "picture":
                m_, b_ = b7_card(r["items"], r.slots, rng, font), blank_card
                p_ = bp_ = b7_chat("image", 1, B7_ASK)
            elif rend == "video":
                m_, b_ = [b7_card([w], [sl], rng, font) for w, sl in zip(r["items"], r.slots)], [blank_card] * r.k
                p_ = bp_ = b7_chat("image", r.k, B7_ASK)
            elif rend == "sound":
                m_, b_ = b7_sound(zf7, esc, r["items"], rng), b7_silence(r.k)
                p_ = bp_ = b7_chat("audio", 1, B7_ASK)
            else:
                m_, b_ = b7_spoken(r["items"]), b7_silence(1)
                p_ = bp_ = b7_chat("audio", 1, B7_ASK)
            md.append(m_)
            bm.append(b_)
            pr.append(p_)
            bl.append(bp_)
        media[rend], prompts[rend], blanks[rend], bmedia[rend] = md, pr, bl, bm
    present7 = np.array([[w in it for w in B7_WORDS] for it in S7["items"]])
    B7_ALL = list(range(FINAL + 1))
    resB7 = dict(by_rendering={}, n_scenes=int(len(S7)))
    for rend in media:
        nrec = b7_receipt_len(prompts[rend][0])
        assert all(b7_receipt_len(p) == nrec for p in prompts[rend][::9]) and b7_receipt_len(blanks[rend][0]) == nrec
        _, RR = b7_forward(prompts[rend], media[rend], B7_ALL, where=lambda ids, n=nrec: list(range(len(ids) - n, len(ids))))
        REC7 = {L: RR[L].half().numpy() for L in B7_ALL}
        del RR
        report = {L: REC7[L][:, -1].astype(np.float32) for L in LAYERS}
        best = b_ranks(report, [B7_SID[w] for w in B7_WORDS], jac=True).min(1)
        inside = best <= B_TOP
        Q = pd.DataFrame([dict(scene=i, k=int(S7.k.iloc[i]), item=w, present=w in S7["items"].iloc[i]) for i in range(len(S7)) for w in S7.ask.iloc[i]])
        t_full = [b7_turn2(prompts[rend][i], w) for i, w in zip(Q.scene, Q.item)]
        t_blank = [b7_turn2(blanks[rend][i], w) for i, w in zip(Q.scene, Q.item)]
        na = np.array([n_tokens(t) - n_tokens(blanks[rend][i]) for t, i in zip(t_blank, Q.scene)])
        P = {}
        P["full scene"], mass = b7_probs(t_full, [media[rend][i] for i in Q.scene])
        P["blank frame"], _ = b7_probs(t_blank, [bmedia[rend][i] for i in Q.scene])
        P["receipt, every layer"], _ = b7_probs(t_blank, [bmedia[rend][i] for i in Q.scene], hooks=b_receipt_hooks(B7_ALL, REC7, Q.scene.values, na, nrec))
        pr_ = Q.present.values
        bq = np.array([best[i, B7_WORDS.index(w)] for i, w in zip(Q.scene, Q.item)])
        read, carried = bq <= B_TOP, P["receipt, every layer"] >= 0.5
        row = dict(by_k={}, lexical=B7_LEXICAL[rend], n_receipt=int(nrec), yes_no_mass=mass)
        for k in B7_KS:
            sel = (S7.k == k).values
            sq = (Q.k == k).values
            r = dict(said=boot_ci((inside & present7)[sel].sum(1)), said_absent=boot_ci((inside & ~present7)[sel].sum(1)))
            for name in P:
                r[name] = b_score(Q[sq], P[name][sq])
            m = sq & pr_
            r["carried_items"] = b_boot_scene(k * carried[m], Q.scene.values[m])
            r["silent"] = b_boot_scene(carried[m] & ~read[m], Q.scene.values[m])
            row["by_k"][k] = r
        c1 = row["by_k"][1]
        row["seen"] = bool(c1["full scene"]["hit"][0] >= 0.9 and c1["full scene"]["fa"][0] <= 0.15)
        row["control_ok"] = bool(c1["receipt, every layer"]["hit"][0] >= 0.7 and abs(c1["blank frame"]["dprime"][0]) < B_FLOOR)
        row["K_said"] = b_capacity(B7_KS, {k: (inside & present7)[(S7.k == k).values].sum(1) for k in B7_KS})[0]
        row["K_carried"] = b_capacity(B7_KS, {k: k * carried[(Q.k == k).values & pr_] for k in B7_KS})[0]
        resB7["by_rendering"][rend] = row
        b8 = row["by_k"][B7_KS[-1]]
        print(f"{rend:8s} seen {'ok' if row['seen'] else 'NO'} (k = 1 hits {c1['full scene']['hit'][0]:.2f}, fa {c1['full scene']['fa'][0]:.2f})  control {'ok' if row['control_ok'] else 'FAILED'}  K_said {b_fmt(row['K_said']['K'])}  K_carried {b_fmt(row['K_carried']['K'])}  "
              f"at k = {B7_KS[-1]}: said {b8['said'][0]:.1f} (null {b8['said_absent'][0]:.1f}), carried {b8['carried_items'][0]:.1f}, silent share {b_fmt(b8['silent'])}  [{time.time() - t0:.0f}s]")
    k8 = B7_KS[-1]
    ok = {r: resB7["by_rendering"][r]["seen"] and resB7["by_rendering"][r]["control_ok"] for r in resB7["by_rendering"]}
    sil = {r: resB7["by_rendering"][r]["by_k"][k8]["silent"] for r in resB7["by_rendering"] if ok[r]}
    lex = [r for r in sil if B7_LEXICAL[r]]
    non = [r for r in sil if not B7_LEXICAL[r]]
    words = bool(non and lex and all(sil[r][1] >= B_SILENT for r in non) and all(sil[r][2] < B_SILENT for r in lex))
    channel = bool("sound" in sil and "picture" in sil and abs(sil["sound"][0] - sil["picture"][0]) >= 0.15 and (sil["sound"][1] > sil["picture"][2] or sil["picture"][1] > sil["sound"][2]))
    verdict = ("the bound follows words: every non-lexical channel carries items without a word and every lexical one says what it carries" if words and not channel else
               "the channel matters: sound and picture differ in what goes unsaid" if channel and not words else
               "both: lexical renderings say what they carry, and the non-lexical channels differ among themselves" if words and channel else
               "neither rule met" + ("" if all(ok.values()) else f" (unusable renderings: {[r for r in ok if not ok[r]]})"))
    resB7.update(verdict=verdict, silent_k8={r: sil[r] for r in sil}, seconds=time.time() - t0)
    save(resB7, "eB7_results")
    print(f"\n{verdict}")
    # --- figure
    rends = list(resB7["by_rendering"])
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.0), gridspec_kw=dict(wspace=0.75, width_ratios=[1, 1.1, 0.9]))
    ends, fr_ends = [], []
    for rend in rends:
        row, c = resB7["by_rendering"][rend], B7_COLOR[rend]
        ls = "-" if ok[rend] else ":"
        y = ws_band(ax[0], B7_KS, [row["by_k"][k]["said"] for k in B7_KS], c, ls=ls)
        ax[0].plot(B7_KS, [row["by_k"][k]["said_absent"][0] for k in B7_KS], "--", color=c, lw=0.9, alpha=0.7)
        ends.append((y, rend + ("" if ok[rend] else " (unusable)"), c))
        xs, ys = [row["by_k"][k]["said"][0] for k in B7_KS], [row["by_k"][k]["carried_items"][0] for k in B7_KS]
        ax[1].plot(xs, ys, "-o", color=c, lw=1.6, ms=3.5, alpha=0.95 if ok[rend] else 0.4)
        fr_ends.append((ys[-1], xs[-1], rend, c))
    ax[0].plot(B7_KS, B7_KS, color="0.75", lw=1, ls="-.")
    ax[0].set(xlabel=f"items in the scene, k (n = {B7_PER_K} scenes per k)", ylabel=f"present items in the J-lens top-{B_TOP} at the report\n(dashed: absent items, the null)", xticks=B7_KS, ylim=(0, k8 + 0.6))
    ws_end_labels(ax[0], ends, 0.45)
    ax[1].plot([0, k8 + 0.6], [0, k8 + 0.6], color="0.75", lw=1, ls="-.")
    fr_ends = sorted(fr_ends)
    ys_ = [e[0] for e in fr_ends]
    for i in range(1, len(ys_)):
        ys_[i] = max(ys_[i], ys_[i - 1] + 0.34)
    for (y0, x0, text, c), y in zip(fr_ends, ys_):
        ax[1].text(x0 + 0.12, y, text, color=c, fontsize=7.5, va="center")
    ax[1].set(xlabel="items said (lens at the report)", ylabel="items used (carried from the receipt alone)", xlim=(0, k8 + 0.6), ylim=(0, k8 + 0.6))
    xs = np.arange(len(rends))
    vals = [resB7["by_rendering"][r]["by_k"][k8]["silent"] for r in rends]
    ax[2].bar(xs, [v[0] for v in vals], 0.6, color=[B7_COLOR[r] for r in rends], alpha=[0.9 if ok[r] else 0.35 for r in rends][0] if len(set(ok.values())) == 1 else 0.9)
    ax[2].errorbar(xs, [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
    ax[2].axhline(B_SILENT, color="0.7", lw=1, ls="-.")
    ax[2].set(xticks=xs, xticklabels=[f"{r}\n{'lexical' if B7_LEXICAL[r] else 'non-lexical'}" for r in rends], ylabel=f"silent share at k = {k8}: carried, no word", ylim=(0, max(0.5, max(v[2] for v in vals) + 0.05)))
    ax[2].tick_params(axis="x", labelsize=7.5)
    short_v = verdict.split(":")[0].split(" (")[0]
    fig.suptitle(f"{MODEL_TAG}, one vocabulary through every channel. " + (f"Silent share at k = {k8}: " + ", ".join(f"{r} {sil[r][0]:.0%}" for r in sil) + ". " if sil else "No rendering passed both gates. ") + short_v.capitalize(), fontsize=9.5, y=1.0)
    ws_finish(fig, "figB7_omni", f"B7 · {len(S7)} scenes per rendering · {short_v}")
