# The notebook's model, lens and setup cells (Copy_of_jlens_working_nayan.ipynb, cells 2, 4, 6 and 14) as one script, run by run.py
# before the B cells. Every edit against the notebook is listed in jbound/CHANGES.md: the model class (the picture cells need the
# vision tower), the lens file table for every size, the batch sizes for a 96 GB card, the run folder, a numpy-safe save(), and a
# headless matplotlib. Nothing scientific is changed here.
#
# %% ---------------------------------------------------------------- model and lens (notebook cells 2, 4, 6)
import os, sys, json, time, glob, hashlib, logging, datetime, subprocess, functools
import numpy as np, pandas as pd, torch, requests
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import jlens
from jlens import JacobianLens
jlens.configure_logging()

MODEL_NAME = globals().get("MODEL_NAME", "Qwen/Qwen3.5-4B")
RUN_TAG = globals().get("RUN_TAG", datetime.datetime.now().strftime("%Y%m%d_%H%M"))
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = globals().get("LENS_REVISION", None)          # None reads main; the notebook used "qwen-n1000" for the Qwen n1000 files (same bytes, see CHANGES.md)
LENS_FILE = {
    "Qwen/Qwen3.5-0.8B": "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt",
    "Qwen/Qwen3.5-2B": "qwen3.5-2b/jlens/Salesforce-wikitext/Qwen3.5-2B_jacobian_lens.pt",
    "Qwen/Qwen3.5-4B": "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
    "Qwen/Qwen3.5-27B": "qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt",
    "Qwen/Qwen3.6-27B": "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    "google/gemma-4-E2B-it": "gemma-4-e2b/jlens/Salesforce-wikitext/gemma-4-E2B_jacobian_lens.pt",
    "google/gemma-4-E4B-it": "gemma-4-e4b/jlens/Salesforce-wikitext/gemma-4-E4B_jacobian_lens.pt",
    "google/gemma-4-31B-it": "gemma-4-31b/jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt",
}[MODEL_NAME]

JPP_FILE = {                                                   # J++ lenses (Ayonrinde & Lindsey recipe, Neuronpedia fits); none exist for Qwen3.5-4B / 27B or Gemma E2B / E4B
    "Qwen/Qwen3.5-0.8B": "qwen3.5-0.8b/jpp/Salesforce-wikitext/Qwen3.5-0.8B_jpp_lens.pt",
    "Qwen/Qwen3.5-2B": "qwen3.5-2b/jpp/Salesforce-wikitext/Qwen3.5-2B_jpp_lens.pt",
    "Qwen/Qwen3.5-9B": "qwen3.5-9b/jpp/Salesforce-wikitext/Qwen3.5-9B_jpp_lens.pt",
    "Qwen/Qwen3.6-27B": "qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt",
    "google/gemma-4-31B-it": "gemma-4-31b/jpp/Salesforce-wikitext/gemma-4-31B_jpp_lens.pt",
}.get(MODEL_NAME)
import transformers
_t0 = time.time()
if "gemma" in MODEL_NAME.lower():
    from transformers import AutoModelForMultimodalLM as _ModelClass        # text, image, video and audio in
else:
    from transformers import AutoModelForImageTextToText as _ModelClass     # the notebook used AutoModelForCausalLM; the picture cells need the vision tower
hf_model = _ModelClass.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).cuda()
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = jlens.from_hf(hf_model, tokenizer, force_bos=False)      # the chat templates supply any BOS themselves; force_bos=True made lens.apply tokenize Gemma prompts with two <bos> (CHANGES.md)
lens = JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION)
lens_jpp = JacobianLens.from_pretrained(LENS_REPO, filename=JPP_FILE, revision=LENS_REVISION) if JPP_FILE else None
LENS_PATH = (glob.glob(f"{os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))}/hub/models--neuronpedia--jacobian-lens/snapshots/*/{LENS_FILE}") or ["?"])[0]
print(f"{model} ({type(hf_model).__name__}) loaded in {time.time() - _t0:.0f}s; {lens}; lens file {LENS_PATH}" + (f"; J++ {lens_jpp} ({JPP_FILE})" if lens_jpp else "; no J++ lens on the hub for this model"))

# %% ---------------------------------------------------------------- setup (notebook cell 14)
torch.manual_seed(0)
RNG = np.random.default_rng(0)
SRC = list(lens.source_layers)
N_LAYERS = model.n_layers
LAYERS = [L for L in (range(20, 61, 4) if N_LAYERS == 64 else range(int(0.3 * N_LAYERS), N_LAYERS, max(1, N_LAYERS // 11))) if L in SRC]
BAND = [L for L in LAYERS if 0.4 * N_LAYERS <= L <= 0.75 * N_LAYERS]   # candidate layers for the swap
DEV = hf_model.device
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
BATCH = (96 if N_LAYERS <= 40 else 32) if GPU_GB >= 80 else (48 if N_LAYERS <= 40 else 16)     # the notebook's 48 / 16 were for a T4; raised for the 96 GB card (CHANGES.md)
BATCH = int(globals().get("BATCH_OVERRIDE") or BATCH)
BATCH_TOKENS = 65536 if GPU_GB >= 80 else (24576 if GPU_GB >= 20 else 6144)   # padded tokens per forward; a batch that still runs out of memory is halved and retried
FINAL = N_LAYERS - 1
TEXT = functools.reduce(getattr, model.layout.path.split("."), hf_model)   # the decoder stack, called directly so no logits are built at every position
JAC = {L: lens.jacobians[L].to(DEV) for L in LAYERS}
JPP = {L: lens_jpp.jacobians[L].to(DEV) for L in LAYERS} if lens_jpp and all(L in lens_jpp.jacobians for L in LAYERS) else None      # the J++ map on the same read layers, or None
U_OUT = hf_model.get_output_embeddings().weight
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
transformers.logging.set_verbosity_error()
_dtype = next(hf_model.parameters()).dtype
print(f"{MODEL_NAME}: {N_LAYERS} layers, {_dtype}, on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}; "
      f"lens fitted on {SRC[0]}..{SRC[-1]}, reading {LAYERS}, swap candidates {BAND}; BATCH {BATCH}, BATCH_TOKENS {BATCH_TOKENS}")
if _dtype == torch.bfloat16 and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
    print("this GPU has no bfloat16 units: reload the model with torch_dtype=torch.float16 or every forward will be slow")
if torch.cuda.is_available():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    _free, _total = torch.cuda.mem_get_info()
    print(f"GPU memory free {_free / 2**30:.1f} of {_total / 2**30:.1f} GB; batches hold at most {BATCH_TOKENS} padded tokens and halve on an out-of-memory error")
    if _free < 3 * 2**30:
        print("under 3 GB free: earlier notebook cells are holding tensors (the slice page keeps full-sequence lens logits); restart the runtime and run only the model and lens cells before this one")


def chat(msgs):
    """chat template over (role, text) pairs, ending where the assistant starts writing"""
    return tokenizer.apply_chat_template([{"role": r, "content": t} for r, t in msgs], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def n_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def single_token(word):
    """id of ' word' when it is one token, else None"""
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def single_ids(variants):
    return sorted({t[0] for v in variants for t in [tokenizer.encode(v, add_special_tokens=False)] if len(t) == 1})


# two ways to read the answer letter: prefill "(" and read A-D, or prefill nothing and read "(A"-style tokens; the check below keeps the one the model uses
SCHEMES = {"(": {l: single_ids((l, " " + l)) for l in "ABCD"}, "": {l: single_ids(("(" + l, "(" + l + ")", l, " " + l)) for l in "ABCD"}}
ANSWER_PREFIX, LETTER_IDS = "(", SCHEMES["("]


def residual_of(out):
    return out[0] if isinstance(out, tuple) else out


def register(hooks, idx):
    """hooks is None or a function idx -> [(layer, fn)] for the prompts at these indices; returns the handles"""
    return [model.layers[L].register_forward_hook(fn) for L, fn in (hooks(torch.as_tensor(idx, device=DEV)) if hooks else [])]


def batch_plan(prompts):
    """index arrays for length-sorted batches of at most BATCH rows and BATCH_TOKENS padded tokens"""
    lens = np.array([n_tokens(p) for p in prompts])
    plan, cur = [], []
    for i in np.argsort(lens, kind="stable"):
        if cur and (len(cur) == BATCH or (len(cur) + 1) * lens[i] > BATCH_TOKENS):
            plan.append(np.array(cur))
            cur = []
        cur.append(int(i))
    if cur:
        plan.append(np.array(cur))
    return plan


def run_batches(prompts, fn):
    """fn(idx) -> one result per prompt in prompts[idx]; results returned in prompt order; a batch that runs out of GPU memory is halved and retried"""
    out = [None] * len(prompts)
    stack = batch_plan(prompts)[::-1]
    bar = tqdm(total=len(stack), desc=f"{len(prompts)} prompts", unit="batch", leave=False, dynamic_ncols=True, file=sys.__stderr__, disable=len(stack) < 3)
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


@torch.no_grad()
def forward_last(prompts, layers=(), hooks=None):
    """left-padded batched forward: the model's logits at the last real token [n, vocab] (float32, cpu) and {layer: residual [n, d]} (float32, on DEV).
    The logits come from the final block's residual through the model's own norm and unembedding, the way lens.apply computes them."""
    keep = set(layers) | {FINAL}

    def fn(idx):
        store = {}

        def capture(L):
            def f(mod, inp, out):
                store[L] = residual_of(out)[:, -1].float()
            return f

        handles = register(hooks, idx) + [model.layers[L].register_forward_hook(capture(L)) for L in keep]
        try:
            enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            TEXT(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        logits = model.unembed(store[FINAL]).float().cpu()
        return [(logits[j], {L: store[L][j] for L in layers}) for j in range(len(idx))]

    rows = run_batches(prompts, fn)
    return torch.stack([r[0] for r in rows]), {L: torch.stack([r[1][L] for r in rows]) for L in layers}


def letter_dists(prompts):
    """[n, 4] distribution over A-D at the last token, and the raw probability mass on the four letters"""
    pr = torch.softmax(forward_last(prompts)[0], -1)
    P = np.stack([sum(pr[:, t].numpy() for t in LETTER_IDS[l]) for l in "ABCD"], 1)
    return P / np.maximum(P.sum(1, keepdims=True), 1e-9), P.sum(1)


@torch.no_grad()
def read_ranks(prompts, option_ids, hooks=None):
    """J-lens rank of each option token at the last prompt token: int array [n, len(LAYERS), n_options], 1 = top"""
    _, resid = forward_last(prompts, LAYERS, hooks)
    ids = torch.as_tensor(np.asarray(option_ids), device=DEV)
    out = np.zeros((len(prompts), len(LAYERS), ids.shape[1]), int)
    for li, L in enumerate(LAYERS):
        for s in range(0, len(prompts), BATCH):
            x = model.unembed(resid[L][s:s + BATCH] @ JAC[L].T).float()
            v = x.gather(1, ids[s:s + BATCH])
            out[s:s + BATCH, li] = ((x[:, None, :] > v[:, :, None]).sum(-1) + 1).cpu().numpy()
    return out


def reference(prompt, option_ids):
    """letter distribution and option ranks through lens.apply, used once to check the batched paths"""
    jl, ml, _ = lens.apply(model, prompt, layers=LAYERS, positions=[-1])
    pr = torch.softmax(ml[0].float(), -1)
    p = np.array([sum(pr[t].item() for t in LETTER_IDS[l]) for l in "ABCD"])
    ranks = np.array([[int((jl[L][0] > jl[L][0][t]).sum()) + 1 for t in option_ids] for L in LAYERS])
    return p / p.sum(), ranks


def lens_vectors(L, ids):
    """the paper's lens vectors at layer L: rows of W_U J_L for the given tokens, [n, d]"""
    return U_OUT[torch.as_tensor(np.asarray(ids), device=DEV)].float().to(DEV) @ JAC[L]


def swap_hook(A, B, K):
    """the paper's coordinate swap on span{a, b}: c = V+ h, h <- h + V (swap(c) - c).
    A, B: [b, d] lens vectors per row. Applied to the last K prompt tokens of each row and to every generated token."""
    aa, bb, ab = (A * A).sum(1, keepdim=True), (B * B).sum(1, keepdim=True), (A * B).sum(1, keepdim=True)
    det = aa * bb - ab * ab

    def fn(mod, inp, out):
        h = residual_of(out)
        x = h.float()
        ha = torch.einsum("bsd,bd->bs", x, A)
        hb = torch.einsum("bsd,bd->bs", x, B)
        ca = (bb * ha - ab * hb) / det
        cb = (aa * hb - ab * ha) / det
        delta = (cb - ca)[..., None] * (A - B)[:, None]
        s = x.shape[1]
        if s > 1:
            inside = torch.arange(s, device=x.device)[None] >= (s - K)[:, None]
            delta = delta * inside[..., None]
        x = (x + delta).to(h.dtype)
        return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
    return fn


@torch.no_grad()
def generate(prompts, n_new, hooks=None):
    """greedy continuations, in prompt order; hooks as in forward_last"""
    def fn(idx):
        handles = register(hooks, idx)
        try:
            enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            g = hf_model.generate(**enc, max_new_tokens=n_new, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        finally:
            for h in handles:
                h.remove()
        return [tokenizer.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for row in g]

    return run_batches(prompts, fn)


def boot_ci(x, n=2000, seed=0):
    """mean with a 95% bootstrap CI over the entries of x (one entry per item)"""
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def f3(t):
    return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"


def auroc(pos, neg):
    """probability that a positive scores above a negative, ties counted half"""
    pos, neg = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def auroc_ci(pos, neg, n=2000, seed=0):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    v = [auroc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    return auroc(pos, neg), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def question(stem, opts):
    return stem + "\n" + "\n".join(f"({l}) {o}" for l, o in zip("ABCD", opts))


def tokens_after_answer(prompt, letter):
    """how many prompt tokens follow the recorded answer ({letter}); the region an intervention acts on"""
    cut = prompt.rindex(f"({letter})") + 3
    return n_tokens(prompt) - n_tokens(prompt[:cut])


YES_IDS = sorted({t[0] for v in ("Yes", " Yes", "yes", " yes") for t in [tokenizer.encode(v, add_special_tokens=False)] if len(t) == 1})
NO_IDS = sorted({t[0] for v in ("No", " No", "no", " no") for t in [tokenizer.encode(v, add_special_tokens=False)] if len(t) == 1})
assert YES_IDS and NO_IDS, "yes / no are not single tokens in this tokenizer"


def yes_prob(prompts):
    """p(yes) against p(no) at the last token, and the raw mass on the two"""
    pr = torch.softmax(forward_last(prompts)[0], -1)
    y = sum(pr[:, t].numpy() for t in YES_IDS)
    n = sum(pr[:, t].numpy() for t in NO_IDS)
    return y / np.maximum(y + n, 1e-9), y + n


def unit_lens_vectors(L, ids):
    V = lens_vectors(L, ids)
    return V / V.norm(dim=1, keepdim=True)


@torch.no_grad()
def coords(prompts, L, ids):
    """coordinate of the residual at the last prompt token on the unit lens vector of each row's token: [n]"""
    _, resid = forward_last(prompts, [L])
    return (resid[L] * unit_lens_vectors(L, ids)).sum(1).cpu().numpy()


def clamp_hook(V, target, K):
    """set the coordinate on the unit lens vectors V [b, d] to target [b], on the last K prompt tokens of each row and on every generated token"""
    def fn(mod, inp, out):
        h = residual_of(out)
        x = h.float()
        c = torch.einsum("bsd,bd->bs", x, V)
        delta = (target[:, None] - c)[..., None] * V[:, None]
        s = x.shape[1]
        if s > 1:
            inside = torch.arange(s, device=x.device)[None] >= (s - K)[:, None]
            delta = delta * inside[..., None]
        x = (x + delta).to(h.dtype)
        return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
    return fn


def add_hook(D, K):
    """add D [b, d] to the residual on the last K prompt tokens of each row and on every generated token"""
    def fn(mod, inp, out):
        h = residual_of(out)
        x = h.float()
        s = x.shape[1]
        delta = D[:, None].expand(-1, s, -1)
        if s > 1:
            inside = torch.arange(s, device=x.device)[None] >= (s - K)[:, None]
            delta = delta * inside[..., None]
        x = (x + delta).to(h.dtype)
        return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
    return fn


# checks: the batched paths against lens.apply on three prompts (short ones get heavy left padding next to the long one)
_qs = ["Pick one of these for dinner.\n(A) pizza\n(B) sushi\n(C) pasta\n(D) salad",
       "Pick one of these colours for a room.\n(A) red\n(B) blue\n(C) green\n(D) white",
       "Background: " + "The weather was mild. " * 40 + "Pick one of these pets.\n(A) cat\n(B) dog\n(C) fish\n(D) bird"]
_opts = [[single_token(w) for w in words] for words in (("pizza", "sushi", "pasta", "salad"), ("red", "blue", "green", "white"), ("cat", "dog", "fish", "bird"))]
assert all(t is not None for o in _opts for t in o), "check words must be single tokens"
_masses = {}
for ANSWER_PREFIX, LETTER_IDS in SCHEMES.items():
    _masses[ANSWER_PREFIX] = float(letter_dists([chat([("user", q + "\nAnswer with the letter in parentheses only.")]) + ANSWER_PREFIX for q in _qs])[1].mean())
ANSWER_PREFIX = max(_masses, key=_masses.get)
LETTER_IDS = SCHEMES[ANSWER_PREFIX]
_prompts = [chat([("user", q + "\nAnswer with the letter in parentheses only.")]) + ANSWER_PREFIX for q in _qs]
_top = torch.softmax(forward_last(_prompts[:1])[0][0], -1).topk(5)
print(f"answer readout: prefill {ANSWER_PREFIX!r} (letter mass {_masses[ANSWER_PREFIX]:.2f}; the other scheme {min(_masses.values()):.2f}); "
      f"top tokens after the prefill on the first check prompt: " + ", ".join(f"{tokenizer.decode([i])!r} {p:.2f}" for p, i in zip(_top.values.tolist(), _top.indices.tolist())))
if _masses[ANSWER_PREFIX] < 0.8:
    print("less than 80% of the mass is on a letter: look at the top tokens above before trusting the answer distributions")
_ref = [reference(p, o) for p, o in zip(_prompts, _opts)]


def batched_deviation():
    """max |dp| on the letter distributions and mean |d log10 best rank| on the twelve option reads"""
    dp = np.abs(np.stack([r[0] for r in _ref]) - letter_dists(_prompts)[0]).max()
    dr = np.abs(np.log10(np.stack([r[1] for r in _ref]).min(1)) - np.log10(read_ranks(_prompts, _opts).min(1))).mean()
    return dp, dr


_dp, _dr = batched_deviation()
print(f"batched vs lens.apply: max |dp| = {_dp:.4f}, mean |d log10 best rank| = {_dr:.3f}")
BATCH_CHECK = dict(max_dp=float(_dp), mean_dlog10rank=float(_dr), batch=BATCH)
if _dp > 0.02 or _dr > 0.25:
    BATCH = 1
    _dp, _dr = batched_deviation()
    print(f"padding changed the numbers; running unbatched: max |dp| = {_dp:.4f}, mean |d log10 best rank| = {_dr:.3f}")
    assert _dp < 0.02 and _dr < 0.25
    BATCH_CHECK.update(unbatched_max_dp=float(_dp), unbatched_mean_dlog10rank=float(_dr), batch=BATCH)

JUDGE = MODEL_NAME + " (self)"      # the B cells use no judge; kept so finish() renders (CHANGES.md)
INSTR = "\nAnswer with the letter in parentheses only."     # defined in the notebook's experiment cells (15, 17, 19, 20), used by M0's picture preflight

# run folder
MODEL_TAG = MODEL_NAME.split("/")[-1]
TAG = f"{RUN_TAG}_{MODEL_TAG}"
RUN_ROOT = globals().get("RUN_ROOT", "/workspace/nayan/runs")
RUN_DIR = f"{RUN_ROOT}/{RUN_TAG}/{MODEL_TAG}"
FIGDIR = f"{RUN_DIR}/figures"
os.makedirs(FIGDIR, exist_ok=True)


def _json_default(o):
    """numpy scalars and arrays as JSON numbers and lists, so results read back with their types (the notebook used default=str)"""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save(obj, name):
    path = os.path.join(RUN_DIR, name)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path + ".csv", index=False)
    elif isinstance(obj, np.ndarray):
        np.save(path + ".npy", obj)
    else:
        json.dump(obj, open(path + ".json", "w"), indent=2, default=_json_default)
    print("saved", path)


# figure style
import matplotlib.pyplot as plt, matplotlib.font_manager as fm, seaborn as sns
plt.show = lambda *a, **k: None       # headless
subprocess.run(["apt-get", "-qq", "install", "-y", "fonts-texgyre"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for f in glob.glob("/usr/share/**/*[Hh]eros*.[ot]tf", recursive=True):
    fm.fontManager.addfont(f)
FONT = next((f for f in ["TeX Gyre Heros", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"] if f in {x.name for x in fm.fontManager.ttflist}), "DejaVu Sans")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
sns.set_theme(context="paper", style="ticks", palette=OKABE, rc={
    "font.family": [FONT], "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
    "ytick.major.width": 0.8, "legend.frameon": False, "axes.titlelocation": "left",
    "axes.titleweight": "normal", "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight"})


def ci_bars(groups):
    st = [boot_ci(g) for g in groups]
    return [s[0] for s in st], [[s[0] - s[1] for s in st], [s[2] - s[0] for s in st]]


def finish(fig, name, text):
    sns.despine(fig=fig)
    fig.text(0.005, 0.005, f"{text} · {MODEL_TAG} · lens layers {LAYERS[0]}–{LAYERS[-1]} · judge {JUDGE} · {TAG}", fontsize=6.5, color="0.5")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{FIGDIR}/{name}.{ext}")
    plt.show()
    print(f"saved {FIGDIR}/{name}.png .pdf .svg")


print(f"run folder {RUN_DIR}, font {FONT}")
