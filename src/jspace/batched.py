"""Generation over many prompts and many interventions at once, built to keep the GPU busy (experiments/media_steer.py).

A job is a list of rows. A row is one prompt (an input kind, a question, a clip) with its own intervention:
    steering       a direction times a dose added at every layer of a layer set, on the media tokens or on the last prompt token
    ablation       a direction (one per clip) or a subspace (one for all) removed at the read layers, replaced by its mean part
    knockout       attention from every text token to the media tokens blocked at a set of attention layers
    none           the clean prompt
Rows with different interventions share a batch: each layer's hook edits only the rows whose intervention includes that layer.
So a batch is always as full as memory allows, however many arms there are.

What keeps the GPU busy, and why each choice does not change an answer:
- Equal-length batches: every prompt of one input kind and question has the same tokens around the same number of media tokens
  (media.py), so a batch has no padding. Session's padding check found that left padding moved Gemma-4's and the 27B's numbers
  and ran them one prompt at a time; with no padding there is nothing for that check to catch, and `assert_unpadded` holds it.
- The processor runs once per prompt, before any batch; a batch is a concatenation of those outputs (`collate`), checked
  against the processor on the same prompts at start-up.
- The vision encoder runs once per clip and input kind; its outputs are handed to the model as `mm_encoder_outputs`
  (transformers 5.19). At start-up a batch is run both ways, and the cache is used only if answers and first-token
  log-probabilities agree (`Engine.check_encoder_cache`); otherwise the encoder runs in every batch.
- The next batches are collated and pinned on a CPU thread while the GPU runs the current one.
- Batch rows start from a token budget set by free GPU memory and halve on out-of-memory; the size that worked is kept.
`GpuMonitor` samples nvidia-smi every 2 seconds into the run folder, so the utilization is measured, not assumed.
"""

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .hooks import _residual_hook

FLOAT_KEYS = ("pixel_values", "pixel_values_videos", "input_features")  # cast to the model's dtype; masks and ids are not
PIXEL_KEY = {"video": "pixel_values_videos", "picture": "pixel_values"}
ENCODER = {"video": "video", "picture": "image"}  # mm_encoder_outputs modality; Gemma-4's audio tower takes no precomputed outputs


def collate(encs):
    """one batch from single-prompt processor outputs: tensors concatenated along dim 0 (rows, patches, frames), lists joined"""
    out = {}
    for k in encs[0]:
        vs = [e[k] for e in encs]
        if torch.is_tensor(vs[0]):
            out[k] = torch.cat(vs, 0)
        elif isinstance(vs[0], list):
            out[k] = [x for v in vs for x in v]
        else:
            assert all(v == vs[0] for v in vs), k
            out[k] = vs[0]
    return out


def assert_unpadded(enc, token_id):
    """no padding in the batch, and the media tokens at the same positions in every row (so one position list serves all)"""
    assert bool(enc["attention_mask"].all()), "a batch has padding: its prompts differ in length"
    m = enc["input_ids"] == token_id
    assert bool((m == m[:1]).all()), "the rows of a batch place their media tokens differently"


def media_positions(ids_row, token_id):
    """positions of the media tokens in one row (a list of ids)"""
    return [j for j, t in enumerate(ids_row) if t == token_id]


class Tables:
    """What the row interventions read, on the GPU.
    dirs[key][L]        [n_scene, d] unit directions, one per scene (e.g. the steering word Y's J-lens direction at layer L)
    dirs[(input, key)]  the same for a direction that depends on the input kind (the clip's top principal direction)
    sigma[input][L]     the dose unit at layer L
    abl[key]            ("scene", {L: [n_scene, d]}) or ("shared", {L: [d, r]}): what an ablation removes at the read layers
    mu[input][L]        [d], the mean media-token state, whose part is put back (mean-ablation)"""

    def __init__(self):
        self.dirs, self.sigma, self.abl, self.mu = {}, {}, {}, {}

    def direction(self, inp, key, L):
        return self.dirs[(inp, key)][L] if (inp, key) in self.dirs else self.dirs[key][L]


def residual_hooks(rows, P, T, read_layers):
    """the residual-stream edits for a batch whose rows share an input kind and a token layout: [(layer, hook)]. P: the media
    positions (a LongTensor on the GPU), the same in every row."""
    inp = rows[0]["input"]
    assert all(r["input"] == inp for r in rows)
    dev = P.device
    sc = torch.as_tensor([r["scene"] for r in rows], device=dev)
    layers = set()
    for r in rows:
        if r["site"]:
            layers |= set(r["layers"])
        if r["ablate"]:
            layers |= set(read_layers)
    out = []
    for L in sorted(layers):
        add = {"media": None, "last": None}
        for site in ("media", "last"):
            for key in sorted({r["key"] for r in rows if r["site"] == site and L in r["layers"]}):
                idx = [i for i, r in enumerate(rows) if r["site"] == site and r["key"] == key and L in r["layers"]]
                it = torch.as_tensor(idx, device=dev)
                coef = torch.as_tensor([r["dose"] * T.sigma[inp][L] for r in (rows[i] for i in idx)], device=dev, dtype=torch.float32)
                D = T.direction(inp, key, L)[sc[it]] * coef[:, None]
                if add[site] is None:
                    add[site] = torch.zeros(len(rows), D.shape[1], device=dev)
                add[site][it] = D
        abl = []
        if L in read_layers:
            for key in sorted({r["ablate"] for r in rows if r["ablate"]}):
                it = torch.as_tensor([i for i, r in enumerate(rows) if r["ablate"] == key], device=dev)
                kind, B = T.abl[key]
                abl.append((it, kind, B[L][sc[it]] if kind == "scene" else B[L], T.mu[inp][L]))

        def edit(x, add=add, abl=abl):
            if add["media"] is not None:
                x[:, P] += add["media"][:, None, :]
            if add["last"] is not None:
                x[:, -1] += add["last"]
            for it, kind, B, mu in abl:
                v = x[it[:, None], P[None, :]]  # [m, P, d]
                if kind == "scene":  # one unit direction per row, B [m, d]
                    v = v - ((v * B[:, None, :]).sum(-1, keepdim=True) - (mu[None, :] * B).sum(-1)[:, None, None]) * B[:, None, :]
                else:  # one subspace, B [d, r] orthonormal
                    v = v - (v @ B) @ B.T + (mu @ B) @ B.T
                x[it[:, None], P[None, :]] = v

        out.append((L, _residual_hook(edit)))
    return out


def attention_layers(lm):
    """the decoder layers whose token mixer is softmax attention (Qwen3.5: one in four; Gemma-4: all), with their modules"""
    out = {}
    for i, layer in enumerate(lm.model.layers):
        mod = getattr(layer, "self_attn", None)
        if mod is not None and getattr(layer, "block_type", "full_attention") != "linear_attention":
            out[i] = mod
    return out


def knockout_mask(ids_row, token_id, blocked):
    """[b, 1, s, s] bool, True where attention is removed: from every non-media query to every media key, for the rows in
    `blocked` ([b] bool). Text between frames (timestamps) is a query too, so no text token carries the media forward."""
    is_media = ids_row == token_id
    base = (~is_media)[:, None] & is_media[None, :]
    return blocked[:, None, None, None] & base[None, None]


def knockout_hooks(rows, ids_row, token_id, attn):
    """forward pre-hooks on the attention modules that remove those reads for the rows whose knockout set includes the layer;
    a generated token's step (one query) is not edited, as in attention_knockout.py"""
    handles = []
    dev = ids_row.device
    s = ids_row.shape[0]
    for L, mod in attn.items():
        blocked = torch.as_tensor([L in r["ko"] for r in rows], device=dev)
        if not bool(blocked.any()):
            continue
        KO = knockout_mask(ids_row, token_id, blocked)

        def pre(mod, args, kwargs, KO=KO):
            if kwargs["hidden_states"].shape[1] == 1:
                return None
            m = kwargs.get("attention_mask")
            b = KO.shape[0]
            if m is None:
                new = torch.tril(torch.ones(s, s, dtype=torch.bool, device=KO.device))[None, None] & ~KO
            else:
                if m.shape[0] == 1 and b > 1:
                    m = m.expand(b, *m.shape[1:])
                K = KO[..., : m.shape[-2], : m.shape[-1]]
                new = (m & ~K) if m.dtype == torch.bool else m.masked_fill(K, torch.finfo(m.dtype).min)
            return args, dict(kwargs, attention_mask=new)

        handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
    return handles


class GpuMonitor:
    """nvidia-smi's utilization and memory every 2 s into a CSV; `stop()` returns the mean utilization while the job ran"""

    def __init__(self, path):
        self.path, self.proc = path, None
        try:
            self.f = open(path, "w")
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits", "-l", "2"], stdout=self.f, stderr=subprocess.DEVNULL
            )
        except OSError:
            print("nvidia-smi not found: GPU utilization is not recorded")

    def stop(self):
        if self.proc is None:
            return None
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.f.close()
        util = []
        for line in open(self.path):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[1].replace(".", "").isdigit():
                util.append(float(parts[1]))
        return dict(samples=len(util), mean_utilization=float(np.mean(util)) if util else None, median_utilization=float(np.median(util)) if util else None)


class Engine:
    """Runs jobs of rows on a Session. encs[(input, task, scene)] holds each prompt's processor output (CPU tensors, one row);
    token[input] the media token id; medium[input] "video", "picture" or "sound"."""

    def __init__(self, lm, encs, token, medium, tables, read_layers, max_rows=512, budget_tokens=None):
        self.lm, self.encs, self.token, self.medium, self.T, self.read_layers = lm, encs, token, medium, tables, read_layers
        self.attn = attention_layers(lm)
        self.max_rows = max_rows
        free = torch.cuda.mem_get_info()[0] / 2**30
        # a first guess that OOM halving corrects: about 2,000 prompt tokens per free GB held up for the models here
        self.budget = budget_tokens or int(2048 * max(free - 4, 1))
        self.cap = {}  # (input, task) -> rows per batch that ran
        self.features = {}  # (input, scene) -> tuple of encoder output pieces (GPU), when the cache is on for that input
        self.cache_on = {}
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.batches = 0
        self.rows_done = 0
        self.t_gpu = 0.0
        print(f"engine: {free:.0f} GB free after loading, first token budget {self.budget} per batch, at most {max_rows} rows")

    # ---- inputs
    def _batch_cpu(self, rows, use_cache):
        enc = collate([self.encs[(r["input"], r["task"], r["scene"])] for r in rows])
        if use_cache:
            enc.pop(PIXEL_KEY[self.medium[rows[0]["input"]]], None)
        return {k: (v.pin_memory() if torch.is_tensor(v) else v) for k, v in enc.items()}

    def _to_gpu(self, enc):
        dev, dt = self.lm.device, self.lm.dtype
        out = {}
        for k, v in enc.items():
            if torch.is_tensor(v):
                v = v.to(dev, non_blocking=True)
                if k in FLOAT_KEYS:
                    v = v.to(dt)
            out[k] = v
        return out

    def _mm(self, rows):
        """mm_encoder_outputs for a batch from the cache, or {}"""
        inp = rows[0]["input"]
        if not self.cache_on.get(inp):
            return {}
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        pieces = tuple(p for r in rows for p in self.features[(inp, r["scene"])])
        return {"mm_encoder_outputs": {ENCODER[self.medium[inp]]: BaseModelOutputWithPooling(pooler_output=pieces)}}

    # ---- the encoder cache
    @torch.no_grad()
    def build_encoder_cache(self, inp, scenes, task="say"):
        """run the encoder once per clip for a picture or video input; the cache is switched on only by check_encoder_cache"""
        med = self.medium[inp]
        if med not in ENCODER:
            return False
        mod = ENCODER[med]
        fn = getattr(self.lm.hf.model, f"get_{mod}_features", None)
        if fn is None:
            return False
        try:
            for s in scenes:
                enc = self._to_gpu(self.encs[(inp, task, s)])
                second = next((enc[k] for k in (f"{mod}_grid_thw", f"{mod}_position_ids") if k in enc), None)
                args = (enc[PIXEL_KEY[med]],) if second is None else (enc[PIXEL_KEY[med]], second)
                out = fn(*args, return_dict=True)
                self.features[(inp, s)] = tuple(p.detach() for p in out.pooler_output)
        except Exception as e:  # noqa: BLE001 - an encoder API this code does not know: the encoder runs in every batch instead
            print(f"{inp}: encoder cache not built ({type(e).__name__}: {str(e)[:160]}); the encoder runs in every batch")
            return False
        return True

    @torch.no_grad()
    def check_encoder_cache(self, inp, rows, n_new):
        """the same rows generated with the encoder in the batch and from the cache: the cache stays on only if the answers are
        identical and the first-token log-probabilities agree within 0.05"""
        self.cache_on[inp] = False
        a = self._generate_batch(rows, n_new)
        self.cache_on[inp] = True
        try:
            b = self._generate_batch(rows, n_new)
        except Exception as e:  # noqa: BLE001
            print(f"{inp}: generation from cached encoder outputs failed ({type(e).__name__}: {str(e)[:160]}); cache off")
            self.cache_on[inp] = False
            return dict(on=False, reason=f"{type(e).__name__}")
        same_text = [x[0] == y[0] for x, y in zip(a, b)]
        dlp = float(max(np.max(np.abs(x[3] - y[3])) for x, y in zip(a, b)))
        ok = all(same_text) and dlp <= 0.05
        self.cache_on[inp] = ok
        print(f"{inp}: encoder cache {'on' if ok else 'off'} (answers identical {sum(same_text)}/{len(same_text)}, max |d log p| top-10 {dlp:.4f})")
        return dict(on=ok, identical=int(sum(same_text)), n=len(same_text), max_dlogp=dlp)

    def check_collate(self, inp, task, scenes, build):
        """the concatenation of single-prompt outputs must equal the processor's own output on the same prompts"""
        a = collate([self.encs[(inp, task, s)] for s in scenes])
        b = build(scenes)
        bad = [k for k in a if k in b and torch.is_tensor(a[k]) and (a[k].shape != b[k].shape or not torch.equal(a[k], b[k]))]
        missing = [k for k in b if k not in a]
        assert not bad and not missing, f"{inp}/{task}: collated batch differs from the processor's in {bad + missing}"

    # ---- running
    def _generate_batch(self, rows, n_new):
        enc = self._to_gpu(self._batch_cpu(rows, self.cache_on.get(rows[0]["input"], False)))
        return self._generate_enc(rows, enc, n_new)

    @torch.no_grad()
    def _generate_enc(self, rows, enc, n_new):
        lm = self.lm
        assert_unpadded(enc, self.token[rows[0]["input"]])
        ids0 = enc["input_ids"][0]
        P = torch.as_tensor(media_positions(ids0.tolist(), self.token[rows[0]["input"]]), device=lm.device)
        assert len(P) > 0, "no media tokens in the prompt"
        hk = residual_hooks(rows, P, self.T, self.read_layers)
        handles = [lm.register(L, f) for L, f in hk]
        if any(r["ko"] for r in rows):
            handles += knockout_hooks(rows, ids0, self.token[rows[0]["input"]], self.attn)
        try:
            kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
            g = lm.hf.generate(**kw, **self._mm(rows), max_new_tokens=n_new, do_sample=False, pad_token_id=lm.tokenizer.pad_token_id, output_scores=True, return_dict_in_generate=True)
        finally:
            for h in handles:
                h.remove()
        lp = torch.log_softmax(g.scores[0].float(), -1)
        top = lp.topk(10, -1).values.cpu().numpy()
        n0 = enc["input_ids"].shape[1]
        res = []
        for j, r in enumerate(rows):
            text = lm.tokenizer.decode(g.sequences[j, n0:], skip_special_tokens=True).strip()
            lx = float(torch.logsumexp(lp[j, r["ids_x"]], 0))
            ly = float(torch.logsumexp(lp[j, r["ids_y"]], 0))
            res.append((text, lx, ly, top[j]))
        return res

    def generate(self, rows, n_new=4, label=""):
        """answers and first-token log-probabilities for every row, in row order: [(text, lp_x, lp_y)]"""
        out = [None] * len(rows)
        groups = {}
        for i, r in enumerate(rows):
            groups.setdefault((r["input"], r["task"]), []).append(i)
        t0 = time.time()
        for (inp, task), idx in groups.items():
            seq = int(self.encs[(inp, task, rows[idx[0]]["scene"])]["input_ids"].shape[1])
            B = self.cap.get((inp, task)) or max(1, min(self.max_rows, self.budget // seq))
            pos = 0
            pending = {}

            def submit(p, b):
                key = (p, b)
                if key not in pending and p < len(idx):
                    use = self.cache_on.get(inp, False)
                    pending[key] = self.pool.submit(self._batch_cpu, [rows[i] for i in idx[p : p + b]], use)

            while pos < len(idx):
                submit(pos, B)
                submit(pos + B, B)
                submit(pos + 2 * B, B)
                chunk = idx[pos : pos + B]
                try:
                    enc_cpu = pending.pop((pos, B)).result()
                    tg = time.time()
                    res = self._generate_enc([rows[i] for i in chunk], self._to_gpu(enc_cpu), n_new)
                    self.t_gpu += time.time() - tg
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if B == 1:
                        raise
                    B = max(1, B // 2)
                    pending.clear()
                    print(f"  out of memory at {len(chunk)} rows of {seq} tokens: {B} rows per batch from here")
                    continue
                for i, x in zip(chunk, res):
                    out[i] = x[:3]
                pos += len(chunk)
                self.batches += 1
                self.rows_done += len(chunk)
            self.cap[(inp, task)] = B
        print(f"  {label}: {len(rows)} rows in {time.time() - t0:.0f}s ({len(rows) / max(time.time() - t0, 1e-9):.0f} rows/s)")
        return out

    @torch.no_grad()
    def capture(self, rows, layers, on_layer):
        """clean forwards over rows with on_layer(L, rows, v) called inside each layer's hook, v [b, P, d] float32 at the media
        positions (nothing is kept unless on_layer keeps it)"""
        lm = self.lm
        groups = {}
        for r in rows:
            groups.setdefault((r["input"], r["task"]), []).append(r)
        for (inp, _), rs in groups.items():
            B = 8  # the hooks hold one layer's media states at a time; small batches keep the SVDs in memory
            for s in range(0, len(rs), B):
                batch = rs[s : s + B]
                enc = self._to_gpu(self._batch_cpu(batch, self.cache_on.get(inp, False)))
                assert_unpadded(enc, self.token[inp])
                P = torch.as_tensor(media_positions(enc["input_ids"][0].tolist(), self.token[inp]), device=lm.device)

                def hook(L):
                    def f(mod, inp_, out):
                        h = out[0] if isinstance(out, tuple) else out
                        on_layer(L, batch, h[:, P].float())

                    return f

                handles = [lm.register(L, hook(L)) for L in layers]
                try:
                    kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
                    kw.update(self._mm(batch))
                    if lm.vision_path == "base":
                        lm.hf.model(**kw, use_cache=False)
                    else:
                        lm.hf(**kw, use_cache=False, logits_to_keep=1)
                finally:
                    for h in handles:
                        h.remove()
