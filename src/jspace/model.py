"""The model under study: a Hugging Face vision-language model with its J-lens, loaded at pinned revisions, and the batched forward
passes every experiment uses.

    lm = Session("Qwen/Qwen3.5-4B")
    logits, resid = lm.forward(texts, images, layers=lm.layers, where=..., hooks=...)
    answers = lm.generate(texts, images, n_new=6, hooks=...)

Prompts are left-padded and batched by length (at most `batch` rows and `batch_tokens` padded tokens); a batch that runs out of
GPU memory is halved and retried. Two checks run at load time: left-padded batches must give the numbers lens.apply gives one
prompt at a time (otherwise batching is turned off), and the model must read a letter off a picture (the vision path works).
"""

import functools
import gc
import os
import sys

import jlens
import numpy as np
import torch
import transformers
from jlens import JacobianLens
from tqdm.auto import tqdm

from . import config, stimuli
from .hooks import residual_of

INSTR = "\nAnswer with the letter in parentheses only."
CPU_THREADS = 16
IMG_TOKENS = (448 // 32) * (224 // 32) + 8  # the batch planner's token allowance per picture


def entry_points(spec):
    """the forward entry points the picture check tries, in order: the base model ("base") or the full model ("logits_to_keep").
    A Qwen omni thinker's base model is its text decoder alone, whose forward takes **kwargs: it would accept the picture and the
    audio, leave them out without an error, and the check would then fail as "not read" (or, by chance, pass with every media
    forward reading no media). Only the full thinker reads them."""
    return ("logits_to_keep",) if spec.loader else ("base", "logits_to_keep")


class Session:
    def __init__(self, name):
        missing = [f for f in stimuli.FONT_FILES if not os.path.exists(f)]
        assert not missing, f"{missing} not found: the stimuli are drawn in DejaVu Sans (Debian / Ubuntu: apt install fonts-dejavu-core)"
        assert torch.cuda.is_available(), "the experiments need a CUDA GPU"
        # The original runs used 16 CPU threads. Multithreaded CPU linear algebra rounds differently with another count: the QR
        # that orthonormalizes a random or lens subspace (lens.orth) gives a different basis with 64 threads.
        torch.set_num_threads(CPU_THREADS)
        self.name, self.spec, self.tag = name, config.MODELS[name], name.split("/")[-1]
        jlens.configure_logging()
        cls = transformers.AutoModelForMultimodalLM if "gemma" in name.lower() else transformers.AutoModelForImageTextToText
        if self.spec.loader:  # a Qwen omni model: its text-generating part ("thinker") alone, from the full checkpoint
            cls = getattr(transformers, self.spec.loader)
        self.hf = cls.from_pretrained(name, revision=self.spec.revision, dtype=torch.bfloat16).cuda()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name, revision=self.spec.revision)
        # the chat templates supply any BOS themselves (force_bos=True gave Gemma prompts two <bos> in lens.apply)
        self.model = jlens.from_hf(self.hf, self.tokenizer, force_bos=False)
        repo, rev = self.spec.lens_source()  # Neuronpedia's repo, unless the model's lens is pinned elsewhere
        self.lens = JacobianLens.from_pretrained(repo, filename=self.spec.lens_file, revision=rev)
        self.lens_jpp = JacobianLens.from_pretrained(repo, filename=self.spec.jpp_file, revision=rev) if self.spec.jpp_file else None
        torch.manual_seed(0)  # after loading, as in the original runs: the PCA control in gqa_ablation draws from this stream

        n = self.n_layers = self.model.n_layers
        source = list(self.lens.source_layers)
        self.layers = [L for L in (range(20, 61, 4) if n == 64 else range(int(0.3 * n), n, max(1, n // 11))) if L in source]  # the read layers
        self.final = n - 1
        self.device = self.hf.device
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        self.batch = (96 if n <= 40 else 32) if gpu_gb >= 80 else (48 if n <= 40 else 16)
        self.batch_tokens = 65536 if gpu_gb >= 80 else (24576 if gpu_gb >= 20 else 6144)
        self.text = functools.reduce(getattr, self.model.layout.path.split("."), self.hf)  # the decoder stack: no logits at every position
        self.jac = {L: self.lens.jacobians[L].to(self.device) for L in self.layers}
        jpp_ok = self.lens_jpp is not None and all(L in self.lens_jpp.jacobians for L in self.layers)
        self.jpp = {L: self.lens_jpp.jacobians[L].to(self.device) for L in self.layers} if jpp_ok else None
        self.unembed_weight = self.hf.get_output_embeddings().weight
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        transformers.logging.set_verbosity_error()
        self.dtype = next(self.hf.parameters()).dtype
        self.d_model = self.hf.config.get_text_config().hidden_size
        self.image_token_id = int(getattr(self.hf.config, "image_token_id", self.tokenizer.convert_tokens_to_ids("<|image_pad|>")))
        self.vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.cache = {}  # lens Gram matrix and eigenbases (lens.py)
        print(
            f"{name} @ {self.spec.revision[:8]}: {n} layers, {self.dtype}, d = {self.d_model}, on {torch.cuda.get_device_name(0)}; "
            f"lens {self.spec.lens_file} ({'with' if self.jpp else 'no'} J++), read layers {self.layers}; batch {self.batch} / {self.batch_tokens} tokens"
        )
        gc.collect()
        torch.cuda.empty_cache()
        self._check_batching()
        self._preflight()

    # ---- tokens and chat templates
    def n_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def single_ids(self, variants):
        """ids of the variants that are a single token"""
        return sorted({t[0] for v in variants for t in [self.tokenizer.encode(v, add_special_tokens=False)] if len(t) == 1})

    def chat(self, msgs):
        """text-only chat template over (role, text) pairs, ending where the assistant starts writing"""
        return self.tokenizer.apply_chat_template([{"role": r, "content": t} for r, t in msgs], tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def chat_media(self, parts):
        """one user turn from parts: text, or "IMAGE" for a picture"""
        content = [{"type": "image"} if p == "IMAGE" else {"type": "text", "text": p} for p in parts]
        return self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def chat_messages(self, messages):
        """a multi-turn conversation in the processor's message format"""
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    # ---- the multimodal forward
    def encode(self, texts, images):
        """images: one picture per prompt, a list of pictures per prompt (frames), or None"""
        if images is None or all(im is None for im in images):
            return self.tokenizer(list(texts), return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        flat = [f for im in images for f in (im if isinstance(im, (list, tuple)) else [im])]
        if type(self.processor).__name__.startswith("Gemma4"):  # Gemma 4's processor wants one list of pictures per prompt
            flat = [list(im) if isinstance(im, (list, tuple)) else [im] for im in images]
        enc = self.processor(text=list(texts), images=flat, return_tensors="pt", padding=True)
        if "pixel_values" in enc:
            enc["pixel_values"] = enc["pixel_values"].to(self.dtype)
        return enc.to(self.device)

    def _run(self, enc):
        """the forward without logits at every position: the decoder stack for text, the multimodal entry point for pictures"""
        kw = {k: v for k, v in enc.items() if k != "token_type_ids"}
        if "pixel_values" not in kw:
            return self.text(input_ids=kw["input_ids"], attention_mask=kw["attention_mask"], use_cache=False)
        if self.vision_path == "base":
            return self.hf.model(**kw, use_cache=False)
        return self.hf(**kw, use_cache=False, logits_to_keep=1)

    def run_batches(self, texts, images, fn):
        """fn(idx) -> one result per prompt; results in prompt order. Length-sorted batches; a batch that runs out of memory is halved"""
        images = images if images is not None else [None] * len(texts)
        lens = np.array([self.n_tokens(t) + IMG_TOKENS * (len(im) if isinstance(im, (list, tuple)) else 1 if im is not None else 0) for t, im in zip(texts, images)])
        plan, cur = [], []
        for i in np.argsort(lens, kind="stable"):
            if cur and (len(cur) == self.batch or (len(cur) + 1) * lens[i] > self.batch_tokens):
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
                stack += [idx[len(idx) // 2 :], idx[: len(idx) // 2]]
                bar.total += 1
        bar.close()
        torch.cuda.empty_cache()
        return out

    def register(self, L, f):
        """a forward hook on decoder layer L's output; L = -1 is the input to layer 0 (the embeddings, image embeddings scattered in),
        presented to f the same way"""
        if L >= 0:
            return self.model.layers[L].register_forward_hook(f)

        def pre(mod, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            new = f(mod, args, h)
            if new is None:
                return None
            new = new[0] if isinstance(new, tuple) else new
            if args:
                return (new,) + tuple(args[1:]), kwargs
            return args, dict(kwargs, hidden_states=new)

        return self.model.layers[0].register_forward_pre_hook(pre, with_kwargs=True)

    @torch.no_grad()
    def forward(self, texts, images=None, layers=(), where=None, hooks=None):
        """The logits at the last token [n, vocab] (float32, cpu) and {layer: residual [n, P, d]} (float32, cpu) at the positions
        where(row_ids) names (the same P for every row), or at the last token alone when where is None. hooks(enc, idx) -> [(layer, fn)]
        are registered before the captures, so a capture sees the edited residual."""
        keep = set(layers) | {self.final}
        images = images if images is not None else [None] * len(texts)

        def fn(idx):
            enc = self.encode([texts[i] for i in idx], [images[i] for i in idx])
            ids = enc["input_ids"]
            b, s = ids.shape
            pos = torch.as_tensor([(where(ids[j].tolist()) if where else []) + [s - 1] for j in range(b)], device=self.device)
            rows = torch.arange(b, device=self.device)[:, None]
            store = {}

            def capture(L):
                def f(mod, inp, out):
                    store[L] = residual_of(out)[rows, pos].float()

                return f

            handles = [self.register(L, f) for L, f in (hooks(enc, idx) if hooks else [])] + [self.register(L, capture(L)) for L in keep]
            try:
                self._run(enc)
            finally:
                for h in handles:
                    h.remove()
            logits = self.model.unembed(store[self.final][:, -1]).float().cpu()
            return [(logits[j], {L: (store[L][j, :-1] if where else store[L][j, -1:]).cpu() for L in layers}) for j in range(b)]

        rows = self.run_batches(texts, images, fn)
        return torch.stack([r[0] for r in rows]), {L: torch.stack([r[1][L] for r in rows]) for L in layers}

    @torch.no_grad()
    def generate(self, texts, images, n_new, hooks=None):
        """greedy continuations; the hooks stay registered through generation"""

        def fn(idx):
            enc = self.encode([texts[i] for i in idx], [images[i] for i in idx])
            handles = [self.register(L, f) for L, f in (hooks(enc, idx) if hooks else [])]
            try:
                g = self.hf.generate(**enc, max_new_tokens=n_new, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
            finally:
                for h in handles:
                    h.remove()
            return [self.tokenizer.decode(row[enc["input_ids"].shape[1] :], skip_special_tokens=True).strip() for row in g]

        return self.run_batches(texts, images, fn)

    @property
    def audio_token_id(self):
        """the token the processor expands into a clip's audio positions: the processor's id where it gives one (Gemma-4), else
        its audio token's id (the Qwen omni processors name the token only)"""
        tid = getattr(self.processor, "audio_token_id", None)
        if tid is None:
            tid = self.tokenizer.convert_tokens_to_ids(self.processor.audio_token)
        return int(tid)

    def all_layers(self):
        """[-1, 0, ..., final]: every layer an intervention can address. Loads the J-lens maps of all of them"""
        for L in range(self.final):
            if L not in self.jac:
                self.jac[L] = self.lens.jacobians[L].to(self.device)
        return [-1] + list(range(self.final + 1))

    # ---- load-time checks
    def _letter_dists(self, prompts, letter_ids):
        pr = torch.softmax(self._forward_last(prompts)[0], -1)
        P = np.stack([sum(pr[:, t].numpy() for t in letter_ids[l]) for l in "ABCD"], 1)
        return P / np.maximum(P.sum(1, keepdims=True), 1e-9), P.sum(1)

    @torch.no_grad()
    def _forward_last(self, prompts, layers=()):
        """text-only forward through the decoder stack: last-token logits [n, vocab] and {layer: residual [n, d]}"""
        keep = set(layers) | {self.final}

        def fn(idx):
            store = {}

            def capture(L):
                def f(mod, inp, out):
                    store[L] = residual_of(out)[:, -1].float()

                return f

            handles = [self.model.layers[L].register_forward_hook(capture(L)) for L in keep]
            try:
                enc = self.tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
                self.text(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
            finally:
                for h in handles:
                    h.remove()
            logits = self.model.unembed(store[self.final]).float().cpu()
            return [(logits[j], {L: store[L][j] for L in layers}) for j in range(len(idx))]

        # the text planner: length-sorted batches of at most `batch` rows and `batch_tokens` padded tokens
        lens = np.array([self.n_tokens(p) for p in prompts])
        plan, cur = [], []
        for i in np.argsort(lens, kind="stable"):
            if cur and (len(cur) == self.batch or (len(cur) + 1) * lens[i] > self.batch_tokens):
                plan.append(np.array(cur))
                cur = []
            cur.append(int(i))
        if cur:
            plan.append(np.array(cur))
        out = [None] * len(prompts)
        for idx in plan:
            for i, r in zip(idx, fn(idx)):
                out[i] = r
        return torch.stack([r[0] for r in out]), {L: torch.stack([r[1][L] for r in out]) for L in layers}

    @torch.no_grad()
    def _read_ranks(self, prompts, option_ids):
        _, resid = self._forward_last(prompts, self.layers)
        ids = torch.as_tensor(np.asarray(option_ids), device=self.device)
        out = np.zeros((len(prompts), len(self.layers), ids.shape[1]), int)
        for li, L in enumerate(self.layers):
            for s in range(0, len(prompts), self.batch):
                x = self.model.unembed(resid[L][s : s + self.batch] @ self.jac[L].T).float()
                v = x.gather(1, ids[s : s + self.batch])
                out[s : s + self.batch, li] = ((x[:, None, :] > v[:, :, None]).sum(-1) + 1).cpu().numpy()
        return out

    def _check_batching(self):
        """Which answer-letter scheme the model uses, and whether left-padded batches give the numbers lens.apply gives one prompt
        at a time (letter distributions within 0.02, option ranks within 0.25 in log10); otherwise batching is turned off."""
        schemes = {"(": {c: self.single_ids((c, " " + c)) for c in "ABCD"}, "": {c: self.single_ids(("(" + c, "(" + c + ")", c, " " + c)) for c in "ABCD"}}
        qs = [
            "Pick one of these for dinner.\n(A) pizza\n(B) sushi\n(C) pasta\n(D) salad",
            "Pick one of these colours for a room.\n(A) red\n(B) blue\n(C) green\n(D) white",
            "Background: " + "The weather was mild. " * 40 + "Pick one of these pets.\n(A) cat\n(B) dog\n(C) fish\n(D) bird",
        ]
        words = (("pizza", "sushi", "pasta", "salad"), ("red", "blue", "green", "white"), ("cat", "dog", "fish", "bird"))
        opts = [[self.tokenizer.encode(" " + w, add_special_tokens=False) for w in ws] for ws in words]
        assert all(len(t) == 1 for o in opts for t in o), "check words must be single tokens"
        opts = [[t[0] for t in o] for o in opts]
        masses = {p: float(self._letter_dists([self.chat([("user", q + INSTR)]) + p for q in qs], ids)[1].mean()) for p, ids in schemes.items()}
        self.answer_prefix = max(masses, key=masses.get)
        self.letter_ids = schemes[self.answer_prefix]
        prompts = [self.chat([("user", q + INSTR)]) + self.answer_prefix for q in qs]
        print(f"answer readout: prefill {self.answer_prefix!r} (letter mass {masses[self.answer_prefix]:.2f}; the other scheme {min(masses.values()):.2f})")

        def reference(prompt, option_ids):
            jl, ml, _ = self.lens.apply(self.model, prompt, layers=self.layers, positions=[-1])
            pr = torch.softmax(ml[0].float(), -1)
            p = np.array([sum(pr[t].item() for t in self.letter_ids[c]) for c in "ABCD"])
            rk = np.array([[int((jl[L][0] > jl[L][0][t]).sum()) + 1 for t in option_ids] for L in self.layers])
            return p / p.sum(), rk

        ref = [reference(p, o) for p, o in zip(prompts, opts)]

        def deviation():
            dp = np.abs(np.stack([r[0] for r in ref]) - self._letter_dists(prompts, self.letter_ids)[0]).max()
            dr = np.abs(np.log10(np.stack([r[1] for r in ref]).min(1)) - np.log10(self._read_ranks(prompts, opts).min(1))).mean()
            return float(dp), float(dr)

        dp, dr = deviation()
        print(f"batched vs lens.apply: max |dp| = {dp:.4f}, mean |d log10 best rank| = {dr:.3f}")
        self.batch_check = dict(max_dp=dp, mean_dlog10rank=dr, batch=self.batch)
        if dp > 0.02 or dr > 0.25:
            self.batch = 1
            dp, dr = deviation()
            print(f"padding changed the numbers; running unbatched: max |dp| = {dp:.4f}, mean |d log10 best rank| = {dr:.3f}")
            assert dp < 0.02 and dr < 0.25
            self.batch_check.update(unbatched_max_dp=dp, unbatched_mean_dlog10rank=dr, batch=self.batch)

    def _letters(self, texts, images):
        pr = torch.softmax(self.forward(texts, images)[0], -1)
        P = np.stack([sum(pr[:, t].numpy() for t in self.letter_ids[c]) for c in "ABCD"], 1)
        return P / np.maximum(P.sum(1, keepdims=True), 1e-9), P.sum(1)

    def _preflight(self):
        """the model must read a letter off a picture: three cards whose answer exists only in the picture"""
        self.processor = transformers.AutoProcessor.from_pretrained(self.name, revision=self.spec.revision)
        self.processor.tokenizer.padding_side = "left"
        q = "Which letter is written in the picture?\n(A) A\n(B) B\n(C) C\n(D) D" + INSTR
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        texts = [self.chat_messages(msgs) + self.answer_prefix] * 3
        imgs = [stimuli.text_picture(c) for c in "ACD"]
        err = ""
        for path in entry_points(self.spec):
            self.vision_path = path
            try:
                P, mass = self._letters(texts, imgs)
                break
            except Exception as e:  # noqa: BLE001 - the other entry point is tried next
                err = f"{path}: {type(e).__name__}: {str(e)[:160]}"
        else:
            raise RuntimeError(f"the multimodal forward failed on both entry points ({err})")
        picks = "".join("ABCD"[j] for j in P.argmax(1))
        n_ok = sum(a == b for a, b in zip(picks, "ACD"))
        if mass.mean() < 0.5 or n_ok < 2:
            raise RuntimeError(f"the picture is not being read: picks {picks} for A, C, D with letter mass {mass.mean():.2f}")
        print(f"vision path '{self.vision_path}' reads the picture (picks {picks} for A, C, D)")
