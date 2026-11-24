#!/usr/bin/env python
"""Fit a J-lens for a model no published lens covers (config.FIT_MODELS: the Qwen omni models, for audio, and Gemma-4-26B-A4B, the
mid-size Gemma-4, for video), with the lens library the experiments use (jlens, pinned in uv.lock), on WikiText-103, Neuronpedia's
corpus.

    uv run scripts/fit_lens.py --model Qwen/Qwen2.5-Omni-3B --n 10 --run fit_lens_timing      # time it first: about 10 prompts
    uv run scripts/fit_lens.py --model Qwen/Qwen2.5-Omni-3B --n 250 --run fit_lens             # the lens
    uv run scripts/fit_lens.py --model Qwen/Qwen3-Omni-30B-A3B-Instruct --n 250 --shard 0/4 --run fit_lens_s0   # one of 4 GPUs
    uv run scripts/fit_lens.py --model Qwen/Qwen3-Omni-30B-A3B-Instruct --merge runs/fit_lens_s0 runs/fit_lens_s1 ... --run fit_lens
    uv run scripts/fit_lens.py --model Qwen/Qwen2.5-Omni-3B --n 1000 --start 250 --run fit_lens_ext            # prompts 250-999
    uv run scripts/fit_lens.py --model Qwen/Qwen2.5-Omni-3B --merge runs/fit_lens runs/fit_lens_ext --run fit_lens_n1000  # = 1000

What is fitted. The model is loaded with FitSpec.loader at the pinned revision, in bf16: a Qwen omni model as its text-generating
part (the "thinker"), Gemma-4-26B-A4B whole, as the experiments load Gemma; jlens reads it at its decoder. The lens is jlens.fit's estimator with the paper's settings and jlens's defaults: the average
Jacobian from every source layer (0 to the last but one) to the final layer, over 128-token prompts, positions before 16
excluded; jlens.fit's dim_batch (output dimensions per backward pass) sets only the speed. This is the convention of
Neuronpedia's lenses, so the experiments read the result as they read theirs (JacobianLens.load).

The prompts (fixed before any fit). WikiText-103 raw, the first training shard at the pinned revision, its sha256 checked: the
paragraphs of at least MIN_CHARS characters that are not headings, without repeats, in the order of a seed-0 permutation; the
first --n of them (with --shard i/k, every k-th of those from the i-th, so k shards together cover the same --n prompts and
JacobianLens.merge combines them into the lens one machine would fit). --start K fits only prompts K to N-1 of the first N: the
first N prompts of a larger N begin with the first N of a smaller one, so a fitted lens is extended to more prompts by fitting
the rest and merging. --merge takes any parts whose prompts, together, are the first N exactly once each (shards, ranges, or
both; check_cover), fitted with the same model, revision, corpus rule and sequence length; JacobianLens.merge weights each part
by its prompt count, so the merged lens is the one a single fit of the N prompts gives. A prompt that tokenizes too short for the
skipped positions is skipped by jlens.fit and not counted (n_prompts in meta.json).

The report (no pass/fail: scripts/check_lens.py's rule decides once the experiments can load the model). On the last 28
positions of 32 paragraphs of WikiText-103's validation split, read at every fitted layer: the share of positions where the
J-lens, and the plain logit lens, give the model's own next-token prediction, and the median rank of that token in each readout.

Outputs in runs/<run>/<model>/: lens.pt (fp16, jlens's format), meta.json (the model, the corpus and the settings, with
sha256s, the seconds per prompt and the projected time for other prompt counts), report.csv, checkpoint.pt while it runs
(resumed after an interruption), run.log. --push-to REPO also uploads lens.pt and meta.json to that Hugging Face repo under
<model name>/ (--push-dir: another folder in the repo, so a new lens does not replace one already pinned; needs a write token in
.env); the experiments then pin that repo's commit and the lens's sha256 in config.py.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jspace import config
from jspace.runs import Run, load_env

MIN_CHARS = 600
SEED = 0
MAX_SEQ_LEN = 128
REPORT_N = 32
CHECKPOINT_EVERY = 10  # prompts: a checkpoint holds every layer's d x d running sum in fp32


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def wikitext(data_dir, split):
    """the split's file at the pinned revision, checked; its path"""
    from huggingface_hub import hf_hub_download

    rel, expected = config.WIKITEXT_FILES[split]
    path = Path(hf_hub_download(config.WIKITEXT_REPO, rel, repo_type="dataset", revision=config.WIKITEXT_REVISION, local_dir=Path(data_dir) / "wikitext"))
    got = sha256(path)
    if got != expected:
        raise SystemExit(f"{rel}: sha256 {got}, pinned {expected}")
    return path


def paragraphs(texts, min_chars=MIN_CHARS, seed=SEED):
    """the corpus rule: paragraphs of at least min_chars characters, no headings (" = Title = "), no repeats, seed-permuted"""
    seen, out = set(), []
    for t in texts:
        t = str(t).strip()
        if len(t) < min_chars or t.startswith("=") or t in seen:
            continue
        seen.add(t)
        out.append(t)
    order = np.random.default_rng(seed).permutation(len(out))
    return [out[i] for i in order]


def covered(n, start=0, shard=None):
    """the indices, in the corpus rule's order, of the prompts a fit with these settings reads"""
    assert 0 <= start < n, (start, n)
    idx = list(range(start, n))
    if shard is None:
        return idx
    i, k = shard
    assert 0 <= i < k, shard
    return idx[i::k]


def select(prompts, n, shard=None, start=0):
    """prompts start to n-1 of the first n, or with shard (i, k) every k-th of those from the i-th"""
    if len(prompts) < n:
        raise SystemExit(f"the corpus has {len(prompts)} prompts, fewer than --n {n}")
    return [prompts[j] for j in covered(n, start, shard)]


def check_cover(metas):
    """the parts of a merge: together, the first N prompts exactly once each, for N their largest --n. Returns N"""
    n = max(m["n_requested"] for m in metas)
    seen = [j for m in metas for j in covered(m["n_requested"], m.get("start", 0), tuple(m["shard"]) if m.get("shard") else None)]
    if sorted(seen) != list(range(n)):
        dup = sorted({j for j in seen if seen.count(j) > 1})[:5]
        raise SystemExit(f"the parts do not cover prompts 0 to {n - 1} exactly once (repeated: {dup}; missing: {sorted(set(range(n)) - set(seen))[:5]})")
    return n


def parse_shard(s):
    if s is None:
        return None
    i, k = (int(x) for x in s.split("/"))
    return i, k


def load(name):
    """the model's text-generating part at the pinned revision, in bf16 on the GPU, and its tokenizer"""
    import transformers

    spec = config.FIT_MODELS[name]
    cls = getattr(transformers, spec.loader)
    hf = cls.from_pretrained(name, revision=spec.revision, dtype=torch.bfloat16).cuda().eval()
    tok = transformers.AutoTokenizer.from_pretrained(name, revision=spec.revision)
    text = hf.config.get_text_config()
    if (text.num_hidden_layers, text.hidden_size) != (spec.n_layers, spec.d_model):
        raise SystemExit(f"{name}: {text.num_hidden_layers} layers, d {text.hidden_size}; config.FIT_MODELS says {spec.n_layers}, {spec.d_model}")
    return hf, tok


REPORT_POSITIONS = list(range(-28, 0))  # each text's last 28 positions: past the attention-sink start, and the logits stay small
REPORT_CHUNK = 8  # layers per forward: a readout is [positions, vocabulary] per layer, on the CPU


def report(lm, lens, texts):
    """per fitted layer: the J-lens's and the logit lens's top-1 agreement with the model, and the median rank of its top token"""
    acc = {}
    layers = list(lens.source_layers)
    with torch.no_grad():
        for t in texts:
            for kind, use in (("jlens", True), ("logit", False)):
                for c in range(0, len(layers), REPORT_CHUNK):
                    lens_logits, model_logits, _ = lens.apply(lm, t, layers=layers[c : c + REPORT_CHUNK], positions=REPORT_POSITIONS, max_seq_len=MAX_SEQ_LEN, use_jacobian=use)
                    top = model_logits.argmax(-1)
                    for L, lg in lens_logits.items():
                        ranks = (lg > lg.gather(1, top[:, None])).sum(1) + 1
                        a = acc.setdefault((kind, L), [0, 0, []])
                        a[0] += int((lg.argmax(-1) == top).sum())
                        a[1] += int(top.numel())
                        a[2] += ranks.tolist()
    rows = [dict(readout=k, layer=L, depth=L / lm.n_layers, top1=h / n, median_rank=float(np.median(r))) for (k, L), (h, n, r) in sorted(acc.items())]
    return pd.DataFrame(rows)


def push(run, repo, folder):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, exist_ok=True)
    for f in ("lens.pt", "meta.json", "report.csv"):
        api.upload_file(path_or_fileobj=str(run.path(f)), path_in_repo=f"{folder}/{f}", repo_id=repo)
    print(f"uploaded to {repo}/{folder}/: pin the repo's commit and the lens's sha256 in config.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(config.FIT_MODELS))
    ap.add_argument("--n", type=int, default=250, help="prompts (the corpus rule's first N)")
    ap.add_argument("--shard", help="i/k: fit every k-th of the N prompts from the i-th; merge the k shards with --merge")
    ap.add_argument("--start", type=int, default=0, help="fit prompts START to N-1 only (to extend a lens of START prompts to N)")
    ap.add_argument("--merge", nargs="+", type=Path, help="run folders of parts (shards, ranges) to merge into one lens (no fitting)")
    ap.add_argument("--dim-batch", type=int, default=8, help="output dimensions per backward pass: speed and memory only")
    ap.add_argument("--run", default="fit_lens")
    ap.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--push-to", help="a Hugging Face model repo to upload lens.pt, meta.json and report.csv to")
    ap.add_argument("--push-dir", help="the folder in that repo (default: the model's name)")
    args = ap.parse_args()
    load_env()
    assert torch.cuda.is_available(), "fitting needs a CUDA GPU"
    import jlens
    from jlens import JacobianLens

    run = Run(argparse.Namespace(model=args.model, run=args.run, runs_dir=args.runs_dir), "fit_lens")
    spec = config.FIT_MODELS[args.model]
    hf, tok = load(args.model)
    lm = jlens.from_hf(hf, tok)
    val = paragraphs(pd.read_parquet(wikitext(args.data_dir, "validation"))["text"])[:REPORT_N]
    meta = dict(model=args.model, revision=spec.revision, loader=spec.loader, n_layers=lm.n_layers, d_model=lm.d_model, gpu=torch.cuda.get_device_name(0))
    meta["corpus"] = dict(
        repo=config.WIKITEXT_REPO, revision=config.WIKITEXT_REVISION, file=config.WIKITEXT_FILES["train"][0], sha256=config.WIKITEXT_FILES["train"][1], min_chars=MIN_CHARS, seed=SEED
    )
    t0 = time.time()
    if args.merge:
        parts = [JacobianLens.load(str(Path(d) / args.model.split("/")[-1] / "lens.pt")) for d in args.merge]
        metas = [json.loads((Path(d) / args.model.split("/")[-1] / "meta.json").read_text()) for d in args.merge]
        if {(m["model"], m["revision"], m["max_seq_len"], json.dumps(m["corpus"], sort_keys=True)) for m in metas} != {
            (args.model, spec.revision, MAX_SEQ_LEN, json.dumps(meta["corpus"], sort_keys=True))
        }:
            raise SystemExit("the parts differ in model, revision, sequence length or corpus rule")
        n = check_cover(metas)
        lens = JacobianLens.merge(parts)
        parts_meta = [
            dict(run=str(d), n_requested=m["n_requested"], start=m.get("start", 0), shard=m.get("shard"), n_prompts=m["n_prompts"], lens_sha256=m["lens_sha256"]) for d, m in zip(args.merge, metas)
        ]
        meta.update(n_requested=n, start=0, shard=None, merged=parts_meta, seconds=sum(m["seconds"] for m in metas))
    else:
        prompts = select(paragraphs(pd.read_parquet(wikitext(args.data_dir, "train"))["text"]), args.n, parse_shard(args.shard), args.start)
        print(f"{args.model} @ {spec.revision[:8]}: {lm.n_layers} layers, d {lm.d_model}; {len(prompts)} prompts (of --n {args.n}, from {args.start}, shard {args.shard})")
        lens = jlens.fit(lm, prompts, dim_batch=args.dim_batch, max_seq_len=MAX_SEQ_LEN, checkpoint_path=str(run.path("checkpoint.pt")), checkpoint_every=CHECKPOINT_EVERY)
        sec = time.time() - t0
        meta.update(n_requested=args.n, start=args.start, shard=parse_shard(args.shard), seconds=sec, seconds_per_prompt=sec / max(lens.n_prompts, 1))
        meta["projected_hours_one_gpu"] = {k: round(meta["seconds_per_prompt"] * k / 3600, 2) for k in (100, 250, 1000)}
    lens.save(str(run.path("lens.pt")))
    meta.update(
        n_prompts=int(lens.n_prompts),
        source_layers=list(lens.source_layers),
        target_layer=lm.n_layers - 1,
        max_seq_len=MAX_SEQ_LEN,
        dim_batch=args.dim_batch,
        lens_sha256=sha256(run.path("lens.pt")),
    )
    R = report(lm, lens, val)
    run.save_csv(R, "report.csv")
    meta["report"] = {k: {int(L): round(float(v), 4) for L, v in zip(g["layer"], g["top1"])} for k, g in R.groupby("readout")}
    run.save_json(meta, "meta.json")
    if run.path("checkpoint.pt").exists():
        run.path("checkpoint.pt").unlink()  # the lens is saved; the running sums are not needed again
    for _, r in R.iterrows():
        print(f"{r['readout']:5s} layer {int(r['layer']):2d} ({r['depth']:.2f}): top-1 {r['top1']:.3f}  median rank {r['median_rank']:.0f}")
    print(f"lens {meta['n_prompts']} prompts, sha256 {meta['lens_sha256'][:12]}; {meta.get('seconds_per_prompt', 0):.1f} s per prompt; projected {meta.get('projected_hours_one_gpu')}")
    if args.push_to:
        push(run, args.push_to, args.push_dir or args.model.split("/")[-1])


if __name__ == "__main__":
    main()
