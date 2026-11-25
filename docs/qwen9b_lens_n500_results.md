# Qwen3.5-9B's lens at 500 prompts: it passes, and the lens was lost

The second pod's step 1 (PR #14 at `8853711`): Qwen3.5-9B's own J-lens, fitted with PR #12's recipe, for the mid-size video runs.
Run 24 to 25 November 2026, 9:18 PM to 4:02 AM PST, on an RTX PRO 6000 in US-PA-1 with no network volume (Qwen3.5's
linear-attention backward refuses Hopper GPUs with the Triton `uv.lock` pins), about $15. Check 1 for the Qwen family had passed
(`docs/lens_checks_results.md`).

| step | prompts | seconds per prompt | lens sha256 (first 12) |
|---|---|---|---|
| timing | 10 | 51.7 | `d0f8c2530fe3` |
| fit, prompts 0–249 | 250 | 46.7 | `e1b6cc1067a7` |
| fit, prompts 250–499 | 250 | 45.8 | `ad63c6877aa3` |
| merged: the 500-prompt lens | 500 | | `9e9b705b43e2` |

**Verdict: passes.** Check 2: the two halves agree at cosine ≥ 0.9946 at every read layer (the bar is 0.99). Check 3 (a caveat,
deciding nothing): pass@10 0.301 against its own logit lens's 0.311, a margin of −0.010, above the void bar of −0.033.

**The lens file was lost.** The pod packed its results at 4:02 AM PST (sha256 `e32ec095…`, 993 MB). My first download failed its
checksum, and before the retry the pod's idle watchdog stopped it (20 minutes without GPU work). The pod had no network volume,
and a stop clears the container disk, so on restart `/workspace` was empty; the pod was then terminated. What remains are the
lines the watcher copied from the run's log, in `docs/qwen9b_lens_n500_watch.txt`: the numbers above. Nothing was uploaded to
Hugging Face. Using this lens needs the fit run again (about 7 hours on an RTX PRO 6000, about $15). The next such run should
download without waiting on the watchdog, or run with a network volume.
