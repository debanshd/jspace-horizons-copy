# Omni audio: the pilots (step 4 of `docs/omni_audio_run.md`)

PR #14 at `24cfa40`, run on 25 November 2026 from 8:45 AM PST on one H200 on the US-NC-1 volume, in the omni worktree, with the
three deciding 500-prompt lenses (Hugging Face `a473c078`; the 30B's by `config.LENS_EXCEPTIONS`). The chain runs, per model,
`audio_clean`, `audio_rules`, the spoken / frozen / typed pilot twice (5 scenes, patching), and the pilot gate, then stops for
inspection. Numbers are copied from each run's `summary.json` and the chain log. **This document is partial: the 3B only, so far**;
the 7B and the 30B follow when their runs finish.

## Qwen2.5-Omni-3B: the pilot gate FAILS (the typed input's controls)

- **`audio_clean`: ready.** Every recording named exactly on all three inputs (15 of 15, none at the token limit); every gated
  clean share 1.00 (say, animal, letter, set, ride; receivers and donors). Silence and noise controls: "phone", "mouse", "object"
  (no object name).
- **`audio_rules`:** the animal rule passes for wordings A, B and D on every input. Diagnosis: spoken question-first "too few
  errors" (B 0 wrong, D 1); audio-first "first named" (all errors pick the first-named option); typed "too few errors" (B 8,
  D 1). Variant A matches `audio_clean` exactly.
- **Pilot, spoken (`q_audio`): controls pass.** Self-patch 1.00, every-layer donor patch makes the say answer follow the donor
  (1.00), cut L_say 27, L_use 23.
- **Pilot, frozen (`q_frozen`): not perceived, untestable** (its clean gates fail: donor say 0.20, animal 0.60), as the rule
  allows; the chain goes on.
- **Pilot, typed (`q_typed`): the patch machinery fails.** Self-patch gives the same answer on **0.92** of rows, below the
  1.00 the control asks, although the every-layer donor patch is followed (1.00). By the gate, "the pilot's controls fail": the
  3B's full sweep does not run. A self-patch that changes an answer points at the typed word's patch position or span on this
  model (or a numerical difference between batches), not at the model's audio: the first thing to look at.

Files: `docs/omni_audio_pilot_3b.tar.gz` (sha256 `b20e4036cb2d32eab42cf8748d439d6fc703d16222da2d3b2e5d4b388ea085fb`): the 3B's `audio_clean`, `audio_rules`, both pilot folders (CSVs,
`summary.json`, `stimuli.png`) and the chain log up to that point.
