# Omni audio: the pilots (step 4 of `docs/omni_audio_run.md`)

PR #14 at `24cfa40`, run on 25 November 2026 from 8:45 AM PST on one H200 on the US-NC-1 volume, in the omni worktree, with the
three deciding 500-prompt lenses (Hugging Face `a473c078`; the 30B's by `config.LENS_EXCEPTIONS`, so its results carry that
caveat). The pod ran 8:45 to 9:10 AM PST (about $1.75). The chain runs, per model,
`audio_clean`, `audio_rules`, the spoken / frozen / typed pilot twice (5 scenes, patching), and the pilot gate, then stops for
inspection. Numbers are copied from each run's `summary.json` and the chain log. Every pilot gave identical
answers when repeated (3B 2,550 rows, 7B 2,100, 30B 3,225).

| model | `audio_clean` | spoken pilot | frozen | typed pilot | pilot gate |
|---|---|---|---|---|---|
| Qwen2.5-Omni-3B | ready | together (L_say 27, L_use 23) | not perceived, untestable | **patch machinery fails** (self-patch 0.92) | **fails** |
| Qwen2.5-Omni-7B | ready | together (19 / 15) | not perceived, untestable | together (19 / 15) | passes |
| Qwen3-Omni-30B-A3B | ready | too late (39 / 31) | not perceived, untestable | too late (39 / 19) | passes |

The pilot's predictions come from 5 scenes and decide nothing: P1 is missed where the verdict is "together" (the 3B's and
7B's spoken input, the 7B's typed) and kept where it is "too late" (the 30B's spoken and typed).

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

## Qwen2.5-Omni-7B: the pilot gate passes

- **`audio_clean`: ready**, 15 of 15 recordings named on every input. Typed `ride` clean 0.60 (donors 0.80), the one share below
  1.00; spoken all 1.00.
- **`audio_rules`:** variant A matches `audio_clean` exactly.
- **Pilot:** spoken and typed controls pass (self-patch 1.00, every-layer donor patch followed 1.00), both "together" at L_say 19,
  L_use 15; frozen not perceived (every-layer donor patch followed 0.00), untestable.

## Qwen3-Omni-30B-A3B: the pilot gate passes (its lens by the recorded exception)

- **`audio_clean`: ready.** Spoken question-first named 12 of 15 exactly: three answers ran to the token limit as "Assistant:
  Bus", "Assistant: Chair", "Assistant: Train" (the right object behind a role prefix, so scored wrong under the strict scorer).
  Audio-first and typed: 15 of 15.
- **`audio_rules`:** variant A matches `audio_clean` exactly.
- **Pilot:** spoken and typed controls pass; spoken "too late" (L_say 39, L_use 31), typed "too late" (L_say 39, L_use 19);
  frozen not perceived, untestable.

## Next

The chain's next step is the full sweep (`FULL=1`) of the models whose pilots passed, the 7B and the 30B, after inspection.

Files: `docs/omni_audio_pilots.tar.gz` (sha256 `45091a92f27a720e8d0e8e5deaf0804b4e48a3cf1771b4a8d13acb637f187eb7`): for all three models `audio_clean`, `audio_rules`, both pilot
folders (CSVs, `summary.json`, `stimuli.png`), the chain logs. The 3B-only archive of 8:55 AM PST (sha256 `b20e4036…`) is replaced by this one, which contains the same 3B files.
