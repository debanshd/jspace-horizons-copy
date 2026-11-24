# Which PR holds what

A one-page map of the open pull requests in `debanshd/jspace-horizons-copy`, as of 25 November 2026. The PRs are stacked: each
is based on the one before it. A **code** PR holds scripts; a **results** PR holds run outputs (`docs/*_results.md` and
`docs/*.tar.gz`). Nothing has been merged into `main` except PR #1.

## The stack

```text
main
└─ #2  plan.md
   └─ #3  media_steer (code + results)
      └─ #4  media_patch / video_sweep (code)
         ├─ #5  video_sweep results
         └─ #6  clip_naming (code)
            └─ #7  clip_naming results
               └─ #8  video_sweep2 (code)
                  ├─ #9  video_sweep2 results
                  └─ #10 audio (code)
                     ├─ #11 audio results
                     ├─ #15 mid-size video (code; also merged into #14)
                     └─ #12 lens fitting (code)
                        ├─ #13 lens fitting results
                        └─ #14 lens checks now (code; holds #15 too)
```

## Video

| PR | kind | what it holds |
|---|---|---|
| #2 | plan | `plan.md`: the steering pilot on video and sound, with its review (council pass). |
| #3 | code + results | `media_steer`: steering on VGGSound video and sound, on six models. |
| #4 | code | `media_patch` and `video_sweep`: `depth_sweep`'s patching on video, and every intervention on Nayan's matched video set. |
| #5 | results | `video_sweep` and `media_patch` results, five models. |
| #6 | code | `clip_naming` 1–4: the naming gate (does the model name each clip?), including the hedge follow-up and Qwen3.5-0.8B's question. |
| #7 | results | `clip_naming` results, and a prefetch change shown to leave outputs identical. |
| #8 | code | `video_sweep2`: `video_sweep` on each model's reliably named clips, shown whole; the moving gate. |
| #9 | results | **The main video results:** six models, with moving, frozen and still inputs. The 27B and 31B have the patching part only. |
| #15 | code | **Mid-size video:** Qwen3.5-2B and 9B. The 9B reads a third-party J-lens (`--lens jlens`, `bcywinski/jacobian-lens-qwen3.5-9b`), and only after its `check_lens` passes. Run guide: `docs/mid_video_run.md`. Not yet run. |

## Audio

| PR | kind | what it holds |
|---|---|---|
| #10 | code | **The audio experiments:** <ul><li>the whole-answer scorer (`strict.py`);</li><li>`audio_clean`, `audio_rules`, and `audio_sweep` with its `_qfirst` and `_static` variants (spoken / frozen / typed);</li><li>the launcher gates;</li><li>Gemma-4-12B with `check_lens` and `diagnose_lens`;</li><li>`rescore_strict` for the saved video answers.</li></ul> Run guide: `docs/audio_stage3_run.md`. |
| #11 | results | **The audio results**, stages 1–3: <ul><li>the clean gate;</li><li>the rule wordings;</li><li>the full spoken / frozen / typed sweep for E2B and E4B;</li><li>the 12B lens diagnostic;</li><li>the before/after check.</li></ul> |

## Lenses

| PR | kind | what it holds |
|---|---|---|
| #12 | code | **`fit_lens`:** fits our own J-lens. <ul><li>Models: the three Qwen omni models, Gemma-4-26B-A4B and Gemma-4-12B.</li><li>A rule written before any 12B fit: the full fit goes ahead only if its 10-prompt test passes.</li><li>Extending a lens to 1,000 prompts: `EXTEND_TO=1000`.</li></ul> |
| #13 | results | The omni lens fits: <ul><li>the timing runs;</li><li>three 250-prompt lenses in `debanshd/jspace-omni-lenses` at `4648419f`, about $18.50.</li></ul> |
| #14 | code | **Lens checks now** (`scripts/lens_checks.sh`): <ul><li>Qwen3.5-9B's J-lens, which decides;</li><li>the omni models' 250-prompt lenses, a loading test that decides nothing.</li></ul> The Qwen omni models load in `Session`, with the thinker, the end-token and chat-template fixes, and the picture check through the full model. The audio experiments for them stay in the code, for later, on the 1,000-prompt lenses. Run guide: `docs/omni_audio_run.md`, step 1 only. |

## What is run, and what is waiting

| | done | waiting |
|---|---|---|
| video | <ul><li>six models (#9)</li><li>the strict rescoring: 13 of 15 verdicts hold</li></ul> | Qwen3.5-2B and 9B (#15), after the 9B's lens check |
| audio | <ul><li>Gemma-4-E2B and E4B: full spoken / frozen / typed sweep (#11)</li></ul> | <ul><li>the Qwen omni models (#14): lens checks now; audio later</li><li>Gemma-4-12B: lens fit test (#12)</li></ul> |
| lenses | <ul><li>three omni 250-prompt fits (#13)</li></ul> | <ul><li>the lens checks (#14)</li><li>the 1,000-prompt extension, about $50 (#12)</li><li>Gemma-4-12B test, then full fit (#12)</li><li>Gemma-4-26B-A4B fit (#12)</li></ul> |
