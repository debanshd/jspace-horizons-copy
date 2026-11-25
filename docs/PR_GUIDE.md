# PR guide

- PR 1 → early setup (merged into main)
- PR 2 → plan.md: the video/sound steering plan
- PR 3 → media_steer: steering code + results on VGGSound video and sound
- PR 4 → video patching code (media_patch, video_sweep) on Nayan's matched video set
- PR 5 → video_sweep and media_patch results
- PR 6 → naming gate code (clip_naming 1–4)
- PR 7 → naming gate results
- PR 8 → video_sweep2 code (rerun on well-named clips)
- PR 9 → video_sweep2 results: the main video results, six models
- PR 10 → audio experiments code (strict scorer, audio_clean, audio_rules, audio sweeps, 12B lens check)
- PR 11 → audio results: stages 1–3, Gemma-4-E2B and E4B
- PR 12 → lens-fitting code (Qwen omni models, Gemma-4-26B-A4B, Gemma-4-12B; extension to 1000 prompts)
- PR 13 → lens-fitting results: the three Qwen omni 250-prompt lenses
- PR 14 → lens checks (Qwen omni lenses, Qwen3.5-9B J-lens: done); the Qwen calibration and Qwen3-Omni diagnosis (done); now the Qwen rule (rank-based, new lenses only) and the omni answer prefix chosen on card B; Qwen omni models loadable; includes PR 15
- PR 15 → mid-size video code: Qwen3.5-2B and 9B
