"""Audio sweep, static inputs: the audio counterparts of video_sweep2's three inputs, so every video experiment has one on audio.

Why. video_sweep2 ran each scene as a moving video, a frozen video (its middle sampled frame repeated, the same tokens without
motion) and a still picture (that frame once, another format), and asked whether the name and rule horizons stay put (P4, P5)
and whether steering acts the same (moving vs frozen, moving vs still). audio_sweep_qfirst runs the spoken word alone. This run
adds the two counterparts, all three with the question first (the order both audio models pass), on the same scenes:
  q_audio   the recording (the moving video's counterpart)
  q_frozen  the recording with its spoken span frozen: every 40-ms block of the word (one audio token's length) replaced by the
            span's middle block, the silence around it kept (spoken.frozen). The same length, the same 75 audio tokens at the
            same positions; what changes within the word over time goes, as motion goes from a frozen video.
  q_typed   the word typed after the question ("Transcript of the audio: <word>"), no audio: the same content in another format,
            as the still picture is. Its media position is the typed word's one token (a batched.Span, its place from the end of
            the prompt); the run stops unless every word is one token, so the receiver's and donor's typed prompts differ there
            only.
Everything else is audio_sweep's (and so video_sweep2's), unchanged: the question-first schedule of audio_sweep_qfirst
(audio_clean.order_ready for q_audio), donor and self states captured under each row's own question with the alignment check
(for q_typed: the prompts differ only at the typed word), the patching arms, media_steer's phase A and B interventions,
strict.py's whole-answer scoring of first answers with hit_limit, dclean rows, controls_pass, the summaries.

The rule (written before the run) is video_sweep's for each input: the patching gate (clean say >= 0.90 and animal >= 0.85 on
that input's own clean rows; an input below it is "not perceived, untestable", as Gemma-4-E2B's still pictures were), the
machinery check, the positive control, the horizons, the cut and verdicts, P1 to P3, and media_steer's frozen rule and
predictions 1 to 6. Then, as video_sweep's P4 and P5 for the moving video:
- P4 (q_audio's prediction): the spoken and the frozen word give the same horizons, within one window (media_patch.same_horizons).
- P5: the spoken and the typed word give the same horizons, within one window.
  Either is untestable when one of its two inputs is untestable.
- steer_static_compare: video_sweep.steer_compare with q_audio for the moving video: per block, the paired difference in say
  following Y at q_audio's dose on the held-out scenes (>= 16), "same" when its 95% interval lies within +-0.15 in every block.
Expected beforehand, and not part of the rule: freezing may make the word unrecognizable (speech, unlike a picture, is made of
change over time), in which case q_frozen is untestable; that outcome is a result, reported as such. The typed input's single
position makes its principal direction (the PC steering key) undefined; its PC rows are kept and read as a second random control.
Recordings recur across scenes (15 recordings, one voice): intervals over scenes are narrower than intervals over recordings.
Models (25 Nov): also the Qwen omni models, once check_lens passes for their lenses. The frozen input is defined in time (40-ms
blocks, one Gemma-4 audio token); Qwen2.5-Omni also gives a 3-second clip 75 audio tokens, Qwen3-Omni 39 (about 77 ms each), so
there two frozen blocks share a token; the spoken span is frozen all the same.

    uv run experiments/audio_sweep_static.py --model google/gemma-4-E4B-it --parts patch --n 5 --run audio_static_pilot
    uv run experiments/audio_sweep_static.py --model google/gemma-4-E4B-it --run audio_static

Outputs in runs/<run>/<model>/: audio_sweep's (audio_sweep.csv, summary.json with static_compare and steer_static_compare,
scenes.json, stimuli.png with the frozen waveform, gpu.csv, config.json, run.log).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_sweep as sw  # noqa: E402  the sweep
import audio_sweep_qfirst as qf  # noqa: E402  the question-first schedule

INPUTS = ["q_audio", "q_frozen", "q_typed"]

if __name__ == "__main__":
    sw.main(experiment="audio_sweep_static", doc=__doc__, schedule=qf.q_first, default_inputs=",".join(INPUTS))
