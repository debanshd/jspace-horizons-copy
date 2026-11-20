"""The matched spoken-audio set for experiments/audio_naming.py and experiments/audio_sweep.py: Nayan Saxena's
saxenan3/jspace-matched-spoken-audio.

The 64 donor-receiver scenes of the matched video set (the same scene ids, labels and set-question names), each side a
recording of the object's name only: 15 distinct recordings of one synthetic voice (af_bella, Kokoro-82M), every one a 3-second,
16 kHz, mono PCM16 WAV that starts with 250 ms of silence, so every recording gives the same number of audio samples. Words and
waveforms recur across scenes and roles: the 64 scenes are not 64 independent utterances (the dataset's protocol.json).

Two prompt orders, the inputs of both experiments (Gemma-4's audio tower; the dataset asks for native chat formatting and for
the order to be an explicit factor):
  q_audio   the question, then the audio: Gemma-4's card's order, as media_steer asked about sounds. The audio comes after the
            question, so its states can depend on the question: donor and self states are captured under the receiver's exact
            question (the dataset's protocol).
  audio_q   the audio, then the question: the order every video experiment used. The audio's states cannot depend on the
            question in a causal decoder; they are captured under each question all the same, so both orders run one code path.
The questions are the dataset's protocol.json wording (QUESTIONS), checked against the file by scripts/prepare_spoken.py.
"""

import hashlib
import json
import wave
from pathlib import Path

import numpy as np

DIR = "matched_audio"
SAMPLE_RATE = 16000
N_SAMPLES = 48000  # 3 s
VOICE = "af_bella"
INPUTS = ["q_audio", "audio_q"]
QUESTIONS = {
    "say": "What object name is spoken in the audio? Reply with the noun only.",
    "animal": "Rule: if the object named in the audio is an animal, answer 7; otherwise answer 3. Reply with the number only.",
    "letter": "What is the first letter of the object name spoken in the audio? Reply with the letter only.",
    "set": "Rule: if the object named in the audio is a {a}, a {b} or a {c}, answer 7; otherwise answer 3. Reply with the number only.",
    "ride": "Rule: if the object named in the audio is something people ride or drive, answer 7; otherwise answer 3. Reply with the number only.",
}


def scenes(data_dir):
    """the 64 scenes of the primary voice, in scene order"""
    path = Path(data_dir) / DIR / "metadata.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found: run `uv run scripts/prepare_spoken.py` first")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted((r for r in rows if r["voice_id"] == VOICE), key=lambda r: r["scene_id"])


def recordings(rows):
    """{audio id: (label, relative path, sha256)} over both roles of every scene"""
    out = {}
    for r in rows:
        for role in ("donor", "receiver"):
            rec = (r[f"{role}_label"], r[f"{role}_audio"], r[f"{role}_sha256"])
            assert out.setdefault(r[f"{role}_audio_id"], rec) == rec, (r["row_id"], role)
    return out


def load_wav(path):
    """a PCM16 mono WAV as float32 samples in [-1, 1); the rate and length are checked, so no recording is resampled or cut"""
    with wave.open(str(path)) as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, SAMPLE_RATE), (path, f.getparams())
        x = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    assert x.shape == (N_SAMPLES,), (path, x.shape)
    return (x.astype(np.float32) / 32768.0).copy()


def silence():
    """the no-speech control: 3 s of digital silence"""
    return np.zeros(N_SAMPLES, np.float32)


def noise(seed=0, rms=0.02):
    """the noise control: 3 s of white noise at a speech-like level, from a fixed seed"""
    return (np.random.default_rng(seed).standard_normal(N_SAMPLES) * rms).astype(np.float32)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content(inp, question, audio_part=None):
    """a user turn's parts in the input's order: the question and the audio (a {"type": "audio"} part)"""
    a = audio_part or {"type": "audio"}
    q = {"type": "text", "text": question}
    if inp == "q_audio":
        return [q, a]
    if inp == "audio_q":
        return [a, q]
    raise ValueError(inp)
