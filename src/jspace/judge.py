"""The LLM judge for answer correctness: Gemini 2.5 Flash with frozen rubrics, raw outputs kept, failed calls never cached.

Verdicts are cached by sha1(model + filled rubric), so the same answer to the same question is judged once. The verdicts of the
original runs ship in results/judge_cache.json and are consulted first: a re-run whose answers are identical needs no API key, and
any new answer is sent to Gemini (GEMINI_API_KEY). Every verdict a run uses is written to its own judge_cache.json.
"""

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

JUDGE_MODEL = "gemini-2.5-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
RUBRIC_NAME = """A vision-language model was asked to name the object in a picture. Decide whether its answer names the target object class.
Target object class: {target}
Model answer: "{answer}"
Rules: YES if the answer names the target class or an unambiguous synonym / hyponym (e.g. "puppy" for dog, "jet" for airplane, "notebook computer" for laptop), even with extra words.
NO if it names a different object, a letter or number, a non-object word, nothing, or hedges between several objects.
Return JSON only: {{"verdict": "YES" or "NO"}}"""
RUBRIC_QA = """A vision-language model answered a question about a picture or video. Decide whether its answer is correct given the reference answer.
Question: {question}
Reference answer: {reference}
Model answer: "{answer}"
Rules: YES if the model's answer has the same meaning as the reference (synonyms, plural/singular, "yes"/"no" phrased as a sentence, a number written as a word all count). NO if it is a different answer, contradicts the reference, hedges between answers, or does not answer.
Return JSON only: {{"verdict": "YES" or "NO"}}"""


class Judge:
    def __init__(self, run_dir, reference_caches=(), model=JUDGE_MODEL, threads=8):
        self.key = os.environ.get("GEMINI_API_KEY", "")
        self.model, self.threads = model, threads
        self.run_dir = str(run_dir)
        self.cache_path = os.path.join(self.run_dir, "judge_cache.json")
        self.raw_path = os.path.join(self.run_dir, "judge_raw.jsonl")
        self.cache = json.load(open(self.cache_path)) if os.path.exists(self.cache_path) else {}
        self.reference = {}
        for path in reference_caches:
            if os.path.exists(path):
                self.reference.update(json.load(open(path)))
        self.lock = threading.Lock()
        self.n_calls = self.n_failed = 0

    def _call(self, prompt):
        for attempt in range(5):
            try:
                gen = {"temperature": 0, "maxOutputTokens": 64, "responseMimeType": "application/json"}
                if attempt == 0:
                    gen["thinkingConfig"] = {"thinkingBudget": 0}
                r = requests.post(
                    f"{GEMINI_URL}/models/{self.model}:generateContent",
                    headers={"x-goog-api-key": self.key},
                    json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen},
                    timeout=60,
                )
                r.raise_for_status()
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:  # noqa: BLE001 - retried, then reported as no verdict
                limited = getattr(getattr(e, "response", None), "status_code", None) == 429
                time.sleep((6 if limited else 1.5) * (attempt + 1))
        return ""

    @staticmethod
    def parse(text):
        """True / False from the judge's JSON (or a bare YES / NO), None when there is no usable verdict"""
        up = (text or "").upper()
        try:
            v = json.loads(up[up.index("{") : up.rindex("}") + 1])["VERDICT"]
        except Exception:  # noqa: BLE001 - malformed JSON falls back to the bare words
            hits = [w for w in ("YES", "NO") if w in up]
            v = hits[0] if len(hits) == 1 else None
        return {"YES": True, "NO": False}.get(v)

    def run(self, prompts):
        """verdicts (True / False / None) for rubric-filled prompts"""
        keys = [hashlib.sha1((self.model + p).encode()).hexdigest() for p in prompts]
        for k in keys:
            if k not in self.cache and k in self.reference:
                self.cache[k] = self.reference[k]
        todo = list({k: p for k, p in zip(keys, prompts) if k not in self.cache}.items())
        if todo and not self.key:
            raise RuntimeError(f"{len(todo)} answers have no recorded verdict: set GEMINI_API_KEY (in .env) so the judge can score them")
        if todo:
            with ThreadPoolExecutor(self.threads) as ex:
                outs = list(ex.map(self._call, [p for _, p in todo]))
            with self.lock, open(self.raw_path, "a") as raw:
                for (k, p), txt in zip(todo, outs):
                    v = self.parse(txt)
                    self.n_calls += 1
                    raw.write(json.dumps(dict(key=k, model=self.model, prompt=p, response=txt, verdict=v)) + "\n")
                    if v is not None:
                        self.cache[k] = v
                    else:
                        self.n_failed += 1
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)
        return [self.cache.get(k) for k in keys]

    def name_match(self, answers, targets):
        return self.run([RUBRIC_NAME.format(target=t, answer=str(a).replace('"', "'")[:200]) for a, t in zip(answers, targets)])

    def qa_match(self, questions, answers, references):
        return self.run([RUBRIC_QA.format(question=q, reference=r, answer=str(a).replace('"', "'")[:200]) for q, a, r in zip(questions, answers, references)])

    def label_sample(self, df, name, n=40, label_col="correct"):
        """n rows to hand-label, stratified by the judge's verdict: fill the 'human' column, then agreement() gives Cohen's kappa"""
        import pandas as pd

        groups = [g.sample(min(len(g), n // 2 + 1), random_state=0) for _, g in df.groupby(label_col, dropna=False)]
        lab = pd.concat(groups).head(n) if groups else df.head(0)
        path = os.path.join(self.run_dir, f"{name}_to_label.csv")
        lab.assign(human="").to_csv(path, index=False)
        return path


def agreement(path, label_col="correct"):
    """Cohen's kappa between the judge's column and the filled 'human' column of a label sample, and the number of rows"""
    import pandas as pd

    t = pd.read_csv(path).dropna(subset=["human"])
    a, b = t[label_col].astype(str).str.upper().str.strip(), t.human.astype(str).str.upper().str.strip()
    po = (a == b).mean()
    pe = sum((a == lab).mean() * (b == lab).mean() for lab in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), len(t)
