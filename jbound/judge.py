"""LLM judge for answer correctness (rubric CI / XLIX): frozen model and rubric, raw outputs preserved, failed calls never cached.

    from judge import Judge
    J = Judge(run_dir)                      # reads GEMINI_API_KEY from the environment (jb sources /workspace/nayan/.env)
    ok = J.name_match(answers, targets)     # list of True / False / None (None = the judge gave no usable verdict)
    ok = J.qa_match(questions, answers, references)
    J.to_label(rows, "e1")                  # a stratified hand-label sample; judge_agreement() gives Cohen's kappa once filled

The model is pinned to JUDGE_MODEL; the rubric strings are frozen here and hashed into the cache key, so a rubric change never reuses
old verdicts. Raw responses go to <run_dir>/judge_raw.jsonl, verdicts to <run_dir>/judge_cache.json.
"""
import hashlib, json, os, threading, time
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
RUBRIC_ENTAIL = """A vision-language model wrote a free description of a picture. Separately, a question about the same picture has a known reference answer. Decide whether the description, on its own, states or clearly implies the reference answer to that question.
Question: {question}
Reference answer: {reference}
Description: "{description}"
Rules: YES only if a reader of the description alone could answer the question with the reference answer (synonyms count; the exact word is not required). NO if the description does not mention the relevant object, attribute or relation, mentions it without settling the answer, or contradicts the reference.
Return JSON only: {{"verdict": "YES" or "NO"}}"""


class Judge:
    def __init__(self, run_dir, model=JUDGE_MODEL, threads=8):
        self.key = os.environ.get("GEMINI_API_KEY", "")
        self.model, self.threads, self.run_dir = model, threads, run_dir
        self.cache_path = os.path.join(run_dir, "judge_cache.json")
        self.raw_path = os.path.join(run_dir, "judge_raw.jsonl")
        self.cache = json.load(open(self.cache_path)) if os.path.exists(self.cache_path) else {}
        self.lock = threading.Lock()
        self.n_calls = self.n_failed = 0

    def available(self):
        return bool(self.key)

    def _call(self, prompt):
        for attempt in range(5):
            try:
                gen = {"temperature": 0, "maxOutputTokens": 64, "responseMimeType": "application/json"}
                if attempt == 0:
                    gen["thinkingConfig"] = {"thinkingBudget": 0}
                r = requests.post(f"{GEMINI_URL}/models/{self.model}:generateContent", headers={"x-goog-api-key": self.key},
                                  json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}, timeout=60)
                r.raise_for_status()
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                limited = getattr(getattr(e, "response", None), "status_code", None) == 429
                time.sleep((6 if limited else 1.5) * (attempt + 1))
        return ""

    @staticmethod
    def _parse(text):
        up = (text or "").upper()
        try:
            v = json.loads(up[up.index("{"): up.rindex("}") + 1])["VERDICT"]
        except Exception:
            hits = [l for l in ("YES", "NO") if l in up]
            v = hits[0] if len(hits) == 1 else None
        return {"YES": True, "NO": False}.get(v)

    def run(self, prompts):
        """verdicts (True / False / None) for a list of rubric-filled prompts; cached by sha1(model + prompt)"""
        keys = [hashlib.sha1((self.model + p).encode()).hexdigest() for p in prompts]
        todo = [(k, p) for k, p in zip(keys, prompts) if k not in self.cache]
        if todo and self.key:
            with ThreadPoolExecutor(self.threads) as ex:
                outs = list(ex.map(self._call, [p for _, p in todo]))
            with self.lock, open(self.raw_path, "a") as raw:
                for (k, p), txt in zip(todo, outs):
                    v = self._parse(txt)
                    self.n_calls += 1
                    raw.write(json.dumps(dict(key=k, model=self.model, prompt=p, response=txt, verdict=v)) + "\n")
                    if v is not None:
                        self.cache[k] = v
                    else:
                        self.n_failed += 1
                json.dump(self.cache, open(self.cache_path, "w"))
        return [self.cache.get(k) for k in keys]

    def name_match(self, answers, targets):
        return self.run([RUBRIC_NAME.format(target=t, answer=str(a).replace('"', "'")[:200]) for a, t in zip(answers, targets)])

    def qa_match(self, questions, answers, references):
        return self.run([RUBRIC_QA.format(question=q, reference=r, answer=str(a).replace('"', "'")[:200]) for q, a, r in zip(questions, answers, references)])

    def entails(self, questions, references, descriptions):
        return self.run([RUBRIC_ENTAIL.format(question=q, reference=r, description=str(d).replace('"', "'")[:1200]) for q, r, d in zip(questions, references, descriptions)])

    def to_label(self, df, name, n=40, label_col="correct"):
        """n rows to hand-label, stratified by the judge's verdict; fill 'human', then judge_agreement(path) gives kappa"""
        import pandas as pd
        groups = [g.sample(min(len(g), n // 2 + 1), random_state=0) for _, g in df.groupby(label_col, dropna=False)]
        lab = pd.concat(groups).head(n) if groups else df.head(0)
        path = os.path.join(self.run_dir, f"{name}_to_label.csv")
        lab.assign(human="").to_csv(path, index=False)
        return path


def judge_agreement(path, label_col="correct"):
    """Cohen's kappa between the judge column and the filled 'human' column"""
    import pandas as pd
    t = pd.read_csv(path).dropna(subset=["human"])
    a, b = t[label_col].astype(str).str.upper().str.strip(), t.human.astype(str).str.upper().str.strip()
    po = (a == b).mean()
    pe = sum((a == l).mean() * (b == l).mean() for l in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), len(t)
