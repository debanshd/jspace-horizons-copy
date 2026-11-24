"""Whole-answer scoring of the five matched-set questions, for the audio experiments and for rescoring saved answers of the
image and video ones (scripts/rescore_strict.py).

Why. The scoring the image and video experiments inherit takes the first vocabulary word of a say answer and the first
character of every other answer, so "not a cat" scores as cat, "7 or 3" as 7 and "Cannot determine" as the letter C. Here the
whole answer must be the answer and nothing else:
- The answer is cleaned: surrounding white space, markdown marks, quotes and brackets stripped, a trailing . ! , ; or :
  stripped, lowercased, inner white space collapsed.
- say: exact when what is left, after one leading article (a, an, the), is the word itself. A plural, or a kind or a spelling
  of the word (SYNONYMS: clip_naming2's list), is reported apart as a synonym, not as exact recognition. phrase(), a
  modifier before the word ("sports car"), is a further level for rescoring pictures and videos, reported apart.
- letter: what is left is the one letter.  set, animal, ride: what is left is the digit, 7 or 3.
- An answer that hit the token limit without ending is not a whole answer: it is neither correct nor valid.
valid says whether the answer has the form of an answer at all (one vocabulary word or synonym, one letter, 7 or 3), to tell a
wrong answer from one that is not an answer.
"""

import re

from . import stimuli

VOCAB = list(stimuli.VOCAB)
SYNONYMS = {
    "plane": "airplane",
    "aeroplane": "airplane",
    "jet": "airplane",
    "aircraft": "airplane",
    "airliner": "airplane",
    "sailboat": "boat",
    "ship": "boat",
    "yacht": "boat",
    "kitten": "cat",
    "puppy": "dog",
    "pony": "horse",
    "mug": "cup",
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "locomotive": "train",
    "automobile": "car",
    "duck": "bird",
    "chicken": "bird",
    "goose": "bird",
    "pigeon": "bird",
    "gull": "bird",
}  # fmt: skip  (clip_naming2's EXTRA_NAMES2, word for word)
EDGE = "\\s*_`\"'\u201c\u201d\u2018\u2019()\\[\\]"
LEAD = re.compile(f"^[{EDGE}]+")
TRAIL = re.compile(f"[{EDGE}.!,;:]+$")
ARTICLE = re.compile(r"^(a|an|the)\s+")


def clean(text):
    """the answer without its wrapping: "**Cat.**" -> "cat", " 7\\n" -> "7" """
    return " ".join(TRAIL.sub("", LEAD.sub("", str(text))).lower().split())


def singulars(a):
    out = {a}
    if a.endswith("es"):
        out.add(a[:-2])
    if a.endswith("s"):
        out.add(a[:-1])
    return out


def say(text, word):
    """(exact, synonym): the answer is the word; the answer is the word, its plural, or one of its SYNONYMS (or their plural)"""
    a = ARTICLE.sub("", clean(text))
    f = singulars(a)
    return a == word, word in f or any(SYNONYMS.get(x) == word for x in f)


NOT_ONE = {"not", "no", "non", "or", "and", "nor", "maybe", "possibly", "either", "neither"}


def phrase(text, word):
    """a sensitivity level for a picture or a video, reported apart, never the primary score: the answer is at most three
    words, names no list and no negation, and its last word is the word at the synonym level ("sports car", "wall clock")"""
    a = ARTICLE.sub("", clean(text))
    toks = a.split()
    if not 1 <= len(toks) <= 3 or re.search(r"[,;/|&:]", a) or NOT_ONE & set(toks):
        return False
    return say(toks[-1], word)[1]


def truth(task, word, membership):
    """the answer `word` gives to `task` (matched.truth's)"""
    if task == "say":
        return word
    if task == "letter":
        return word[0].upper()
    if task == "set":
        return "7" if word in membership else "3"
    return "7" if word in (stimuli.RIDDEN if task == "ride" else stimuli.ANIMALS) else "3"


def matches(task, text, word, membership, hit_limit=False):
    """is the whole answer `word`'s answer to `task` (exact recognition for say)"""
    if hit_limit:
        return False
    if task == "say":
        return say(text, word)[0]
    return clean(text).upper() == truth(task, word, membership)


def valid(task, text, hit_limit=False):
    """has the answer the form of an answer to `task`"""
    if hit_limit:
        return False
    a = clean(text)
    if task == "say":
        return any(say(text, w)[1] for w in VOCAB)
    if task == "letter":
        return len(a) == 1 and a.isalpha()
    return a in ("7", "3")


def score(task, text, word, membership, hit_limit=False):
    """the receiver's word: correct (exact), synonym (say only: exact or a plural, kind or spelling), valid"""
    syn = (not hit_limit and say(text, word)[1]) if task == "say" else matches(task, text, word, membership, hit_limit)
    return dict(correct=matches(task, text, word, membership, hit_limit), synonym=bool(syn), valid=valid(task, text, hit_limit))
