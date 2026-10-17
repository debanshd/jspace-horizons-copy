"""The pictures the models see: COCO object crops in numbered slots, printed words, labelled crops, and resized photos.

A canvas is a grid of numbered 224 px slots; on Qwen3.5 each slot covers exactly 7 x 7 = 49 image tokens, so the tokens of an
object are known. Slot numbers and printed words are drawn in DejaVu Sans, which is part of every stimulus: a missing font would
change the pictures, so the font check in Session fails loudly instead.
"""

import collections
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .analysis import noun_matches

VOCAB = ["dog", "cat", "car", "bus", "chair", "clock", "cup", "bird", "train", "horse", "bicycle", "airplane", "boat", "bed", "pizza", "laptop"]
ANIMALS = {"dog", "cat", "bird", "horse"}
RIDDEN = {"horse", "car", "bus", "train", "bicycle", "airplane", "boat"}  # "something people ride or drive"
SLOT, CROP, MARGIN = 224, 184, 0.10  # slot size, crop tile size, margin around a COCO box
BG, EMPTY = (235, 235, 235), (210, 210, 210)
COLS, ROWS = 3, 2  # the six-slot canvas
N_SLOTS = COLS * ROWS
PER_CLASS = 12  # crops per class in a crop set
SEED = 0
GATE_QUESTION = "What object is this? Reply with the noun only."
FONT_FILES = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def font(size):
    """the stimulus font (slot numbers, printed words): DejaVu Sans Bold"""
    for path in FONT_FILES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def sheet_font(size):
    """the label font of contact sheets and the vision preflight card: DejaVu Sans"""
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def canvas(n_cols, n_rows, slots):
    """a grid of numbered slots; slots = {slot number: PIL tile or None}, numbered row-major from 1"""
    img = Image.new("RGB", (n_cols * SLOT, n_rows * SLOT), BG)
    d = ImageDraw.Draw(img)
    f = font(30)
    for s in range(1, n_cols * n_rows + 1):
        r, c = divmod(s - 1, n_cols)
        x0, y0 = c * SLOT, r * SLOT
        if slots.get(s) is None:
            d.rectangle([x0 + 4, y0 + 4, x0 + SLOT - 5, y0 + SLOT - 5], fill=EMPTY)
        else:
            img.paste(slots[s], (x0 + (SLOT - CROP) // 2, y0 + (SLOT - CROP) // 2))
        d.rectangle([x0 + 6, y0 + 6, x0 + 44, y0 + 44], fill="white", outline="black")
        d.text((x0 + 14, y0 + 8), str(s), fill="black", font=f)
    return img


def scene(slot, tile):
    """the six-slot canvas with one tile in one slot (tile None: the empty canvas)"""
    return canvas(COLS, ROWS, {s: (tile if s == slot else None) for s in range(1, N_SLOTS + 1)})


def slot_tokens(n_cols, n_rows, s):
    """row-major image-token indices (within the image's tokens) covered by slot s, each slot 7 x 7 tokens"""
    r, c = divmod(s - 1, n_cols)
    W = n_cols * (SLOT // 32)
    return [(r * 7 + i) * W + (c * 7 + j) for i in range(7) for j in range(7)]


def word_tile(word):
    """the word printed on a crop-sized tile: the same fact as a picture of text"""
    tile = Image.new("RGB", (CROP, CROP), BG)
    d = ImageDraw.Draw(tile)
    f = font(40)
    bb = d.textbbox((0, 0), word, font=f)
    d.text(((CROP - (bb[2] - bb[0])) // 2 - bb[0], (CROP - (bb[3] - bb[1])) // 2 - bb[1]), word, fill="black", font=f)
    return tile


def label_tile(crop, word):
    """a crop with a word printed on a white band across its lower part"""
    tile = crop.copy()
    d = ImageDraw.Draw(tile)
    f = font(34)
    bb = d.textbbox((0, 0), word, font=f)
    h = bb[3] - bb[1] + 18
    d.rectangle([0, CROP - h, CROP, CROP], fill="white")
    d.text(((CROP - (bb[2] - bb[0])) // 2 - bb[0], CROP - h + 9 - bb[1]), word, fill="black", font=f)
    return tile


def fit_photo(src, max_side=448):
    """a photo (path or bytes) at most max_side on its longer side, both sides a multiple of 32 (the vision merge) and at least 64"""
    im = Image.open(io.BytesIO(src) if isinstance(src, bytes) else src).convert("RGB")
    im.thumbnail((max_side, max_side))
    w, h = (im.width // 32) * 32, (im.height // 32) * 32
    return im.resize((max(w, 64), max(h, 64)))


def text_picture(text, size=56, w=448, h=224):
    """black text centred on a white card (the vision preflight)"""
    img = Image.new("RGB", (w, h), "white")
    d, f = ImageDraw.Draw(img), sheet_font(size)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f)
    d.text(((w - (x1 - x0)) / 2 - x0, (h - (y1 - y0)) / 2 - y0), text, fill="black", font=f)
    return img


def contact_sheet(frames, cols, path):
    """the pictures a model saw, labelled, in one file: frame j is pasted at ((j % cols)(w + 12), (j // cols)(h + 34) + 28)"""
    w, h = frames[0][1].size
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (w + 12), rows * (h + 34)), "white")
    d, f = ImageDraw.Draw(sheet), sheet_font(16)
    for j, (label, im) in enumerate(frames):
        x, y = (j % cols) * (w + 12), (j // cols) * (h + 34)
        sheet.paste(im, (x, y + 28))
        d.text((x + 2, y + 6), label, fill="black", font=f)
    sheet.save(path)
    print("saved", path)


def sheet_frame(path, j, cols, w, h):
    """frame j of a contact sheet, pixel for pixel"""
    x, y = (j % cols) * (w + 12), (j // cols) * (h + 34) + 28
    return Image.open(path).convert("RGB").crop((x, y, x + w, y + h))


# ---- the COCO crop bank


def crop_candidates(coco_dir, crop_set):
    """{class: [(PIL tile, (COCO image id, [x, y, w, h]))]} from COCO val2017: single instances, not crowd, both sides >= 96 px,
    10% margin, fitted into a 184 px tile on grey. Per class the 36 largest boxes are shuffled once (seed 0); crop set k is the
    k-th run of 12 in that order (set 0 for the first runs, set 1 for the fresh crops of the later ones)."""
    coco_dir = Path(coco_dir)
    rng = np.random.default_rng(SEED)
    ann = json.load(open(coco_dir / "annotations" / "instances_val2017.json"))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    files = {im["id"]: im["file_name"] for im in ann["images"]}
    per = collections.defaultdict(list)
    for a in ann["annotations"]:
        n = cats[a["category_id"]]
        if n in VOCAB and not a.get("iscrowd", 0):
            x, y, w, h = a["bbox"]
            if w >= 96 and h >= 96:
                per[n].append((w * h, a["image_id"], (x, y, w, h)))
    zf = zipfile.ZipFile(coco_dir / "val2017.zip")
    out = {}
    for n in VOCAB:
        rows = sorted(per[n], reverse=True)[: PER_CLASS * 3]
        rng.shuffle(rows)
        out[n] = []
        for _, iid, (x, y, w, h) in rows[crop_set * PER_CLASS : (crop_set + 1) * PER_CLASS]:
            im = Image.open(zf.open(f"val2017/{files[iid]}")).convert("RGB")
            mx, my = w * MARGIN, h * MARGIN
            box = (max(0, int(x - mx)), max(0, int(y - my)), min(im.width, int(x + w + mx)), min(im.height, int(y + h + my)))
            c = im.crop(box)
            c.thumbnail((CROP, CROP))
            tile = Image.new("RGB", (CROP, CROP), BG)
            tile.paste(c, ((CROP - c.width) // 2, (CROP - c.height) // 2))
            out[n].append((tile, (int(iid), [int(v) for v in (x, y, w, h)])))
    return out


@dataclass
class Crops:
    words: list  # classes with a single-token name and at least 4 crops the model names correctly
    tiles: dict  # class -> the tiles that passed the gate, in bank order
    coco: dict  # class -> (COCO image id, box) of each of those tiles
    ids: dict  # class -> single-token ids of the name's forms (word, ' word', Word, ' Word')
    gate: dict = field(default_factory=dict)  # class -> {kept, total}

    def save(self, run):
        run.save_json(dict(vocab=self.words, crop_gate=self.gate, gate_question=GATE_QUESTION, coco=self.coco), "crops.json")
        contact_sheet([(w, self.tiles[w][0]) for w in self.words], 8, run.path("crops.png"))


def load_crops(lm, coco_dir, crop_set):
    """the crop bank for a model: every candidate shown alone on a one-slot canvas must be named correctly (greedy, 6 tokens)"""
    ids = {}
    for w in VOCAB:
        v = lm.single_ids([w, " " + w, w.capitalize(), " " + w.capitalize()])
        if v:
            ids[w] = v
    words = [w for w in VOCAB if w in ids]
    print(f"vocabulary: {len(words)} single-token classes" + (f"; dropped (multi-token): {[w for w in VOCAB if w not in ids]}" if len(words) < len(VOCAB) else ""))
    cand = crop_candidates(coco_dir, crop_set)
    flat = [(w, im, meta) for w in words for im, meta in cand[w]]
    outs = lm.generate([lm.chat_media(["IMAGE", GATE_QUESTION])] * len(flat), [canvas(1, 1, {1: im}) for _, im, _ in flat], 6)
    ok = noun_matches(outs, [w for w, _, _ in flat], set(words))
    gate = {w: dict(kept=int(sum(o for (ww, _, _), o in zip(flat, ok) if ww == w)), total=len(cand[w])) for w in words}
    tiles = {w: [im for (ww, im, _), o in zip(flat, ok) if ww == w and o] for w in words}
    coco = {w: [meta for (ww, _, meta), o in zip(flat, ok) if ww == w and o] for w in words}
    print("crop gate (kept/total): " + ", ".join(f"{w} {v['kept']}/{v['total']}" for w, v in gate.items()))
    words = [w for w in words if len(tiles[w]) >= 4]
    print(f"crop bank: {len(words)} classes, crop set {crop_set}")
    return Crops(words=words, tiles=tiles, coco=coco, ids=ids, gate=gate)
