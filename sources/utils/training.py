import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from backbone import l2
from config import DEVICE, TRAIN_DATA, TRAIN_IMAGES, VAL_DATA
from utils.pairs import ATTRIBUTES, a_an

COLS = ("caption_pos", "caption_neg_true", "caption_neg_false", "mcq_correct", "mcq_d1", "mcq_d2", "mcq_d3")
CANDIDATES = COLS[:3]
MCQ = COLS[3:]
TAU = 100.0


def class_prompt(name):
    if name.endswith("ing"):
        return f"a photo of someone {name}."
    if name in ATTRIBUTES:
        return f"a photo of something {name}."
    return f"a photo of {a_an(name)}."


def load_data():
    df = pd.read_parquet(TRAIN_DATA)
    val = pd.read_parquet(VAL_DATA)
    is_val = (df.split == "val").to_numpy()
    for col in COLS:
        df.loc[is_val, col] = val.loc[is_val, col].values

    bank = np.load(TRAIN_IMAGES, allow_pickle=True)
    bank_row = {file: i for i, file in enumerate(bank["files"].tolist())}

    negated = {word for col in COLS for word in df[f"neg_obj_{col}"].unique() if word}
    classes = sorted(negated | set(df.obj_absent) | set(df.obj_present))
    has_negation = {col: torch.from_numpy(df[f"neg_obj_{col}"].to_numpy() != "") for col in COLS}

    return {
        "images": torch.from_numpy(bank["emb"]).float(),
        "image_idx": torch.from_numpy(df.image_file.map(bank_row).to_numpy()),
        "text": {col: df[col].to_numpy() for col in COLS},
        "has_negation": has_negation,
        "train": torch.from_numpy(np.where(~is_val)[0]),
        "val": torch.from_numpy(np.where(is_val)[0]),
        "classes": classes,
    }


def split_by_column(x, rows):
    return dict(zip(COLS, x.split(rows)))


def encode(model, clip, data, idx):
    texts = [text for col in COLS for text in data["text"][col][idx.numpy()]]
    hidden, e_orig, mask, eot = clip.texts(texts)
    e_aff, e_neg, p = model.encode(hidden, e_orig, mask, eot)
    rows = len(idx)
    return (split_by_column(e_aff, rows), split_by_column(e_neg, rows), split_by_column(p, rows),
            split_by_column(l2(e_orig), rows))


def images(data, idx):
    return l2(data["images"][data["image_idx"][idx]]).to(DEVICE)


def one_after_another(x, cols):
    return torch.cat([x[col] for col in cols])


def side_by_side(x, cols):
    return torch.stack([x[col] for col in cols], 1)


def hide(scores, columns):
    hidden = torch.zeros_like(scores).scatter_(1, columns[:, None], float("-inf"))
    return scores + hidden


def contrastive_loss(model, img, e_aff, e_neg, p):
    n = len(img)
    sentences = (one_after_another(e_aff, CANDIDATES), one_after_another(e_neg, CANDIDATES),
                 one_after_another(p, CANDIDATES))
    scores = TAU * model.score(img, *sentences)
    caption = torch.arange(n, device=DEVICE)
    true_negation = caption + n

    image_finds_caption = F.cross_entropy(hide(scores, true_negation), caption)
    image_finds_true_negation = F.cross_entropy(hide(scores, caption), true_negation)
    sentence_finds_image = F.cross_entropy(scores[:, :2 * n].T, torch.cat([caption, caption]))
    return (image_finds_caption + image_finds_true_negation + sentence_finds_image) / 3


def mcq_loss(model, img, e_aff, e_neg, p):
    options = TAU * model.score_options(img, side_by_side(e_aff, MCQ), side_by_side(e_neg, MCQ),
                                        side_by_side(p, MCQ))
    correct = torch.zeros(len(img), dtype=torch.long, device=DEVICE)
    return F.cross_entropy(options, correct)


def detector_loss(p, has_negation):
    p = one_after_another(p, COLS).clamp(1e-5, 1 - 1e-5)
    return F.binary_cross_entropy(p, has_negation.float())


def loss(model, clip, data, idx):
    img = images(data, idx)
    e_aff, e_neg, p, _ = encode(model, clip, data, idx)
    has_negation = torch.cat([data["has_negation"][col][idx] for col in COLS]).to(DEVICE)

    contrastive = contrastive_loss(model, img, e_aff, e_neg, p)
    mcq = mcq_loss(model, img, e_aff, e_neg, p)
    detector = detector_loss(p, has_negation)
    return contrastive + mcq + 0.5 * detector


def encode_all(model, clip, data, idx, batch=512):
    parts = [encode(model, clip, data, idx[i:i + batch]) for i in range(0, len(idx), batch)]
    merged = []
    for k in range(4):
        merged.append({col: torch.cat([part[k][col] for part in parts]) for col in COLS})
    return merged


def retrieval(scores, true_image):
    image_ids = torch.arange(scores.shape[0], device=DEVICE)
    text_to_image = (scores.argmax(0) == true_image).float().mean().item()
    image_to_text = (true_image[scores.argmax(1)] == image_ids).float().mean().item()
    return text_to_image, image_to_text


@torch.no_grad()
def validate(model, clip, data):
    idx = data["val"]
    e_aff, e_neg, p, e_orig = encode_all(model, clip, data, idx)
    gallery_rows, true_image = np.unique(data["image_idx"][idx].numpy(), return_inverse=True)
    gallery = l2(data["images"][torch.from_numpy(gallery_rows)]).to(DEVICE)
    true_image = torch.from_numpy(true_image).to(DEVICE)

    result = {}
    for col in ("caption_pos", "caption_neg_true"):
        scores = {"clip": gallery @ e_orig[col].T,
                  "negsplit": model.score(gallery, e_aff[col], e_neg[col], p[col])}
        for name, s in scores.items():
            result[f"{col} t2i {name}"], result[f"{col} i2t {name}"] = retrieval(s, true_image)

    img = images(data, idx)
    options = model.score_options(img, side_by_side(e_aff, MCQ), side_by_side(e_neg, MCQ), side_by_side(p, MCQ))
    result["mcq negsplit"] = (options.argmax(-1) == 0).float().mean().item()
    return result
