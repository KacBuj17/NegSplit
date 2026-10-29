import ast
import json
import random
import re
import zipfile
from collections import defaultdict
from functools import cache

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from benchmark_model import BenchmarkModel
from config import ANALYSIS, CACHE, DEVICE, RAW, SEEDS, TRAIN_DATA, checkpoint, heldout_checkpoint
from head import load
from utils.benchmarks import (BATCH, CCNEG, NEGBENCH_CSV, NEGBENCH_MCQ, PREPROCESS, WORKERS, CCNegEvalDataset, collate_ccneg,
                              configs, extract_ccneg_test)
from utils.data import COCO_ANNOTATIONS, download
from utils.pairs import a_an, cap1
from utils.training import COLS, load_data

VAL_ANNOTATIONS = RAW / "annotations" / "instances_val2017.json"
VAL_IMAGES = CACHE / "analysis_val2017_images.pt"
CCNEG_FEATURES = CACHE / "analysis_ccneg.pt"
PREDICTIONS = ANALYSIS / "predictions.json"
MCQ_COLUMNS = ["caption_0", "caption_1", "caption_2", "caption_3"]
NO_X = ["There is no {o} in this image.", "No {o} is present in this image.", "This image does not contain {a}."]
A_X = ["There is {a} in this image.", "{A} is present in this image.", "This image contains {a}."]


@cache
def coco_val():
    if not VAL_ANNOTATIONS.exists():
        archive = RAW / "annotations_trainval2017.zip"
        download(COCO_ANNOTATIONS, archive)
        with zipfile.ZipFile(archive) as z:
            z.extract("annotations/instances_val2017.json", RAW)
        archive.unlink()
    annotations = json.loads(VAL_ANNOTATIONS.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in annotations["categories"]}
    classes = sorted(names.values())
    present = defaultdict(set)
    for a in annotations["annotations"]:
        present[a["image_id"]].add(names[a["category_id"]])
    image_ids = sorted(image["id"] for image in annotations["images"])
    labels = torch.tensor([[c in present[i] for c in classes] for i in image_ids])
    return classes, image_ids, labels


def held_out_categories():
    return sorted(random.Random(0).sample([c for c in coco_val()[0] if c != "person"], 10))


@cache
def held_out_pattern():
    return re.compile(r"\b(?:" + "|".join(re.escape(c) for c in held_out_categories()) + r")(?:s|es)?\b", re.I)


def mentions(text):
    return bool(held_out_pattern().search(text))


def load_data_without_held_out():
    data = load_data()
    keep = np.ones(len(data["image_idx"]), dtype=bool)
    for col in COLS:
        keep &= ~np.array([mentions(t) for t in data["text"][col]])
    for key in ("train", "val"):
        data[key] = data[key][torch.from_numpy(keep[data[key].numpy()])]
    data["classes"] = [c for c in data["classes"] if not mentions(c)]
    print(f"held out: {held_out_categories()}; train {len(data['train'])}, val {len(data['val'])}, "
          f"classes {len(data['classes'])}", flush=True)
    return data


def load_head(clip, seed, held_out=False, gamma=True):
    head = load(clip, seed)
    if held_out:
        head.load_state_dict(torch.load(heldout_checkpoint(seed), map_location=DEVICE))
    if not gamma:
        head.log_gamma.data.fill_(-float("inf"))
    return head


def heads(clip, held_out=False, gamma=True):
    path = heldout_checkpoint if held_out else checkpoint
    prefix = "held-out" if held_out else "NegSplit"
    suffix = "" if gamma else " gamma=0"
    for seed in SEEDS:
        if path(seed).exists():
            yield f"{prefix} s{seed}{suffix}", load_head(clip, seed, held_out, gamma)


@torch.no_grad()
def encode_texts(model, texts, batch=1024):
    return torch.cat([model.encode_text(model.tokenizer(texts[i:i + batch])).cpu() for i in range(0, len(texts), batch)])


def auc(scores, positive):
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    n_pos, n_neg = positive.sum().item(), (~positive).sum().item()
    return ((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


@torch.no_grad()
def val_images(clip):
    if not VAL_IMAGES.exists():
        plain = BenchmarkModel(clip)
        image_ids = coco_val()[1]
        parts = []
        for i in range(0, len(image_ids), BATCH):
            batch = torch.stack([PREPROCESS(Image.open(RAW / "val2017" / f"{j:012d}.jpg").convert("RGB"))
                                 for j in image_ids[i:i + BATCH]])
            parts.append(plain.encode_image(batch).cpu())
        torch.save(torch.cat(parts), VAL_IMAGES)
    return torch.load(VAL_IMAGES)


@cache
def mcq_questions():
    df = pd.read_csv(NEGBENCH_CSV / NEGBENCH_MCQ["NegBench COCO MCQ"])
    df["image_id"] = [int(re.search(r"(\d+)\.jpg", p).group(1)) for p in df.image_path]
    df["held_out"] = [any(mentions(c) for c in q) for q in df[MCQ_COLUMNS].values]
    return df


@torch.no_grad()
def mcq(model, images):
    df = mcq_questions()
    row_of = {i: k for k, i in enumerate(coco_val()[1])}
    texts = [t for col in MCQ_COLUMNS for t in df[col]]
    features = encode_texts(model, texts).view(4, len(df), -1).transpose(0, 1).to(DEVICE)
    logits = model.score_options(images[[row_of[i] for i in df.image_id]].to(DEVICE), features).cpu()
    return (logits.argmax(1) == torch.tensor(df.correct_answer.values)).tolist()


@torch.no_grad()
def ccneg_data(clip):
    if not CCNEG_FEATURES.exists():
        extract_ccneg_test()
        configs.ccneg_root_folder = str(CCNEG / "ccneg_images")
        configs.finetuning_dataset_path = str(CCNEG / "ccneg_preprocessed.pt")
        dataset = CCNegEvalDataset(transform=PREPROCESS)
        loader = DataLoader(dataset, batch_size=BATCH, num_workers=WORKERS, collate_fn=collate_ccneg)
        plain = BenchmarkModel(clip)
        images, texts = [], []
        for batch_images, batch_texts in loader:
            images.append(plain.encode_image(batch_images).cpu())
            texts += batch_texts
        objects = []
        for annotation in dataset.annotations:
            sop = annotation["sop_data"]
            objects.append((ast.literal_eval(sop) if isinstance(sop, str) else sop)["sop_chosen_object"])
        torch.save((torch.cat(images), texts, objects), CCNEG_FEATURES)
    return torch.load(CCNEG_FEATURES)


@torch.no_grad()
def ccneg(model, clip):
    images, texts, _ = ccneg_data(clip)
    features = encode_texts(model, texts).view(len(images), 2, -1)
    scores = torch.cat([model.score_options(images[i:i + 4096], features[i:i + 4096]).cpu()
                        for i in range(0, len(images), 4096)])
    return (scores[:, 0] > scores[:, 1]).tolist()


def ccneg_groups(clip):
    df = pd.read_parquet(TRAIN_DATA)
    vocab = {w for col in COLS for w in df[f"neg_obj_{col}"].unique() if w} | set(df.obj_absent) | set(df.obj_present)
    coco = set(df.obj_absent) & set(coco_val()[0])
    objects = [o.lower().strip() for o in ccneg_data(clip)[2]]

    def known(o, words):
        return o in words or o.rstrip("s") in words or o[:-2] in words and o.endswith("es")

    groups = {"all": np.ones(len(objects), bool),
              "concept from the 80 COCO categories": np.array([known(o, coco) for o in objects]),
              "concept in the training vocabulary": np.array([known(o, vocab) for o in objects])}
    groups["concept outside the training vocabulary"] = ~groups["concept in the training vocabulary"]
    return groups


def fill(template, c):
    return template.format(o=c, a=a_an(c), A=cap1(a_an(c)))


@torch.no_grad()
def no_x(model, images):
    classes, _, labels = coco_val()
    texts = [fill(t, c) for t in NO_X + A_X for c in classes]
    scores = model.score(images, encode_texts(model, texts).to(DEVICE)).view(len(images), len(NO_X + A_X), len(classes)).cpu()
    neg, pos = scores[:, :len(NO_X)], scores[:, len(NO_X):]
    per_class = defaultdict(list)
    for k in range(len(classes)):
        absent = ~labels[:, k]
        for t in range(len(NO_X)):
            top = neg[:, t, k].argsort(descending=True)
            choice_neg = neg[:, t, k] > pos[:, t, k]
            per_class["AUC no X"].append(auc(neg[:, t, k], absent))
            per_class["AUC a X"].append(auc(pos[:, t, k], ~absent))
            per_class["top 10 without X"].append(absent[top[:10]].float().mean().item())
            per_class["top 100 without X"].append(absent[top[:100]].float().mean().item())
            per_class["without X: picks no X"].append(choice_neg[absent].float().mean().item())
            per_class["with X: picks a X"].append((~choice_neg[~absent]).float().mean().item())
    return {key: [sum(v[i:i + len(NO_X)]) / len(NO_X) for i in range(0, len(v), len(NO_X))] for key, v in per_class.items()}


def predictions(clip, models):
    store = json.loads(PREDICTIONS.read_text(encoding="utf-8")) if PREDICTIONS.exists() else {}
    images = val_images(clip)
    for name, head in models:
        if name in store:
            continue
        model = BenchmarkModel(clip, head)
        store[name] = {"mcq": mcq(model, images), "ccneg": ccneg(model, clip), "no_x": no_x(model, images.to(DEVICE))}
        ANALYSIS.mkdir(parents=True, exist_ok=True)
        PREDICTIONS.write_text(json.dumps(store), encoding="utf-8")
        print(f"{name}: done", flush=True)
    return store


def mean_of(table, names):
    names = [n for n in names if n in table.index]
    return table.loc[names].mean()
