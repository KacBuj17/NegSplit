import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from backbone import FrozenCLIP, l2
from benchmark_model import BenchmarkModel
from config import ANALYSIS, DEVICE
from utils.analysis import auc, coco_val, encode_texts, heads, val_images
from utils.pairs import a_an

ADJECTIVES = ["black", "white", "red", "brown", "small", "big"]
NO_X = ["There is no {x} in this image.", "No {x} is present in this image.", "This image does not contain {a}."]
COMPOSITE = "a photo of {a} without {b}."
MIN_IMAGES = 20
OUT = ANALYSIS / "attribute_negation.json"


def word_positions(clip, ids, word):
    target = clip.tokenizer(word, add_special_tokens=False)["input_ids"]
    row = ids.tolist()
    for start in range(len(row) - len(target) + 1):
        if row[start:start + len(target)] == target:
            return list(range(start, start + len(target)))
    return []


def phrase(adjective, c):
    return f"{adjective} {c}" if adjective else c


def no_x_texts(classes):
    return [(adj, k, t.format(x=phrase(adj, c), a=a_an(phrase(adj, c))))
            for adj in [None, *ADJECTIVES] for k, c in enumerate(classes) for t in NO_X]


def composite_pairs(classes, labels):
    pairs = []
    for a in range(len(classes)):
        for b in range(len(classes)):
            both, only_a = (labels[:, a] & labels[:, b]).sum().item(), (labels[:, a] & ~labels[:, b]).sum().item()
            if a != b and min(both, only_a) >= MIN_IMAGES:
                pairs.append((a, b))
    return pairs


@torch.no_grad()
def penalty_side(clip, head, classes, labels, images):
    texts = no_x_texts(classes)
    ids = clip.tokenize([t for _, _, t in texts])
    concept = {c: l2(clip.texts([f"a photo of {a_an(c)}."])[1]).cpu()[0] for c in classes}
    rows = defaultdict(list)
    baseline = {}
    for start in range(0, len(texts), 256):
        hidden, e_orig, mask, eot = clip.encode_ids(ids[start:start + 256])
        attention = head.attention(head.mix_layers(hidden), mask).cpu()
        e_aff, e_neg, p = (x.cpu() for x in head.encode(hidden, e_orig, mask, eot))
        for j in range(len(attention)):
            adj, k, _ = texts[start + j]
            c, row_ids = classes[k], ids[start + j]
            if adj is None:
                baseline.setdefault(k, []).append(e_neg[j])
            rows["adjective"].append(adj or "none")
            rows["k"].append(k)
            rows["e_neg"].append(e_neg[j])
            rows["e_aff"].append(e_aff[j])
            rows["p"].append(p[j].item())
            rows["attention: object"].append(attention[j, word_positions(clip, row_ids, c)].sum().item())
            rows["attention: adjective"].append(attention[j, word_positions(clip, row_ids, adj)].sum().item() if adj else 0.0)
    classes_matrix = torch.stack([concept[c] for c in classes])
    adjective_only = {adj: l2(clip.texts([f"a photo of something {adj}."])[1]).cpu()[0] for adj in ADJECTIVES}
    adjective_object = {(adj, c): l2(clip.texts([f"a photo of a {adj} {c}."])[1]).cpu()[0]
                        for adj in ADJECTIVES for c in classes}
    out = defaultdict(list)
    for i, adj in enumerate(rows["adjective"]):
        k, e = rows["k"][i], rows["e_neg"][i]
        c = classes[k]
        plain = l2(torch.stack(baseline[k]).mean(0))
        image_scores = images @ e
        out["adjective"].append(adj)
        out["p"].append(rows["p"][i])
        out["attention: object"].append(rows["attention: object"][i])
        out["attention: adjective"].append(rows["attention: adjective"][i])
        out["e_neg hits the object (top-1 of 80)"].append(100 * float((e @ classes_matrix.T).argmax().item() == k))
        out["penalty: <I, e_neg> higher with the object (AUC)"].append(100 * auc(image_scores, labels[:, k]))
        out["e_aff higher without the object (AUC)"].append(100 * auc(images @ rows["e_aff"][i], ~labels[:, k]))
        out["cos(e_neg, e_neg without adjective)"].append((e @ plain).item())
        out["cos(e_neg, CLIP 'a photo of a {adj} {o}')"].append((e @ adjective_object[(adj, c)]).item() if adj != "none" else np.nan)
        out["cos(e_neg, CLIP 'a photo of a {o}')"].append((e @ concept[c]).item())
        out["cos(e_neg, CLIP 'something {adj}')"].append((e @ adjective_only[adj]).item() if adj != "none" else np.nan)
        out["image corr: <I, e_neg> vs without adjective"].append(np.corrcoef(image_scores, images @ plain)[0, 1])
        out["image corr: <I, e_neg> vs 'something {adj}'"].append(
            np.corrcoef(image_scores, images @ adjective_only[adj])[0, 1] if adj != "none" else np.nan)
    return pd.DataFrame(out).groupby("adjective", sort=False).mean()


@torch.no_grad()
def score_side(model, classes, labels, images, pairs):
    no_x = no_x_texts(classes)
    composite = [(adj, a, b, COMPOSITE.format(a=a_an(classes[a]), b=a_an(phrase(adj, classes[b]))))
                 for adj in [None, *ADJECTIVES] for a, b in pairs]
    scores = model.score(images, encode_texts(model, [t for *_, t in no_x + composite]))
    rows = defaultdict(list)
    for j, (adj, k, _) in enumerate(no_x):
        rows["adjective"].append(adj or "none")
        rows["test"].append("'There is no X' (AUC: images without the object higher)")
        rows["value"].append(100 * auc(scores[:, j], ~labels[:, k]))
    for j, (adj, a, b, _) in enumerate(composite):
        keep = labels[:, a]
        rows["adjective"].append(adj or "none")
        rows["test"].append("'a photo of A without X' (AUC on images with A: without B higher)")
        rows["value"].append(100 * auc(scores[keep, len(no_x) + j], ~labels[keep, b]))
    return pd.DataFrame(rows).groupby(["test", "adjective"], sort=False)["value"].mean().unstack(0)


def main():
    clip = FrozenCLIP()
    classes, _, labels = coco_val()
    images = val_images(clip).float()
    pairs = composite_pairs(classes, labels)
    print(f"device {DEVICE}; {len(pairs)} (A, B) pairs with at least {MIN_IMAGES} images of A with and without B", flush=True)
    examples = {"no X": [t for adj, k, t in no_x_texts(classes) if classes[k] == "dog"][::3],
                "composite": COMPOSITE.format(a=a_an("cat"), b=a_an("black dog"))}

    behaviour = {"CLIP": score_side(BenchmarkModel(clip), classes, labels, images, pairs)}
    mechanism = {}
    for name, head in heads(clip):
        behaviour[name] = score_side(BenchmarkModel(clip, head), classes, labels, images, pairs)
        mechanism[name] = penalty_side(clip, head, classes, labels, images)
        print(f"{name}: done", flush=True)

    seeds = [name for name in behaviour if name != "CLIP"]
    behaviour_mean = {"CLIP": behaviour["CLIP"], "NegSplit (4 seeds)": sum(behaviour[s] for s in seeds) / len(seeds)}
    mechanism_mean = sum(mechanism[s] for s in seeds) / len(seeds)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", None)
    for name, table in behaviour_mean.items():
        print(f"\n{name}, behaviour:")
        print(table.round(2).to_string())
    print("\nNegSplit (4 seeds), negated-concept vector:")
    print(mechanism_mean.round(3).T.to_string())
    OUT.write_text(json.dumps({
        "adjectives": ADJECTIVES, "templates": NO_X, "composite_template": COMPOSITE, "pairs": len(pairs),
        "pair_names": [[classes[a], classes[b]] for a, b in pairs], "examples": examples, "device": DEVICE,
        "behaviour": {name: table.to_dict() for name, table in behaviour.items()},
        "behaviour_mean": {name: table.to_dict() for name, table in behaviour_mean.items()},
        "mechanism": {name: table.to_dict(orient="index") for name, table in mechanism.items()},
        "mechanism_mean": mechanism_mean.to_dict(orient="index")}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
