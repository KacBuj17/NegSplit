import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from backbone import FrozenCLIP, l2
from config import ANALYSIS, SEEDS, heldout_checkpoint
from trainer import train
from utils.analysis import (NO_X, ccneg_groups, coco_val, fill, held_out_categories, heads, load_data_without_held_out,
                            load_head, mcq_questions, mean_of, predictions, val_images, auc)
from utils.pairs import a_an
from utils.training import load_data

DIAGNOSIS_SEEDS = (0, 1)
OUT = ANALYSIS / "heldout_categories.json"


def word_positions(clip, ids, word):
    target = clip.tokenizer(word, add_special_tokens=False)["input_ids"]
    row = ids.tolist()
    for start in range(len(row) - len(target) + 1):
        if row[start:start + len(target)] == target:
            return list(range(start, start + len(target)))
    return []


@torch.no_grad()
def diagnose(clip, head, images):
    classes, _, labels = coco_val()
    concepts = l2(clip.texts([f"a photo of {a_an(c)}." for c in classes])[1]).cpu()
    texts = [fill(t, c) for t in NO_X for c in classes]
    ids = clip.tokenize(texts)
    hidden, e_orig, mask, eot = clip.encode_ids(ids)
    attention = head.attention(head.mix_layers(hidden), mask).cpu()
    e_aff, e_neg, p = (x.cpu() for x in head.encode(hidden, e_orig, mask, eot))
    rows = defaultdict(list)
    for j in range(len(texts)):
        k = j % len(classes)
        without = ~labels[:, k]
        rows["class"].append(classes[k])
        rows["p (negation detected)"].append(p[j].item())
        rows["attention on the object word"].append(attention[j, word_positions(clip, ids[j], classes[k])].sum().item())
        rows["e_neg hits the concept (top-1 of 80)"].append(100 * float((e_neg[j] @ concepts.T).argmax().item() == k))
        rows["penalty: <I, e_neg> higher with X (AUC)"].append(100 * auc(images @ e_neg[j], ~without))
        rows["e_aff higher without X (AUC)"].append(100 * auc(images @ e_aff[j], without))
    df = pd.DataFrame(rows)
    df["group"] = np.where(df["class"].isin(held_out_categories()), "held-out (10)", "known (70)")
    return df.drop(columns="class").groupby("group").mean()


def main():
    for seed in SEEDS:
        if not heldout_checkpoint(seed).exists():
            train(seed, load=load_data_without_held_out, path=heldout_checkpoint(seed))

    full, kept = load_data(), load_data_without_held_out()
    removed = 1 - (len(kept["train"]) + len(kept["val"])) / (len(full["train"]) + len(full["val"]))
    del full, kept

    clip = FrozenCLIP()
    store = predictions(clip, [("CLIP", None), *heads(clip), *heads(clip, held_out=True)])
    names = {"CLIP": ["CLIP"], "NegSplit (saw the categories)": [f"NegSplit s{s}" for s in SEEDS],
             "NegSplit without the 10 categories": [f"held-out s{s}" for s in SEEDS]}

    df = mcq_questions()
    held, kind = df.held_out.values, df.correct_answer_template.values
    groups = {"MCQ, other questions": ~held}
    for t, label in (("positive", "affirmative"), ("negative", "negated"), ("hybrid", "hybrid")):
        groups[f"held-out: {label}"] = held & (kind == t)
    classes = coco_val()[0]
    held_classes = np.array([c in held_out_categories() for c in classes])
    rows = {}
    for name, r in store.items():
        ok = np.array(r["mcq"])
        rows[name] = {g: 100 * ok[m].mean() for g, m in groups.items()}
        no_x_auc = np.array(r["no_x"]["AUC no X"])
        rows[name]["no X AUC, held-out"] = 100 * no_x_auc[held_classes].mean()
        rows[name]["no X AUC, known"] = 100 * no_x_auc[~held_classes].mean()
    per_model = pd.DataFrame(rows).T
    main_table = pd.DataFrame({label: mean_of(per_model, n) for label, n in names.items()}).T
    counts = {g: int(m.sum()) for g, m in groups.items()}

    ccneg = ccneg_groups(clip)
    ccneg_rows = {}
    for name, r in store.items():
        ok = np.array(r["ccneg"])
        ccneg_rows[name] = {g: 100 * ok[m].mean() for g, m in ccneg.items()}
    ccneg_per_model = pd.DataFrame(ccneg_rows).T
    ccneg_table = pd.DataFrame({label: mean_of(ccneg_per_model, n) for label, n in names.items()}).T

    images = l2(val_images(clip).float())
    diagnosis = {}
    for seed in DIAGNOSIS_SEEDS:
        for held_out in (False, True):
            name = f"{'held-out' if held_out else 'NegSplit'} s{seed}"
            diagnosis[name] = diagnose(clip, load_head(clip, seed, held_out).eval(), images)
            print(f"diagnosis {name}: done", flush=True)

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", None)
    print(f"Held-out categories: {held_out_categories()}; removed {100 * removed:.1f}% of the training rows")
    print(f"\nNegBench COCO MCQ (own scoring) and 'There is no X' AUC, mean over seeds; questions: {counts}")
    print(main_table.round(2).to_string())
    print(f"\nCC-Neg by negated concept; examples: { {g: int(m.sum()) for g, m in ccneg.items()} }")
    print(ccneg_table.round(2).to_string())
    print("\nDiagnosis on 'There is no X' sentences:")
    for name, table in diagnosis.items():
        print(f"\n{name}\n{table.T.round(3).to_string()}")

    OUT.write_text(json.dumps({
        "held_out": held_out_categories(),
        "removed_rows": removed,
        "question_counts": counts,
        "mcq_and_no_x": main_table.to_dict(orient="index"),
        "mcq_and_no_x_per_model": per_model.to_dict(orient="index"),
        "ccneg_counts": {g: int(m.sum()) for g, m in ccneg.items()},
        "ccneg": ccneg_table.to_dict(orient="index"),
        "ccneg_per_model": ccneg_per_model.to_dict(orient="index"),
        "diagnosis": {name: table.to_dict(orient="index") for name, table in diagnosis.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
