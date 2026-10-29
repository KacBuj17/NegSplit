import json
import math

import numpy as np
import pandas as pd

from backbone import FrozenCLIP
from benchmark_model import BenchmarkModel
from config import ABLATION, ANALYSIS, DEVICE, SEEDS, heldout_checkpoint
from utils.analysis import coco_val, held_out_categories, heads, load_head, mcq, mcq_questions, no_x, predictions, val_images

FACTORS = [0, 1, 1.5, 2, 3, 4, 6]
SCAN_SEEDS = (0, 1)
OUT = ANALYSIS / "subtraction.json"


def summary(ok, auc_no_x):
    df = mcq_questions()
    held, kind = df.held_out.values, df.correct_answer_template.values
    held_classes = np.array([c in held_out_categories() for c in coco_val()[0]])
    ok, auc_no_x = np.array(ok), np.array(auc_no_x)
    return {"MCQ, other questions": 100 * ok[~held].mean(), "MCQ, held-out": 100 * ok[held].mean(),
            "MCQ, held-out negated": 100 * ok[held & (kind == "negative")].mean(),
            "MCQ, held-out hybrid": 100 * ok[held & (kind == "hybrid")].mean(), "MCQ, all": 100 * ok.mean(),
            "no X AUC, known": 100 * auc_no_x[~held_classes].mean(),
            "no X AUC, held-out": 100 * auc_no_x[held_classes].mean()}


def main():
    clip = FrozenCLIP()
    store = predictions(clip, [*heads(clip), *heads(clip, gamma=False),
                               *heads(clip, held_out=True), *heads(clip, held_out=True, gamma=False)])
    held_classes = np.array([c in held_out_categories() for c in coco_val()[0]])
    negated = mcq_questions().held_out.values & (mcq_questions().correct_answer_template.values == "negative")

    def contribution(prefix, metric):
        return {f"s{seed}": metric(store[f"{prefix} s{seed}"]) - metric(store[f"{prefix} s{seed} gamma=0"])
                for seed in SEEDS if f"{prefix} s{seed}" in store}

    rows = {
        "CC-Neg (official model)": contribution("NegSplit", lambda r: 100 * np.mean(r["ccneg"])),
        "CC-Neg (model without the 10 categories)": contribution("held-out", lambda r: 100 * np.mean(r["ccneg"])),
        "MCQ held-out, negated (model without the 10 categories)":
            contribution("held-out", lambda r: 100 * np.array(r["mcq"])[negated].mean()),
        "no X AUC, held-out (model without the 10 categories)":
            contribution("held-out", lambda r: 100 * np.array(r["no_x"]["AUC no X"])[held_classes].mean()),
    }
    if ABLATION.exists():
        ablation = json.loads(ABLATION.read_text(encoding="utf-8"))
        rows = {"NegBench COCO MCQ, official benchmark (official model)":
                {f"s{seed}": ablation[f"full model s{seed}"]["NegBench COCO MCQ"]
                 - ablation[f"without subtraction s{seed}"]["NegBench COCO MCQ"] for seed in SEEDS}, **rows}
    table = pd.DataFrame(rows).T
    table.insert(0, "mean", table.mean(axis=1))

    images = val_images(clip)
    scan = []
    for seed in SCAN_SEEDS:
        for held_out in (False, True):
            if held_out and not heldout_checkpoint(seed).exists():
                continue
            for factor in FACTORS:
                head = load_head(clip, seed, held_out)
                head.log_gamma.data.add_(math.log(factor) if factor > 0 else -math.inf)
                model = BenchmarkModel(clip, head.eval())
                row = {"model": "without the 10 categories" if held_out else "official", "seed": seed, "gamma x": factor,
                       **summary(mcq(model, images), no_x(model, images.to(DEVICE))["AUC no X"])}
                scan.append(row)
                print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    scan = pd.DataFrame(scan).drop(columns="seed").groupby(["model", "gamma x"], sort=False).mean()

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", None)
    print("\nContribution of the subtraction (with - gamma=0), pp:")
    print(table.round(2).to_string())
    print(f"\nScaling gamma without training, mean over seeds {SCAN_SEEDS}:")
    print(scan.round(2).to_string())
    OUT.write_text(json.dumps({"contribution": table.to_dict(orient="index"),
                               "scan": {f"{m} x{f}": v for (m, f), v in scan.to_dict(orient="index").items()}},
                              indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
