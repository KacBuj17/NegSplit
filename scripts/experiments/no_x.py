import json

import numpy as np
import pandas as pd

from backbone import FrozenCLIP
from config import ANALYSIS, SEEDS
from utils.analysis import coco_val, heads, predictions

EXAMPLE = "cat"
OUT = ANALYSIS / "no_x.json"


def main():
    clip = FrozenCLIP()
    store = predictions(clip, [("CLIP", None), *heads(clip)])
    classes, _, labels = coco_val()
    seeds = [f"NegSplit s{seed}" for seed in SEEDS]
    without = 100 * (1 - labels.float().mean(0)).numpy()

    table = pd.DataFrame({name: {k: 100 * np.mean(v) for k, v in store[name]["no_x"].items()} for name in ["CLIP", *seeds]}).T
    table.loc["NegSplit (4 seeds)"] = table.loc[seeds].mean()
    table.loc["random"] = {"AUC no X": 50, "AUC a X": 50, "top 10 without X": without.mean(),
                           "top 100 without X": without.mean(), "without X: picks no X": 50, "with X: picks a X": 50}
    table = table.loc[["random", "CLIP", "NegSplit (4 seeds)", *seeds]]

    k = classes.index(EXAMPLE)
    example = pd.DataFrame({name: {m: 100 * v[k] for m, v in store[name]["no_x"].items()} for name in ("CLIP", seeds[0])})
    auc = pd.DataFrame({name: store[name]["no_x"]["AUC no X"] for name in ["CLIP", *seeds]}, index=classes)
    auc["NegSplit (4 seeds)"] = auc[seeds].mean(axis=1)
    weakest = 100 * auc[["CLIP", "NegSplit (4 seeds)"]].sort_values("NegSplit (4 seeds)").head(8)

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", None)
    print("'There is no X' on 5000 COCO val2017 images, mean over 80 categories and 3 sentence forms:")
    print(table.round(2).to_string())
    print(f"\nExample '{EXAMPLE}' (images without it: {without[k]:.2f}%):")
    print(example.round(2).to_string())
    print("\nWeakest categories (AUC no X):")
    print(weakest.round(1).to_string())
    OUT.write_text(json.dumps({"table": table.to_dict(orient="index"), "example": EXAMPLE,
                               "example_without": float(without[k]), "example_table": example.to_dict(),
                               "weakest": weakest.to_dict(orient="index")}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
