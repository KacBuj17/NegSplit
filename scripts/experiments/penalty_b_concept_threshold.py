import numpy as np
import pandas as pd
import torch

from backbone import FrozenCLIP
from config import SEEDS
from penalty_variants import (ConceptThreshold, PenaltyVariant, compare, load_variant, run_variant, save_diagnostics,
                              variant_checkpoint)
from utils.analysis import coco_val, val_images

NAME = "b_concept_threshold"
REFERENCE = "a_without_subtraction"


@torch.no_grad()
def thresholds(clip, head, images):
    classes, _, labels = coco_val()
    hidden, e_orig, mask, eot = clip.texts([f"There is no {c} in this image." for c in classes])
    _, e_neg, _ = head.encode(hidden, e_orig, mask, eot)
    tau = head.tau_of(e_neg)
    tau = (tau if tau.dim() else tau.expand(len(classes))).cpu()
    x = (images.to(e_neg.device) @ e_neg.T).cpu()
    penalized = [(x[~labels[:, k], k] > tau[k]).float().mean().item() for k in range(len(classes))]
    return pd.DataFrame({"threshold": tau.numpy(), "penalized": 100 * np.array(penalized)}, index=classes)


def main():
    run_variant(NAME, ConceptThreshold)
    compare([REFERENCE, NAME])

    clip = FrozenCLIP()
    images = val_images(clip)
    tables = {}
    for name, cls in ((REFERENCE, PenaltyVariant), (NAME, ConceptThreshold)):
        seeds = [seed for seed in SEEDS if variant_checkpoint(name, seed).exists()]
        tables[name] = sum(thresholds(clip, load_variant(clip, cls, name, seed), images) for seed in seeds) / len(seeds)
    b = tables[NAME].sort_values("threshold")
    penalized = {name: t["penalized"].mean() for name, t in tables.items()}
    pd.set_option("display.width", 300)
    print("\nThresholds for 'There is no X in this image.' (mean over seeds), lowest and highest;"
          " 'penalized' = % of images without X on which the sentence gets a penalty:")
    print(pd.concat([b.head(8), b.tail(8)]).round(3).to_string())
    print(f"\nTrue 'no X' (X absent) that get a penalty, % mean over 80 classes: {penalized}")
    save_diagnostics(NAME, {"thresholds": b.to_dict(orient="index"), "penalized": penalized})


if __name__ == "__main__":
    main()
