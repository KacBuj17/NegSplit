import json

import numpy as np
import pandas as pd

from backbone import FrozenCLIP
from config import ANALYSIS
from generic_head import GenericCorrection
from penalty_variants import VARIANTS, WithoutSubtraction, load_variant, variant_head
from trainer import train
from utils.analysis import coco_val, held_out_categories, load_data_without_held_out, mcq_questions, predictions

SEED = 0
MODELS = {"a_without_subtraction": WithoutSubtraction, "generic_correction": GenericCorrection}
OUT = ANALYSIS / "generic_heldout.json"


def heldout_name(name):
    return f"{name}_heldout"


def heldout_path(name):
    return VARIANTS / f"{heldout_name(name)}_s{SEED}.pt"


def table(store):
    df = mcq_questions()
    held, kind = df.held_out.values, df.correct_answer_template.values
    groups = {"MCQ, other questions": ~held}
    for t, label in (("positive", "affirmative"), ("negative", "negated"), ("hybrid", "hybrid")):
        groups[f"held-out: {label}"] = held & (kind == t)
    held_classes = np.array([c in held_out_categories() for c in coco_val()[0]])
    rows = {}
    for name, r in store.items():
        ok = np.array(r["mcq"])
        rows[name] = {g: 100 * ok[m].mean() for g, m in groups.items()}
        no_x_auc = np.array(r["no_x"]["AUC no X"])
        rows[name]["no X AUC, held-out"] = 100 * no_x_auc[held_classes].mean()
        rows[name]["no X AUC, known"] = 100 * no_x_auc[~held_classes].mean()
        rows[name]["CC-Neg"] = 100 * np.mean(r["ccneg"])
    return pd.DataFrame(rows).T


def main():
    for name, cls in MODELS.items():
        if not heldout_path(name).exists():
            train(SEED, load=load_data_without_held_out, path=heldout_path(name), make_head=variant_head(cls))

    clip = FrozenCLIP()
    models = []
    for name, cls in MODELS.items():
        models.append((f"{name} s{SEED}", load_variant(clip, cls, name, SEED)))
        models.append((f"{name} held-out s{SEED}", load_variant(clip, cls, heldout_name(name), SEED)))
    store = predictions(clip, models)
    result = table({name: store[name] for name, _ in models})

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", None)
    print(f"Held-out categories: {held_out_categories()}; seed {SEED}")
    print(result.round(2).T.to_string())
    OUT.write_text(json.dumps(result.to_dict(orient="index"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
