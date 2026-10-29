import pandas as pd
import torch

from backbone import FrozenCLIP, l2
from config import SEEDS
from head import load
from penalty_variants import save_diagnostics
from utils.analysis import auc
from utils.training import class_prompt, encode_all, images, load_data


@torch.no_grad()
def main():
    clip = FrozenCLIP()
    data = load_data()
    mu = l2(l2(clip.texts([class_prompt(c) for c in data["classes"]])[1]).mean(0))
    idx = data["val"]
    img = images(data, idx)
    rows = {}
    for seed in SEEDS:
        head = load(clip, seed)
        _, e_neg, _, _ = encode_all(head, clip, data, idx)
        absent = (img * e_neg["caption_neg_true"]).sum(-1)
        present = (img * e_neg["caption_neg_false"]).sum(-1)
        x = torch.cat([absent, present]).cpu()
        x_rel = (torch.cat([absent, present]) - torch.cat([img @ mu, img @ mu])).cpu()
        is_present = torch.cat([torch.zeros(len(absent), dtype=torch.bool), torch.ones(len(present), dtype=torch.bool)])
        rows[f"NegSplit s{seed}"] = {"AUROC x": 100 * auc(x, is_present), "AUROC x_rel": 100 * auc(x_rel, is_present)}
    table = pd.DataFrame(rows).T
    table.loc["mean"] = table.mean()
    print(f"Presence of the negated object on the validation split ({len(idx)} rows, negated object absent vs present):")
    print(table.round(2).to_string())
    save_diagnostics("e_detector_auroc", table.to_dict(orient="index"))


if __name__ == "__main__":
    main()
