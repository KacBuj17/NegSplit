import copy
import json

import pandas as pd
import torch

from backbone import FrozenCLIP
from benchmark_model import BenchmarkModel
from config import ABLATION, RESULTS, SEEDS
from head import load
from utils.benchmarks import run_all

VARIANTS = ("full model", "without aff_mlp", "without penalty", "without subtraction", "tau = 0", "without gate (p = 1)")


def disable(head, variant):
    head = copy.deepcopy(head)
    with torch.no_grad():
        if variant == "without aff_mlp":
            head.aff_mlp[-1].weight.zero_()
            head.aff_mlp[-1].bias.zero_()
        elif variant == "without penalty":
            head.log_beta.fill_(-1e9)
        elif variant == "without subtraction":
            head.log_gamma.fill_(-1e9)
        elif variant == "tau = 0":
            head.tau.fill_(0)
        elif variant == "without gate (p = 1)":
            head.p_head[-1].bias.fill_(1e9)
    return head


def main():
    clip = FrozenCLIP()
    results = json.loads(ABLATION.read_text(encoding="utf-8")) if ABLATION.exists() else {}
    for seed in SEEDS:
        head = load(clip, seed)
        for variant in VARIANTS:
            name = f"{variant} s{seed}"
            if name in results:
                continue
            results[name] = {k: 100 * float(v) for k, v in run_all(BenchmarkModel(clip, disable(head, variant))).items()}
            ABLATION.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{name}: done", flush=True)

    runs = {variant: pd.DataFrame({seed: results[f"{variant} s{seed}"] for seed in SEEDS}) for variant in VARIANTS}
    clip_results = pd.Series(json.loads(RESULTS.read_text(encoding="utf-8"))["CLIP"])
    scores = pd.DataFrame({"CLIP": clip_results.reindex(runs["full model"].index).map("{:.2f}".format)})
    changes = pd.DataFrame()
    for variant, df in runs.items():
        scores[variant] = df.mean(axis=1).map("{:.2f}".format) + " ± " + df.std(axis=1).map("{:.2f}".format)
        if variant != "full model":
            change = df - runs["full model"]
            changes[variant] = change.mean(axis=1).map("{:+.2f}".format) + " ± " + change.std(axis=1).map("{:.2f}".format)
    pd.set_option("display.width", 400)
    pd.set_option("display.max_columns", None)
    print("Mean ± std over seeds:")
    print(scores.to_string())
    print("\nChange vs full model (pp, same seed), mean ± std over seeds:")
    print(changes.to_string())


if __name__ == "__main__":
    main()
