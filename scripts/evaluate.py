import json

import pandas as pd
import torch

from backbone import FrozenCLIP
from benchmark_model import BenchmarkModel
from config import FIGURES, RESULTS, SEEDS
from head import load
from utils.benchmarks import run_all
from utils.plots import plot_results


def main():
    clip = FrozenCLIP()
    models = {"CLIP": lambda: BenchmarkModel(clip),
              **{f"NegSplit s{seed}": lambda seed=seed: BenchmarkModel(clip, load(clip, seed)) for seed in SEEDS}}
    results = {}
    for name, make in models.items():
        model = make()
        results[name] = {k: 100 * float(v) for k, v in run_all(model).items()}
        del model
        torch.cuda.empty_cache()
        print(f"{name}: done", flush=True)
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    seeds = [f"NegSplit s{seed}" for seed in SEEDS]
    plot_results(results, seeds, FIGURES)

    runs = pd.DataFrame({s: results[s] for s in seeds})
    table = pd.DataFrame({"CLIP": results["CLIP"], "NegSplit": runs[seeds[0]]})
    table["NegSplit, 4 seeds"] = runs.mean(axis=1).map("{:.2f}".format) + " ± " + runs.std(axis=1).map("{:.2f}".format)
    pd.set_option("display.width", 300)
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
