# NegSplit

## Reproducing the results

Requires Python 3.12, git and a CUDA GPU. Commands are for Windows PowerShell, run from the repository root.
In PyCharm, mark `sources` as Sources Root and run any script with Run.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "sources"

.\.venv\Scripts\python.exe scripts\pipeline\generate_data.py     # downloads datasets, builds training data
.\.venv\Scripts\python.exe scripts\pipeline\setup_benchmarks.py  # clones and patches the benchmarks
.\.venv\Scripts\python.exe scripts\pipeline\train.py             # trains 4 seeds
.\.venv\Scripts\python.exe scripts\pipeline\evaluate.py          # runs all benchmarks
```

- `generate_data.py` writes the data to `resources/raw` and `resources/cache`.
- `setup_benchmarks.py` clones NegBench, CLIP_benchmark and CoN-CLIP next to this repository, at pinned commits.
- `train.py` writes the checkpoints to `resources/checkpoints`.
- `evaluate.py` evaluates frozen CLIP and NegSplit with the official benchmark code. It writes `resources/results.json`
  and `resources/figures/`, and prints a results table.

`scripts/demo` shows one query step by step (`run.py`) and the training data (`show_data.py`).

## Experiments

Scripts in `scripts/experiments`, run after the pipeline. Results go to `resources/ablation.json` and `resources/analysis/`.

- `ablation.py` disables one element of each trained head (aff_mlp, penalty, subtraction, penalty threshold τ = 0,
  gate p = 1) without retraining, runs all benchmarks and prints the mean ± std over 4 seeds and the change vs the full model.
- `heldout_categories.py` trains 4 seeds without 10 COCO categories and measures how negation transfers to them.
- `subtraction.py` measures what the subtraction term (γ) adds, on the released model and on the held-out model.
  Needs `heldout_categories.py` and `ablation.py` first.
- `no_x.py` scores "There is no X in this image." on COCO val2017 for CLIP and NegSplit.
- `multiple_negations.py` counts test sentences with more than one negation.
- `penalty_*.py` train and evaluate variants of the penalty (4 seeds each): `a` without subtraction (the reference),
  `b` concept-dependent threshold, `c` reward for absence, `d` both, `e` presence relative to the mean concept.
  `penalty_e_detector_auroc.py` needs no training. `penalty_compare.py` prints all variants against `a`.
- `generic_correction.py` replaces the exclusion branch of `a` (e_neg projection, β, τ) with a small MLP that predicts a
  signed score correction from the text context, the image and image × e_aff, with a parameter count within 5%;
  trains 4 seeds from scratch, compares with `a`, and writes parameter counts and scoring time.
- `generic_heldout.py` trains `a` and the generic correction on seed 0 without the 10 held-out COCO categories and
  compares how negation transfers to them (needs `generic_correction.py` first).
