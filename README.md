# NegSplit

CLIP matches "a cat without a dog" to pictures of cats with dogs: its text vector barely changes when a sentence
negates something. NegSplit is a small text-side head (1.58 M trained parameters) on top of a frozen CLIP ViT-B/32 that
splits a sentence into what should be in the image and what should not. The image encoder and the CLIP weights are
untouched, so image vectors from plain CLIP can be reused.

How a sentence is scored:

1. An attention query over the tokens (CLIP text layers 6 and 8) finds the negated part and turns it into a vector `e_neg`;
   a small classifier gives `p`, the probability that the sentence negates anything.
2. The sentence vector is corrected towards what should be in the image (`e_aff`).
3. The image score is `image · e_aff` minus a penalty that grows when the image resembles `e_neg`, weighted by `p`.
   Sentences without negation (`p ≈ 0`) are scored almost like in CLIP.

One query, step by step (`scripts/demo/run.py`, seed 0):

```
query: A photo of a white cat without a black dog
neg attention: black 0.56, dog 0.15, a 0.09, the rest ≤ 0.04
p (is there a negation) = 0.975
e_neg nearest concepts: black, sheep, sports ball, bear, dog
similarity of the sentence vector to "black": CLIP 0.746 → NegSplit e_aff 0.488
```

The head finds the negated phrase, but `e_neg` is closest to "black", not to "dog": negated concepts with an
attribute are its weak spot (see `attribute_negation.py` below).

## Results

Official benchmark code (NegBench, CLIP_benchmark, CC-Neg from CoN-CLIP), mean ± std over 4 training seeds.

| Benchmark | CLIP | NegSplit | change |
|---|---:|---:|---:|
| NegBench COCO MCQ | 39.26 | 79.48 ± 0.84 | +40.22 |
| — affirmative / negated / hybrid questions | 69.14 / 6.84 / 39.21 | 98.63 / 66.46 / 72.24 | |
| NegBench VOC MCQ | 38.72 | 80.91 ± 0.55 | +42.19 |
| NegBench COCO retrieval R@5 | 54.77 | 64.23 ± 0.33 | +9.46 |
| NegBench COCO negated retrieval R@5 | 47.93 | 58.39 ± 0.69 | +10.46 |
| CC-Neg | 65.34 | 87.74 ± 0.25 | +22.40 |
| Flickr30K t2i R@1 | 58.76 | 64.55 ± 0.20 | +5.79 |
| CIFAR-100 zero-shot | 64.28 | 61.70 ± 0.32 | **−2.58** |

NegSplit loses 2.6 points on CIFAR-100 classification. All metrics are in `resources/results.json`.

## Repository

```
sources/              model and shared code (add to PYTHONPATH)
  head.py             NegSplit head
  backbone.py         frozen CLIP
  trainer.py          training loop (resumable)
  benchmark_model.py  wrapper used by the official benchmark code
  utils/              data, losses, benchmarks, analysis, plots
scripts/pipeline/     data → benchmarks setup → training → evaluation
scripts/demo/         one query step by step, a look at the training data
scripts/experiments/  ablation, held-out categories, penalty variants, generic correction
benchmarks/           patches applied to the cloned benchmark repositories
resources/            checkpoints, results, analysis
```

Checkpoints of the 4 seeds are in `resources/checkpoints`, so the demo and the evaluation run without training.

## Setup

Requires Python 3.12, git and a CUDA GPU (results were computed on an RTX 4060 Laptop, 8 GB). Commands are for Windows
PowerShell, run from the repository root. In PyCharm, mark `sources` as Sources Root and run any script with Run.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "sources"
```

Quick look, no data download needed:

```powershell
.\.venv\Scripts\python.exe scripts\demo\run.py
```

## Reproducing the results

```powershell
.\.venv\Scripts\python.exe scripts\pipeline\generate_data.py     # downloads datasets, builds training data
.\.venv\Scripts\python.exe scripts\pipeline\setup_benchmarks.py  # clones and patches the benchmarks
.\.venv\Scripts\python.exe scripts\pipeline\train.py             # trains 4 seeds
.\.venv\Scripts\python.exe scripts\pipeline\evaluate.py          # runs all benchmarks
```

- `generate_data.py` writes about 6 GB to `resources/raw` and `resources/cache`.
- `setup_benchmarks.py` clones NegBench, CLIP_benchmark and CoN-CLIP **next to** this repository, at pinned commits, and
  applies the patches from `benchmarks/`.
- `train.py` writes the checkpoints to `resources/checkpoints`. An interrupted run continues from the last finished epoch.
- `evaluate.py` evaluates frozen CLIP and NegSplit and writes `resources/results.json` and `resources/figures/`.

## Experiments

Scripts in `scripts/experiments`, run after the pipeline; `run_all_experiments.py` runs them all in the right order.
Results go to `resources/ablation.json` and `resources/analysis/`.

| Script | What it checks | Finding |
|---|---|---|
| `ablation.py` | turns off one part of the trained head at a time (aff_mlp, penalty, subtraction, τ = 0, p = 1) | aff_mlp carries most of the gain (COCO MCQ 78 → 47 without it); without the gate p, retrieval of ordinary captions collapses |
| `heldout_categories.py` | trains without 10 COCO categories, tests negation of them | negated questions about unseen categories: 31% (CLIP 2%, trained on them 68%) |
| `subtraction.py` | what the subtraction term γ adds (needs the two above) | small: COCO MCQ −0.9, CC-Neg +1.4; on unseen categories negated MCQ +8.8 |
| `no_x.py` | "There is no X in this image." on COCO val2017 | images without X ranked higher: AUC 90 (CLIP 11) |
| `multiple_negations.py` | how many test sentences have more than one negation | very few (48 of 23 656 in COCO MCQ), so one negation query is enough for the benchmarks |
| `attribute_negation.py` | negated concept with an attribute ("no black dog") | weak spot: AUC drops from 90 to 46–76; with colours the head negates the colour instead of the object |
| `penalty_*.py` | penalty variants: `a` without subtraction (reference), `b` concept-dependent threshold, `c` reward for absence, `d` both, `e` presence relative to the mean concept; `penalty_compare.py` prints them against `a` | none beats `a` |
| `generic_correction.py` | replaces the fixed-form penalty of `a` with a small MLP that predicts a score correction for each image–text pair, same parameter count | tie on MCQ, CC-Neg −3, scoring ~70× slower |
| `generic_heldout.py` | the same two models trained without the 10 held-out categories (seed 0) | `a` better by 9–14 points on unseen categories |
