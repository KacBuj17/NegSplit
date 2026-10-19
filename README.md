# NegSplit

## Reproducing the results

Requires Python 3.12, git and a CUDA GPU. Commands are for Windows PowerShell, run from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "sources"

.\.venv\Scripts\python.exe scripts\generate_data.py     # downloads datasets, builds training data
.\.venv\Scripts\python.exe scripts\setup_benchmarks.py  # clones and patches the benchmarks
.\.venv\Scripts\python.exe scripts\train.py             # trains 4 seeds
.\.venv\Scripts\python.exe scripts\evaluate.py          # runs all benchmarks
```

- `generate_data.py` writes the data to `resources/raw` and `resources/cache`.
- `setup_benchmarks.py` clones NegBench, CLIP_benchmark and CoN-CLIP next to this repository, at pinned commits.
- `train.py` writes the checkpoints to `resources/checkpoints`.
- `evaluate.py` evaluates frozen CLIP and NegSplit with the official benchmark code. It writes `resources/results.json`
  and `resources/figures/`, and prints a results table.
