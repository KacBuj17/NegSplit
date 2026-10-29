from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
RESOURCES = PROJECT / "resources"
PATCHES = PROJECT / "benchmarks"
NEGBENCH_REPO = PROJECT.parent / "negbench"
CLIP_BENCHMARK_REPO = PROJECT.parent / "CLIP_benchmark"
CONCLIP_REPO = PROJECT.parent / "CoN-CLIP"
RAW = RESOURCES / "raw"
CACHE = RESOURCES / "cache"
CHECKPOINTS = RESOURCES / "checkpoints"
VAL_IDS = RESOURCES / "val_coco_ids.txt"
RESULTS = RESOURCES / "results.json"
ABLATION = RESOURCES / "ablation.json"
ANALYSIS = RESOURCES / "analysis"
FIGURES = RESOURCES / "figures"

TRAIN_DATA = CACHE / "coco_neg_pairs.parquet"
VAL_DATA = CACHE / "coco_neg_pairs_val.parquet"
TRAIN_IMAGES = CACHE / "img_train_full.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_ID = "openai/clip-vit-base-patch32"
SEEDS = range(4)


def checkpoint(seed):
    return CHECKPOINTS / f"negsplit_s{seed}.pt"


def heldout_checkpoint(seed):
    return CHECKPOINTS / f"heldout_s{seed}.pt"
