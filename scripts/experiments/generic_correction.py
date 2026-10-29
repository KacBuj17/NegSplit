import json
import statistics
import time

import torch

from backbone import FrozenCLIP
from config import ANALYSIS, DEVICE
from generic_head import D, HIDDEN, GenericCorrection
from penalty_variants import WithoutSubtraction, compare, load_variant, run_variant
from utils.analysis import coco_val, val_images

NAME = "generic_correction"
REFERENCE = "a_without_subtraction"
REPEATS = 20


def count(params):
    return sum(param.numel() for param in params if param.requires_grad)


def parameter_counts(clip):
    reference = WithoutSubtraction(clip, torch.zeros(D))
    generic = GenericCorrection(clip, torch.zeros(D))
    exclusion = count(reference.proj.parameters()) + reference.log_beta.numel() + reference.tau.numel()
    correction = count(generic.correction.parameters())
    shared = count(generic.parameters()) - correction
    assert shared + exclusion + reference.mu.numel() == count(reference.parameters())
    return {
        "shared adapter": shared,
        "exclusion branch (proj, beta, tau)": exclusion,
        "generic correction MLP": correction,
        "branch ratio generic / exclusion": correction / exclusion,
        "total NegSplit without subtraction": shared + exclusion,
        "total generic correction": shared + correction,
    }


def synchronize():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def scoring_ms(head, images, texts):
    e_aff, text_side, p = head.encode(*texts)
    head.score(images, e_aff, text_side, p)
    times = []
    for _ in range(REPEATS):
        synchronize()
        start = time.perf_counter()
        head.score(images, e_aff, text_side, p)
        synchronize()
        times.append(1000 * (time.perf_counter() - start))
    return statistics.median(times)


def costs(clip):
    classes = coco_val()[0]
    texts = clip.texts([f"There is no {c} in this image." for c in classes] + [f"A photo of a {c}." for c in classes])
    images = val_images(clip).to(DEVICE)
    timing = {name: scoring_ms(load_variant(clip, cls, name, 0), images, texts)
              for name, cls in ((REFERENCE, WithoutSubtraction), (NAME, GenericCorrection))}
    return {
        "gallery images": len(images),
        "queries": len(classes) * 2,
        "scoring ms (median of 20), " + REFERENCE: timing[REFERENCE],
        "scoring ms (median of 20), " + NAME: timing[NAME],
        "multiply-adds per (image, text) pair, " + REFERENCE: 2 * D,
        "multiply-adds per (image, text) pair, " + NAME: 2 * D + D * HIDDEN + HIDDEN,
    }


def main():
    run_variant(NAME, GenericCorrection)
    compare([REFERENCE, NAME])

    clip = FrozenCLIP()
    result = {"parameters": parameter_counts(clip), "cost": costs(clip)}
    print(json.dumps(result, indent=2))
    (ANALYSIS / f"variant_{NAME}_cost.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
