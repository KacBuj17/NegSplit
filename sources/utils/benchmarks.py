import sys
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import CACHE, CLIP_BENCHMARK_REPO, CONCLIP_REPO, NEGBENCH_REPO, RAW

# Prepended so that open_clip resolves to the copy in the NegBench repo, not the package installed in .venv.
# CoN-CLIP/src/data separately: that package's __init__ imports the uninstalled `clip`, so we take only the dataset file.
sys.path[:0] = [str(path) for path in (NEGBENCH_REPO / "benchmarks" / "src", NEGBENCH_REPO / "benchmarks", CLIP_BENCHMARK_REPO,
                                       CONCLIP_REPO / "src", CONCLIP_REPO / "src" / "data")]

from clip_benchmark.datasets.builder import build_dataset, get_dataset_collate_fn  # noqa: E402
from clip_benchmark.metrics import zeroshot_classification, zeroshot_retrieval  # noqa: E402
from configs import configs  # noqa: E402
from evaluation_datasets import CCNegEvalDataset  # noqa: E402
from open_clip.transform import image_transform  # noqa: E402
from src.evaluation import mcq, retrieval  # noqa: E402
from training.data import get_eval_dataset  # noqa: E402

NEGBENCH_CSV = RAW / "negbench_csv"
NEGBENCH_PATHS = {"data/coco/images/val2017": RAW / "val2017",
                  "data/voc2007/VOCdevkit/VOC2007/JPEGImages": RAW / "VOCdevkit" / "VOC2007" / "JPEGImages"}
NEGBENCH_MCQ = {"NegBench COCO MCQ": "COCO_val_mcq_llama3.1_rephrased.csv",
                "NegBench VOC MCQ": "VOC2007_mcq_llama3.1_rephrased.csv"}
NEGBENCH_RETRIEVAL = {"NegBench COCO t2i": "COCO_val_retrieval.csv",
                      "NegBench COCO negated t2i": "COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"}
CCNEG = RAW / "ccneg"
FLICKR = RAW / "flickr30k"
CIFAR = RAW / "cifar100_torchvision"
BATCH, WORKERS = 128, 4

PREPROCESS = image_transform(224, is_train=False)


class Args:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "fp32"
    batch_size = BATCH
    workers = WORKERS
    distributed = False
    video = False
    name = "negsplit"


def negbench_csv(file):
    local = CACHE / "negbench" / file
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        text = (NEGBENCH_CSV / file).read_text(encoding="utf-8")
        for prefix, path in NEGBENCH_PATHS.items():
            text = text.replace(prefix, path.as_posix())
        local.write_text(text, encoding="utf-8")
    return str(local)


def negbench(model):
    result = {}
    for name, file in NEGBENCH_MCQ.items():
        loader = get_eval_dataset(Args, negbench_csv(file), PREPROCESS, "mcq").dataloader
        m = mcq.evaluate_model(model, loader, Args, tokenizer=model.tokenizer)
        result[name] = m["total_accuracy"]
        result[f"{name} affirmative"] = m["positive_accuracy"]
        result[f"{name} negated"] = m["negative_accuracy"]
        result[f"{name} hybrid"] = m["hybrid_accuracy"]
    for name, file in NEGBENCH_RETRIEVAL.items():
        loader = get_eval_dataset(Args, negbench_csv(file), PREPROCESS, "retrieval").dataloader
        m = retrieval.evaluate_model(model, loader, Args, model.tokenizer, recall_k_list=[1, 5])
        result[f"{name} R@1"] = m["image_retrieval_recall@1"]
        result[f"{name} R@5"] = m["image_retrieval_recall@5"]
    return result


def extract_ccneg_test():
    folder = CCNEG / "ccneg_images"
    marker = folder / "test_extracted.txt"
    if marker.exists():
        return
    data = torch.load(CCNEG / "ccneg_preprocessed.pt", weights_only=False)
    wanted = {Path(path).name for path in data["image_paths"][-configs.num_ccneg_eval_samples:]}
    with zipfile.ZipFile(CCNEG / "ccneg_images.zip") as zf:
        for member in zf.namelist():
            if not member.startswith("__MACOSX/") and Path(member).name in wanted:
                zf.extract(member, CCNEG)
    marker.write_text("ok")


def collate_ccneg(batch):
    images = torch.stack([image for image, _, _ in batch])
    texts = [text for _, caption, negation in batch for text in (caption[0], negation[0])]
    return images, texts


@torch.no_grad()
def ccneg(model):
    extract_ccneg_test()
    configs.ccneg_root_folder = str(CCNEG / "ccneg_images")
    configs.finetuning_dataset_path = str(CCNEG / "ccneg_preprocessed.pt")
    dataset = CCNegEvalDataset(transform=PREPROCESS)
    loader = DataLoader(dataset, batch_size=BATCH, num_workers=WORKERS, collate_fn=collate_ccneg)
    correct, total = 0, 0
    for images, texts in loader:
        text_features = model.encode_text(model.tokenizer(texts)).view(len(images), 2, -1)
        scores = model.score_options(model.encode_image(images), text_features)
        correct += (scores[:, 0] > scores[:, 1]).sum().item()
        total += len(images)
    return {"CC-Neg": correct / total}


def clip_benchmark(model):
    flickr_type = FLICKR / "dataset_type.txt"
    if not flickr_type.exists():
        flickr_type.write_text("retrieval")
    flickr = build_dataset("wds/flickr30k", root=str(FLICKR), transform=PREPROCESS, split="test", task="zeroshot_retrieval")
    loader = DataLoader(flickr.batched(BATCH), batch_size=None, shuffle=False, num_workers=WORKERS)
    m = zeroshot_retrieval.evaluate(model, loader, model.tokenizer, recall_k_list=[1], device=Args.device, amp=False)
    result = {"Flickr30K t2i R@1": m["image_retrieval_recall@1"]}

    cifar = build_dataset("cifar100", root=str(CIFAR), transform=PREPROCESS, split="test", download=True,
                         task="zeroshot_classification")
    loader = DataLoader(cifar, batch_size=BATCH, shuffle=False, num_workers=WORKERS,
                        collate_fn=get_dataset_collate_fn("cifar100"))
    m = zeroshot_classification.evaluate(model, loader, model.tokenizer, cifar.classes, cifar.templates,
                                         device=Args.device, amp=False)
    result["CIFAR-100"] = m["acc1"]
    return result


def run_all(model):
    return {**negbench(model), **ccneg(model), **clip_benchmark(model)}
