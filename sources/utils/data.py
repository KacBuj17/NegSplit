import io
import json
import shutil
import tarfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gdown
import numpy as np
from PIL import Image

from config import RAW, TRAIN_IMAGES

HEADERS = {"User-Agent": "Mozilla/5.0"}
NEGBENCH_FOLDER = "1kSEq0mkV1t1T8GuOAM65iz_iAA7e5gxB"
NEGBENCH_FILES = ["COCO_val_mcq_llama3.1_rephrased.csv", "VOC2007_mcq_llama3.1_rephrased.csv",
                  "COCO_val_retrieval.csv", "COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"]
ARCHIVES = {
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip": "annotations/instances_train2017.json",
    "http://images.cocodataset.org/zips/val2017.zip": "val2017/000000000139.jpg",
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar": "VOCdevkit/VOC2007/JPEGImages/000001.jpg",
}
HF = "https://huggingface.co/datasets"
FILES = {
    f"{HF}/jaisidhsingh/ccneg/resolve/main/ccneg_preprocessed.pt": "ccneg/ccneg_preprocessed.pt",
    f"{HF}/jaisidhsingh/ccneg/resolve/main/ccneg_images.zip": "ccneg/ccneg_images.zip",
    **{f"{HF}/clip-benchmark/wds_flickr30k/resolve/main/test/{name}": f"flickr30k/test/{name}"
       for name in ["nshards.txt", *(f"{i}.tar" for i in range(5))]},
}


def download_negbench():
    out = RAW / "negbench_csv"
    out.mkdir(parents=True, exist_ok=True)
    for f in gdown.download_folder(id=NEGBENCH_FOLDER, skip_download=True, quiet=True, use_cookies=False):
        name = Path(f.path).name
        if name in NEGBENCH_FILES and not (out / name).exists():
            gdown.download(id=f.id, output=str(out / name), quiet=True)


def download(url, path):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS)) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f)


def download_files():
    for url, path in FILES.items():
        path = RAW / path
        if not path.exists():
            print(f"downloading {path.name}", flush=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            download(url, path)


def download_archive(url, marker):
    if (RAW / marker).exists():
        return
    archive = RAW / url.rsplit("/", 1)[1]
    print(f"downloading {archive.name}", flush=True)
    download(url, archive)
    with (tarfile.open if archive.suffix == ".tar" else zipfile.ZipFile)(archive) as a:
        a.extractall(RAW)
    archive.unlink()


def decode(data):
    return Image.open(io.BytesIO(data)).convert("RGB")


def fetch_image(url, tries=3):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as r:
                return decode(r.read())
        except Exception:
            time.sleep(1 + attempt)
    return None


def encode_train(clip):
    if TRAIN_IMAGES.exists():
        return
    with open(RAW / "annotations" / "instances_train2017.json", encoding="utf-8") as f:
        instances = json.load(f)
    annotated = {a["image_id"] for a in instances["annotations"]}
    images = sorted((im for im in instances["images"] if im["id"] in annotated), key=lambda im: im["id"])
    files, emb = [], []
    with ThreadPoolExecutor(32) as pool:
        for i in range(0, len(images), 256):
            batch = images[i:i + 256]
            fetched = pool.map(fetch_image, [im["coco_url"] for im in batch])
            loaded = [(im["file_name"], x) for im, x in zip(batch, fetched) if x is not None]
            if loaded:
                files += [name for name, _ in loaded]
                emb.append(clip.images([x for _, x in loaded]).cpu().numpy().astype(np.float16))
            print(f"COCO train2017: {len(files)}/{len(images)}", flush=True)
    np.savez(TRAIN_IMAGES, files=np.array(files), emb=np.concatenate(emb))

