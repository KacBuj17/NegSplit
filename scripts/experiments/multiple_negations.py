import ast
import json
import re

import pandas as pd
import torch

from config import ANALYSIS, RAW
from utils.benchmarks import NEGBENCH_CSV, NEGBENCH_MCQ, NEGBENCH_RETRIEVAL

CUE = re.compile(r"\b(?:no|not|without|none|nothing|nobody|never|neither|nor|absent|absence|lacks?|lacking|missing|devoid)\b"
                 r"|n't\b", re.I)
CLAUSE = re.compile(r"[,;.]|\bbut\b|\bwhile\b|\bwhereas\b|\byet\b", re.I)
VOC = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table", "dog", "horse",
       "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv monitor", "tv-monitor", "tvmonitor"]
OUT = ANALYSIS / "multiple_negations.json"


def pattern(words):
    return re.compile(r"\b(" + "|".join(sorted(map(re.escape, set(words)), key=len, reverse=True)) + r")(?:s|es)?\b", re.I)


def negated_objects(text, objects):
    found = set()
    for clause in CLAUSE.split(text):
        cue = CUE.search(clause)
        if cue:
            found |= {m.lower() for m in objects.findall(clause[cue.start():])}
    return found


def count(name, texts, objects):
    cues = pd.Series([len(CUE.findall(t)) for t in texts])
    negated = [negated_objects(t, objects) if objects is not None else set() for t in texts]
    many = [(t, sorted(n)) for t, n in zip(texts, negated) if len(n) >= 2]
    print(f"\n{name}: {len(texts)} sentences; negation words 0: {(cues == 0).sum()}, 1: {(cues == 1).sum()}, "
          f"2+: {(cues >= 2).sum()}; two or more negated objects: {len(many)}")
    for text, objs in many:
        print(f"   {objs} | {text}")
    return {"sentences": len(texts), "no negation word": int((cues == 0).sum()), "one negation word": int((cues == 1).sum()),
            "two or more negation words": int((cues >= 2).sum()), "two or more negated objects": len(many),
            "examples": [t for t, _ in many]}


def main():
    coco = [c["name"] for c in json.loads((RAW / "annotations" / "instances_train2017.json").read_text(encoding="utf-8"))["categories"]]
    result = {}
    for (name, file), words in zip(NEGBENCH_MCQ.items(), (coco, VOC)):
        df = pd.read_csv(NEGBENCH_CSV / file)
        texts = [t for col in ("caption_0", "caption_1", "caption_2", "caption_3") for t in df[col]]
        result[name] = count(name, texts, pattern(words))

    df = pd.read_csv(NEGBENCH_CSV / NEGBENCH_RETRIEVAL["NegBench COCO negated t2i"])
    texts, objects = [], []
    for captions, negative in zip(df.captions, df.negative_objects):
        for caption in ast.literal_eval(captions):
            texts.append(caption)
            objects.append(ast.literal_eval(negative))
    cues = pd.Series([len(CUE.findall(t)) for t in texts])
    many = [(t, sorted(negated_objects(t, pattern(o)))) for t, o in zip(texts, objects)]
    many = [(t, n) for t, n in many if len(n) >= 2]
    print(f"\nNegBench COCO negated t2i: {len(texts)} sentences; negation words 0: {(cues == 0).sum()}, "
          f"1: {(cues == 1).sum()}, 2+: {(cues >= 2).sum()}; two or more negated objects (from the negative_objects list): {len(many)}")
    for text, objs in many:
        print(f"   {objs} | {text}")
    result["NegBench COCO negated t2i"] = {
        "sentences": len(texts), "no negation word": int((cues == 0).sum()), "one negation word": int((cues == 1).sum()),
        "two or more negation words": int((cues >= 2).sum()), "two or more negated objects": len(many),
        "examples": [t for t, _ in many]}

    data = torch.load(RAW / "ccneg" / "ccneg_preprocessed.pt", weights_only=False)
    sop = [a["sop_data"] if isinstance(a["sop_data"], dict) else ast.literal_eval(a["sop_data"]) for a in data["annotations"][-40000:]]
    texts = [s["negative-prompt"].replace(",", "") for s in sop]
    result["CC-Neg"] = count("CC-Neg (negated captions, one chosen object per caption)", texts, None)

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
