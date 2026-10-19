import json
import random
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from config import RAW, TRAIN_IMAGES, VAL_IDS

SUBJECTS = ["this image", "the image", "this scene", "the scene", "the frame"]
POS_VERBS = ["features", "shows", "depicts", "includes", "displays", "captures", "has"]
TEMPLATES = ["positive", "negative", "hybrid"]
NEIGHBOURS = 25

VERBS = ["sitting", "standing", "holding", "riding", "playing", "walking", "flying", "wearing", "laying",
         "eating", "driving", "grazing", "skiing", "hanging", "cutting", "watching", "jumping", "smiling",
         "talking", "posing", "swinging", "carrying", "waiting", "running", "pulling", "sleeping", "throwing",
         "leaning", "crossing", "drinking", "hitting", "resting", "working", "cooking", "brushing", "reading",
         "performing", "feeding", "catching", "reaching", "floating", "staring", "painting", "swimming",
         "landing", "skating", "touching", "displaying", "preparing", "pointing"]
VAL_VERBS = ["surfing", "snowboarding", "skateboarding", "kneeling", "petting", "sailing", "kicking",
             "shopping", "writing", "racing", "serving"]
ATTRIBUTES = ["red", "blue", "green", "yellow", "white", "black", "brown", "orange", "pink", "purple", "gray",
              "silver", "wooden", "metal", "striped", "empty", "snowy", "sunny", "wet"]

VAL_NEG_APPEND = ["{cap} You won't find any {o} in the picture.", "{cap} The photo lacks {a}.",
                  "{cap} Nowhere in the shot is there {a}.", "{cap} It is missing {a}.",
                  "{cap} There isn't a single {o} here.", "Missing from this picture: {a}. {cap}"]
VAL_POS = ["You can see {a} in this photo.", "The picture contains {a}.", "{A} can be spotted here."]
VAL_POS2 = ["You can see {a1} and {a2} in this photo.", "The picture contains {a1} as well as {a2}."]
VAL_NEG = ["You cannot see {a} in this photo.", "The picture lacks {a}.",
           "There isn't {a} anywhere here.", "{A} is missing from this picture."]
VAL_HYB = ["You can see {ap}, yet there is no {o}.", "The picture contains {ap} while lacking {a}.",
           "{Ap} can be spotted here, but {a} is missing."]


def a_an(noun):
    return ("an " if noun[0].lower() in "aeiou" else "a ") + noun


def cap1(text):
    return text[0].upper() + text[1:]


def normalize(caption):
    caption = " ".join(caption.strip().split())
    if caption and caption[-1] not in ".!?":
        caption += "."
    return cap1(caption) if caption else caption


def single(template, obj):
    return template.format(a=a_an(obj), A=cap1(a_an(obj)), o=obj)


def pair(template, first, second):
    return template.format(a1=a_an(first), A1=cap1(a_an(first)), a2=a_an(second))


def hybrid(template, present, absent):
    return template.format(ap=a_an(present), Ap=cap1(a_an(present)), p=present, a=a_an(absent), o=absent)


def neg_clause(rng, obj):
    a, subject = a_an(obj), rng.choice(SUBJECTS)
    return rng.choice([
        f"there is no {obj} in {subject}", f"no {obj} is present", f"no {obj} appears in {subject}",
        f"not a single {obj} is visible", f"{subject} shows no {obj}", f"{subject} has no {obj}",
        f"{subject} does not include {a}", f"{a} is absent", f"{a} does not appear in {subject}",
        f"there is no sign of {a}", f"no {obj} can be seen", f"{subject} is free of any {obj}",
        f"{subject} features no {obj}",
    ])


def neg_sentence(rng, obj):
    return cap1(neg_clause(rng, obj)) + "."


def neg_append(rng, caption, obj):
    without_dot = caption.rstrip(".")
    clause = neg_clause(rng, obj)
    form = rng.randrange(5)
    if form == 0:
        return f"{caption} {cap1(clause)}."
    if form == 1:
        return f"{without_dot}, {rng.choice(['but', 'and', 'though'])} {clause}."
    if form == 2:
        return f"{cap1(clause)}. {caption}"
    if form == 3:
        return f"{without_dot}, without any {obj}."
    return f"With no {obj} around, {caption[0].lower() + caption[1:]}"


def pos_sentence(rng, obj, obj2=None):
    subject, verb = rng.choice(SUBJECTS), rng.choice(POS_VERBS)
    if obj2:
        return rng.choice([f"{cap1(subject)} {verb} {a_an(obj)} and {a_an(obj2)}.",
                           f"Both {a_an(obj)} and {a_an(obj2)} appear in {subject}.",
                           f"{cap1(a_an(obj))} and {a_an(obj2)} are present in {subject}."])
    return rng.choice([f"{cap1(subject)} {verb} {a_an(obj)}.", f"{cap1(a_an(obj))} is present in {subject}.",
                       f"{cap1(a_an(obj))} appears in {subject}.", f"There is {a_an(obj)} in {subject}."])


def hybrid_sentence(rng, present, absent):
    subject, verb, ap = rng.choice(SUBJECTS), rng.choice(POS_VERBS), a_an(present)
    return rng.choice([
        f"{cap1(subject)} {verb} {ap}, but {neg_clause(rng, absent)}.",
        f"{cap1(subject)} {verb} {ap} and no {absent}.",
        f"{cap1(ap)} is present in {subject}, but {a_an(absent)} is absent.",
        f"{cap1(subject)} {verb} {ap} without any {absent}.",
        f"There is {ap} but no {absent} in {subject}.",
        f"{cap1(subject)} {verb} {ap}; however, there is no {absent}.",
        f"Although {subject} {verb} {ap}, it has no {absent}.",
    ])


def correct_sentence(rng, template, present, present2, absent):
    if template == "positive":
        return pos_sentence(rng, present, present2)
    if template == "negative":
        return neg_sentence(rng, absent)
    return hybrid_sentence(rng, present, absent)


def words_in(caption, vocab):
    low = caption.lower()
    return [word for word in vocab if re.search(rf"\b{word}\b", low)]


def predicate_row(row, rng, caption, image_captions, template, verbs):
    present = words_in(caption, VERBS)
    if not present:
        return False
    verb = rng.choice(present)
    all_captions = " ".join(image_captions).lower()
    absent = [word for word in verbs if not re.search(rf"\b{word}\b", all_captions)]
    if not absent:
        return False
    absent_verb = rng.choice(absent)

    row["obj_present"], row["obj_absent"] = verb, absent_verb
    row["caption_neg_true"] = f"{caption} " + rng.choice([
        f"No one is {absent_verb} here.", f"Nobody is {absent_verb} in this scene.",
        f"There is no {absent_verb} going on.", f"Nothing here involves {absent_verb}."])
    row["neg_obj_caption_neg_true"] = absent_verb

    attributes = words_in(caption, ATTRIBUTES)
    if attributes and rng.random() < 0.3:
        attribute = rng.choice(attributes)
        row["caption_neg_false"] = f"{caption.rstrip('.')}, but nothing here is {attribute}."
        row["neg_obj_caption_neg_false"] = attribute
    else:
        i = caption.lower().find(verb)
        row["caption_neg_false"] = rng.choice([caption[:i] + "not " + caption[i:],
                                               f"{caption.rstrip('.')}, but nobody is {verb}.",
                                               f"{caption.rstrip('.')}, with no one {verb}."])
        row["neg_obj_caption_neg_false"] = verb

    correct = {
        "positive": (f"Someone is {verb} in this image.", ""),
        "negative": (f"No one is {absent_verb} in this image.", absent_verb),
        "hybrid": (f"Someone is {verb} in this image, but no one is {absent_verb}.", absent_verb),
    }
    row["mcq_correct"], row["neg_obj_mcq_correct"] = correct[template]
    row["mcq_d1"], row["neg_obj_mcq_d1"] = f"Someone is {absent_verb} in this image.", ""
    row["mcq_d2"], row["neg_obj_mcq_d2"] = f"No one is {verb} in this image.", verb
    row["mcq_d3"], row["neg_obj_mcq_d3"] = f"Someone is {absent_verb} in this image, but no one is {verb}.", verb
    return True


def val_templates(row, rng, caption, absent, present, present2, template):
    row["caption_neg_true"] = rng.choice(VAL_NEG_APPEND).format(o=absent, a=a_an(absent), cap=caption)
    row["caption_neg_false"] = rng.choice(VAL_NEG_APPEND).format(o=present, a=a_an(present), cap=caption)
    if template == "positive":
        if present2:
            row["mcq_correct"] = pair(rng.choice(VAL_POS2), present, present2)
        else:
            row["mcq_correct"] = single(rng.choice(VAL_POS), present)
    elif template == "negative":
        row["mcq_correct"] = single(rng.choice(VAL_NEG), absent)
    else:
        row["mcq_correct"] = hybrid(rng.choice(VAL_HYB), present, absent)
    row["mcq_d1"] = single(rng.choice(VAL_POS), absent)
    row["mcq_d2"] = single(rng.choice(VAL_NEG), present)
    row["mcq_d3"] = hybrid(rng.choice(VAL_HYB), absent, present)


def skip_draws(rng, *sizes):
    for n in sizes:
        rng.randrange(n)


def load_coco():
    with open(RAW / "annotations" / "instances_train2017.json", encoding="utf-8") as f:
        instances = json.load(f)
    with open(RAW / "annotations" / "captions_train2017.json", encoding="utf-8") as f:
        captions = json.load(f)
    names = {category["id"]: category["name"] for category in instances["categories"]}

    objects = defaultdict(set)
    for annotation in instances["annotations"]:
        objects[annotation["image_id"]].add(names[annotation["category_id"]])

    together = defaultdict(Counter)
    for image_objects in objects.values():
        for x in sorted(image_objects):
            for y in sorted(image_objects):
                if x != y:
                    together[x][y] += 1
    neighbours = {name: [y for y, _ in together[name].most_common(NEIGHBOURS)] for name in names.values()}

    image_captions = defaultdict(list)
    for annotation in captions["annotations"]:
        image_captions[annotation["image_id"]].append(annotation["caption"])

    files = {image["id"]: image["file_name"] for image in instances["images"]}
    image_id_of = {file: image_id for image_id, file in files.items()}
    val_ids = {int(x) for x in VAL_IDS.read_text().split()}
    split = {}
    for file in np.load(TRAIN_IMAGES, allow_pickle=True)["files"].tolist():
        image_id = image_id_of.get(file)
        if image_id in objects and image_captions.get(image_id):
            split[image_id] = "val" if image_id in val_ids else "train"
    return objects, neighbours, image_captions, files, split, sorted(set(names.values()))


def negation_candidates(image_objects, neighbours, all_names):
    candidates = {neighbour for obj in image_objects for neighbour in neighbours[obj]} - image_objects
    return sorted(candidates) or sorted(set(all_names) - image_objects)


def object_row(rng, rng_text, image_id, file, split, caption, image_objects, candidates):
    absent = rng.choice(candidates)
    present = rng.choice(sorted(image_objects))
    skip_draws(rng, 8, 8)
    template = rng.choice(TEMPLATES)
    present2 = None
    if template == "positive" and len(image_objects) >= 2 and rng.random() < 0.5:
        skip_draws(rng, 5)
        present2 = rng.choice(sorted(image_objects - {present}))
    else:
        skip_draws(rng, 6)
    skip_draws(rng, 6, 6, 6)

    row = {
        "image_id": image_id, "image_file": file, "split": split,
        "obj_absent": absent, "obj_present": present,
        "caption_pos": caption, "neg_obj_caption_pos": "",
        "caption_neg_true": neg_append(rng_text, caption, absent), "neg_obj_caption_neg_true": absent,
        "caption_neg_false": neg_append(rng_text, caption, present), "neg_obj_caption_neg_false": present,
        "mcq_correct": correct_sentence(rng_text, template, present, present2, absent),
        "neg_obj_mcq_correct": "" if template == "positive" else absent,
        "mcq_d1": pos_sentence(rng_text, absent), "neg_obj_mcq_d1": "",
        "mcq_d2": neg_sentence(rng_text, present), "neg_obj_mcq_d2": present,
        "mcq_d3": hybrid_sentence(rng_text, absent, present), "neg_obj_mcq_d3": present,
        "mcq_template": template,
    }
    return row, absent, present, present2, template


def build(coco, val_templates_on):
    objects, neighbours, image_captions, files, split, all_names = coco
    rng, rng_val, rng_text, rng_pred = (random.Random(seed) for seed in (1, 1001, 2001, 3001))
    rows = []
    for image_id in sorted(split):
        image_objects = objects[image_id]
        candidates = negation_candidates(image_objects, neighbours, all_names)
        use_val = val_templates_on and split[image_id] == "val"
        captions = image_captions[image_id]
        for caption in rng.sample(captions, min(2, len(captions))):
            caption = normalize(caption)
            row, absent, present, present2, template = object_row(
                rng, rng_text, image_id, files[image_id], split[image_id], caption, image_objects, candidates)
            verbs = VAL_VERBS if use_val else VERBS
            predicate = rng_pred.random() < 0.5 and predicate_row(row, rng_pred, caption, captions, template, verbs)
            if use_val and not predicate:
                val_templates(row, rng_val, caption, absent, present, present2, template)
            rows.append(row)
    return pd.DataFrame(rows)
