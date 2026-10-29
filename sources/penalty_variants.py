import json
import math

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import FrozenCLIP, l2
from benchmark_model import BenchmarkModel
from config import ANALYSIS, CHECKPOINTS, DEVICE, SEEDS
from head import NegSplit
from trainer import BATCH, train
from utils.benchmarks import run_all
from utils.training import encode, images

VARIANTS = CHECKPOINTS / "variants"


class WithoutSubtraction(NegSplit):
    def __init__(self, clip, mu):
        super().__init__(clip, mu)
        self.log_gamma.data.fill_(-math.inf)
        self.log_gamma.requires_grad_(False)


class PenaltyVariant(WithoutSubtraction):
    concept_threshold = False
    reward = False
    relative = False

    def __init__(self, clip, mu):
        super().__init__(clip, mu)
        if self.concept_threshold:
            self.threshold_weight = nn.Parameter(torch.zeros(512))
            self.threshold_bias = nn.Parameter(self.tau.detach().clone())
        if self.reward:
            self.log_beta_reward = nn.Parameter(torch.tensor(math.log(0.01)))
        if self.relative:
            self.register_buffer("mu_fixed", l2(mu.clone().float()))

    def extra_scalars(self):
        threshold = [self.threshold_weight, self.threshold_bias] if self.concept_threshold else []
        return threshold + ([self.log_beta_reward] if self.reward else [])

    def tau_of(self, e_neg):
        return e_neg @ self.threshold_weight + self.threshold_bias if self.concept_threshold else self.tau

    def presence(self, img, s_neg):
        return s_neg - (img @ self.mu_fixed)[:, None] if self.relative else s_neg

    def negation_term(self, x, tau):
        term = -self.log_beta.exp() * F.relu(x - tau)
        if self.reward:
            term = term + self.log_beta_reward.exp() * F.relu(tau - x)
        return term

    def score(self, img, e_aff, e_neg, p):
        tau = self.tau_of(e_neg)
        return img @ e_aff.T + p[None, :] * self.negation_term(self.presence(img, img @ e_neg.T), tau[None, :] if tau.dim() else tau)

    def score_options(self, img, e_aff, e_neg, p):
        s_aff = torch.einsum("nd,nkd->nk", img, e_aff)
        s_neg = torch.einsum("nd,nkd->nk", img, e_neg)
        return s_aff + p * self.negation_term(self.presence(img, s_neg), self.tau_of(e_neg))


class ConceptThreshold(PenaltyVariant):
    concept_threshold = True


class AbsenceReward(PenaltyVariant):
    reward = True


class ThresholdAndReward(PenaltyVariant):
    concept_threshold = True
    reward = True


class RelativePresence(PenaltyVariant):
    relative = True


@torch.no_grad()
def start_tau_at_median(model, clip, data):
    idx = data["train"][:BATCH]
    img = images(data, idx)
    _, e_neg, _, _ = encode(model, clip, data, idx)
    x = model.presence(img, img @ e_neg["caption_neg_true"].T).diagonal()
    model.tau.fill_(x.median().item())
    return model


def variant_head(cls):
    def make_head(clip, concepts, data):
        model = cls(clip, concepts.mean(0))
        if getattr(cls, "relative", False):
            model = start_tau_at_median(model.to(DEVICE), clip, data)
        return model

    return make_head


def variant_checkpoint(name, seed):
    return VARIANTS / f"{name}_s{seed}.pt"


def load_variant(clip, cls, name, seed):
    model = cls(clip, torch.zeros(512)).to(DEVICE)
    model.load_state_dict(torch.load(variant_checkpoint(name, seed), map_location=DEVICE))
    return model.eval()


def results_file(name):
    return ANALYSIS / f"variant_{name}.json"


def save_diagnostics(name, diagnostics):
    out = ANALYSIS / f"variant_{name}_diagnostics.json"
    out.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")


def run_variant(name, cls):
    VARIANTS.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    out = results_file(name)
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    validation_file = ANALYSIS / f"variant_{name}_validation.json"
    validation = json.loads(validation_file.read_text(encoding="utf-8")) if validation_file.exists() else {}
    for seed in SEEDS:
        key = f"s{seed}"
        if key in results:
            continue
        if not variant_checkpoint(name, seed).exists():
            validation[key] = 100 * train(seed, path=variant_checkpoint(name, seed), make_head=variant_head(cls))
            validation_file.write_text(json.dumps(validation, indent=2), encoding="utf-8")
        clip = FrozenCLIP()
        model = BenchmarkModel(clip, load_variant(clip, cls, name, seed))
        results[key] = {"validation MCQ": validation.get(key), **{k: 100 * float(v) for k, v in run_all(model).items()}}
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        del model, clip
        torch.cuda.empty_cache()
        print(f"{name} s{seed}: done", flush=True)
    return results


def compare(names):
    runs = {name: pd.DataFrame(json.loads(results_file(name).read_text(encoding="utf-8")))
            for name in names if results_file(name).exists()}
    scores = pd.DataFrame({name: df.mean(axis=1).map("{:.2f}".format) + " ± " + df.std(axis=1).map("{:.2f}".format)
                           for name, df in runs.items()})
    reference = names[0]
    changes = pd.DataFrame()
    for name, df in runs.items():
        if name != reference and reference in runs:
            change = df - runs[reference]
            changes[name] = change.mean(axis=1).map("{:+.2f}".format) + " ± " + change.std(axis=1).map("{:.2f}".format)
    pd.set_option("display.width", 400)
    pd.set_option("display.max_columns", None)
    print("Mean ± std over seeds:")
    print(scores.to_string())
    if not changes.empty:
        print(f"\nChange vs {reference} (pp, same seed), mean ± std over seeds:")
        print(changes.to_string())
    return scores, changes
