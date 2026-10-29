import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import LAYERS, l2
from config import DEVICE, checkpoint


class NegSplit(nn.Module):
    def __init__(self, clip, mu):
        super().__init__()
        d = 512
        self.ln = copy.deepcopy(clip.model.text_model.final_layer_norm).float().requires_grad_(True)
        self.proj = copy.deepcopy(clip.model.text_projection).float().requires_grad_(True)
        self.layer_logits = nn.Parameter(torch.zeros(len(LAYERS)))
        self.key = nn.Linear(d, d, bias=False)
        self.query = nn.Parameter(torch.randn(d) * 0.02)
        self.aff_mlp = nn.Sequential(nn.Linear(2 * d, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, d))
        self.p_head = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, 1))
        nn.init.zeros_(self.aff_mlp[-1].weight)
        nn.init.zeros_(self.aff_mlp[-1].bias)
        nn.init.constant_(self.p_head[-1].bias, -4.0)
        self.mu = nn.Parameter(mu.clone().float())
        self.log_gamma = nn.Parameter(torch.tensor(math.log(0.5)))
        self.log_beta = nn.Parameter(torch.tensor(math.log(0.5)))
        self.tau = nn.Parameter(torch.tensor(0.2))

    def extra_scalars(self):
        return []

    def mix_layers(self, hidden):
        weights = torch.softmax(self.layer_logits, 0)
        mixed = (weights[None, :, None, None] * hidden).sum(1)
        return self.ln(mixed)

    def attention(self, tokens, mask):
        scores = torch.einsum("bld,d->bl", self.key(tokens), self.query) / math.sqrt(tokens.shape[-1])
        scores = scores.masked_fill(~mask, float("-inf"))
        return scores.softmax(-1)

    def encode(self, hidden, e_orig, mask, eot):
        tokens = self.mix_layers(hidden)
        h_neg = torch.einsum("bl,bld->bd", self.attention(tokens, mask), tokens)
        t_eot = tokens[torch.arange(len(tokens), device=tokens.device), eot]
        ctx = torch.cat([t_eot, h_neg], -1)

        e_neg = l2(self.proj(h_neg))
        p = torch.sigmoid(self.p_head(ctx)).squeeze(-1)
        correction = p[:, None] * self.aff_mlp(ctx)
        subtraction = self.log_gamma.exp() * p[:, None] * (e_neg - self.mu)
        e_aff = l2(l2(e_orig) - subtraction + correction)
        return e_aff, e_neg, p

    def penalty(self, s_neg):
        return -self.log_beta.exp() * F.relu(s_neg - self.tau)

    def score(self, img, e_aff, e_neg, p):
        return img @ e_aff.T + p[None, :] * self.penalty(img @ e_neg.T)

    def score_options(self, img, e_aff, e_neg, p):
        s_aff = torch.einsum("nd,nkd->nk", img, e_aff)
        s_neg = torch.einsum("nd,nkd->nk", img, e_neg)
        return s_aff + p * self.penalty(s_neg)


def load(clip, seed):
    model = NegSplit(clip, torch.zeros(512)).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint(seed), map_location=DEVICE))
    return model.eval()
