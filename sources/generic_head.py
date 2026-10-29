import torch
import torch.nn as nn

from backbone import l2
from head import NegSplit

D = 512
CONTEXT = 2 * D
HIDDEN = 128
IMAGE_CHUNK = 64


class GenericCorrection(NegSplit):
    def __init__(self, clip, mu):
        super().__init__(clip, mu)
        del self.proj, self.mu, self.log_gamma, self.log_beta, self.tau
        with torch.random.fork_rng(devices=[]):
            self.correction = nn.Sequential(nn.Linear(CONTEXT + 2 * D, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def encode(self, hidden, e_orig, mask, eot):
        tokens = self.mix_layers(hidden)
        h_neg = torch.einsum("bl,bld->bd", self.attention(tokens, mask), tokens)
        t_eot = tokens[torch.arange(len(tokens), device=tokens.device), eot]
        ctx = torch.cat([t_eot, h_neg], -1)

        p = torch.sigmoid(self.p_head(ctx)).squeeze(-1)
        e_aff = l2(l2(e_orig) + p[:, None] * self.aff_mlp(ctx))
        first = self.correction[0]
        text_part = ctx @ first.weight[:, :CONTEXT].T + first.bias
        return e_aff, text_part, p

    def image_weights(self):
        weight = self.correction[0].weight
        return weight[:, CONTEXT:CONTEXT + D], weight[:, CONTEXT + D:]

    def output(self, h):
        return self.correction[2](self.correction[1](h)).squeeze(-1)

    def score(self, img, e_aff, text_part, p):
        w_img, w_pair = self.image_weights()
        parts = []
        for x in img.split(IMAGE_CHUNK):
            pair = ((x[:, None] * w_pair) @ e_aff.T).transpose(1, 2)
            h = text_part[None] + (x @ w_img.T)[:, None] + pair
            parts.append(x @ e_aff.T + p[None] * self.output(h))
        return torch.cat(parts)

    def score_options(self, img, e_aff, text_part, p):
        w_img, w_pair = self.image_weights()
        h = text_part + (img @ w_img.T)[:, None] + (img[:, None] * e_aff) @ w_pair.T
        return torch.einsum("nd,nkd->nk", img, e_aff) + p * self.output(h)

    def class_embedding(self, e_aff, text_part, p):
        aff = l2(e_aff).mean(0)
        return torch.cat([aff / aff.norm(), text_part.mean(0), p.mean()[None]])
