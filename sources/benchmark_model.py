import torch

from backbone import l2
from config import DEVICE

D = 512


class BenchmarkModel(torch.nn.Module):
    def __init__(self, clip, head=None):
        super().__init__()
        self.clip, self.head = clip, head

    def tokenizer(self, texts):
        return self.clip.tokenize([str(text) for text in texts])

    @torch.no_grad()
    def encode_image(self, pixels):
        return l2(self.clip.encode_pixels(pixels))

    @torch.no_grad()
    def encode_text(self, ids):
        hidden, e_orig, mask, eot = self.clip.encode_ids(ids)
        if self.head is None:
            return l2(e_orig)
        e_aff, e_neg, p = self.head.encode(hidden, e_orig, mask, eot)
        return torch.cat([e_aff, e_neg, p[:, None]], -1)

    def unpack(self, texts):
        return texts[..., :D], texts[..., D:-1], texts[..., -1]

    @torch.no_grad()
    def score(self, images, texts, batch=4096):
        images = images.to(DEVICE).float()
        parts = []
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch].to(DEVICE).float()
            if self.head is None:
                parts.append((images @ chunk.T).cpu())
            else:
                parts.append(self.head.score(images, *self.unpack(chunk)).cpu())
        return torch.cat(parts, 1)

    @torch.no_grad()
    def score_options(self, images, texts):
        images, texts = images.to(DEVICE).float(), texts.to(DEVICE).float()
        if self.head is None:
            return torch.einsum("nd,nkd->nk", images, texts)
        return self.head.score_options(images, *self.unpack(texts))

    @torch.no_grad()
    def class_embedding(self, texts):
        if self.head is None:
            mean = l2(texts).mean(0)
            return mean / mean.norm()
        e_aff, e_neg, p = self.unpack(texts)
        if hasattr(self.head, "class_embedding"):
            return self.head.class_embedding(e_aff, e_neg, p)
        aff, neg = l2(e_aff).mean(0), l2(e_neg).mean(0)
        return torch.cat([aff / aff.norm(), neg / neg.norm(), p.mean()[None]])
