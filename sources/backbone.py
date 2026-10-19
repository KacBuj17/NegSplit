import torch
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast

from config import CLIP_ID, DEVICE

LAYERS = (6, 8)


def l2(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def save_output(store, layer):
    def hook(module, inputs, output):
        store[layer] = output[0] if isinstance(output, tuple) else output

    return hook


class FrozenCLIP:
    def __init__(self):
        self.dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=self.dtype).to(DEVICE).eval()
        self.model.requires_grad_(False)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(CLIP_ID)
        self.processor = CLIPImageProcessor.from_pretrained(CLIP_ID)

    @torch.no_grad()
    def images(self, images):
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        return self.encode_pixels(pixels)

    @torch.no_grad()
    def encode_pixels(self, pixels):
        out = self.model.get_image_features(pixel_values=pixels.to(DEVICE, self.dtype))
        return (out if torch.is_tensor(out) else out.pooler_output).float()

    def tokenize(self, texts):
        return self.tokenizer(list(texts), padding="max_length", max_length=77, truncation=True,
                              return_tensors="pt")["input_ids"]

    @torch.no_grad()
    def texts(self, texts):
        return self.encode_ids(self.tokenize(texts))

    @torch.no_grad()
    def encode_ids(self, ids):
        ids = ids.to(DEVICE)
        eot = (ids == self.tokenizer.eos_token_id).int().argmax(-1)
        positions = torch.arange(ids.shape[1], device=ids.device)[None]
        mask = positions <= eot[:, None]

        captured = {}
        blocks = self.model.text_model.encoder.layers
        hooks = [blocks[layer - 1].register_forward_hook(save_output(captured, layer)) for layer in LAYERS]
        try:
            out = self.model.text_model(input_ids=ids, attention_mask=mask.long())
        finally:
            for hook in hooks:
                hook.remove()

        rows = torch.arange(len(ids), device=ids.device)
        e_orig = self.model.text_projection(out.last_hidden_state[rows, eot]).float()
        hidden = torch.stack([captured[layer] for layer in LAYERS], 1).float()
        return hidden, e_orig, mask, eot
