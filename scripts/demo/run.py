import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from backbone import FrozenCLIP, l2
from config import TRAIN_DATA
from head import load
from utils.training import COLS, class_prompt

QUERY = "A photo of a white cat without a black dog"
SEED = 0
GRAY, BLUE, ORANGE = "#888888", "#2a78d6", "#eb6834"


def concepts():
    columns = ["obj_present", "obj_absent"] + [f"neg_obj_{col}" for col in COLS]
    df = pd.read_parquet(TRAIN_DATA, columns=columns)
    return sorted({word for col in columns for word in df[col].unique() if word})


def token_words(clip, query, length):
    ids = clip.tokenizer(query)["input_ids"]
    words = [token.replace("</w>", "") for token in clip.tokenizer.convert_ids_to_tokens(ids)]
    return ["[SOT]"] + words[1:length - 1] + ["[EOT]"]


@torch.no_grad()
def walk_through(clip, head, query):
    hidden, e_orig, mask, eot = clip.texts([query])
    length = int(eot[0]) + 1

    tokens = head.mix_layers(hidden)
    attention = head.attention(tokens, mask)
    h_neg = torch.einsum("bl,bld->bd", attention, tokens)
    ctx = torch.cat([tokens[0, eot[0]], h_neg[0]])[None]

    e_aff, e_neg, p = head.encode(hidden, e_orig, mask, eot)
    correction = p[:, None] * head.aff_mlp(ctx)
    subtraction = head.log_gamma.exp() * p[:, None] * (e_neg - head.mu)

    names = concepts()
    concept_emb = l2(clip.texts([class_prompt(name) for name in names])[1])
    nearest = (e_neg[0] @ concept_emb.T).topk(5)
    nearest_names = [names[i] for i in nearest.indices.tolist()]
    negated_emb = concept_emb[nearest.indices[0]]
    e_orig = l2(e_orig)

    return {
        "words": token_words(clip, query, length),
        "layers": torch.softmax(head.layer_logits, 0).cpu().numpy(),
        "attention": attention[0, :length].cpu().numpy(),
        "p": p.item(),
        "nearest": list(zip(nearest_names, nearest.values.tolist())),
        "subtraction": subtraction.norm().item(),
        "correction": correction.norm().item(),
        "cos_aff_orig": (e_aff @ e_orig.T).item(),
        "negated": nearest_names[0],
        "orig_to_negated": (e_orig[0] @ negated_emb).item(),
        "aff_to_negated": (e_aff[0] @ negated_emb).item(),
        "beta": head.log_beta.exp().item(),
        "tau": head.tau.item(),
    }


def print_steps(query, s):
    print(f"query: {query}")
    print(f"1. tokens: {' · '.join(s['words'])}")
    print(f"2. layer weights: layer 6 = {s['layers'][0]:.3f}, layer 8 = {s['layers'][1]:.3f}")
    print("3. neg attention: " + ", ".join(f"{w} {a:.2f}" for w, a in zip(s["words"], s["attention"])))
    print(f"4. p (is there a negation) = {s['p']:.3f}")
    print("5. e_neg nearest: " + ", ".join(f"{name} {v:.3f}" for name, v in s["nearest"]))
    print(f"6. subtraction |γ·p·(e_neg − μ)| = {s['subtraction']:.3f}, correction |p·MLP_aff| = {s['correction']:.3f}")
    print(f"7. cos(e_aff, e_orig) = {s['cos_aff_orig']:.3f}")
    print(f"   similarity to \"{s['negated']}\": e_orig {s['orig_to_negated']:.3f} → e_aff {s['aff_to_negated']:.3f}")
    print(f"8. penalty: β = {s['beta']:.3f}, τ = {s['tau']:.3f}")


def plot_attention(ax, words, weights):
    x = np.arange(len(words))
    ax.bar(x, weights, color=BLUE)
    ax.set_xticks(x, words, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("neg query attention")


def plot_penalty(ax, s):
    similarity = np.linspace(0.05, 0.35, 200)
    penalty = s["beta"] * s["p"] * np.maximum(similarity - s["tau"], 0)
    ax.plot(similarity, penalty, color=ORANGE)
    ax.axvline(s["tau"], color=GRAY, linestyle="--")
    ax.set_xlabel("⟨image, e_neg⟩ — how much the image resembles the negated thing")
    ax.set_title(f"penalty subtracted from the score (β = {s['beta']:.2f}, τ = {s['tau']:.3f})")


def plot_steps(query, s):
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"NegSplit step by step: \"{query}\"", fontsize=13)
    gs = ax[0, 0].get_gridspec()
    ax[0, 0].remove()
    ax[0, 1].remove()
    plot_attention(fig.add_subplot(gs[0, :2]), s["words"], s["attention"])

    ax[0, 2].bar(["layer 6", "layer 8"], s["layers"], color=GRAY)
    ax[0, 2].bar(["p"], [s["p"]], color=ORANGE)
    ax[0, 2].set_ylim(0, 1)
    ax[0, 2].set_title("layer weights and negation detector p")

    names = [name for name, _ in s["nearest"]][::-1]
    values = [v for _, v in s["nearest"]][::-1]
    ax[1, 0].barh(names, values, color=BLUE)
    ax[1, 0].set_title("what e_neg is (cosine with \"a photo of …\")")

    ax[1, 1].bar(["|subtraction|", "|MLP_aff correction|"], [s["subtraction"], s["correction"]], color=[GRAY, BLUE])
    ax[1, 1].set_title(f"change of the sentence vector; cos(e_aff, e_orig) = {s['cos_aff_orig']:.3f}\n"
                       f"similarity to \"{s['negated']}\": e_orig {s['orig_to_negated']:.3f} → "
                       f"e_aff {s['aff_to_negated']:.3f}", fontsize=9)

    plot_penalty(ax[1, 2], s)
    plt.tight_layout()
    plt.show()


def main():
    clip = FrozenCLIP()
    head = load(clip, SEED)
    steps = walk_through(clip, head, QUERY)
    print_steps(QUERY, steps)
    plot_steps(QUERY, steps)


if __name__ == "__main__":
    main()
