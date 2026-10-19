import copy

import torch

from backbone import FrozenCLIP, l2
from config import CHECKPOINTS, DEVICE, SEEDS, checkpoint
from head import NegSplit
from utils.training import class_prompt, load_data, loss, validate

EPOCHS = 4
BATCH = 256
LR, COPY_LR, SCALAR_LR, WEIGHT_DECAY = 2e-4, 5e-5, 5e-3, 0.01


def make_optimizer(model, total_steps):
    copies = [*model.ln.parameters(), *model.proj.parameters()]
    scalars = [model.log_gamma, model.log_beta, model.tau, model.layer_logits]
    special = {id(param) for param in copies + scalars}
    network = [param for param in model.parameters() if id(param) not in special]
    optimizer = torch.optim.AdamW([
        {"params": network, "lr": LR, "weight_decay": WEIGHT_DECAY},
        {"params": copies, "lr": COPY_LR, "weight_decay": 0.0},
        {"params": scalars, "lr": SCALAR_LR, "weight_decay": 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[LR, COPY_LR, SCALAR_LR], total_steps=total_steps, pct_start=0.05)
    return optimizer, scheduler


def train_one_epoch(model, clip, data, optimizer, scheduler):
    model.train()
    order = data["train"][torch.randperm(len(data["train"]))]
    for start in range(0, len(order) - BATCH + 1, BATCH):
        optimizer.zero_grad(set_to_none=True)
        loss(model, clip, data, order[start:start + BATCH]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()


def retrieval_not_worse_than_clip(result):
    for col in ("caption_pos", "caption_neg_true"):
        for task in ("t2i", "i2t"):
            if result[f"{col} {task} negsplit"] < result[f"{col} {task} clip"]:
                return False
    return True


def train(seed):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9)
    clip = FrozenCLIP()
    data = load_data()
    concepts = l2(clip.texts([class_prompt(c) for c in data["classes"]])[1])
    model = NegSplit(clip, concepts.mean(0)).to(DEVICE)
    optimizer, scheduler = make_optimizer(model, EPOCHS * (len(data["train"]) // BATCH))

    best_mcq, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, clip, data, optimizer, scheduler)
        model.eval()
        result = validate(model, clip, data)
        guard = retrieval_not_worse_than_clip(result)
        print(f"seed {seed}, epoch {epoch}: validation MCQ {result['mcq negsplit']:.4f}, "
              f"retrieval >= CLIP: {'yes' if guard else 'no'}", flush=True)
        if guard and result["mcq negsplit"] > best_mcq:
            best_mcq, best_state = result["mcq negsplit"], copy.deepcopy(model.state_dict())

    torch.save(best_state, checkpoint(seed))


def main():
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        train(seed)


if __name__ == "__main__":
    main()
