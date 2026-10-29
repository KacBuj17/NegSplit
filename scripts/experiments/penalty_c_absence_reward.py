from backbone import FrozenCLIP
from config import SEEDS
from penalty_variants import AbsenceReward, compare, load_variant, run_variant, save_diagnostics, variant_checkpoint

NAME = "c_absence_reward"
REFERENCE = "a_without_subtraction"


def main():
    run_variant(NAME, AbsenceReward)
    compare([REFERENCE, NAME])

    clip = FrozenCLIP()
    ratio = {}
    for seed in SEEDS:
        if variant_checkpoint(NAME, seed).exists():
            head = load_variant(clip, AbsenceReward, NAME, seed)
            ratio[f"s{seed}"] = (head.log_beta_reward.exp() / head.log_beta.exp()).item()
    print(f"\nLearned beta_reward / beta: {ratio}")
    save_diagnostics(NAME, {"beta_reward / beta": ratio})


if __name__ == "__main__":
    main()
