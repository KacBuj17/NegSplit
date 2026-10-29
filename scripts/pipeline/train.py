from config import CHECKPOINTS, SEEDS
from trainer import train


def main():
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        train(seed)


if __name__ == "__main__":
    main()
