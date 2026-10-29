from penalty_variants import ThresholdAndReward, compare, run_variant

NAME = "d_threshold_and_reward"


def main():
    run_variant(NAME, ThresholdAndReward)
    compare(["a_without_subtraction", "b_concept_threshold", "c_absence_reward", NAME])


if __name__ == "__main__":
    main()
