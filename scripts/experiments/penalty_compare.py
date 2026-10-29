from penalty_variants import compare

NAMES = ["a_without_subtraction", "b_concept_threshold", "c_absence_reward", "d_threshold_and_reward", "e_relative_presence"]


def main():
    compare(NAMES)


if __name__ == "__main__":
    main()
