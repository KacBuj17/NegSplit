from penalty_variants import RelativePresence, compare, run_variant

NAME = "e_relative_presence"


def main():
    run_variant(NAME, RelativePresence)
    compare(["a_without_subtraction", "b_concept_threshold", NAME])


if __name__ == "__main__":
    main()
