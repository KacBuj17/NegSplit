from penalty_variants import WithoutSubtraction, compare, run_variant

NAME = "a_without_subtraction"


def main():
    run_variant(NAME, WithoutSubtraction)
    compare([NAME])


if __name__ == "__main__":
    main()
