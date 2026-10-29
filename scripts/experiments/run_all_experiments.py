import os
import subprocess
import sys
from pathlib import Path

from config import PROJECT

SCRIPTS = ["ablation", "no_x", "heldout_categories", "subtraction", "multiple_negations", "penalty_e_detector_auroc",
           "penalty_a_without_subtraction", "penalty_b_concept_threshold", "penalty_c_absence_reward",
           "penalty_d_threshold_and_reward", "penalty_e_relative_presence", "penalty_compare", "attribute_negation", "generic_correction", "generic_heldout"]


def main():
    env = {**os.environ, "PYTHONPATH": str(PROJECT / "sources"), "PYTHONIOENCODING": "utf-8"}
    for name in SCRIPTS:
        print(f"=== {name}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name(f"{name}.py"))], env=env, check=True)


if __name__ == "__main__":
    main()
