"""Clones the external benchmarks next to the NegSplit repo and applies our patches from benchmarks/."""
import subprocess

from config import CLIP_BENCHMARK_REPO, CONCLIP_REPO, NEGBENCH_REPO, PATCHES

REPOS = [
    (NEGBENCH_REPO, "https://github.com/m1k2zoo/negbench.git", "2ac009b5adfb18d1d7c185c466b2bced10f43f6a", "negbench.patch"),
    (CLIP_BENCHMARK_REPO, "https://github.com/LAION-AI/CLIP_benchmark.git", "ee14a9a8a9bcde09d51f4a4b1dff5a75182a3596",
     "CLIP_benchmark.patch"),
    (CONCLIP_REPO, "https://github.com/jaisidhsingh/CoN-CLIP.git", "fce0ac9452e26f3c5877e6f2eca3eee9c2c8b806", "CoN-CLIP.patch"),
]


def git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True)


def main():
    for folder, url, commit, patch in REPOS:
        if not folder.exists():
            git("clone", url, str(folder))
        git("checkout", "--force", commit, cwd=folder)
        git("apply", str(PATCHES / patch), cwd=folder)
        print(f"{folder.name}: {commit[:7]} + {patch}")


if __name__ == "__main__":
    main()
