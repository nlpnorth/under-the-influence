#!/usr/bin/env python3
"""Example: score pile-10k against mean WMDP bio gradients using trackstar.

Computes adafactor normalizers and preconditioners on both query and value
datasets (up to 10k samples each). Preconditioners are not recomputed during
build or score.
"""

import subprocess
import sys
from pathlib import Path

from datasets import load_dataset

from bergson.data import load_scores

cmd = [
    sys.executable,
    "-m",
    "bergson",
    "trackstar",
    "runs/trackstar_wmdp",
    "--model",
    "EleutherAI/pythia-160m",
    "--normalizer",
    "adafactor",
    "--stats_sample_size",
    "10000",
    # Value dataset
    "--data.dataset",
    "NeelNanda/pile-10k",
    "--data.split",
    "train",
    "--data.truncation",
    # Query dataset
    "--query.dataset",
    "cais/wmdp",
    "--query.split",
    "test",
    "--query.subset",
    "wmdp-bio",
    "--query.prompt_column",
    "question",
    # Score settings
    "--unit_normalize",
    "--score",
    "mean",
    "--overwrite",
]

print(" ".join(cmd))
subprocess.run(cmd, check=True)

print(
    "If everything worked, your scores should be in "
    "runs/trackstar_wmdp/scores/scores.bin"
)
print(
    "You can load the memmap using "
    "bergson.data.load_scores('runs/trackstar_wmdp/scores')"
)
print("First 10 scores: ")

scores = load_scores(Path("runs/trackstar_wmdp/scores"))
print(scores[:10])

print("Highest scoring items: ")

pile_ds = load_dataset("NeelNanda/pile-10k", split="train")

top_indices = scores[:].flatten().argsort()[-10:][::-1]
for idx in top_indices:
    print(f"Index: {idx}, Scores: {scores[idx]}")
    print(f"Start of text: {pile_ds[idx]['text'][:50]}")
