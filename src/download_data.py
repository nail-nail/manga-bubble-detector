"""Download the raw dataset from Hugging Face into data/raw/.

Only images and labels are fetched. The dataset's own loading script is skipped
on purpose: it is not needed, and code from a dataset repo should not be run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "PSImera/manga_bubbles_detect"
# Pinned so that every run, including on Kaggle, sees exactly the same files.
REVISION = "0d885e0eb6dfa42880e937936e135f84f045f145"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/manga_bubbles_detect"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=args.out,
        allow_patterns=["images/*", "labels/*", "README.md"],
        max_workers=8,
    )
    print(f"Downloaded to {path}")


if __name__ == "__main__":
    main()
