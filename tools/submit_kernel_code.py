from __future__ import annotations

import argparse

from kaggle.api.kaggle_api_extended import KaggleApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True, help="owner/kernel-slug")
    parser.add_argument("--message", required=True)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--competition", default="biohub-cell-tracking-during-development")
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    result = api.competition_submit_code(
        "submission.csv",
        args.message,
        args.competition,
        args.kernel,
        args.version,
    )
    print(result)


if __name__ == "__main__":
    main()
