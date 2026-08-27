"""Print a small, read-only summary of a generated Parquet file."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    frame = pl.read_parquet(args.path)
    print(frame.schema)
    print(frame.head(10))


if __name__ == "__main__":
    main()
