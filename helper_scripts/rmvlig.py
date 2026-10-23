#!/usr/bin/env python3
"""
Recursively copy PDB files from an input folder (including subfolders) into an
output folder (preserving subfolder structure) while removing:
  - all HETATM lines
  - all CONECT (and CONNECT, if present) lines

Rules:
  - ALWAYS keep the first line of each PDB file (write it unchanged).
  - ALWAYS write an output copy for every PDB found, even if nothing is removed.
  - Stream processing: handles files one-by-one and line-by-line (no big temp folders,
    no loading entire file into memory).

Defaults can be set at the top (DEFAULT_INPUT_DIR / DEFAULT_OUTPUT_DIR), but can be
overridden by CLI args.

Usage:
  python3 rmvlig.py --input /path/to/input --output /path/to/output
  python3 rmvlig.py   # uses defaults
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional


# ========= Set defaults here =========
DEFAULT_INPUT_DIR = Path("./input_pdbs")
DEFAULT_OUTPUT_DIR = Path("./output_pdbs_cleaned")
# =====================================


def should_remove_pdb_line(line: str) -> bool:
    """
    Return True if this PDB line should be removed.
    PDB record name is columns 1-6 (0-5 in Python), usually left-justified.
    Robust to short lines.
    """
    record = line[:6].strip().upper()
    return record in {"HETATM", "CONECT", "CONNECT"}


def clean_pdb_file_streaming(input_path: Path, output_path: Path) -> None:
    """
    Stream-clean a single PDB:
      - write first line unconditionally
      - then skip HETATM/CONECT/CONNECT lines
      - write everything else
    Always creates an output file (copy), even if no lines are removed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # latin-1 + replace keeps script resilient to odd bytes.
    with input_path.open("r", encoding="latin-1", errors="replace") as fin, \
         output_path.open("w", encoding="latin-1", errors="replace") as fout:

        first_line = fin.readline()
        if first_line != "":
            fout.write(first_line)  # keep first line no matter what

        for line in fin:
            if should_remove_pdb_line(line):
                continue
            fout.write(line)


def process_folder(input_dir: Path | str, output_dir: Path | str, *, extensions: Optional[set[str]] = None) -> int:
    """
    Recursively walk input_dir, and for each matching file extension, write a cleaned
    copy to output_dir, preserving relative subfolder structure.

    Processes files one at a time; does not create temp folders.
    Returns number of PDB files processed.
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if extensions is None:
        extensions = {".pdb"}

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for root, _, files in os.walk(input_dir):
        root_path = Path(root)
        for name in files:
            in_path = root_path / name
            if in_path.suffix.lower() in extensions:
                rel = in_path.relative_to(input_dir)
                out_path = output_dir / rel
                clean_pdb_file_streaming(in_path, out_path)
                processed += 1

    return processed


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Recursively copy PDBs to an output folder, removing HETATM and CONECT/CONNECT lines (keeping first line)."
    )
    p.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input folder containing PDBs in subfolders (default: {DEFAULT_INPUT_DIR})",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output folder to write cleaned copies (default: {DEFAULT_OUTPUT_DIR})",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    count = process_folder(args.input, args.output)
    print(f"Processed {count} PDB file(s). Output written to: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())