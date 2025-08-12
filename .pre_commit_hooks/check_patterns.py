"""Hook to check for patterns in staged changes."""

import argparse
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Union


def main(argv: Union[Sequence[str], None] = None) -> int:
    """Function to check for patterns in staged changes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*", help="Filenames to check")
    args = parser.parse_args(argv)

    # Path to current script
    script_dir = Path(__file__).resolve().parent
    pattern_file = script_dir / "patterns.txt"

    # Get patterns
    with pattern_file.open("r") as f:
        lines = [
            line.partition("#")[0].rstrip()
            for line in f
            if line.partition("#")[0].rstrip()
        ]
        COMPILED_PATTERN = [(line, re.compile(line)) for line in lines]

    violations = []
    for filename in args.filenames:
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            for lineno, line in enumerate(f, 1):
                for pattern_str, pattern in COMPILED_PATTERN:
                    if pattern.search(line):
                        violations.append((filename, lineno, pattern_str, line.strip()))

    violations_commiter = []
    # Get commit metadata
    metadata_fields = {
        "Author Name": subprocess.getoutput("git config user.name"),
        "Author Email": subprocess.getoutput("git config user.email"),
    }

    for label, value in metadata_fields.items():
        for pattern_str, pattern in COMPILED_PATTERN:
            if pattern.search(value):
                violations_commiter.append(f"[{label}] {value.strip()}")
                break

    if violations or violations_commiter:
        print("--")
        print("Pattern(s) found:")
        print("--")
        for filename, lineno, pattern_str, line in violations:
            print(f"Match in {filename}:{lineno} -> pattern: '{pattern_str}'")
            print(f"Line: {line}")
            print("--")
        for v in violations_commiter:
            print(f"Match in commit metadata -> {v}")
        return 1
    else:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
