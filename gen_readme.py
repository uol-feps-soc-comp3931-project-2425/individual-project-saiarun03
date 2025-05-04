#!/usr/bin/env python3
"""
gen_readme.py

Automatically generates a README.md containing a tree view of your
project directory (depth=2), excluding common unwanted folders.
"""

import subprocess
import sys
from pathlib import Path

# ─── Configuration ─────────────────────────────────────────────────────────────

# Pattern of directories/files to ignore (passed to `tree -I`)
EXCLUDE_PATTERN = ".git|__pycache__|venv"

# How deep to display (levels)
TREE_DEPTH = "2"

# Output README file
README_PATH = Path("README.md")

# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_tree() -> str:
    """
    Runs the `tree` command and returns its output as a string.
    Exits with an error if `tree` is not installed or fails.
    """
    try:
        proc = subprocess.run(
            ["tree", "-I", EXCLUDE_PATTERN, "-L", TREE_DEPTH],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except FileNotFoundError:
        print("Error: `tree` command not found. Please install it (e.g. `brew install tree`).", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running tree:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    return proc.stdout

def write_readme(tree_str: str) -> None:
    """
    Writes the project structure into README.md
    """
    content = [
        "# Project Structure\n",
        "\n",
        "```",
        tree_str.rstrip("\n"),
        "```",
        "\n"
    ]
    README_PATH.write_text("\n".join(content))
    print(f"Wrote project tree to {README_PATH}")

def main():
    tree_output = generate_tree()
    write_readme(tree_output)

if __name__ == "__main__":
    main()
