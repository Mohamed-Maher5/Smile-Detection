"""Quick static smoke test: run is_smiling() on image paths from argv."""

import sys
from pathlib import Path

import cv2

from smile_detector import is_smiling


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python src/test_smile.py path/to/image1.jpg [...]")
        return 2

    for path in argv:
        img = cv2.imread(path)
        if img is None:
            print(f"{path} -> ERROR (unreadable)")
            continue
        print(f"{path} -> {is_smiling(img)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))