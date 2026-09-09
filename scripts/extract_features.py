"""Compatibility entry for official physical-feature preparation."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import main
from src.data.feature_extractor import extract_visual_descriptor


def extract_visual_representation(img_path, dim=512):
    if dim != 512:
        raise ValueError("Official cover descriptor dimension is 512")
    return extract_visual_descriptor(img_path)


def run_feature_extraction():
    return main(["prepare"])


if __name__ == "__main__":
    main(["prepare", *sys.argv[1:]])
