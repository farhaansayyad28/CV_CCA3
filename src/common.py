"""Shared paths and small helpers for the Computer Vision CCA 3 project."""
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "input" / "images"
VIDEO_DIR = ROOT / "data" / "input" / "video"
OUTPUT_DIR = ROOT / "outputs"


def read_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}\nCheck the filename and that the file is a supported image.")
    return image


def require_file(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}\nAdd your own file at the path shown, then run again.")
    return path


def find_video():
    """Return the supplied MP4/MOV video, if present."""
    for name in ("motion_video.mp4", "motion_video.mov", "IMG_2811.MOV"):
        candidate = VIDEO_DIR / name
        if candidate.is_file():
            return candidate
    return None


def save_image(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save output image: {path}")
