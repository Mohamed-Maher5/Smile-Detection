"""Detection fallback (zoom-crop retry) and image preprocessing transforms."""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Zoom crop
# ---------------------------------------------------------------------------

def zoom_crop(img: np.ndarray, scale: float) -> np.ndarray:
    """Return a center crop of *img* at the given fractional *scale* (0 < scale <= 1).

    A scale of 1.0 returns the original image; 0.5 returns the central 50%
    of both width and height.
    """
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    return img[y0 : y0 + new_h, x0 : x0 + new_w].copy()


# ---------------------------------------------------------------------------
# Detection with zoom-crop fallback
# ---------------------------------------------------------------------------

def _restore_crop_coords(faces, img: np.ndarray, cropped: np.ndarray) -> None:
    """Translate crop-local bbox/kps back into the original image's coordinates."""
    h_orig, w_orig = img.shape[:2]
    h_crop, w_crop = cropped.shape[:2]
    y_off = (h_orig - h_crop) // 2
    x_off = (w_orig - w_crop) // 2
    for face in faces:
        face.bbox[0] += x_off
        face.bbox[1] += y_off
        face.bbox[2] += x_off
        face.bbox[3] += y_off
        if face.kps is not None:
            face.kps[:, 0] += x_off
            face.kps[:, 1] += y_off


def detect_with_fallback(detector, img: np.ndarray,
                         scales: tuple[float, ...] = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)):
    """Detect faces with progressive center-crop fallback.

    Runs the detector on the original image first.  If that finds at least
    one face the result is returned immediately — the fallback chain is
    **never** triggered for images that already have a detection.

    Otherwise, each scale from ``scales`` is tried in two interleaved steps:
    the plain center crop first, then — only if that exact scale failed — the
    same crop converted to grayscale (3-channel).  This order recovers
    e.g. file2669.jpg in 3 attempts instead of 8.  All scales are exhausted
    before returning empty.
    """
    faces, _ = detect_with_fallback_traced(detector, img, scales)
    return faces


def detect_with_fallback_traced(detector, img: np.ndarray,
                                scales: tuple[float, ...] = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)):
    """Like :func:`detect_with_fallback` but also reports which stage produced
    the (first non-empty) face list, for latency-path auditing.

    Returns ``(faces, path)`` where *path* is ``"direct"`` (first detect on the
    original image), ``"crop_<scale>"`` (recovered by a plain zoom-crop),
    ``"grayscale_crop_<scale>"`` (the interleaved grayscale attempt at that same
    scale), or ``"none"`` when every stage failed.  Behavior otherwise matches
    ``detect_with_fallback``.
    """
    faces = detector.detect(img)
    if faces:
        return faces, "direct"

    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    for scale in scales:
        cropped = zoom_crop(img, scale)
        faces = detector.detect(cropped)
        if faces:
            _restore_crop_coords(faces, img, cropped)
            return faces, f"crop_{scale}"

        # Same scale, grayscale (3-channel): the plain crop failed, so try the
        # grayscale variant at this same scale before moving to a smaller crop.
        gray_crop = zoom_crop(gray, scale)
        faces = detector.detect(gray_crop)
        if faces:
            _restore_crop_coords(faces, img, gray_crop)
            return faces, f"grayscale_crop_{scale}"

    return [], "none"
