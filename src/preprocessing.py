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

def detect_with_fallback(detector, img: np.ndarray,
                         scales: tuple[float, ...] = (0.8, 0.7, 0.6, 0.5, 0.4)):
    """Detect faces with progressive center-crop fallback.

    Runs the detector on the original image first.  If that finds at least
    one face the result is returned immediately — the fallback chain is
    **never** triggered for images that already have a detection.

    Otherwise, successive center crops at the given *scales* are tried until
    one produces a detection or all scales are exhausted.
    """
    faces = detector.detect(img)
    if faces:
        return faces

    for scale in scales:
        cropped = zoom_crop(img, scale)
        faces = detector.detect(cropped)
        if faces:
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
            return faces

    return []


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast-limited adaptive histogram equalisation on L channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def apply_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction.  gamma < 1 brightens, gamma > 1 darkens."""
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(img, table)


def to_grayscale_3ch(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale and broadcast back to 3 channels for detector compatibility."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
