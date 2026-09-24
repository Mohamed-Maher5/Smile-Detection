"""End-to-end smile detection: detector -> crop&refine -> align -> features -> model.

Carries the Stage-4 decision from `04_modeling.ipynb`:
  Model     : LogisticRegression (C=0.1), trained on the 3 final aligned features
  Scaler    : fitted StandardScaler (applied before inference)
  Decision  : predict_proba(smile) >= bundle threshold (0.5 when threshold is None)
  Test F1   : 0.8939 (precision 0.9067) on the held-out clean df_test (795);
              0.8942 on the full 823-image delivery benchmark
              (outputs/benchmark_summary.csv).

Live pipeline notes:
  * The single detected face is cropped out, upscaled to a canonical size and
    re-detected on the crop alone (background removed, scale normalised) so the
    5 landmarks are accurate at any distance — see REFINE_MIN_EDGE etc.
  * Features are still the exact scale-invariant geometric features built from
    ``align_kps``-aligned landmarks, normalised by the same StandardScaler the
    model was trained on (identical preprocessing to training).
"""

from pathlib import Path
import os

import cv2
import numpy as np

from face_detector import get_detector
from features import align_kps, extract_features
from preprocessing import detect_with_fallback

BUNDLE_PATH = Path(__file__).resolve().parents[1] / "models" / "classifier" / "final_model.pkl"

import joblib

_BUNDLE = joblib.load(BUNDLE_PATH)
_MODEL = _BUNDLE["model"]
_SCALER = _BUNDLE["scaler"]
_FEATURES = list(_BUNDLE["features"])
_BUNDLE_TYPE = _BUNDLE["type"]
_THRESHOLD = _BUNDLE.get("threshold")

# ── Crop-and-refine landmark stage ──────────────────────────────────────────
# SCRFD landmark quality degrades for small faces (far from the camera) and
# can be distracted by background.  After the coarse full-frame pass the single
# face is cropped out, upscaled to a canonical size and detected again *on the
# crop only*: the background is gone and the face fills the detector input, so
# the 5 keypoints land more precisely at any distance/scale.  The crop is only
# re-detected when it can actually help (face below the size at which landmarks
# are already accurate, or a low-confidence detection); big high-quality faces
# keep the single-pass path and its latency.
REFINE_MIN_EDGE = 224          # upscale the crop so its smaller edge >= this
REFINE_MARGIN = 0.06           # margin around the bbox (fraction of bbox size)
REFINE_MAX_INTER_EYE = 140     # px: faces with inter-eye >= this skip the 2nd pass
REFINE_MIN_DET_SCORE = 0.7     # below this the coarse detection is refined too

# Latency escape hatch: set SMILE_NO_REFINE=1 (or flip at runtime) to fall back to
# the single-pass path.  Single-pass is ~4.5 ms median per call but ~2pp less
# accurate (0.8663 vs 0.8870 on the 823 benchmark); crop+refine is ~7.9 ms median —
# still ~25x inside the live 200 ms decision window (webcam_demo --fps 5).
ENABLE_CROP_REFINE = os.environ.get("SMILE_NO_REFINE", "") != "1"


def _refine_landmarks(image: np.ndarray, bbox, coarse_kps: np.ndarray) -> np.ndarray:
    """Re-detect landmarks on an upscaled face crop; fall back to coarse on failure."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return coarse_kps
    margin = REFINE_MARGIN * max(bw, bh)
    cx1 = max(0, int(x1 - margin)); cy1 = max(0, int(y1 - margin))
    cx2 = min(w, int(np.ceil(x2 + margin))); cy2 = min(h, int(np.ceil(y2 + margin)))
    if cx2 - cx1 <= 0 or cy2 - cy1 <= 0:
        return coarse_kps
    crop = image[cy1:cy2, cx1:cx2]
    cw, ch = crop.shape[1], crop.shape[0]
    u = REFINE_MIN_EDGE / min(cw, ch)
    if u > 1.0:
        crop = cv2.resize(crop, (max(1, int(round(cw * u))), max(1, int(round(ch * u)))),
                          interpolation=cv2.INTER_LINEAR)
    refined = detect_with_fallback(get_detector(), crop)
    if len(refined) != 1 or refined[0].kps is None:
        return coarse_kps
    kps_up = np.asarray(refined[0].kps, dtype=float)
    kps_crop = kps_up / u if u > 1.0 else kps_up
    return kps_crop + np.array([cx1, cy1], dtype=float)


def _score_sample(feature_row: dict) -> tuple[bool, float | None]:
    """Return (prediction, decision score) for a single feature row.

    For an ``sklearn`` bundle the score is the smile-class ``predict_proba``;
    for a ``threshold`` bundle it is the raw single-feature value that the
    rule thresholds on (there is no probability model).
    """
    row = np.array([[feature_row[f] for f in _FEATURES]], dtype=float)
    if np.isnan(row).any():
        return False, None
    if _SCALER is not None:
        row = _SCALER.transform(row)
    if _BUNDLE_TYPE == "sklearn":
        prob = float(_MODEL.predict_proba(row)[0, 1])
        cut = 0.5 if _THRESHOLD is None else float(_THRESHOLD)
        return bool(prob >= cut), prob
    # threshold bundle: a single-feature rule, model carries the decision value.
    raw = float(row[0, 0])
    return bool(raw >= float(_MODEL)), raw


def detect_and_score(image: np.ndarray) -> tuple[bool, np.ndarray | None, float | None, int]:
    """Run the production pipeline and return (is_smiling, kps, score, n_faces).

    Returns the judgement, the (5, 2) landmarks that produced it (None when
    there is no usable single face), the decision score as reported by
    :func:`_score_sample`, and the total number of faces SCRFD found (so the
    webcam demo can display why the smile check was skipped when count != 1).

    Callers like the webcam demo can render the exact landmarks and confidence
    the model used without re-running detection.  is_smiling() returns False
    for both zero-face and multi-face frames, but the count distinguishes them.
    """
    return detect_and_score_detailed(image)[:4]


def detect_and_score_detailed(
    image: np.ndarray,
) -> tuple[bool, np.ndarray | None, float | None, int, float | None, np.ndarray | None]:
    """Like :func:`detect_and_score` but also returns the SCRFD detection score
    and the detected face bounding box.

    The second-to-last element (offset -2) is ``faces[0].det_score`` — the
    detector's own confidence for the single face that was classified, or
    ``None`` when there is no usable single face.  It lets callers like the
    webcam demo apply a low-confidence hold without re-running detection or
    touching ``face_detector.py``.  The last element is ``faces[0].bbox``
    (SCRFD order: x1, y1, x2, y2) or ``None`` and lets the demo draw the face
    bounding box overlay.
    """
    faces = detect_with_fallback(get_detector(), image)
    n_faces = len(faces)
    if n_faces != 1 or faces[0].kps is None:
        return False, None, None, n_faces, None, None
    kps = faces[0].kps
    det_score = float(faces[0].det_score)
    bbox = faces[0].bbox
    left_eye, right_eye = kps[0], kps[1]
    inter_eye = float(np.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]))
    if ENABLE_CROP_REFINE and (inter_eye < REFINE_MAX_INTER_EYE or det_score < REFINE_MIN_DET_SCORE):
        kps = _refine_landmarks(image, bbox, kps)
    aligned_kps = align_kps(kps)
    pred, score = _score_sample(extract_features(aligned_kps))
    return pred, kps, score, n_faces, det_score, bbox


def is_smiling(image: np.ndarray) -> bool:
    """Return True iff *image* contains exactly one face that is judged smiling.

    Pipeline: SCRFD landmark detection (with zoom-crop fallback) on the whole
    image, then the single face's 5 landmarks are similarity-aligned to the
    canonical eye frame (:func:`features.align_kps`), and geometric features
    from the aligned landmarks run through the same scaler + classifier as
    training.

    False when there are zero or more than one faces, no usable landmarks, or
    degenerate (NaN) features. The carried classifier is a C=0.1
    LogisticRegression + StandardScaler on
    ['mouth_width', 'mouth_vertical_lift', 'mouth_nose_ratio']
    (threshold: predict_proba >= bundle threshold; 0.5 default).
    """
    return detect_and_score(image)[0]