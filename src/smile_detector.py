"""End-to-end smile detection: detector -> align -> features -> model.

Carries the Stage-4 decision from `04_modeling.ipynb`:
  Model     : LogisticRegression (C=0.1), trained on the 4 final aligned features
  Scaler    : fitted StandardScaler (applied before inference)
  Decision  : predict_proba(smile) >= bundle threshold (0.5 when threshold is None)
  Test F1   : 0.8939 (precision 0.9067) on the held-out clean df_test (795);
              0.8733 on the full 823-image delivery set.
"""

from pathlib import Path

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
    ['mouth_eye_ratio', 'mouth_vertical_lift', 'mouth_nose_ratio', 'mouth_width']
    (threshold: predict_proba >= bundle threshold; 0.5 default).
    """
    return detect_and_score(image)[0]