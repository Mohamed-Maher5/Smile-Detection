"""End-to-end smile detection: detector -> features -> trained model/threshold.

Carries the Stage-4 decision from `04_modeling.ipynb`:
  Model     : LogisticRegression (C=1), trained on the 4 final features
  Scaler    : fitted StandardScaler (applied before inference)
  Decision  : predict_proba(smile) >= 0.5
  Test F1   : 0.8839 (precision 0.9010) on the held-out df_test.
"""

from pathlib import Path

import numpy as np

from face_detector import get_detector
from features import extract_features
from preprocessing import detect_with_fallback

BUNDLE_PATH = Path(__file__).resolve().parents[1] / "models" / "classifier" / "final_model.pkl"

import joblib

_BUNDLE = joblib.load(BUNDLE_PATH)
_MODEL = _BUNDLE["model"]
_SCALER = _BUNDLE["scaler"]
_FEATURES = list(_BUNDLE["features"])
_BUNDLE_TYPE = _BUNDLE["type"]


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
        return bool(prob >= 0.5), prob
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
    faces = detect_with_fallback(get_detector(), image)
    n_faces = len(faces)
    if n_faces != 1 or faces[0].kps is None:
        return False, None, None, n_faces
    kps = faces[0].kps
    pred, score = _score_sample(extract_features(kps))
    return pred, kps, score, n_faces


def is_smiling(image: np.ndarray) -> bool:
    """Return True iff *image* contains exactly one face that is judged smiling.

    Pipeline: SCRFD landmark detection (with zoom-crop fallback) on the whole
    image, then geometric features from the single face's 5 landmarks run
    through the same scaler + classifier as training.

    False when there are zero or more than one faces, no usable landmarks, or
    degenerate (NaN) features. The carried classifier is a C=1
    LogisticRegression + StandardScaler on
    ['mouth_eye_ratio', 'mouth_vertical_lift', 'mouth_nose_ratio', 'mouth_width']
    (threshold: predict_proba >= 0.5; test F1 0.8839 / precision 0.9010).
    """
    return detect_and_score(image)[0]