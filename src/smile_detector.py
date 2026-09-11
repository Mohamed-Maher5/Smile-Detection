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


def _predict_from_features(feature_row: dict) -> bool:
    row = np.array([[feature_row[f] for f in _FEATURES]], dtype=float)
    if np.isnan(row).any():
        return False
    if _SCALER is not None:
        row = _SCALER.transform(row)
    if _BUNDLE_TYPE == "sklearn":
        prob = float(_MODEL.predict_proba(row)[0, 1])
        return bool(prob >= 0.5)
    # threshold bundle: a single-feature rule, model carries the decision value.
    return bool(row[0, 0] >= float(_MODEL))


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
    faces = detect_with_fallback(get_detector(), image)
    if len(faces) != 1:
        return False

    kps = faces[0].kps
    if kps is None:
        return False

    return _predict_from_features(extract_features(kps))