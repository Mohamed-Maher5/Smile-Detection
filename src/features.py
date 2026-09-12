"""Geometric feature extraction from SCRFD 5-point landmarks."""

import numpy as np


def extract_features(kps: np.ndarray) -> dict:
    """Extract scale-invariant geometric features from a (5, 2) landmark array.

    Landmark order (must match SCRFD convention):
        0 – left eye, 1 – right eye, 2 – nose,
        3 – left mouth corner, 4 – right mouth corner.

    Coordinates are in original-image pixel space (y grows downward).

    SIGN CONVENTION for mouth_vertical_lift
    ----------------------------------------
    In image coordinates y increases *downward*.  A mouth that sits below the
    eye line therefore has a *larger* y than the eye line.  We define:

        mouth_vertical_lift = (eye_midpoint_y - mouth_midpoint_y) / inter_eye_dist

    • Positive  → mouth is *above* the eye line (unusual / lifted smile).
    • Zero       → mouth is level with the eyes.
    • Negative   → mouth is *below* the eye line (normal resting face).

    So we *expect* smiling faces to show a *larger* (less negative) value as
    the mouth corners lift toward or past the eye line.

    >>> DO NOT flip this sign without updating every downstream consumer. <<<
    """
    assert kps.shape == (5, 2), f"Expected kps shape (5, 2), got {kps.shape}"

    left_eye, right_eye, nose, left_mouth, right_mouth = kps

    inter_eye_dist = float(np.linalg.norm(left_eye - right_eye))
    mouth_width = float(np.linalg.norm(left_mouth - right_mouth))

    eye_midpoint = (left_eye + right_eye) / 2.0
    eye_midpoint_y = float(eye_midpoint[1])
    mouth_midpoint_y = float((left_mouth[1] + right_mouth[1]) / 2.0)

    nose_to_eye_mid = float(np.linalg.norm(nose - eye_midpoint))

    if inter_eye_dist == 0.0:
        nan = np.nan
        return {
            "mouth_width": mouth_width,
            "inter_eye_dist": inter_eye_dist,
            "mouth_eye_ratio": nan,
            "mouth_vertical_lift": nan,
            "mouth_nose_ratio": nan,
        }

    return {
        "mouth_width": mouth_width,
        "inter_eye_dist": inter_eye_dist,
        "mouth_eye_ratio": mouth_width / inter_eye_dist,
        "mouth_vertical_lift": (eye_midpoint_y - mouth_midpoint_y) / inter_eye_dist,
        "mouth_nose_ratio": mouth_width / nose_to_eye_mid,
    }
