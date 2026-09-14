"""Geometric feature extraction from SCRFD 5-point landmarks."""

import numpy as np


def align_kps(kps: np.ndarray) -> np.ndarray:
    """Similarity-align a (5, 2) landmarks array into the canonical eye frame.

    Maps ``left_eye`` -> ``(-0.5, 0)`` and ``right_eye`` -> ``(0.5, 0)`` using
    an in-plane similarity transform: translate the eye midpoint to the origin,
    rotate the eye vector onto the +x axis, then uniformly scale so the
    inter-eye distance becomes 1.  Rotation + uniform scale preserves all
    distances and angles, so relative face geometry is unchanged — the output
    is the input de-rotated / de-scaled / re-centered.

    Aligning before :func:`extract_features` makes the features measured along
    the face's own axes (perpendicular to the eye line) rather than the
    image's y-axis, which is the intended robustness fix for head roll.

    Parameters
    ----------
    kps : array-like of shape (5, 2)
        Rows [left_eye, right_eye, nose, left_mouth, right_mouth] in image
        pixel space.

    Returns
    -------
    np.ndarray of shape (5, 2) aligned to the canonical frame (float64).
    """
    kps = np.asarray(kps, dtype=float)
    if kps.shape != (5, 2):
        raise ValueError(f"expected shape (5, 2), got {kps.shape}")
    left_eye, right_eye = kps[0], kps[1]

    midpoint = (left_eye + right_eye) / 2.0
    centered = kps - midpoint  # eye midpoint -> origin

    eye_vec = right_eye - left_eye
    angle = np.arctan2(eye_vec[1], eye_vec[0])  # signed angle of eye vector
    cth, sth = np.cos(-angle), np.sin(-angle)   # rotate by -angle -> +x axis
    rot = np.array([[cth, -sth], [sth, cth]])
    de_rotated = centered @ rot.T

    scale = 1.0 / np.hypot(eye_vec[0], eye_vec[1])  # |eye|/2 -> 0.5 => 1/|eye|
    aligned = de_rotated * scale
    return aligned


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

    CANDIDATE FEATURES (see 03_feature_engineering.ipynb)
    -----------------------------------------------------
    • mouth_corner_angle   : the angle (in degrees) subtended at the nose by
      the vectors to the left and right mouth corners.  Scale-invariant by
      construction (ratio of norms cancels).
    • mouth_symmetry       : |left_mouth_y - right_mouth_y| / inter_eye_dist.
      Asymmetric mouth heights (one corner lifted) drive this away from zero.
    • mouth_triangle_area  : 0.5 * |cross(v_left, v_right)| / inter_eye_dist^2,
      the area of the nose–corner–corner triangle, normalized for scale.
    """
    assert kps.shape == (5, 2), f"Expected kps shape (5, 2), got {kps.shape}"

    left_eye, right_eye, nose, left_mouth, right_mouth = kps

    inter_eye_dist = float(np.linalg.norm(left_eye - right_eye))
    mouth_width = float(np.linalg.norm(left_mouth - right_mouth))

    eye_midpoint = (left_eye + right_eye) / 2.0
    eye_midpoint_y = float(eye_midpoint[1])
    mouth_midpoint_y = float((left_mouth[1] + right_mouth[1]) / 2.0)

    nose_to_eye_mid = float(np.linalg.norm(nose - eye_midpoint))

    # Vectors from nose to the mouth corners (image coords, y grows downward).
    v_left = left_mouth - nose
    v_right = right_mouth - nose

    left_mouth_y, right_mouth_y = float(left_mouth[1]), float(right_mouth[1])

    # Angle subtended at the nose between the two mouth-corner vectors.
    n_left = float(np.linalg.norm(v_left))
    n_right = float(np.linalg.norm(v_right))
    if n_left > 0.0 and n_right > 0.0:
        cos_ang = float(np.dot(v_left, v_right) / (n_left * n_right))
        mouth_corner_angle = float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))
    else:
        mouth_corner_angle = np.nan

    # Triangle (nose, left corner, right corner) area via 2D scalar cross product.
    cross_2d = float(v_left[0] * v_right[1] - v_left[1] * v_right[0])
    triangle_area = 0.5 * abs(cross_2d)

    if inter_eye_dist == 0.0:
        nan = np.nan
        return {
            "mouth_width": mouth_width,
            "inter_eye_dist": inter_eye_dist,
            "mouth_eye_ratio": nan,
            "mouth_vertical_lift": nan,
            "mouth_nose_ratio": nan,
            "mouth_corner_angle": nan,
            "mouth_symmetry": nan,
            "mouth_triangle_area": nan,
        }

    return {
        "mouth_width": mouth_width,
        "inter_eye_dist": inter_eye_dist,
        "mouth_eye_ratio": mouth_width / inter_eye_dist,
        "mouth_vertical_lift": (eye_midpoint_y - mouth_midpoint_y) / inter_eye_dist,
        "mouth_nose_ratio": mouth_width / nose_to_eye_mid,
        "mouth_corner_angle": mouth_corner_angle,
        "mouth_symmetry": abs(left_mouth_y - right_mouth_y) / inter_eye_dist,
        "mouth_triangle_area": triangle_area / inter_eye_dist**2,
    }
