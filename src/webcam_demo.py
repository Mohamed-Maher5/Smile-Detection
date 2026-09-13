"""Live webcam demo: classify each frame with is_smiling() and overlay result + latency
plus the 5 face landmarks (kps) the model used, drawn live with short tags.

The on-screen verdict is stabilised against per-frame flicker: the raw smile
predict_proba is smoothed over a rolling 5-frame window and decided with
hysteresis (up-threshold T+0.05 / down-threshold T-0.05 around T=0.5), instead
of thresholding the instantaneous probability on every frame.

When a single face is detected but with low detector confidence (det_score < 0.6)
the verdict is withheld: the last stable result is held on screen with a yellow
border signalling that it is not from the current frame.

The display is decoupled from the capture resolution: the window is created
resizable at 1280x720 and each frame is upscaled (never downscaled) to at least
1280x720 before drawing, so the on-screen size is guaranteed regardless of the
resolution the camera actually delivers."""

import signal
import sys
import time
from collections import deque

import cv2
import numpy as np

from smile_detector import detect_and_score_detailed

# SCRFD landmark order (matches features.py): left eye, right eye, nose,
# left mouth corner, right mouth corner.
KPS_TAGS = ["LE", "RE", "N", "LM", "RM"]
KPS_COLOR = (255, 255, 0)  # cyan (BGR)

# Stabilisation knobs (the underlying model/classification is not modified).
SMILE_THRESHOLD = 0.5      # model's predict_proba decision point
SMILE_HYSTERESIS = 0.05    # hold band: keep label inside [T-0.05, T+0.05]
WINDOW_SIZE = 5            # rolling average length for the smoothed decision
DET_CONF_THRESHOLD = 0.6   # hold result when the single-face det_score drops below this
HOLD_COLOR = (0, 255, 255)  # yellow (BGR) frame border for the hold indicator

# Requested capture resolution (webcams may not honour it exactly).
CAM_WIDTH, CAM_HEIGHT = 1280, 720

# Display is decoupled from the capture resolution: the window is forced to
# (at least) this size and every frame is upscaled to fit, so the image on
# screen stays 1280x720 regardless of what the camera actually delivers.
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1280, 720
WINDOW_TITLE = "Smile Detection"


def _hysteresis_verdict(smoothed: float | None, prev: bool) -> bool:
    """Map the smoothed probability to a label with a hold band.

    Flipping to True only when ``smoothed > T + 0.05``, to False only when
    ``smoothed < T - 0.05``, and otherwise keeping the previous label — so
    borderline values no longer toggle on a frame-by-frame basis.
    """
    if smoothed is None:
        return prev
    if smoothed > SMILE_THRESHOLD + SMILE_HYSTERESIS:
        return True
    if smoothed < SMILE_THRESHOLD - SMILE_HYSTERESIS:
        return False
    return prev


def _status_overlay(n_faces: int, verdict: bool) -> tuple[str, tuple[int, int, int], bool]:
    """Return (label, BGR color, show_probability_lines) for the main label.

    is_smiling() returns False identically for no-face, multi-face, and
    genuine "not smiling" — so the screen distinguishes them: no face is shown
    in gray, multiple faces in yellow, and a single face uses the smile verdict
    (SMILING green / NOT SMILING red). The probability lines are only
    meaningful when there is exactly one face to score.
    """
    if n_faces == 0:
        return "NO FACE", (128, 128, 128), False
    if n_faces >= 2:
        return "MULTIPLE FACES", (0, 255, 255), False
    return ("SMILING" if verdict else "NOT SMILING"), (0, 255, 0) if verdict else (0, 0, 255), True


def _draw_kps(frame: np.ndarray, kps: np.ndarray) -> None:
    """Overlay each landmark as a small cyan dot with its tag next to it."""
    for tag, (px, py) in zip(KPS_TAGS, kps):
        px, py = float(px), float(py)
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        xi, yi = int(round(px)), int(round(py))
        if not (0 <= xi < frame.shape[1] and 0 <= yi < frame.shape[0]):
            continue
        cv2.circle(frame, (xi, yi), 4, KPS_COLOR, -1)
        cv2.putText(frame, tag, (xi + 6, yi - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, KPS_COLOR, 1, cv2.LINE_AA)


def main() -> int:
    stop = False

    def _request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam (cv2.VideoCapture(0))")
        return 1

    # Most webcams only expose 1280x720 under MJPEG; request it first, then the
    # size, so the desired resolution is honoured when the device supports it.
    print(f"Before set: W={cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f} "
          f"H={cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    print(f"After set : W={cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f} "
          f"H={cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Capture resolution: requested {CAM_WIDTH}x{CAM_HEIGHT}, got {got_w}x{got_h}")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    print("Press 'q' to quit.")
    window = deque(maxlen=WINDOW_SIZE)  # rolling raw probabilities
    verdict = False                     # previous displayed label (NOT SMILING)
    # cv2's Qt windows overwrite the default SIGINT/SIGTERM dispositions and
    # park them at SIG_IGN while the loop runs; re-arm Python's handlers every
    # frame so Ctrl-C / `kill` cleanly release the camera instead of leaving a
    # stuck process holding /dev/video0.
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    while not stop:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("WARNING: failed to read frame, skipping")
            continue

        try:
            t0 = time.perf_counter()
            _, kps, raw, n_faces, det_score = detect_and_score_detailed(frame)
            latency_ms = (time.perf_counter() - t0) * 1e3
        except Exception as err:
            print(f"WARNING: is_smiling() failed on frame: {err!r}")
            continue

        # Upscale the display frame to at least 1280x720 (never downscale) so
        # the window size is decoupled from what the camera captures. Keypoints
        # are scaled by the same factor to stay aligned with the face.
        orig_h, orig_w = frame.shape[:2]
        disp_w, disp_h = max(orig_w, DISPLAY_WIDTH), max(orig_h, DISPLAY_HEIGHT)
        if (disp_w, disp_h) != (orig_w, orig_h):
            frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            if kps is not None:
                sx, sy = disp_w / orig_w, disp_h / orig_h
                kps = kps.copy()
                kps[:, 0] *= sx
                kps[:, 1] *= sy

        if kps is not None:
            _draw_kps(frame, kps)

        # Faces count from the same detection result driving classification, so
        # the screen shows why the smile check was skipped when count != 1
        # (is_smiling returns False for both, but the reason distinguishes
        # "genuinely not smiling" from "couldn't check").
        if n_faces != 1:
            reason = "no face" if n_faces == 0 else "multiple faces, skipping"
            faces_line = f"Faces detected: {n_faces} — {reason}"
        else:
            faces_line = f"Faces detected: {n_faces}"

        # Low-confidence guard: a single face whose detection score is below the
        # floor is unreliable — withhold the update and keep showing the last
        # stable result (window, smoothed value and verdict all stay frozen).
        hold_active = (
            n_faces == 1
            and det_score is not None
            and det_score < DET_CONF_THRESHOLD
        )

        # Rolling average of the raw smile probability (frames without a usable
        # single face contribute no sample — the window holds what it has).
        if not hold_active and raw is not None:
            window.append(raw)
        smoothed = float(np.mean(window)) if window else None

        # No usable single face -> cannot check the smile: show NOT SMILING and
        # the reason line above.  Otherwise decide on the smoothed value with
        # hysteresis to avoid flicker around the 0.5 boundary.  A held frame
        # skips this entirely, preserving the previous verdict.
        if not hold_active:
            if raw is not None:
                verdict = _hysteresis_verdict(smoothed, verdict)
            else:
                verdict = False

        label, color, show_prob = _status_overlay(n_faces, verdict)
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.putText(frame, faces_line, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Probability lines only make sense with a single face to score; when
        # there are 0 or 2+ faces there is nothing to score, so skip them.
        if show_prob:
            raw_line = f"raw: {raw:.2f}" if raw is not None else "raw: --"
            smoothed_line = f"smoothed: {smoothed:.2f}" if smoothed is not None else "smoothed: --"
            cv2.putText(frame, raw_line, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, smoothed_line, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"{latency_ms:.1f} ms", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Held frame: yellow border + note make it explicit the displayed verdict
        # is the last stable one, not a judgement from this frame.
        if hold_active:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), HOLD_COLOR, 4)
            cv2.putText(frame, "HELD (low-confidence detection)", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, HOLD_COLOR, 2)

        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())