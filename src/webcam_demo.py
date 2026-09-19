"""Live webcam demo: classify each frame with the shipped smile pipeline and
overlay result + latency plus the 5 face landmarks (kps) the model used.

This is the SINGLE live-camera entry point. It contains the two previously
separate behaviors behind one flag set:

  * DEFAULT — the GATED view: a green centered square auto-sizes to the detected
    face and the smile verdict is only produced while the face sits inside it.
    A user instructed to "put your face in the box" holds still, centered and at
    roughly constant scale — exactly the regime the 5-landmark geometric
    features were measured in (GENKI-4K portraits). The square is the only box
    drawn (no face bounding box): its edge mirrors the face's size.
  * `--no-square` — the classic whole-frame view: the smile check runs on any
    single face anywhere in the frame, and the detected face bounding box IS
    drawn.

Both views run the identical pipeline (detection -> align -> 4 features ->
LogisticRegression) and the same decision layer (see below). There is no code
duplication between them — they are two branches of one loop.

DECISION LAYER (identical for both views; the model and features are untouched):
  * The raw smile predict_proba is smoothed over a rolling 5-frame window and
    decided with hysteresis (bands T+0.05 / T-0.05 around the cut), so
    borderline expressions don't flicker.
  * Default cut T = the production operating point (0.5 for the sklearn bundle).
  * --threshold raises/lowers the cut; --debounce K switches to debounced
    switching (flip only after K consecutive frames agree — safe against
    single-frame spikes, but adds response latency); --raw removes EVERYTHING
    and labels the instantaneous probability each frame (A/B control).

SAFETY:
  * When there is no usable single face inside the gate the verdict is withheld
    (the screen says why).
  * When the single-face detection confidence drops below 0.6 the verdict is
    held: the last stable result stays on screen with a yellow border saying
    it is not from the current frame.

LOGGING (optional, off by default): `--log PATH` appends one CSV row per frame
with frame_idx, n_faces, det_score, in-square/gate state, raw & smoothed
probabilities, the verdict, latency and the bbox/kps arrays — every flush
per frame, so an interrupted run loses nothing.

Display is decoupled from the capture resolution: the window is resizable at
1280x720 and each frame is upscaled (never downscaled) to at least 1280x720
before drawing, so the on-screen size is guaranteed regardless of the
resolution the camera actually delivers.

Keys: 'q' quit, 'c' reset the smoothed/verdict state.
"""

import argparse
import csv
import signal
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from features import align_kps
from smile_detector import detect_and_score_detailed, _BUNDLE

# The default decision threshold follows the loaded bundle: for an sklearn
# bundle it is the production 0.5 probability cut; for a threshold bundle the
# rule decides on a single aligned feature directly (e.g. mouth_eye_ratio >=
# 0.8357). --threshold overrides it.
SMILE_THRESHOLD = (
    float(_BUNDLE["model"])
    if _BUNDLE["type"] == "threshold"
    else 0.5
)

KPS_TAGS = ["LE", "RE", "N", "LM", "RM"]
KPS_COLOR = (255, 255, 0)        # keypoint dots (cyan)
BBOX_COLOR = (255, 255, 200)     # face bbox (light cyan-white) — whole-frame view
BBOX_THICKNESS = 2
SQUARE_COLOR = (0, 255, 0)       # gate square (green)
SQUARE_FILL = (0, 255, 0)        # face inside -> faint green wash over the square
SQUARE_FILLED_ALPHA = 0.12
GATE_COLOR = (0, 165, 255)       # orange — waiting for the face to enter the square

# Stabilisation knobs (the underlying model/classification is not modified).
SMILE_HYSTERESIS = 0.05    # hold band: keep label inside [T-0.05, T+0.05]
WINDOW_SIZE = 5            # rolling average length for the smoothed decision
DET_CONF_THRESHOLD = 0.6   # hold result when the single-face det_score drops below this
HOLD_COLOR = (0, 255, 255)  # yellow (BGR) frame border for the hold indicator

# Requested capture resolution (webcams may not honour it exactly).
CAM_WIDTH, CAM_HEIGHT = 1280, 720

# Display is decoupled from the capture resolution (see module docstring).
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1280, 720
WINDOW_TITLE = "Smile Detection"

EXP_DIR = Path(__file__).resolve().parents[1] / "data" / "exper"
DEFAULT_LOG = EXP_DIR / "webcam_demo_log.csv"
LOG_HEADER = [
    "frame_idx", "timestamp_ms", "n_faces", "det_score",
    "face_in_square", "gate_open", "raw", "smoothed", "verdict",
    "latency_ms", "bbox", "kps",
]


def _hysteresis_verdict(smoothed: float | None, prev: bool,
                        threshold: float) -> bool:
    """Map the smoothed probability to a label with a hold band.

    Flipping to True only when ``smoothed > T + 0.05``, to False only when
    ``smoothed < T - 0.05``, and otherwise keeping the previous label — so
    borderline values no longer toggle on a frame-by-frame basis.
    """
    if smoothed is None:
        return prev
    if smoothed > threshold + SMILE_HYSTERESIS:
        return True
    if smoothed < threshold - SMILE_HYSTERESIS:
        return False
    return prev


def _bbox_in_square(bbox: np.ndarray, x1s: int, y1s: int, x2s: int, y2s: int,
                    strict: bool) -> bool:
    """True iff *bbox* (x1,y1,x2,y2) satisfies the centered-square rule."""
    if bbox is None or len(bbox) != 4:
        return False
    bx1, by1, bx2, by2 = (float(v) for v in bbox)
    if not all(np.isfinite(v) for v in (bx1, by1, bx2, by2)):
        return False
    cx = (bx1 + bx2) / 2.0
    cy = (by1 + by2) / 2.0
    if strict:
        return bx1 >= x1s and by1 >= y1s and bx2 <= x2s and by2 <= y2s
    return (x1s <= cx <= x2s) and (y1s <= cy <= y2s)


def _draw_square(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                 face_in: bool) -> None:
    """Draw the centered gate square; faint green fill when a face is inside."""
    if face_in:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), SQUARE_FILL, -1)
        cv2.addWeighted(overlay, SQUARE_FILLED_ALPHA, frame, 1 - SQUARE_FILLED_ALPHA,
                        0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), SQUARE_COLOR, 2, cv2.LINE_AA)


def _draw_bbox(frame: np.ndarray, bbox: np.ndarray) -> None:
    """Overlay the detected face bounding box as a thin rectangle."""
    if bbox is None or len(bbox) != 4:
        return
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
        return
    pt1 = (int(round(x1)), int(round(y1)))
    pt2 = (int(round(x2)), int(round(y2)))
    if not (0 <= pt1[0] < frame.shape[1] and 0 <= pt1[1] < frame.shape[0]):
        return
    cv2.rectangle(frame, pt1, pt2, BBOX_COLOR, BBOX_THICKNESS, cv2.LINE_AA)


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


def _draw_aligned(frame: np.ndarray, aligned: np.ndarray) -> None:
    """Overlay the aligned landmark coordinates (canonical eye frame) top-right."""
    if aligned is None:
        return
    h, w = frame.shape[:2]
    lines = [f"{tag}: " + ",".join(f"{c:+.3f}" for c in pt)
             for tag, pt in zip(KPS_TAGS, aligned)]
    y0 = 40
    line_h = 22
    panel_h = line_h * len(lines) + 10
    x0 = w - 240
    cv2.rectangle(frame, (x0 - 6, y0 - 34), (w - 4, y0 + panel_h - 8), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x0, y0 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 200), 1, cv2.LINE_AA)


def _draw_status(frame: np.ndarray, text: str, color: tuple[int, int, int],
                 y: int = 40) -> None:
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)


def _draw_face_label(frame: np.ndarray, bbox: np.ndarray, label: str,
                     color: tuple[int, int, int], scale: float = 1.0) -> None:
    """Draw *label* just above the top edge of *bbox*, outlined for readability."""
    if bbox is None or len(bbox) != 4:
        return
    x1 = float(bbox[0])
    y1 = float(bbox[1])
    if not (np.isfinite(x1) and np.isfinite(y1)):
        return
    thickness = 3 if scale >= 1.0 else 2
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    tx = int(round(max(1, min(x1, frame.shape[1] - tw - 1))))
    ty = int(round(max(th + 4, y1 - 8)))
    cv2.rectangle(frame, (tx - 3, ty - th - 3),
                  (tx + tw + 3, ty + 3), (0, 0, 0), -1)
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Live smile demo (single file): centered square gate by "
                    "default; --no-square for the classic whole-frame view.")
    ap.add_argument("--no-square", action="store_true",
                    help="classic view: check any single face anywhere in the frame "
                         "and draw the face bounding box (no centered gate)")
    ap.add_argument("--square-frac", type=float, default=0.3,
                    help="fallback square side as a fraction of frame width when no "
                         "face is detected (default 0.3). Ignored while a face is "
                         "present: the square auto-sizes to the detected face.")
    ap.add_argument("--fixed", action="store_true",
                    help="keep the square at --square-frac regardless of the face size")
    ap.add_argument("--strict", action="store_true",
                    help="require the WHOLE face bbox inside the square, not just its center")
    ap.add_argument("--raw", action="store_true",
                    help="remove stabilization: verdict = the instantaneous prediction "
                         "thresholded per frame (A/B control)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the decision threshold (default: the bundle's "
                         "production operating point, e.g. 0.5)")
    ap.add_argument("--debounce", type=int, default=0,
                    help="flip only after this many consecutive frames agree "
                         "(0 = hysteresis, the default)")
    ap.add_argument("--log", nargs="?", const=str(DEFAULT_LOG), default=None,
                    help="append per-frame rows to <path> for analysis "
                         "(default path: %s)" % DEFAULT_LOG)
    args = ap.parse_args()
    if not 0.1 <= args.square_frac <= 0.9:
        ap.error("--square-frac must be in [0.1, 0.9]")
    if args.debounce < 0:
        ap.error("--debounce must be >= 0")
    threshold = SMILE_THRESHOLD if args.threshold is None else args.threshold

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
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"After set : W={cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f} "
          f"H={cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Capture resolution: requested {CAM_WIDTH}x{CAM_HEIGHT}, got {got_w}x{got_h}")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    log = writer = None
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        new_file = not Path(args.log).exists()
        log = open(args.log, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(log, fieldnames=LOG_HEADER)
        if new_file:
            writer.writeheader()
        log.flush()

    window = deque(maxlen=WINDOW_SIZE)  # rolling raw probabilities
    verdict = False                     # previous displayed label (NOT SMILING)
    up_count = 0                        # debounce: consecutive frames >= threshold
    down_count = 0                      # debounce: consecutive frames < threshold
    frame_idx = 0

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    print("Press 'q' to quit, 'c' to reset the smoothed verdict.")
    while not stop:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("WARNING: failed to read frame, skipping")
            continue

        # Display is decoupled from capture: upscale to at least 1280x720. All
        # detection/overlays below therefore run in DISPLAY coordinates.
        orig_h, orig_w = frame.shape[:2]
        disp_w, disp_h = max(orig_w, DISPLAY_WIDTH), max(orig_h, DISPLAY_HEIGHT)
        if (disp_w, disp_h) != (orig_w, orig_h):
            frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        # Square fallback geometry (recomputed from the face below when visible).
        side = int(round(args.square_frac * disp_w))
        x1s, y1s = (disp_w - side) // 2, (disp_h - side) // 2
        x2s, y2s = x1s + side, y1s + side

        try:
            t0 = time.perf_counter()
            _, kps, raw, n_faces, det_score, bbox = detect_and_score_detailed(frame)
            latency_ms = (time.perf_counter() - t0) * 1e3
        except Exception as err:
            print(f"WARNING: pipeline failed on frame: {err!r}")
            continue

        aligned = align_kps(kps) if kps is not None else None

        # Auto-size the centered square to the face so the gate "just fits" it.
        if not args.no_square and not args.fixed and bbox is not None and len(bbox) == 4:
            bx1, by1, bx2, by2 = (float(v) for v in bbox)
            if all(np.isfinite(v) for v in (bx1, by1, bx2, by2)) and (bx2 - bx1) > 0:
                side = int(round(max(bx2 - bx1, by2 - by1)))
                x1s, y1s = (disp_w - side) // 2, (disp_h - side) // 2
                x2s, y2s = x1s + side, y1s + side

        # Gate: in the square view the verdict requires a single face whose
        # bbox satisfies the square rule; in the classic view any single face
        # (anywhere) qualifies — one code path, one branch.
        if args.no_square:
            in_square = n_faces == 1
        else:
            in_square = _bbox_in_square(bbox, x1s, y1s, x2s, y2s, args.strict)
        gate_open = (n_faces == 1 and in_square)
        hold_active = (not args.raw and gate_open and det_score is not None
                       and det_score < DET_CONF_THRESHOLD)

        # Decision layer — identical regardless of view.
        if args.raw:
            if gate_open and raw is not None:
                verdict = bool(raw >= threshold)
            elif gate_open and raw is None:
                verdict = False
            window.clear()
            smoothed = raw
        else:
            if not hold_active and gate_open and raw is not None:
                window.append(raw)
            smoothed = float(np.mean(window)) if window else None

            # Fresh decision state every time the gate re-opens, so one
            # attempt's history never leaks into the next.
            if not gate_open:
                up_count = 0
                down_count = 0

            if args.debounce > 0 and gate_open and not hold_active and raw is not None:
                if smoothed is not None and smoothed >= threshold:
                    up_count += 1
                    down_count = 0
                elif smoothed is not None:
                    down_count += 1
                    up_count = 0
                if up_count >= args.debounce:
                    verdict = True
                elif down_count >= args.debounce:
                    verdict = False
            elif gate_open and raw is not None:
                verdict = _hysteresis_verdict(smoothed, verdict, threshold)
            elif gate_open and raw is None:
                verdict = False

        if writer is not None:
            writer.writerow({
                "frame_idx": frame_idx, "timestamp_ms": f"{time.time() * 1e3:.1f}",
                "n_faces": n_faces, "det_score": f"{det_score:.4f}" if det_score is not None else "",
                "face_in_square": int(in_square), "gate_open": int(gate_open),
                "raw": f"{raw:.4f}" if raw is not None else "",
                "smoothed": f"{smoothed:.4f}" if smoothed is not None else "",
                "verdict": int(verdict),
                "latency_ms": f"{latency_ms:.2f}",
                "bbox": "" if bbox is None else np.array2string(np.asarray(bbox, dtype=float), precision=1),
                "kps": "" if kps is None else np.array2string(np.asarray(kps, dtype=float), precision=1),
            })
            log.flush()  # interrupted run (Ctrl-C / crash) never loses the tail
            frame_idx += 1

        # Overlays — always landmarks + aligned panel; the box differs by view.
        if not args.no_square:
            _draw_square(frame, x1s, y1s, x2s, y2s, gate_open)
        elif bbox is not None:
            _draw_bbox(frame, bbox)
        if kps is not None:
            _draw_kps(frame, kps)
        _draw_aligned(frame, aligned)

        # Status text and probability lines. In both views the smile verdict
        # (and its probabilities) only make sense while gate_open — exactly one
        # face inside the gate — so they share this one branch.
        if not gate_open:
            if args.no_square:
                reason = ("NO FACE" if n_faces == 0
                          else "MULTIPLE FACES" if n_faces >= 2
                          else "NO FACE")
                _draw_status(frame, reason, (128, 128, 128))
                detail = (f"Faces detected: {n_faces} — "
                          + ("no face" if n_faces == 0
                             else "multiple faces, skipping"))
                cv2.putText(frame, detail, (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
            else:
                reason = ("NO FACE" if n_faces == 0
                          else "MULTIPLE FACES" if n_faces >= 2
                          else "STEP INTO THE SQUARE")
                _draw_status(frame, reason, GATE_COLOR)
                cv2.putText(frame, f"Faces detected: {n_faces}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            label = "SMILING" if verdict else "NOT SMILING"
            color = (0, 255, 0) if verdict else (0, 0, 255)
            if args.no_square and bbox is not None:
                _draw_face_label(frame, bbox, label, color, scale=1.0)
            else:
                _draw_status(frame, label, color)
            if raw is not None:
                cv2.putText(frame, f"raw: {raw:.2f}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if args.raw:
                    cv2.putText(frame, "MODE=RAW (no smoothing)", (20, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, f"smoothed: {smoothed:.2f}" if smoothed is not None
                                else "smoothed: --", (20, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if args.debounce > 0:
                        cv2.putText(frame, f"debounce: {up_count}/{args.debounce} "
                                    f"(down {down_count})", (20, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"{latency_ms:.1f} ms", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        size_src = f"fixed {args.square_frac:.2f} frame" if args.fixed else "auto = face size"
        if args.raw:
            mode_tag = "MODE=RAW"
        elif args.debounce > 0:
            mode_tag = f"MODE=DEBOUNCE K={args.debounce}"
        else:
            mode_tag = "MODE=HYST"
        if args.no_square:
            mode_tag += " WHOLE-FRAME"
        cv2.putText(frame, f"square = {side}px [{size_src}] "
                    f"{'strict' if args.strict else 'center'} | {mode_tag}",
                    (20, disp_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, SQUARE_COLOR, 2)

        if hold_active:
            cv2.rectangle(frame, (0, 0), (disp_w - 1, disp_h - 1), HOLD_COLOR, 4)
            cv2.putText(frame, "HELD (low-confidence detection)", (20, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, HOLD_COLOR, 2)

        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            window.clear()
            verdict = False
            print("reset smoothed verdict")

    if log is not None:
        log.close()
    cap.release()
    cv2.destroyAllWindows()
    if args.log:
        print(f"Frame log appended to {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())