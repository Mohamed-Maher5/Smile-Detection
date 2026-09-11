"""Live webcam demo: classify each frame with is_smiling() and overlay result + latency."""

import sys
import time

import cv2

from smile_detector import is_smiling


def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam (cv2.VideoCapture(0))")
        return 1

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("WARNING: failed to read frame, skipping")
            continue

        try:
            t0 = time.perf_counter()
            smiling = is_smiling(frame)
            latency_ms = (time.perf_counter() - t0) * 1e3
        except Exception as err:
            print(f"WARNING: is_smiling() failed on frame: {err!r}")
            continue

        smiling_str = "SMILING" if smiling else "NOT SMILING"
        color = (0, 255, 0) if smiling else (0, 0, 255)
        cv2.putText(frame, smiling_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(
            frame,
            f"{latency_ms:.1f} ms",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("is_smiling() demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())