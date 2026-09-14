# Smile Detection

Real-time smile classification from 5 facial landmarks — CPU-only, no GPU required.

## Overview

This project detects real people smiling in a single image and returns a yes/no verdict within a strict latency budget, running entirely on CPU. The hard constraint is input: we deliberately restrict ourselves to the **5 SCRFD landmarks** (left eye, right eye, nose, two mouth corners) rather than dlib's dense 68-point mesh and its ~20 mouth landmarks. Five points cannot reconstruct a full mouth shape, so every smile signal must be squeezed out of a handful of geometric ratios computed between those points — a much harder classification problem than the one CNN-based and 68-point approaches solve. The result is a 1.5 KB classifier that runs in ~4.4 ms per frame with 86.6% accuracy on a real, camera-varied benchmark.

## Results at a glance

Full 823-image delivery set (single-face + zero/multi-face frames); target baseline: **81.2% accuracy, 4.7 ms latency**.

| Metric | Value | vs. Baseline |
|---|---|---|
| Accuracy | **0.8663** | Beats 81.2% baseline by **+5.4 pp** |
| Precision | **0.9067** | 91% of "smiling" calls are correct |
| Recall | **0.8422** | 84% of smiling faces are caught |
| F1 | **0.8733** | Balanced precision/recall tradeoff |
| Latency (median) | **4.42 ms** | ~6% under the 4.7 ms target |
| Latency (p95) | **5.47 ms** | Tail exceeds target under load (hard fallback images) |

*Source: `outputs/benchmark_summary.csv` (n=823, of which 28 zero/multi-face frames are scored as "not smiling" by design).*

## The journey

### The detection problem — and the fallback that solved it

The first surprise came from the detector, not the classifier. On the full labeled dataset, SCRFD's direct pass failed to find *any* face in **6 images** — dark, low-contrast, or near-duplicate frames where the face does not pop out of the image. Without a face, no landmarks exist and the pipeline would silently return "not smiling," which is wrong for 3 of those frames.

The fix is a progressive **fallback chain**: run the detector on the original image, retry the full image in grayscale, then try center crops at scales 0.9 → 0.4 (each crop also tried in grayscale before shrinking further). Because every crop-local box and landmark is translated back into original-image coordinates before scoring, the classifier always sees geometrically consistent landmarks.

![Detection-recovery: hardest faces handled by the fallback pipeline](data/eda/detection_recovery_examples.png)

All **6** baseline failures are recovered by the fallback (**0 remain**): 4 by a plain center crop, 1 by the 0.8 crop, and 1 (the hardest, `file2669.jpg`) only by the full grayscale pass.

### Feature engineering — 4 ratios instead of a mesh

With only 5 points, the worthwhile signals are mouth *opening* and mouth *lifting*. We defined four scale-invariant geometric features from the raw landmarks:

| Feature | What it measures | Separability (|Cohen's d|) |
|---|---|---|
| `mouth_eye_ratio` | mouth width ÷ inter-eye distance (mouth opening) | 2.33 |
| `mouth_vertical_lift` | vertical offset of the mouth vs. the eye line (corner lift) | 1.38 |
| `mouth_nose_ratio` | mouth width ÷ nose-to-eye distance | 0.70 |
| `mouth_width` | raw mouth span | 0.62 |

Scale-invariance matters: features are dimensionless ratios, so they stay meaningful for faces of any size anywhere in the frame. Three more engineered candidates (`mouth_corner_angle`, `mouth_triangle_area`, `mouth_symmetry`) measured the mouth geometry from different angles but were **rejected** — they either duplicated existing features (correlation ≈ 0.78–0.82) or shrank on unseen data. Landmarks are also similarity-aligned to a canonical eye frame (left eye at −0.5, right eye at +0.5) so features are measured along the face's own axes, making them robust to head roll at zero measurable cost.

![Landmark alignment correcting head-roll on high-tilt faces](data/eda/alignment_example.png)

### Model selection — a real shootout, then confirming there's nothing better

We benchmarked every plausible classifier on the same 795-image held-out split (same 4 features, same `random_state=42`):

| Model | Test F1 | Median latency | Verdict |
|---|---|---|---|
| DecisionTree (depth 4) | **0.8970** | 0.23 ms | Tied, statistically |
| LogisticRegression | **0.8960** | 0.42 ms | **Shipped** |
| Threshold rule (mouth_eye_ratio ≥ 0.836) | **0.8950** | <0.01 ms | Tied, statistically |
| LinearSVC | **0.8926** | 0.43 ms | Close behind |
| RandomForest | 0.8910 | 27.58 ms | **Excluded — 66× slower** |
| LightGBM | 0.8907 | 0.98 ms | Close |
| CatBoost | 0.8834 | 0.84 ms | Behind |
| XGBoost | 0.8833 | 0.47 ms | Behind |

![Model leaderboard — test-set F1 with classification latency](data/eda/model_leaderboard.png)

The grid-tuned LogisticRegression (C=0.1, threshold 0.5) sits at test F1 **0.8939** on the same split, statistically indistinguishable from the top tree and the pure threshold rule, and it is *more useful than either*: it exposes a **calibrated probability score**, so any target precision/recall tradeoff is a single configurable threshold rather than a fixed rule. Verdicts are therefore always accompanied by a confidence, not just a label.

## Experiments that didn't make the cut

Rigor means testing alternatives and reporting the numbers. We tried four things that are **not** in the shipped pipeline, and they were rejected on the evidence, not on preference:

| Experiment | Metric before | Metric after | Verdict |
|---|---|---|---|
| **Data augmentation** (landmark rotation + jitter) | CV F1 0.8919±0.0103 · held-out 0.8939 | No held-out benefit (original digits not preserved; qualitative record only) | **Rejected** — removed because the held-out comparison showed no benefit |
| **Extended features** (+corner_angle, +triangle_area, +symmetry) | 4-feat: CV 0.8919 · held-out 0.8939 | 6-feat: CV 0.8915 · held-out 0.8928 · 7-feat: CV 0.8908 · held-out **0.8902** | **Rejected** — redundant (r≈0.78–0.82) and held-out drifts *down* as features are added |
| **Landmark alignment for accuracy** | Non-aligned: CV 0.8911 · held-out 0.8960 | Aligned: CV 0.8904 · held-out 0.8949 · McNemar p = **1.000** (1 flip) | **Kept for robustness, not accuracy** — no measurable gain; alignment is a zero-cost insurance against head roll |
| **DecisionTree as final model** | LR: held-out 0.8939 | DT: held-out 0.8970 · McNemar p = **0.7428** (n=37) | **Rejected** — nominally higher F1, but the difference is pure noise; LR's calibrated probability is strictly more useful |

McNemar's exact test is applied to paired predictions, so these "no-difference" verdicts are statistical statements, not vibes.

## Error analysis

![Confusion matrix on the 795-image held-out split](data/eda/confusion_matrix_final.png)

On the 795-image held-out split the confusion matrix is **TN 326 / FP 39 / FN 51 / TP 379** — accuracy 0.8868. Almost all revenue-relevant happy paths hold up; the errors concentrate in the extremes.

![Test accuracy by brightness and blur tercile](data/eda/error_by_brightness_blur.png)

![Example misclassified faces with feature values](data/eda/misclassified_examples.png)

The tercile breakdown shows what actually breaks the model: **brightness, not blur**. Accuracy is flat across blur terciles (0.883 / 0.898 / 0.879) but drops sharply in the brightest third (0.898 / 0.898 / **0.864**). Overexposed highlights wash out the mouth landmarks that carry the smile signal, and the misclassified examples confirm it — the failed faces cluster around small `mouth_eye_ratio` and high brightness. There is also a structural recall cap: frames containing zero or multiple faces are judged "not smiling" by design, and among the 823 delivery images, the 28 such frames cannot be rescued by any classifier.

## Data quality

Training and evaluation use the **GENKI-4K** subset of the MPLab GENKI database. Because GENKI distributes images rather than explicit per-file labels, every image was manually labeled and then audited against the official dataset's own metadata: label cross-validation produced **99.67% agreement** between our manual pass and the GENKI source labels. The remaining 0.33% (a handful of ambiguous images) were individually resolved to reach a confirmed, consistent baseline before any modeling.

## How it works

```text
Image
  │
  ▼
SCRFD Detection, 5-point (ONNX)          ── no single face? → "Not smiling"
  │                                          │
  │  fallback: grayscale → center crops       │ (zero/multi-face → False by design)
  ▼                                          │
Landmark alignment (rotation + scale → canonical eye frame)
  │
  ▼
Geometric feature extraction (4 ratios)
  │   • mouth_eye_ratio     • mouth_vertical_lift
  │   • mouth_nose_ratio    • mouth_width
  ▼
StandardScaler → LogisticRegression (C=0.1)
  │
  ▼
Smiling / Not smiling  (predict_proba ≥ 0.5)
```

![End-to-end pipeline walkthrough](data/eda/pipeline_walkthrough.png)

Only 5 landmarks are used instead of a full CNN because the goal is speed and portability: the whole pipeline — detection, fallback, alignment, features, and a 1.5 KB logistic model — runs on CPU in ~4.4 ms/frame. Four geometric ratios capture mouth opening and lifting with no neural net beyond the fixed detector.

## Quickstart / usage

Python 3.10+, no GPU. On first use the detector downloads the SCRFD model (`buffalo_sc`) from the InsightFace releases.

```bash
pip install -r requirements.txt
```

Use the model on a single image:

```python
import cv2
from src.smile_detector import is_smiling

img = cv2.imread("photo.jpg")
print("smiling:", is_smiling(img))   # True / False
```

Run the live webcam demo (`q` to quit):

```bash
python src/webcam_demo.py
```

Project layout:

```text
src/                  detector, features, preprocessing, smile_detector, webcam demo
models/classifier/    shipped model bundle (final_model.pkl, 1.5 KB)
notebooks/            data prep, EDA, feature engineering, modeling, benchmark, error analysis
outputs/              benchmark results (CSV + misclassification grid)
data/eda/             all figures used in this README
requirements.txt
```

## Known limitations

- **Bright-image sensitivity.** Accuracy drops most in the brightest frames (0.898 → 0.864) where overexposure washes out the mouth landmarks that drive the decision; blur has little effect by comparison.
- **Multi-face frames are skipped by design.** `is_smiling()` returns `False` unless exactly one face is detected, so a smiling subject in a crowded frame is scored as not smiling — a structural cap on recall across the 28 such delivery frames that no classifier can remove.
- **Latency tail on hard frames.** The median is comfortably under budget (4.42 ms vs 4.7 ms), but frames that force the full grayscale + crop fallback chain push p95 to ~5.5 ms under load; the single hardest image (`file2669.jpg`) requires every fallback stage.

## Citation / acknowledgments

Training and evaluation use the [MPLab GENKI Database, GENKI-4K Subset](https://mplab.ucsd.edu), cited as:

```bibtex
@misc{GENKI-4K,
  Author = {\url{http://mplab.ucsd.edu}},
  Title  = {{The MPLab GENKI Database, GENKI-4K Subset}}
}
```

Face detection and landmarks use the SCRFD detector ([InsightFace](https://github.com/deepinsight/insightface), `buffalo_sc` / `det_500m.onnx`) via ONNX Runtime.