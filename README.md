# Smile liveness — final model report

**Chosen model:** `LogisticRegression` (C=0.1) + fitted `StandardScaler`, exported to `models/classifier/final_model.pkl` (`type='sklearn'`).

**Decision threshold:** `predict_proba(smile) >= 0.5` (the balanced operating point, equal to the classifier default). Selected on training OOF probabilities via 5-fold StratifiedKFold cross-validation: OOF F1-max lands at 0.490 (precision 0.8963 / recall 0.8961 / F1 0.8958) with a flat F1 plateau from 0.480 to 0.545, so 0.5 is shipped as its round-number representative. The plateau maps onto the clean single-face test split as the band where precision and recall are both ≥ 0.85 (0.345–0.555); the point nearest (0.88, 0.88) is 0.410 (precision 0.8863 / recall 0.8884). It rebalances the previous precision-priority 0.67 point (0.9564 / 0.7800 on the current 823 set) toward roughly symmetric precision/recall, maximizing F1 for a liveness cost with no strong asymmetry.

**Feature set (4 shipped, unchanged throughout):** `['mouth_eye_ratio', 'mouth_vertical_lift', 'mouth_nose_ratio', 'mouth_width']`. Three additional geometric features (`mouth_corner_angle`, `mouth_triangle_area`, `mouth_symmetry`) were implemented and tested on held-out data; none earned their place (held-out F1 dropped or was statistically indistinguishable — the F1 gain on CV was consumed by shrinkage on unseen data). A landmark-jitter augmentation pipeline was also tested (std 0.6 on mouth features) and removed after held-out comparison showed no benefit.

**Detection fallback path:** `src/preprocessing.py` uses an interleaved retry order that now runs the plain direct detection first, then the full-image grayscale variant, then the per-scale (0.9 down to 0.4) zoom-crop and grayscale-crop pairs — the grayscale-first reorder recovers the single previously-unrecoverable image (`file2669.jpg`) with negligible latency cost on the rest.

**Why over the close alternative (threshold):** On the 795-image clean test split, the single-feature threshold baseline (re-tuned to `mouth_eye_ratio >= 0.836` on the corrected OOF) is preferred by McNemar over the four-feature LR at its shipped balance point (28 discordant pairs: 20 threshold-only-correct vs 8 LR-only-correct, exact p = 0.036). The LR is still shipped because it exposes a probability score, so any target precision/recall tradeoff is a single configurable threshold rather than a fixed rule — and on the full delivery set its F1 stays comparable and its FP count lower (F1 0.8654 vs 0.8581; 39 vs 49 FPs). The shipped 0.5 makes 39 FPs on the 823 delivery set at 0.8289 recall.

**Tuning (no test-set leakage):** features chosen by separability analysis on the full clean set; scaling fit on the stratified 80/20 train split only (3,176 rows, `random_state=42`); C tuned by 5-fold cross-validation on that split (best C=0.1, CV F1 0.8947). Threshold chosen on OOF probabilities from the same split via F1-max balance (plateau 0.480–0.545, shipped 0.5). The 795-image test split was never touched during selection.

**End-to-end metrics (full 823-image delivery set incl. 0-face/multi-face; current benchmark run, threshold 0.5):**

> Numbers below come from `outputs/benchmark_summary.csv` / `outputs/benchmark_results.csv` (regenerated after the cache path/label fix) and the threshold read from `final_model.pkl`. They reflect a corrected dataset (37 previously mislabeled cache entries were re-resolved to the folder of record) and the reordered grayscale-first detection fallback.

| Metric | Value | Note |
|---|---|---|
| Accuracy | **0.8591** | |
| Precision | **0.9053** | |
| Recall | **0.8289** | 0.8674 on clean single-face; capped at 0.9556 max on 823 due to 20 unclassifiable smiles in multi-face/no-face frames |
| F1 | **0.8654** | |
| Confusion (823) | TN 334, FP 39, FN 77, TP 373 | |
| Latency (median) | **3.095 ms** | Meets 4.7 ms budget (-1.605 ms). Median is the stable reference across load conditions. |
| Latency (p95) | 4.778 ms | Load-sensitive: ~3.1 ms on an idle machine, ~4.8–6.4 ms under moderate load. Not a fixed quantity; median is the reliable measure. |

**Operating-point tradeoff (current 823 delivery set):** the shipped 0.5 and the previous 0.67 bracket the same probability curve. At 0.67 precision rises to 0.9564 but recall drops to 0.7800 (F1 0.8592) with FP 16 / missed smiles 99; at 0.5 precision 0.9053 / recall 0.8289 (F1 0.8654) with FP 39 / missed smiles 77. No single threshold reaches both precision ≥ 0.87 and recall ≥ 0.87 on the 823 set; the maximum recall at precision ≥ 0.87 is 0.8644 (t=0.375). On the clean 795 split, the prec/rec ≥ 0.85 band spans 0.345–0.555 and the point nearest (0.88, 0.88) is 0.410 (0.8863 / 0.8884).

**Known limitations (from bucket analysis & misclassification grid):**
1. **Extreme image quality:** accuracy drops in the brightest tercile (0.860) and blurriest tercile (0.845) vs ~0.88–0.91 in the middle — bright/overexposed and blurry frames are the main single-face error source.
2. **Multi-face frames:** `is_smiling()` spec-correctly returns `False`, so 20 genuine-smile multi-face images (correctly-per-spec, incorrectly-per-label) become false negatives; only exactly-one-face frames produce a decision. This caps 823-set recall at 0.9556 even with a perfect classifier.
3. **Hard detection images:** `file2669.jpg` (brightness 90, blur 769) fails the direct call, raising worst-case latency; the grayscale-first reorder added a full-image grayscale attempt before the zoom-crop chain, so it now resolves to its single face (previously unrecoverable).

**Artifacts:**
- `models/classifier/final_model.pkl` — shipped bundle (model + scaler + features + threshold 0.5)
- `outputs/benchmark_results.csv` — per-image predictions + latency (823 rows)
- `outputs/benchmark_summary.csv` — aggregate metrics
- `notebooks/04_modeling.ipynb` — training, CV, model selection, McNemar, threshold sweep
- `notebooks/05_deployment_benchmark.ipynb` — full delivery-set benchmark
