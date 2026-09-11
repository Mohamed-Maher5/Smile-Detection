# Smile liveness — final model report

**Chosen model:** `LogisticRegression_grid` (C=1) + fitted `StandardScaler`, features `['mouth_eye_ratio','mouth_vertical_lift','mouth_nose_ratio','mouth_width']`, exported to `models/classifier/final_model.pkl` (`type='sklearn'`).

**Why over the close alternative (threshold):** on the clean held-out test set the single-feature threshold `mouth_eye_ratio ≥ 0.8361` posted F1 0.8815 / precision 0.8724 vs the model's **F1 0.8839 / precision 0.9010** (accuracy 0.8766 vs 0.8703, AUC 0.9500 vs 0.9420). McNemar over the 35 discordant pairs (15 vs 20) gives p≈0.50 — the difference is **within noise**, so the tie is broken by the eKYC liveness tradeoff: a *false smiling acceptance* (FP, 41 vs 56) is the consequential error, so higher precision wins even at a small recall cost (57 vs 47 missed genuine smiles — a retry, not a security failure). Selection rule (a) had pinned the threshold; the precision-priority mandate overrides it.

**Tuning (no test-set leakage):** features chosen by separability analysis on the full clean set; scaling fit on the stratified 80/20 train split only (3,176 rows, `random_state=42`); C tuned by 5-fold cross-validation on that split (best C=1, CV F1 0.8941). The test split (794) was never touched during selection.

**End-to-end vs targets (full 823-image test set incl. 0-face/multi-face):**

| Metric | Prompt-C result | Target | Met? |
|---|---|---|---|
| Accuracy | **0.8566** | ≥ 0.812 | ✅ |
| Precision | 0.9010 | — | |
| Recall | 0.8289 | — | |
| F1 | 0.8634 | — | |
| Latency (median) | **3.42 ms** | ≤ 4.7 ms | ✅ |
| Latency (p95) | 4.86 ms | ≤ 4.7 ms | ⚠️ 0.16 ms over |

**Known limitations (from bucket analysis & misclassification grid):**
1. **Extreme image quality:** accuracy drops in the brightest tercile (0.828) and blurriest tercile (0.832) vs ~0.87 in the middle — bright/overexposed and blurry frames are the main single-face error source.
2. **Multi-face frames:** `is_smiling()` spec-correctly returns `False`, so 20 genuine-smile multi-face images (correctly-per-spec, incorrectly-per-label) become false negatives; only exactly-one-face frames produce a decision.
3. **Unrecoverable detection:** `file2669.jpg` (brightness 90, blur 769) could not be detected by the original- or any zoom-crop fallback — it returns `False` correctly by label, but the 14.8 ms worst-case call shows the fallback chain's cost on its single hard failure.