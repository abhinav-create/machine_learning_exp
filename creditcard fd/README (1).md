# Credit Card Fraud Detection

A binary classification project to flag fraudulent credit card transactions in a highly imbalanced dataset, with an evaluation methodology designed to avoid the two most common failure modes in fraud modeling: misleading metrics and data leakage.

## Dataset

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) — 284,807 anonymized European card transactions from September 2013, of which 492 (0.17%) are fraud. Features `V1`–`V28` are PCA components of the original transaction data; `Time` and `Amount` are the only two raw, interpretable fields.

## The Core Challenge: Severe Class Imbalance

At 0.17% fraud, a model that predicts "not fraud" every single time scores 99.83% accuracy while catching zero fraud. Every design choice below follows from taking that seriously — from how the metrics are chosen, to how the data is sampled, to how (and where) resampling is applied.

## Methodology

### 1. Exploratory analysis
Distribution of `Amount` by class, and a correlation heatmap across the PCA components, to check for obviously separable features and confirm there's no missing data.

### 2. Sampling strategy: shrink the majority class, not the minority

Running the full 284,807-row dataset through multiple models × 5-fold CV × SMOTE is slow, but blindly taking a 10% random sample of the whole dataset is worse — it would have cut the already-rare 492 fraud cases down to roughly 49, too few to get a stable read on precision/recall at any threshold.

Instead, **every fraud case is kept**, and only the legitimate class is downsampled (to 30,000 rows). This keeps runtime manageable while preserving a large enough fraud sample to trust the resulting metrics.

### 3. Preprocessing
- `Amount` is standardized with `StandardScaler` into `Amount_scaled`; the raw `Amount` and `Time` columns are then dropped, so the feature set is fully numeric and on a comparable scale (important for Logistic Regression, which is sensitive to feature magnitude in a way tree models aren't).
- Duplicate rows are checked for and dropped (275 found) — this dataset is known to contain exact duplicates, and leaving them in risks the same transaction appearing in both the train and test split.
- Final working dataset: **30,217 rows, 473 fraud (1.57%)**.

### 4. Train/test split
An 80/20 **stratified** split, preserving the fraud ratio in both sets: 24,173 train rows (378 fraud) / 6,044 test rows (95 fraud). `X_test` is set aside here and not touched again until the final evaluation step.

### 5. Why accuracy is the wrong metric
A first-pass Logistic Regression model is used to demonstrate this directly: high accuracy, poor recall on the fraud class. From here on, model comparison relies on **ROC-AUC and Average Precision (AP)** — AP in particular, since it's far more informative than ROC-AUC when the positive class is rare.

### 6. Model comparison via out-of-fold (OOF) cross-validation
Rather than comparing models by scanning thresholds against the test set (which would leak test-set information into model selection), Random Forest and XGBoost are compared using **5-fold stratified cross-validation on the training set only**, collecting out-of-fold predictions. This gives an unbiased read on which model and which resampling strategy generalizes best, entirely without touching `X_test`.

| Model | ROC-AUC (OOF) | AP (OOF) |
|---|---|---|
| Random Forest | 0.9685 | 0.8743 |
| XGBoost | 0.9804 | **0.8935** |
| Random Forest + SMOTE | 0.9822 | 0.8936 |
| XGBoost + SMOTE | 0.9785 | 0.8917 |

### 7. Resampling done correctly: SMOTE inside the fold, never outside it
SMOTE is applied **only to each fold's training portion**, after the fold split — never to the validation fold, and never before cross-validation begins. Oversampling before splitting would let synthetic points derived from validation-fold data leak into training, inflating the apparent performance.

### 8. Threshold selection, still without touching the test set
Using the OOF probabilities from step 6, a full precision/recall/F1 table is built across 99 thresholds (0.01–0.99). The threshold is chosen from this table — filtered to recall ≥ 0.70 and sorted by precision — based on the trade-off between catching fraud and minimizing false declines, **before ever looking at `X_test`**.

For XGBoost + SMOTE, threshold `0.95` gives OOF precision 0.960 / recall 0.833 / F1 0.892 — a strong balance of the two.

### 9. Final model and one held-out evaluation
A **single, fresh** XGBoost model is fit on SMOTE applied to the *entire* training set (not one of the five CV fold-models, which only ever existed to generate OOF predictions). Because SMOTE already rebalances the classes, no additional class-weighting is applied to this model — combining both would double-correct for imbalance and typically hurts precision.

This one model, at the threshold chosen in step 8, is evaluated **once** against the previously untouched `X_test`.

## Final Results (held-out test set)

| | Precision | Recall | F1 |
|---|---|---|---|
| Fraud (class 1) | 0.963 | 0.832 | 0.893 |

Confusion matrix (6,044 test transactions, 95 fraud):

| | Predicted legit | Predicted fraud |
|---|---|---|
| **Actual legit** | 5,946 | 3 |
| **Actual fraud** | 16 | 79 |

3 false positives out of 5,949 legitimate transactions (0.05% false-positive rate), catching 79 of 95 fraud cases (83%).

These numbers land within a point or two of the OOF estimate from step 8 (0.960 precision / 0.833 recall) — the cross-validation process predicted almost exactly how the model would perform on unseen data, which is the real point of doing it this way rather than just tuning against the test set directly.

## Design decisions worth calling out

- **Downsample the majority, not the minority.** Preserves every real fraud example instead of discarding most of them for the sake of runtime.
- **Model and threshold selection happen on OOF predictions, never on the test set.** The test set is touched exactly once, at the very end.
- **SMOTE lives inside the CV fold boundary.** No synthetic sample is ever generated from data that includes points in that fold's validation split.
- **Weight OR resample, never both.** The final model skips class-weighting because SMOTE already rebalanced its training data.

## Known limitations

- The `StandardScaler` for `Amount` is fit before the train/test split, on the full dataset rather than `X_train` alone. Given it's a single feature's mean/std over ~30K rows, the practical leakage is minimal, but it's not textbook-correct.
- One of the earlier exploratory models passes `class_weight="balanced"` to `XGBClassifier`, which isn't a parameter XGBoost's sklearn API recognizes (it's silently ignored — XGBoost uses `scale_pos_weight` instead). This model is used only for early illustrative comparison, not for the final result.
- The train/test split is random rather than time-based. In production, a fraud model is always trained on the past and scored on the future; a chronological split would be a more realistic test of generalization than a random one.
- Hyperparameters (`n_estimators=300`, etc.) were fixed rather than tuned via grid/random search.
- The dataset is a single anonymized two-day snapshot from one issuer; performance may not transfer directly to other populations or time periods.

## Repo contents

- `credit_card_fraud_detection.ipynb` — full analysis notebook (EDA → baseline models → CV model comparison → SMOTE comparison → final model and evaluation)
- `README.md` — this file

## Running it

1. Download `creditcard.csv` from the Kaggle link above and place it in the working directory.
2. Install dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `matplotlib`, `seaborn`.
3. Run the notebook top to bottom. On the sampled dataset (~30K rows) this completes in a few minutes on a standard CPU runtime (e.g. Google Colab, no GPU required).
