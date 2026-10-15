# DR Vision AI project audit and improvement record

## Initial audit

The repository already had a React 18/Vite frontend, FastAPI backend, SQLite prediction history, image upload and validation, generated heatmaps, PDF reports, health/dashboard/dataset/performance endpoints, and ResNet-50 + DenseNet-121 image feature extraction. Training and prediction shared the same crop, CLAHE, resize, RGB ordering, ImageNet normalization, feature concatenation, scaler, and PCA pipeline. Supplied train/validation/test feature caches all contained 3,072 features with matching pretrained, 224 px, CLAHE metadata.

Confirmed problems:

- `ml/train.py` and `ml/train_splits.py` evaluated the test set as part of ordinary training, making repeated candidate selection leak test information. Both also defaulted to writing a production-named model bundle.
- SVM validation results existed, but no separate SVM bundle was integrated. The experiment inspector used a machine-specific absolute path.
- `scripts/tune_svm.py` contained only null bytes and could not be compiled or run.
- The interface and PDF described classifier outputs as “confidence” or “probability,” including uncalibrated AdaBoost scores.
- Model-loading and prediction errors returned internal exception details, which could reveal filesystem paths.
- Prediction history did not save the model version; reports omitted it and did not safely escape uploaded filenames.
- Existing model-load failure messages included implementation detail intended for logs.

## Changes made

- Changed `ml.train` and `ml.train_splits` to train candidates without reading test images, write to candidate directories by default, and keep validation metrics separate. `ml.evaluate` performs one test run and refuses repeat evaluation.
- Added a validation-only RBF-SVM sweep using cached train/validation features. Scaler, PCA, classifier, and sigmoid calibration are fitted using training data; selection uses validation macro F1. Each candidate is recorded separately.
- Added a one-time independent test evaluator that does not fit model components. The selected `svm-v1` bundle and metrics are stored under `backend/models/experiment_svm_v1/`. The prior `backend/models/dr_model.joblib` remains intact for rollback.
- Updated training workflows so they read only training and validation images and default to candidate output directories. Added a CPU thread cap (`DR_TORCH_THREADS`) for efficient inference on CPU systems.
- Added `DR_MODEL_PATH` and `DR_METRICS_PATH` support for switching bundles without replacing the original production artifact.
- Added model-version persistence and reporting, safer user-facing errors, escaped report text, and explicit class-score descriptions. Updated prediction, history, dashboard, and performance screens to call these values class scores.
- Repaired `scripts/tune_svm.py` as a compatibility entry point for the validation-only workflow.
- Added `.env.example` and Windows PowerShell setup, training, evaluation, and startup guidance to the README.

## Model selection and evaluation

Selection used validation macro F1 only. The selected candidate is RBF-SVM with PCA 256, `C=1`, `gamma=scale`, balanced class weights, and sigmoid calibration using internal five-fold cross-validation on training data.

| Split/model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Previous AdaBoost validation | 76.23% | 60.31% | 58.67% | 59.17% |
| Selected SVM validation | 81.42% | 66.71% | 68.51% | 67.36% |
| Selected SVM independent test | 83.33% | 68.84% | 70.80% | 69.10% |

The selected SVM plus calibration fit took 1.93 seconds (excluding scaler/PCA/CAM fitting). Classifier-only inference on cached features was 0.20 ms/image on validation and 0.26 ms/image on test; these timings exclude CNN feature extraction and image decoding. Test log loss was 0.5883 and multiclass Brier score was 0.2697.

Test class metrics (precision / recall / F1; support):

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| No DR | 0.985 | 0.975 | 0.980 | 199 |
| Mild DR | 0.500 | 0.700 | 0.583 | 30 |
| Moderate DR | 0.772 | 0.701 | 0.735 | 87 |
| Severe DR | 0.455 | 0.588 | 0.513 | 17 |
| Proliferative DR | 0.731 | 0.576 | 0.644 | 33 |

Confusion matrix (rows = true class, columns = predicted class; order 0–4):

```text
[[194,  4,  1,  0,  0],
 [  2, 21,  4,  1,  2],
 [  1, 13, 61,  7,  5],
 [  0,  0,  7, 10,  0],
 [  0,  4,  6,  4, 19]]
```

This is one held-out split of 366 images, with only 17 severe cases. It does not establish external validity or clinical readiness. Scores and heatmaps are research outputs, not probabilities of disease or diagnoses. The app remains an educational/research prototype.

## Verification

- Python compile check: passed.
- Backend test suite: 14 passed. The default pytest temp directory was inaccessible, so the suite was rerun with a fresh temporary directory inside the workspace. One upstream Starlette/httpx deprecation warning remains.
- Candidate API integration: passed with a real held-out image; five scores summed to 100%, result/history stored `svm-v1`, and PDF output was valid.
- Frontend production build: passed. Vite reports the main JavaScript bundle is 691 kB and suggests code splitting; this is a performance advisory.
- Live app: backend health is `ok`, model is loaded as RBF-SVM, model-performance endpoint returns the selected test metrics, and Vite returns HTTP 200.

## Files changed or added

- `backend/scripts/train_svm_candidate.py`, `backend/scripts/evaluate_selected_test.py`: reproducible validation and one-time test workflows.
- `backend/models/experiment_svm_v1/`: separate SVM bundle, validation search results, and test metrics.
- `backend/ml/train.py`, `backend/ml/train_splits.py`, `backend/ml/evaluate.py`, `backend/ml/features.py`: candidate/test separation, one-time evaluation, and CPU inference thread control.
- `backend/app/config.py`, `backend/app/database.py`, `backend/app/schemas.py`, `backend/app/services/model_service.py`, `backend/app/services/dataset_service.py`, `backend/app/services/report.py`, `backend/app/routers/predictions.py`, `backend/app/routers/insights.py`: model selection, persisted model version, error handling, score semantics, reports, and metrics selection.
- `frontend/src/pages/Predict.jsx`, `frontend/src/pages/Results.jsx`, `frontend/src/pages/History.jsx`, `frontend/src/pages/Dashboard.jsx`, `frontend/src/pages/ModelPerformance.jsx`: score terminology and model information.
- `backend/scripts/tune_svm.py`, `backend/scripts/inspect_svm.py`: repaired the unusable all-null tuner entry point and removed the inspector's hard-coded local path.
- `.env.example`, `README.md`, `backend/tests/test_api.py`: configuration/setup documentation and regression assertions.
