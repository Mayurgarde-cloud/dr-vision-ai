# DR Vision AI — How It Works

Educational full-stack app that grades a retinal fundus photo into 5 diabetic-retinopathy (DR)
severity classes: 0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative.

> Not a medical device. Outputs are research `class scores`, never a diagnosis
> (`README.md`, `backend/PROJECT_AUDIT.md`).

Reference paper: Mustafa et al., *Multi-Stream Deep Neural Network for Diabetic Retinopathy
Severity Classification Under a Boosting Framework*, IEEE Access 2022
(`frontend/src/pages/About.jsx`, `README.md`). The app copies its core idea:
frozen ResNet-50 + DenseNet-121 features → PCA → boosted ensemble.

## 1. Big picture

```
fundus photo
  -> crop black margins -> CLAHE contrast -> resize 224x224        (ml/preprocess.py)
  -> ResNet-50 (2048-d) + DenseNet-121 (1024-d) -> concat 3072-d   (ml/features.py)
  -> StandardScaler -> PCA (128/256-d) -> classifier -> grade 0-4  (ml/bundle.py)
  -> + CAM heat-map from ResNet layer-4 + PDF report + history
```

Two backbones are complementary (residual vs dense connectivity); PCA separates
inter/intra-class variation and shrinks the vector; the classifier learns grade
boundaries. Production bundle today: RBF-SVM (`svm-v1`), calibrated, selected on
validation macro-F1 (`backend/models/experiment_svm_v1/`).

## 2. User flow (what happens on upload)

1. `frontend/src/pages/Predict.jsx`: drag-and-drop JPG/PNG <= 5 MB, >= 64 px/side.
2. `POST /api/predict` (`backend/app/routers/predictions.py`): validates file,
   `ModelService.predict()` (`backend/app/services/model_service.py`) runs:
   - trained path: `prepare_image` -> `extractor.forward_maps` -> `fuse` ->
     `predict_proba` (scaler->PCA->`predict_proba`, checks 3072-d, finite, sums to 1)
     -> `cam_from_map` heat-map -> resized overlay.
   - demo path (no model / no torch): `demo_model.py` statistics heuristic,
     tagged `mode: demo`, no diagnostic value.
3. Result saved to SQLite with model version, images to `backend/storage/`;
   `frontend/src/pages/Results.jsx` shows class, per-class scores, original /
   heat-map / overlay; PDF via `GET /api/predictions/{id}/report`
   (`backend/app/services/report.py`).
4. Other pages: Dashboard, Dataset, Model Performance (accuracy, macro
   P/R/F1, confusion matrix), History, About (`frontend/src/pages/`,
   `backend/app/routers/insights.py`).

Backend entry `backend/app/main.py` loads the bundle once at startup, serves
`/api/*`, `/files/*`, and the built React app from `frontend/dist`.

## 3. Training and evaluation (the honest protocol)

- Layouts: class folders `backend/data/dataset/0_No_DR ... 4_...` or CSV
  (`id_code,diagnosis`) + image dir, e.g. APTOS 2019 (`backend/ml/dataset.py`).
- `python -m ml.train` (`backend/ml/train.py`): dedups exact-image copies by
  SHA-256, rejects label conflicts, stratified 70/15/15 split via
  `StratifiedGroupKFold` so copies never cross splits; scaler/PCA/classifier fit
  on train only; writes candidate bundle + validation metrics only.
  Test images are never opened here.
- `python -m ml.evaluate --model-dir <candidate>` (`backend/ml/evaluate.py`):
  one-time held-out test, refuses re-runs, checks content-disjointness, writes
  `test_metrics.json` (accuracy, macro P/R/F1, per-class, confusion matrix,
  log-loss, Brier, timings).
- Fast path on cached features: `backend/scripts/train_svm_candidate.py`
  (validation sweep) + `evaluate_selected_test.py` (one-time test).
- Current real-data split: 2930/366/366 (APTOS-2019 distribution).

## 4. Current numbers (real data, single split)

`backend/models/experiment_svm_v1/test_metrics.json` (promoted to production):

- Test accuracy 83.33%, macro-P 68.8%, macro-R 70.8%, macro-F1 69.1%,
  log-loss 0.588, Brier 0.270.
- Per-class F1: No DR 0.98 (n=199), Mild 0.58 (30), Moderate 0.73 (87),
  Severe 0.51 (17), PDR 0.64 (33). Accuracy is carried by class 0.
- Paper 5-class for context: EyePACS 85.46%, Messidor-2 86.78%.

## 6. Procedure (step-by-step)

Prerequisites: Python 3.10+, Node.js 18+ (`README.md:42`).

```powershell
# 0. Fresh setup (project root)
py -3.11 -m venv backend\.venv
.\backend\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt   # CPU torch: install torch CPU build first, see README troubleshooting
npm.cmd --prefix frontend install
```

```powershell
# 1. Prepare real data (pick A or B)
# A. Class folders: backend/data/dataset/0_No_DR/*.jpg ... 4_Proliferative_DR/*.jpg
# B. CSV labels (APTOS style: id_code,diagnosis):
cd backend
python -m ml.train --csv 'E:\Diabetic Dataset\train_1.csv' --image-dir 'E:\Diabetic Dataset\train_images\train_images' --dataset-name "APTOS 2019" --out-dir models\my_candidate
```

```powershell
# 2. Train a candidate (validation only, test never opened)
cd backend
$env:DR_TORCH_THREADS='8'
python -m ml.train --out-dir models\my_candidate --batch-size 16 --pca-components 128 --ada-estimators 10
# optional tuning: --augment 2 --pca-components 256 --ada-estimators 20
# fast SVM path on cached features (no re-extraction):
python -m scripts.train_svm_candidate --cache-dir models\experiment_aug1 --splits models\splits.json --out-dir models\experiment_svm_v1
```

```powershell
# 3. Evaluate once on held-out test (single use, refuses repeats)
python -m ml.evaluate --model-dir models\my_candidate
# check: models\my_candidate\test_metrics.json (accuracy, macro P/R/F1, confusion matrix, log-loss, Brier)
```

```powershell
# 4. Promote winner to production (backup first)
Copy-Item models\dr_model.joblib models\backup_dr_model.joblib -Force
Copy-Item models\my_candidate\dr_model.joblib models\dr_model.joblib -Force
Copy-Item models\my_candidate\test_metrics.json models\metrics.json -Force
```

```powershell
# 5. Serve (backend loads bundle once at startup: backend/app/main.py)
$env:DR_TORCH_THREADS='8'
python -m uvicorn app.main:app --port 8000        # API docs: http://localhost:8000/docs, health: /api/health
# frontend dev (separate terminal): npm.cmd --prefix frontend run dev   # http://localhost:5173
# OR single-process: npm.cmd --prefix frontend run build, then open http://localhost:8000
```

```powershell
# 6. Predict + verify (JPG/PNG, <= 5 MB, >= 64 px/side)
# UI: /predict -> /results/:id -> PDF download; or:
curl.exe -F "file=@C:\path\to\fundus.jpg" http://localhost:8000/api/predict
curl.exe http://localhost:8000/api/health            # expect model_mode: trained
curl.exe http://localhost:8000/api/model/performance # accuracy / macro-F1 / confusion matrix
cd backend; python -m pytest -q                      # regression suite
```

## 5. Limitations (why it is not clinical)

Single 366-image test, tiny minority counts, frozen backbones (no fine-tuning),
one split (no CV / CIs / external camera-shift test), surrogate CAM head
(GAP+linear, equals Grad-CAM only for that head — `ml/bundle.py:47`), class
imbalance handled only by weights. See `PROJECT_AUDIT.md`.
