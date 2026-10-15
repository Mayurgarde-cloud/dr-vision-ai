from pathlib import Path
import json
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "models" / "experiment_aug1"
OUT = ROOT / "models" / "classifier_experiments"
OUT.mkdir(parents=True, exist_ok=True)

split_file = ROOT / "models" / "splits.json"
if not split_file.exists():
    raise FileNotFoundError(f"Missing labels file: {split_file}")

label_map = json.loads(split_file.read_text(encoding="utf-8"))["labels"]

def load_split(name):
    with np.load(CACHE / f"features_{name}.npz") as data:
        paths = data["paths"].tolist()
        X = data["feats"].astype(np.float32)
    y = np.array([label_map[str(Path(p).resolve())] for p in paths])
    return X, y

X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
results = []

for components in [64, 128, 256]:
    scaler = StandardScaler()
    A = scaler.fit_transform(X_train)
    B = scaler.transform(X_val)

    pca = PCA(n_components=components, random_state=42)
    A = pca.fit_transform(A)
    B = pca.transform(B)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            max_features="sqrt", min_samples_leaf=2,
            n_jobs=-1, random_state=42
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced",
            max_features="sqrt", min_samples_leaf=2,
            n_jobs=-1, random_state=42
        ),
        "SVM_RBF": SVC(C=2, kernel="rbf", class_weight="balanced"),
    }

    for name, model in models.items():
        print(f"\nTraining {name}, PCA={components}...")
        model.fit(A, y_train)
        pred = model.predict(B)

        row = {
            "model": name,
            "pca_components": components,
            "accuracy": float(accuracy_score(y_val, pred)),
            "macro_f1": float(f1_score(y_val, pred, average="macro")),
            "macro_recall": float(recall_score(y_val, pred, average="macro")),
            "class_report": classification_report(
                y_val, pred, output_dict=True, zero_division=0
            ),
        }
        results.append(row)
        print(
            f"Accuracy={row['accuracy']:.4f} | "
            f"Macro-F1={row['macro_f1']:.4f} | "
            f"Macro-recall={row['macro_recall']:.4f}"
        )

result_path = OUT / "validation_results.json"
result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

print("\n=== RESULTS RANKED BY VALIDATION MACRO-F1 ===")
for r in sorted(results, key=lambda x: x["macro_f1"], reverse=True):
    print(
        f"{r['model']:12s} PCA={r['pca_components']:3d} "
        f"Accuracy={r['accuracy']:.4f} "
        f"Macro-F1={r['macro_f1']:.4f} "
        f"Macro-recall={r['macro_recall']:.4f}"
    )

print(f"\nResults saved to: {result_path}")
