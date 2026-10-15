"""Print a compact validation summary from a classifier experiment."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "results", nargs="?", type=Path,
    default=ROOT / "models" / "classifier_experiments" / "validation_results.json",
)
args = parser.parse_args()
results = json.loads(args.results.read_text(encoding="utf-8"))

if isinstance(results, dict) and "selected" in results:
    best = results["selected"]
    metrics = best["validation"]
    classifier = best["classifier"]
    class_rows = metrics["per_class"]
else:
    best = max(results, key=lambda row: row["macro_f1"])
    metrics = best
    classifier = best["model"]
    class_rows = best["class_report"]

print("Best candidate:", classifier)
print("PCA components:", best["pca_components"])
print("Validation accuracy:", round(metrics["accuracy"], 4))
print("Validation macro F1:", round(metrics["macro_f1"], 4))
print("\nClass-wise validation results:")
names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
for index, name in enumerate(names):
    row = class_rows[index] if isinstance(class_rows, list) else class_rows.get(str(index), {})
    print(f"{name:22s} Precision={row.get('precision', 0):.3f} "
          f"Recall={row.get('recall', 0):.3f} "
          f"F1={row.get('f1', row.get('f1-score', 0)):.3f}")
