"""Backward-compatible entry point for validation-only RBF-SVM tuning."""
from __future__ import annotations

from scripts.train_svm_candidate import main


if __name__ == "__main__":
    raise SystemExit(main())
