"""
Recalcula as métricas do README a partir das previsões salvas no teste temporal.

Uso (o CSV está no Git LFS):
    git lfs pull --include="artifacts/tabular_predictions.csv"
    python scripts/recompute_metrics.py

Compara o resultado com artifacts/competition_results.json e sai com erro se divergir.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "artifacts" / "tabular_predictions.csv"
REFERENCE = ROOT / "artifacts" / "competition_results.json"
TOLERANCE = 1e-6


def main() -> int:
    df = pd.read_csv(PREDICTIONS, index_col=0)
    y = df["True_Label"].to_numpy()
    score = df["Tabular_Prediction"].to_numpy()
    top = y[np.argsort(-score, kind="stable")]

    fpr, tpr, _ = roc_curve(y, score)
    metrics = {
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "precision@100": top[:100].mean(),
        "precision@500": top[:500].mean(),
    }

    print(f"Transações no teste: {len(y):,}")
    print(f"Positivos: {int(y.sum()):,} ({y.mean():.3%})")
    print(f"Recall com 5% de FPR: {tpr[fpr <= 0.05].max():.4f}")

    reference = next(m for m in json.loads(REFERENCE.read_text())["all_models"] if m["model"] == "XGBoost")
    ok = True
    for name, value in metrics.items():
        match = abs(value - reference[name]) <= TOLERANCE
        ok &= match
        print(f"{name:15s} recalculado={value:.4f}  json={reference[name]:.4f}  {'ok' if match else 'DIVERGE'}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
