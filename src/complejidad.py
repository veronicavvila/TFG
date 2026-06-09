"""
Medidas de complejidad de dataset (F1, N1, C1) usando la librería data-complexity.

Uso:
    python -m src.complejidad --dataset waveform
    python -m src.complejidad --dataset waveform --subsample 5000
    python -m src.complejidad --dataset waveform --pca 50
    python -m src.complejidad --dataset waveform --subsample 5000 --pca 50

Medidas:
    F1  – Maximum Fisher's Discriminant Ratio (separabilidad lineal)
    N1  – Fraction of Borderline Points via MST (solapamiento geométrico)
    C1  – Normalized Class Entropy (balance de clases)

Calcula sobre el dataset completo con etiquetas reales.
Para datasets grandes, --subsample y --pca deben indicarse manualmente.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from dcm import dcm
from mlflow.tracking import MlflowClient

from src.datasets import load_dataset

# ── Configuración ──────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "complejidad"
RESULTS_CSV = os.path.join("results", "complejidad.csv")


# ── Medidas ────────────────────────────────────────────────────────────────────

def _compute_f1(X: np.ndarray, y: np.ndarray) -> float:
    _, value = dcm.F1(X, y)
    return float(value)


def _compute_c1(X: np.ndarray, y: np.ndarray) -> float:
    value, _ = dcm.C12(X, y)
    return float(value)


def _compute_n1(
    X: np.ndarray,
    y: np.ndarray,
    subsample: Optional[int] = None,
    pca: Optional[int] = None,
) -> Tuple[float, str]:
    n, p = X.shape
    method = "exact"

    if subsample is not None:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=subsample, random_state=42)
        idx, _ = next(sss.split(X, y))
        X, y = X[idx], y[idx]
        method = f"subsample_{subsample}"
        print(f"  [N1] Subsample estratificado a {subsample} muestras (de {n}).")

    if pca is not None:
        from sklearn.decomposition import PCA
        X = PCA(n_components=pca, random_state=42).fit_transform(X)
        method = f"pca_{pca}" if subsample is None else f"subsample_{subsample}_pca_{pca}"
        print(f"  [N1] PCA a {pca} componentes (de {p}).")

    print(f"  [N1] Calculando N1 (n={X.shape[0]}, p={X.shape[1]})...")
    return float(dcm.N1(X, y)), method


# ── MLflow ─────────────────────────────────────────────────────────────────────

def _log_mlflow(row: dict) -> None:
    client = MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        client.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=row["dataset"]):
        mlflow.log_metric("complexity_f1",         row["f1"])
        mlflow.log_metric("complexity_n1",         row["n1"])
        mlflow.log_metric("complexity_c1",         row["c1"])
        mlflow.log_metric("complexity_n_samples",  float(row["n_samples"]))
        mlflow.log_metric("complexity_n_features", float(row["n_features"]))
        mlflow.log_param("n1_exact",  str(row["n1_method"] == "exact").lower())
        mlflow.log_param("n1_method", row["n1_method"])

    print(f"  MLflow: run registrado en experimento '{EXPERIMENT_NAME}'.")


# ── CSV acumulado ──────────────────────────────────────────────────────────────

def _update_csv(row: dict) -> None:
    os.makedirs("results", exist_ok=True)
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        df = df[df["dataset"] != row["dataset"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(RESULTS_CSV, index=False, float_format="%.6f")
    print(f"  CSV actualizado → {RESULTS_CSV}")


# ── Entry point ────────────────────────────────────────────────────────────────

def compute_complexity(
    dataset_key: str,
    subsample: Optional[int] = None,
    pca: Optional[int] = None,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_key}")
    print(f"{'='*60}")

    X, y, _, _ = load_dataset(dataset_key)
    n, p = X.shape
    pct_pos = float(y.mean() * 100)
    print(f"  n={n}  p={p}  %pos={pct_pos:.1f}%")

    print("  Calculando F1...")
    f1 = _compute_f1(X, y)
    print(f"  F1 = {f1:.6f}")

    n1, n1_method = _compute_n1(X, y, subsample=subsample, pca=pca)
    print(f"  N1 = {n1:.6f}  [método: {n1_method}]")

    print("  Calculando C1...")
    c1 = _compute_c1(X, y)
    print(f"  C1 = {c1:.6f}")

    return {
        "dataset":    dataset_key,
        "n_samples":  n,
        "n_features": p,
        "pct_pos":    round(pct_pos, 1),
        "f1":         f1,
        "n1":         n1,
        "n1_method":  n1_method,
        "c1":         c1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcula F1, N1, C1 de complejidad para un dataset."
    )
    parser.add_argument("--dataset", required=True,
                        help="Clave del dataset (ej: waveform, musk_v2, kddcup99)")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Submuestreo estratificado a N muestras antes de calcular N1")
    parser.add_argument("--pca", type=int, default=None,
                        help="Reducción PCA a K componentes antes de calcular N1")
    args = parser.parse_args()

    row = compute_complexity(args.dataset, subsample=args.subsample, pca=args.pca)
    _log_mlflow(row)
    _update_csv(row)

    print(f"\nResumen: F1={row['f1']:.4f}  N1={row['n1']:.4f} [{row['n1_method']}]  C1={row['c1']:.4f}")


if __name__ == "__main__":
    main()
