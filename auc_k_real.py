"""
Calcula la curva AUC(k) usando el ranking supervisado real (MI sobre etiquetas verdaderas).

Uso:
    python auc_k_real.py --dataset madelon_extenso
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.datasets import load_dataset
from src.evaluacion import calcular_mi_real, evaluar_clasificador_final

KS = [1, 2, 5, 10, 20, 50, 100, 200]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    parser = argparse.ArgumentParser(description="Curva AUC(k) con ranking supervisado real.")
    parser.add_argument("--dataset", required=True, help="Nombre del dataset (e.g. madelon_extenso)")
    args = parser.parse_args()

    dataset_name = args.dataset

    print(f"Cargando dataset: {dataset_name}")
    X, y, feature_names, meta = load_dataset(dataset_name)
    n_features = X.shape[1]
    print(f"  {X.shape[0]} muestras, {n_features} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print("Calculando ranking real (MI sobre etiquetas verdaderas)...")
    _, ranking_real = calcular_mi_real(X_train, y_train, random_state=RANDOM_STATE)

    ks_validos = [k for k in KS if k <= n_features]
    if len(ks_validos) < len(KS):
        omitidos = [k for k in KS if k > n_features]
        print(f"  k omitidos por exceder n_features={n_features}: {omitidos}")

    resultados = []
    for k in ks_validos:
        auc = evaluar_clasificador_final(X_train, X_test, y_train, y_test, ranking_real, k)
        print(f"  k={k:>4d}  AUC={auc:.4f}")
        resultados.append({"k": k, "auc": auc})

    out_dir = os.path.join("resultados", dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(resultados)
    csv_path = os.path.join(out_dir, "auc_k_real.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV guardado: {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["k"], df["auc"], marker="o", linewidth=1.8, color="#2563eb")
    for _, row in df.iterrows():
        ax.annotate(
            f"{row['auc']:.3f}",
            (row["k"], row["auc"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("k (número de features seleccionadas)", fontsize=11)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_title(f"Curva AUC(k) — ranking supervisado real\n{dataset_name}", fontsize=12)
    ax.set_xticks(df["k"])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=max(0, df["auc"].min() - 0.05), top=min(1.0, df["auc"].max() + 0.05))
    fig.tight_layout()

    png_path = os.path.join(out_dir, "auc_k_real.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Gráfica guardada: {png_path}")


if __name__ == "__main__":
    main()
