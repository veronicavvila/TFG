import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ALPHA = 0.5
RESULTS_DIR = "resultados"
COMPLEJIDAD_CSV = os.path.join("results", "complejidad.csv")
OUTPUT_DIR = "results"

MODELOS = {
    "PU_corregido": ("spearman_pu_topk_mean",    "^"),
    "MI_naive":     ("spearman_naive_topk_mean",  "s"),
    "MI_real":      (None,                         "o"),  # siempre 1.0
}

NOMBRE_MODELO = {
    "PU_corregido": "PU-MI",
    "MI_naive":     "Naive",
    "MI_real":      "Real",
}

NOMBRE_DATASET = {
    "gisette_extenso": "Gisette",
    "madelon_extenso": "Madelon",
    "kddcup99":        "KDDCup99",
    "musk_v2":         "Musk v2",
    "waveform":        "Waveform-5000",
    "corral100":       "Corral100",
    "dexter":          "Dexter",
    "dorothea":        "Dorothea",
    "jasmine":         "Jasmine",
}

MEDIDAS = ["f1", "n1", "c1"]
EXCLUIR = {"arcene"}


def cargar_datos():
    complejidad = pd.read_csv(COMPLEJIDAD_CSV)

    filas = []
    for _, row in complejidad.iterrows():
        dataset = row["dataset"]
        if dataset in EXCLUIR:
            continue
        summary_path = os.path.join(RESULTS_DIR, dataset, "mean", "alpha_sweep_summary.csv")
        if not os.path.exists(summary_path):
            print(f"  [skip] {dataset}: no se encontró {summary_path}")
            continue

        df = pd.read_csv(summary_path)
        fila_alpha = df[df["alpha_true"] == ALPHA]
        if fila_alpha.empty:
            print(f"  [skip] {dataset}: alpha={ALPHA} no encontrado")
            continue
        fila_alpha = fila_alpha.iloc[0]

        spearman_pu    = fila_alpha["spearman_pu_topk_mean"]
        spearman_naive = fila_alpha["spearman_naive_topk_mean"]

        for medida in MEDIDAS:
            valor = row[medida]
            for modelo, (col, _) in MODELOS.items():
                spearman = 1.0 if col is None else (spearman_pu if modelo == "PU_corregido" else spearman_naive)
                filas.append({
                    "dataset": dataset,
                    "medida":  medida,
                    "valor":   valor,
                    "modelo":  modelo,
                    "spearman": spearman,
                })

    return pd.DataFrame(filas)


def graficar(df):
    datasets = sorted(df["dataset"].unique())
    colores  = {d: c for d, c in zip(datasets, plt.cm.tab10.colors)}

    for medida in MEDIDAS:
        df_m = df[df["medida"] == medida]

        fig, ax = plt.subplots(figsize=(9, 5))

        for modelo, (_, marker) in MODELOS.items():
            df_mod = df_m[df_m["modelo"] == modelo]
            for _, row in df_mod.iterrows():
                ax.scatter(
                    row["spearman"], row["valor"],
                    marker=marker,
                    color=colores[row["dataset"]],
                    s=80,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=3,
                )

        ax.set_xlabel(f"Spearman (top-k, α={ALPHA})", fontsize=11)
        ax.set_ylabel(medida, fontsize=11)
        ax.set_title(f"Spearman vs {medida}", fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Leyenda datasets (colores)
        dataset_handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=colores[d], markeredgecolor="black",
                       markersize=8, label=NOMBRE_DATASET.get(d, d))
            for d in datasets
        ]
        # Leyenda métodos (formas)
        model_handles = [
            plt.Line2D([0], [0], marker=mk, color="w",
                       markerfacecolor="gray", markeredgecolor="black",
                       markersize=8, label=NOMBRE_MODELO.get(mod, mod))
            for mod, (_, mk) in MODELOS.items()
        ]

        leg1 = ax.legend(handles=dataset_handles, title="Conjunto de datos",
                         bbox_to_anchor=(1.01, 1), loc="upper left",
                         fontsize=8, title_fontsize=9)
        ax.add_artist(leg1)
        ax.legend(handles=model_handles, title="Método",
                  bbox_to_anchor=(1.01, 0), loc="lower left",
                  fontsize=8, title_fontsize=9)

        fig.subplots_adjust(right=0.70)
        alpha_str = str(ALPHA).replace(".", "_")
        out = os.path.join(OUTPUT_DIR, f"scatter_{medida}_alpha{alpha_str}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Guardado: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    ALPHA = args.alpha

    df = cargar_datos()
    if df.empty:
        print("No hay datos para graficar.")
    else:
        graficar(df)
        print("Hecho.")
