"""
Scatter plot: ratio de dimensionalidad p/n vs estimacion alpha_hat.

Para cada dataset y cada valor de alpha_true (0.1, 0.2, 0.3, 0.5) dibuja
un punto en (p/n, alpha_hat). Datasets con p/n elevado muestran saturacion
del estimador (alpha_hat -> 1), mientras los de p/n bajo siguen el valor real.

Uso:
    python graficar_scatter_alpha_vs_dimensionalidad.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTADOS_DIR = "resultados"
OUTPUT_DIR = "resultados"

# n y p extraidos de src/datasets.py; kddcup99 p~118 tras OHE de 3 variables categoricas
DATASET_META = {
    "kddcup99":        {"n": 494021, "p": 118,    "label": "KDDCup99"},
    "corral100":       {"n": 100000, "p": 100,    "label": "Corral100"},
    "waveform":        {"n": 5000,   "p": 40,     "label": "Waveform"},
    "musk_v2":         {"n": 6598,   "p": 166,    "label": "Musk v2"},
    "jasmine":         {"n": 2984,   "p": 144,    "label": "Jasmine"},
    "madelon_extenso": {"n": 4400,   "p": 500,    "label": "Madelon"},
    "gisette_extenso": {"n": 13500,  "p": 5000,   "label": "Gisette"},
    "dexter":          {"n": 2600,   "p": 20000,  "label": "Dexter"},
    "dorothea":        {"n": 1950,   "p": 100000, "label": "Dorothea"},
}

# Un color por nivel de alpha_true
ALPHA_COLORS = {
    0.1: "#3b82f6",
    0.2: "#f59e0b",
    0.3: "#10b981",
    0.5: "#ef4444",
}

# Posiciones de etiqueta de dataset: (x_text, y_text, ha, va)
LABEL_POSITIONS = {
    "KDDCup99":  (0.00017, 0.40,  "right", "center"),
    "Corral100": (0.0013,  0.15,  "left",  "center"),
    "Waveform":  (0.011,   0.065, "left",  "bottom"),
    "Musk v2":   (0.017,   0.42,  "right", "center"),
    "Jasmine":   (0.065,   0.20,  "left",  "center"),
    "Madelon":   (0.155,   0.58,  "left",  "center"),
    "Gisette":   (0.50,    0.975, "left",  "top"),
    "Dexter":    (5.2,     1.005, "right", "bottom"),
    "Dorothea":  (70,      0.970, "left",  "center"),
}


def cargar_datos() -> pd.DataFrame:
    filas = []
    for dataset, meta in DATASET_META.items():
        csv_path = os.path.join(RESULTADOS_DIR, dataset, "mean", "alpha_sweep_summary.csv")
        if not os.path.exists(csv_path):
            print(f"  [skip] {dataset}: no encontrado {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        p_n = meta["p"] / meta["n"]
        for _, row in df.iterrows():
            filas.append({
                "dataset":    dataset,
                "label":      meta["label"],
                "p_n":        p_n,
                "alpha_true": round(float(row["alpha_true"]), 1),
                "alpha_hat":  float(row["alpha_hat_mean"]),
            })
    return pd.DataFrame(filas)


def graficar(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Zona de saturacion sombreada
    ax.axvspan(0.30, 200, color="#fee2e2", alpha=0.30, zorder=0)
    ax.text(2.0, 0.65, "Zona de\nsaturación",
            ha="center", va="center", fontsize=8.5, color="#b91c1c",
            style="italic", zorder=1)

    # Lineas de referencia horizontales (valor real de alpha)
    for alpha_val, color in ALPHA_COLORS.items():
        ax.axhline(alpha_val, color=color, linestyle="--", linewidth=0.9, alpha=0.50, zorder=1)

    # Conectores verticales: une los 4 puntos de cada dataset con una linea gris
    for _, grp in df.groupby("dataset"):
        x_val = grp["p_n"].iloc[0]
        ax.plot([x_val, x_val],
                [grp["alpha_hat"].min(), grp["alpha_hat"].max()],
                color="#94a3b8", linewidth=1.0, zorder=2)

    # Puntos coloreados por alpha_true
    for alpha_val, color in ALPHA_COLORS.items():
        sub = df[df["alpha_true"] == alpha_val]
        ax.scatter(sub["p_n"], sub["alpha_hat"],
                   color=color, s=65, edgecolors="white", linewidths=0.8,
                   zorder=4, label=f"α = {alpha_val}")

    # Etiquetas de dataset
    for _, grp in df.groupby("dataset"):
        lbl = grp["label"].iloc[0]
        if lbl not in LABEL_POSITIONS:
            continue
        x_t, y_t, ha, va = LABEL_POSITIONS[lbl]
        ax.text(x_t, y_t, lbl, ha=ha, va=va,
                fontsize=8.0, color="#1e293b", fontweight="medium")

    ax.set_xscale("log")
    ax.set_xlim(5e-5, 200)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Ratio de dimensionalidad $p/n$", fontsize=12)
    ax.set_ylabel(r"$\hat{\alpha}$ estimado", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.legend(title="Valor real de $\\alpha$", fontsize=9, title_fontsize=9,
              loc="lower right", framealpha=0.85)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(OUTPUT_DIR, f"scatter_alpha_vs_dimensionalidad.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Guardado: {out}")
    plt.close(fig)


if __name__ == "__main__":
    df = cargar_datos()
    if df.empty:
        print("No hay datos.")
    else:
        print(f"  {len(df)} puntos cargados ({df['dataset'].nunique()} datasets).")
        graficar(df)
        print("Hecho.")
