# Selección de Características Basada en Información Mutua para Datos Positive-Unlabeled

TFG que implementa un framework de **Positive-Unlabeled (PU) Learning** para selección de features mediante Información Mutua.

## Estructura

```
├── main.py
├── src/
│   ├── config.py              # Todos los parámetros
│   ├── datasets.py            # Carga de conjuntos de datos
│   ├── data_utiles.py         # Ruido gaussiano, generación de etiquetas PU
│   ├── pu_model.py            # Modelos PU, estimación de alpha
│   ├── mi_utiles.py           # Información Mutua
│   ├── evaluacion.py          # Métricas (AUC, Spearman, Overlap)
│   ├── complejidad.py         # Cálculo de complejidad
│   └── graficar_complejidad.py  # Graficar complejidad
├── data/                      # Conjuntos de datos locales
└── mlruns/                    # Artefactos MLflow
```

## Inicio rápido

```bash
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
mlflow ui  # → http://localhost:5000
```

## Configuración (`src/config.py`)

```python
RUN_MODE = 'sweep'             # 'single' | 'sweep'
SWEEP_MODE = 'alpha'           # 'alpha' | 'percent'
DATASET = "waveform"           # ver conjuntos de datos disponibles
ALPHA_ESTIMATION_METHOD = 'mean'  # 'mean' | 'robust'
ALPHA_TOP_Q_PERCENT = 30
ALPHA_TRUE = 0.2
TOP_K = 5
NOISE_LEVEL = 0
```

## Modos de ejecución

| `RUN_MODE` | `SWEEP_MODE` | Descripción |
|------------|--------------|-------------|
| `single`   | —            | Una ejecución con parámetros fijos |
| `sweep`    | `alpha`      | Barre valores de alpha (25 ejecuciones) |
| `sweep`    | `percent`    | Barre percentiles top-q (requiere `robust`) |

## Estimación de alpha

- **`mean`**: promedio de scores de todos los positivos etiquetados.
- **`robust`**: promedio del top-q% de positivos etiquetados (caso de estudio con `SWEEP_MODE='percent'`).

## Conjuntos de datos

| `DATASET`    | Muestras | Features  | Tipo       | Dominio              |
|--------------|----------|-----------|------------|----------------------|
| `waveform`   | 5 000    | 40        | Sintético  | Señal (benchmark FS) |
| `corral100`  | 100 000  | 100       | Sintético  | Benchmark FS         |
| `jasmine`    | 2 984    | 144       | Real       | AutoML Challenge     |
| `musk_v2`    | 6 598    | 166       | Real       | Química (UCI)        |
| `kddcup99`   | ~494 000 | 41 (~118 OHE) | Real   | Intrusión de red     |
| `gisette`    | 7 000    | 5 000     | Real       | Imagen (dígitos)     |
| `dexter`     | 2 600    | 20 000    | Real       | Texto (TF-IDF)       |
| `dorothea`   | 1 950    | 100 000   | Real       | Fármacos (sparse)    |
| `madelon`    | 2 000    | 500       | Sintético  | Benchmark FS         |
