# Selección de Características Basada en Información Mutua para Datos Positive-Unlabeled

TFG que implementa un framework de **Positive-Unlabeled (PU) Learning** para selección de features mediante Información Mutua.

## Estructura

```
├── main.py
├── src/
│   ├── config.py        # Todos los parámetros
│   ├── datasets.py      # Carga de datasets
│   ├── data_utiles.py   # Ruido gaussiano, generación de etiquetas PU
│   ├── pu_model.py      # Modelos PU, estimación de alpha
│   ├── mi_utiles.py     # Información Mutua
│   └── evaluacion.py    # Métricas (AUC, Spearman, Overlap)
├── data/                # Datasets locales
└── mlruns/              # Artefactos MLflow
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
RUN_MODE = 'sweep'              # 'single' | 'sweep'
SWEEP_MODE = 'percent'          # 'alpha' | 'percent'
DATASET = "breast_cancer"
ALPHA_ESTIMATION_METHOD = 'mean'  # 'mean' | 'robust'
ALPHA_TOP_Q_PERCENT = 20
ALPHA_TRUE = 0.2
TOP_K = 10
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

## Datasets

| Fuente     | Valores de `DATASET` |
|------------|----------------------|
| Sklearn    | `breast_cancer` |
| OpenML     | `magic_telescope`, `ionosphere`, `spambase`, `sonar`, `miniboone`, `phoneme` |
| LibSVM     | `gas_sensor_drift` (requiere descargar) |
| Microarray | `colon_cancer`, `prostate_cancer`, `lungs_cancer` (requieren descargar) |
