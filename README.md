# PU Learning con Selección de Features Robusta

Proyecto de Trabajo Fin de Grado (TFG) que implementa un framework para **Positive-Unlabeled (PU) Learning** con énfasis en **estimación robusta de alpha** y **selección de features basada en Información Mutua**.

##  Contenido del Proyecto

```
├── main.py                      # Script principal de ejecución
├── src/
│   ├── config.py               # Configuración centralizada de todos los parámetros
│   ├── datasets.py             # Carga de datasets (OpenML, sklearn, libsvm, microarray)
│   ├── data_utiles.py          # Ruido gaussiano, generación de etiquetas PU
│   ├── pu_model.py             # Modelos PU, estimación de alpha (mean/robust)
│   ├── mi_utiles.py            # Cálculo de Información Mutua
│   └── evaluacion.py           # Métricas (AUC, Spearman, Overlap)
├── data/
│   ├── gas_sensor_drift/       # Dataset GAS (requiere descargar)
│   └── microarray_datasets/    # Microarray datasets
├── mlruns/                     # Artefactos de MLflow
└── README.md                   # Este archivo
```

##  Inicio Rápido

### 1. Configuración

Todos los parámetros se controlan desde **`src/config.py`**:

```python
# Modo de ejecución
RUN_MODE = 'sweep'              # 'single' o 'sweep'
SWEEP_MODE = 'percent'          # 'alpha' o 'percent'

# Dataset
DATASET = "breast_cancer"       # Ver sección "Datasets disponibles"

# Método de estimación de alpha
ALPHA_ESTIMATION_METHOD = 'robust'   # 'mean' o 'robust'
ALPHA_TOP_Q_PERCENT = 20             # Percentil para método robust

# Parámetros de sweep
SWEEP_ALPHAS = [0.5, 0.3, 0.2, 0.1, 0.05]
TOP_Q_PERCENT_VALUES = [10, 20, 30, 40, 50]
SWEEP_SEEDS = [0, 1, 2, 3, 4]

# Parámetros generales
TOP_K = 10                      # Features top-K a evaluar
NOISE_LEVEL = 0                 # 0 = sin ruido; 0.3 = 30% ruido
ALPHA_TRUE = 0.2                # Alpha verdadero (solo modo single)
```

### 2. Ejecutar

```bash
# Activar entorno virtual
.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar con configuración actual
python main.py

# Para ver resultados en tiempo real
mlflow ui
# Abre http://localhost:5000 en el navegador
```

##  Modos de Ejecución

El proyecto soporta **3 configuraciones válidas** según la combinación de `RUN_MODE`, `SWEEP_MODE` y `ALPHA_ESTIMATION_METHOD`:

### Modo 1: Single (Ejecución Única)

```python
RUN_MODE = 'single'
ALPHA_TRUE = 0.2
ALPHA_ESTIMATION_METHOD = 'mean'  # o 'robust'
```

**Qué hace:**
- Ejecuta UNA sola vez con parámetros fijos
- Genera etiquetas PU con alpha=0.2
- Prueba 5 métodos de selección de features: PU_corregido, MI_naive, MI_real, Varianza
- Calcula AUC, Overlap, correlación Spearman para top-10 features
- Guarda rankings en archivos CSV

**Outputs:**
- `top_features_*.csv` - Top-10 features por método
- Métricas en MLflow

```python
# Cambio en config.py
RUN_MODE = 'single'
ALPHA_ESTIMATION_METHOD = 'robust'  # Usa percentil 20 para estimar alpha
ALPHA_TOP_Q_PERCENT = 20
```

---

### Modo 2: Sweep Alpha (Variar α)

```python
RUN_MODE = 'sweep'
SWEEP_MODE = 'alpha'
SWEEP_ALPHAS = [0.5, 0.3, 0.2, 0.1, 0.05]
SWEEP_SEEDS = [0, 1, 2, 3, 4]
ALPHA_ESTIMATION_METHOD = 'mean'   # o 'robust' (percentil=20, fijo)
```

**Qué hace:**
- Barre 5 valores de alpha con 5 semillas → **25 ejecuciones**
- Para cada (alpha, seed): 5-fold cross-validation
- Calcula: alpha_hat, AUC, Spearman, Overlap, Estabilidad
- Genera 3 gráficos: AUC vs Alpha, Spearman vs Alpha, Estabilidad vs Alpha

**Outputs:**
- `alpha_sweep_runs.csv` - Todos los 25 resultados
- `alpha_sweep_summary.csv` - Resumen agregado por alpha
- `auc_vs_alpha.png`, `spearman_vs_alpha.png`, `stability_vs_alpha.png`
- Métricas en MLflow

**Casos de uso:**
- Entender cómo el alpha verdadero afecta la estimación
- Validar robustez del método con diferentes alphas
- Comparar 'mean' vs 'robust' bajo variación de alpha

```python
# Ejemplo: comparar métodos
ALPHA_ESTIMATION_METHOD = 'mean'    # Primera ejecución
RUN_NAME = "sweep_alpha_mean"

# Luego cambiar y ejecutar nuevamente:
ALPHA_ESTIMATION_METHOD = 'robust'
RUN_NAME = "sweep_alpha_robust"
```

---

### Modo 3: Sweep Percentil (Variar top_q_percent)  **REQUIERE 'robust'**

```python
RUN_MODE = 'sweep'
SWEEP_MODE = 'percent'
TOP_Q_PERCENT_VALUES = [10, 20, 30, 40, 50]
SWEEP_SEEDS = [0, 1, 2, 3, 4]
ALPHA_ESTIMATION_METHOD = 'robust'  #  OBLIGATORIO (si no, error)
ALPHA_TRUE = 0.2                     # Alpha fijo
```

**Qué hace:**
- Barre 5 percentiles con 5 semillas → **25 ejecuciones**
- El top_q_percent varía cómo se estima alpha (10%, 20%, 30%, 40%, 50%)
- Calcula: alpha_hat(percentil), AUC, Spearman, Error relativo alpha
- Genera gráfico 2x2: Alpha, AUC, Spearman, Error relativo

**Outputs:**
- `top_q_percent_sweep_runs.csv` - Todos los 25 resultados
- `top_q_percent_sweep_summary.csv` - Resumen agregado por percentil
- `top_q_percent_sweep.png` - Gráfico 2x2
- Métricas en MLflow

**Caso de uso:**
- Encontrar el **percentil óptimo** para estimar alpha
- ¿Es mejor usar top-20% de scores o top-40%?
- Optimizar robustez vs sesgo en la estimación

```python
RUN_MODE = 'sweep'
SWEEP_MODE = 'percent'
ALPHA_ESTIMATION_METHOD = 'robust'  # OBLIGATORIO
TOP_Q_PERCENT_VALUES = [10, 15, 20, 25, 30]
ALPHA_TRUE = 0.2
```

---

##  Métodos de Selección de Features

El proyecto compara **4 métodos de ranking de features**:

| Método | Descripción | Label |
|--------|-------------|-------|
| **MI_real** | Información Mutua con etiquetas TRUE (oráculo) | `MI_real` |
| **PU_corregido** | MI estimada desde etiquetas PU + alpha_hat | `PU_corregido` |
| **MI_naive** | MI directa de etiquetas PU (sin corrección) | `MI_naive` |
| **Varianza** | Varianza simple de features (baseline) | `Varianza` |

**Métricas de evaluación:**
- **AUC** - Area Under ROC Curve en top-K features
- **Spearman** - Correlación de ranking vs MI_real
- **Overlap** - Cantidad de features en común vs MI_real top-K
- **Estabilidad** - Robustez del ranking entre semillas

---

##  Estimación de Alpha

### Método 'mean'

```
alpha_hat = mean(scores[S==1])
```
Promedio simple de scores de positivos etiquetados.

**Pros:** Rápido, simple, interpretable  
**Contras:** Sensible a outliers, bajo n_positivos

---

### Método 'robust'

```
alpha_hat = mean(scores[S==1][-top_q_percent:])
```
Promedio de scores de positivos en el **top-q% percentil**.

**Ejemplo (top_q_percent=20):**
- Si hay 50 positivos, toma los 10 mejores (top-20%)
- Calcula promedio de esos 10 scores

**Pros:** Robusto a outliers, más estable  
**Contras:** Requiere threshold (q%)

**Configuración:**
- `ALPHA_TOP_Q_PERCENT = 20` - Para sweep_alpha (fijo)
- `TOP_Q_PERCENT_VALUES = [10,20,30,40,50]` - Para sweep_percent (variable)

---

##  Datasets Disponibles

### OpenML

```python
DATASET = "magic_telescope"     # Magic Gamma Telescope
DATASET = "ionosphere"          # Ionosphere
DATASET = "spambase"            # Spam emails
DATASET = "sonar"               # Sonar targets
DATASET = "miniboone"           # MiniBooNE physics
DATASET = "phoneme"             # Phoneme classification
```

**Parámetro opcional:**
```python
DATASET_POSITIVE_LABEL = None   # Usa label por defecto del registry
DATASET_POSITIVE_LABEL = 1      # O especifica manualmente
```

### Sklearn

```python
DATASET = "breast_cancer"       # Wisconsin Breast Cancer
```

### LibSVM

```python
DATASET = "gas_sensor_drift"    # Gas sensor drift (30+ clases)
GAS_POSITIVE_CLASS = 1          # Clase considerada positivo
```

### Microarray

```python
# Requiere descargar primero
DATASET = "colon_cancer"        
DATASET= "prostate_cancer"
DATASET= "lungs_cancer"

```

##  Restricciones y Validaciones

### Combinaciones VÁLIDAS

| RUN_MODE | SWEEP_MODE | ALPHA_ESTIMATION_METHOD | ✓ |
|----------|-----------|------------------------|---|
| single   | - | mean | ✓ |
| single   | - | robust | ✓ |
| sweep | alpha | mean | ✓ |
| sweep | alpha | robust | ✓ |
| sweep | percent | mean | ✗ ERROR |
| sweep | percent | robust | ✓ |

### Por qué hay restricciones

- **SWEEP_MODE='percent' + 'mean':** Sin sentido. El `top_q_percent` no afecta nada con 'mean', todos los resultados son idénticos.
- **SWEEP_MODE='percent' + 'robust':**  Tiene sentido. El percentil varía cómo se estima alpha.

Si intenta ejecutar una combinación inválida:
```
ERROR: SWEEP_MODE='percent' requiere ALPHA_ESTIMATION_METHOD='robust'
(de lo contrario el top_q_percent no se usa y los resultados se repiten)
```