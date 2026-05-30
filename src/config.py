RANDOM_STATE = 42
RUN_MODE = 'sweep'  # 'single' o 'sweep'
SWEEP_MODE = 'alpha'  # 'alpha' o 'percent'
                      # - 'alpha': varía ALPHA_TRUE, usa ALPHA_ESTIMATION_METHOD
                      # - 'percent': varía TOP_Q_PERCENT (fuerza ALPHA_ESTIMATION_METHOD='mean' automáticamente)

# Dataset a utilizar
# Opciones soportadas:
#   - Benchmark feature-selection:
#       madelon        → 2000 muestras, 500 features (5 clave + 15 redund. + 480 ruido)  ← RECOMENDADO
#       waveform       → 5000 muestras,  40 features (21 señal + 19 ruido), clase 0 vs resto
#       gisette        → 7000 muestras, 5000 features (2500 reales + 2500 probe ruido), digito 4 vs 9
#       synthetic_pu   → 3000 muestras, 100 features (10 inform. + 5 redund. + 85 ruido) [controlado, class_sep=0.8]
#   - OpenML: magic_telescope, ionosphere, spambase, sonar, miniboone, phoneme, musk_v1
#   - Sklearn: breast_cancer
#   - Libsvm: gas_sensor_drift
#   - Microarray (GEO): colon_cancer, prostate_cancer, lung_cancer
#   - MAT files: cns, dlbcl
DATASET = "gisette"

# Opcional: fuerza la etiqueta considerada como positiva en datasets OpenML
# (si es None, se usa el valor por defecto del registry en src/datasets.py)
DATASET_POSITIVE_LABEL = None

# Solo para gas_sensor_drift (dataset multiclase original): clase (gas) considerada positiva (one-vs-rest)
GAS_POSITIVE_CLASS = 1

# Parámetros modo single
ALPHA_TRUE = 0.2

# Parámetros modo sweep alpha
SWEEP_ALPHAS = [0.1, 0.2, 0.5]
SWEEP_SEEDS = [0, 1, 2]

# Método de estimación de alpha
ALPHA_ESTIMATION_METHOD = 'mean'  # 'mean' o 'robust'
ALPHA_TOP_Q_PERCENT = 30  # percentil superior a considerar en método robusto

# Parámetros modo sweep percent
TOP_Q_PERCENT_VALUES = [5, 20, 30, 50, 100]  # Porcentajes a evaluar (usado en modo 'sweep')

# Ruido gaussiano en features (0 = sin ruido; p.ej. 0.3 = 30% de la std por feature)
NOISE_LEVEL = 0

# Parámetros generales
TOP_K = 100  # Número de features top-K para evaluar overlap/AUC
EXPERIMENT_NAME = DATASET
RUN_NAME = "ALPHA TOP-100"  # Nombre del run (cambia esto para diferenciar los runs en mlflow)

# ── Paleta de colores canónica para todas las gráficas ────────────────────────
# Usada en _sweep_alpha, _sweep_percent y cualquier plot futuro.
# Importada automáticamente en main.py via `from src.config import *`.
COLORES_METODOS = {
    'PU_corregido': '#3498DB',  # azul   – método propuesto
    'MI_naive':     '#FF8C42',  # naranja – baseline naive
    'MI_real':      '#2ECC71',  # verde   – cota superior (oráculo)
    'Varianza':     '#E74C3C',  # rojo    – baseline no supervisado
}
