RANDOM_STATE = 42
RUN_MODE = 'sweep'  # 'single' o 'sweep'
SWEEP_MODE = 'alpha'  # 'alpha' o 'percent'
                      # - 'alpha': varía ALPHA_TRUE, usa ALPHA_ESTIMATION_METHOD
                      # - 'percent': varía TOP_Q_PERCENT (fuerza ALPHA_ESTIMATION_METHOD='mean' automáticamente)

# Dataset a utilizar 
# Opciones soportadas:
#   - OpenML: magic_telescope, ionosphere, spambase, sonar, miniboone, phoneme
#   - Sklearn: breast_cancer
#   - Libsvm: gas_sensor_drift
#   - Microarray: colon_cancer, prostate_cancer, lung_cancer (requiere download primero)
DATASET = "prostate_cancer"

# Opcional: fuerza la etiqueta considerada como positiva en datasets OpenML
# (si es None, se usa el valor por defecto del registry en src/datasets.py)
DATASET_POSITIVE_LABEL = None

# Solo para gas_sensor_drift (dataset multiclase original): clase (gas) considerada positiva (one-vs-rest)
GAS_POSITIVE_CLASS = 1

# Parámetros modo single
ALPHA_TRUE = 0.2

# Parámetros modo sweep alpha
SWEEP_ALPHAS = [0.3, 0.2, 0.1]
SWEEP_SEEDS = [0, 1, 2]

# Método de estimación de alpha
ALPHA_ESTIMATION_METHOD = 'mean'  # 'mean' o 'robust'
ALPHA_TOP_Q_PERCENT = 20  # percentil superior a considerar en método robusto
 
# Parámetros modo sweep percent 
TOP_Q_PERCENT_VALUES = [5, 20, 30, 50, 100]  # Porcentajes a evaluar (usado en modo 'sweep')

# Ruido gaussiano en features (0 = sin ruido; p.ej. 0.3 = 30% de la std por feature)
NOISE_LEVEL = 0

# Parámetros generales
TOP_K = 10  # Número de features top-K para evaluar overlap/AUC
EXPERIMENT_NAME = DATASET
RUN_NAME = "mean 3alphas (kfold 3) corregido"  # Nombre del run (cambia esto para diferenciar los runs en mlflow)

