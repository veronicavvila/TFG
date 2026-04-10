RANDOM_STATE = 42
RUN_MODE = 'sweep'  # 'single' o 'sweep'

# Dataset a utilizar (cambia esto para ejecutar otro dataset desde la misma rama)
# Opciones soportadas: magic_telescope, ionosphere, spambase, sonar, miniboone,
# phoneme, breast_cancer, gas_sensor_drift
DATASET = "breast_cancer"

# Opcional: fuerza la etiqueta considerada como positiva en datasets OpenML
# (si es None, se usa el valor por defecto del registry en src/datasets.py)
DATASET_POSITIVE_LABEL = None

# Solo para gas_sensor_drift (dataset multiclase original): clase (gas) considerada positiva (one-vs-rest)
GAS_POSITIVE_CLASS = 1

# Parámetros modo single
ALPHA_TRUE = 0.2

# Parámetros modo sweep
SWEEP_ALPHAS = [0.5, 0.3, 0.2, 0.1, 0.05]
SWEEP_SEEDS = [0, 1, 2, 3, 4]

# Ruido gaussiano en features (0 = sin ruido; p.ej. 0.3 = 30% de la std por feature)
NOISE_LEVEL = 0

# Parámetros generales
TOP_K = 10
EXPERIMENT_NAME = DATASET
RUN_NAME = "Pruebav2.1.0positivos"

# Método de estimación de alpha
ALPHA_ESTIMATION_METHOD = 'robust'  # 'mean' o 'robust'
ALPHA_TOP_Q_PERCENT = 20  # percentil superior a considerar en método robusto
 