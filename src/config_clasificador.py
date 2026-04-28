# Dataset a utilizar 
# Opciones soportadas:
#   - OpenML: magic_telescope, ionosphere, spambase, sonar, miniboone, phoneme
#   - Sklearn: breast_cancer
#   - Libsvm: gas_sensor_drift
#   - Microarray (descargados): colon_cancer, prostate_cancer, lung_cancer
#   - Microarray (locales .mat): cns, dlbcl
DATASET = "dlbcl"  
VALID_DATASETS = [
    'breast_cancer', 'ionosphere', 'sonar', 'gas_sensor_drift', 'miniboone', 
    'spambase', 'magic_telescope', 'phoneme',
    'colon_cancer', 'cns', 'dlbcl'
]
assert DATASET in VALID_DATASETS, f"Dataset debe ser uno de: {VALID_DATASETS}"

                           
# Parámetros generales
N_SEEDS = [0, 1, 2, 3, 4]  
ALPHAS = [0.5, 0.3, 0.2]
N_KFOLDS = 3
NOISE_LEVEL = 0.0
TOP_K = 10  # Número de features top-K para evaluar overlap/AUC
USE_FEATURE_SELECTION = True  # Usar selección de features basada en MI

# Parámetros de PU
# Método de estimación de alpha
# Cambiar entre 'robust' o 'mean':
#   - 'robust': usa estimar_alpha_robusto con top_q_percent=30
#   - 'mean': usa media simple de scores de positivos etiquetados
ALPHA_ESTIMATION_METHOD = 'robust'  
ALPHA_TOP_Q_PERCENT = 30  # percentil superior a considerar en método robusto

# Mlflow
EXPERIMENT_NAME = DATASET
RUN_NAME = "robust comparativa_clasificadores corregido"