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
#       corral100      → 100000 muestras, 100 features (5 inform. + 45 redund. correladas + 50 ruido) [TOP_K recomendado: 5-10]
#       synthetic_pu   → 3000 muestras, 100 features (10 inform. + 5 redund. + 85 ruido) [controlado, class_sep=0.8]
#       synthetic_pu2  → 5000 muestras, 100 features (10 inform. MI 0.08-0.12 + 90 ruido MI 0.04-0.08) [isótropo, naive falla con alpha pequeño]
#   - NIPS 2003 Feature Selection Challenge (UCI directo, todos los splits):
#       arcene          →   300 muestras,  10000 features (microarray, cancer vs normal)   [UCI #167]
#       gisette_extenso → 13500 muestras,   5000 features (imagen, digito 4 vs 9)          [UCI #170]
#       dexter          →  2600 muestras,  20000 features (texto TF-IDF, sparse)           [UCI #168]
#       dorothea        →  1950 muestras, 100000 features (farmacos, binary sparse)        [UCI #169] [RAM ~800 MB]
#       madelon_extenso →  4400 muestras,    500 features (sintetico, UCI completo)        [UCI #171]
#   - OpenML: magic_telescope, ionosphere, spambase, sonar, miniboone, phoneme, musk_v1
#   - LIBSVM streaming: epsilon → 400K muestras, 2000 features densas, 50/50 (descarga solo MAX_SAMPLES líneas)
#   - UCI directo: musk_v2 → 6598 conformaciones, 166 features, ~40% musk (RECOMENDADO)
#   - Nuevos reales (binario moderado):
#       jasmine           → 2984 muestras, 144 features (AutoML Challenge 2)
#       phishing_websites → 11055 muestras, 30 features (UCI, phishing=-1 vs legitimo=1)
#       internet_ads      → 3279 muestras, 1558 features (3 cont. + 1555 bin.), alpha≈0.14
#   - Intrusión (IDS):
#       kddcup99       → ~494K muestras, 41 features orig. (3 categ. OHE + 38 num. ≈118 tras OHE), ataque=1 vs normal=0
#   - Sklearn: breast_cancer
#   - Libsvm: gas_sensor_drift
#   - Microarray (GEO): colon_cancer, prostate_cancer, lung_cancer
#   - MAT files: cns, dlbcl
#   - Cheminformatics (ECFP4 fingerprints binarias, sparse, requiere rdkit):
#       tox21          → ~7800 compuestos, 2048 features ECFP4 binarias (NR-AR)
#       tox21_sr_mmp   → ~7800 compuestos, 2048 features ECFP4 binarias (SR-MMP)
#       tox21_nr_ahr   → ~7800 compuestos, 2048 features ECFP4 binarias (NR-AhR)
#   - Malware (features binarias de API calls/permisos, requiere descarga manual):
#       drebin         → ~5K malware+goodware, ~500K features binarias (Arp et al. 2014)
#                        Descarga: https://drebin.mlsec.org/  (requiere registro)
DATASET = "kddcup99"  # Cambia esto para elegir el dataset a usar (ver opciones arriba)

# Opcional: fuerza la etiqueta considerada como positiva en datasets OpenML
# (si es None, se usa el valor por defecto del registry en src/datasets.py)
DATASET_POSITIVE_LABEL = None

# Solo para gas_sensor_drift (dataset multiclase original): clase (gas) considerada positiva (one-vs-rest)
GAS_POSITIVE_CLASS = 1

# Subsampling (None = sin límite; útil para datasets grandes como epsilon)
MAX_SAMPLES = None  # muestras aleatorias a usar (reproducible con RANDOM_STATE)

# Número de folds para KFold en modo sweep
N_SPLITS = 3

# Parámetros modo single
ALPHA_TRUE = 0.2

# Parámetros modo sweep alpha
SWEEP_ALPHAS = [0.5, 0.1, 0.2, 0.3]
SWEEP_SEEDS = [0, 1, 2]

# Método de estimación de alpha
ALPHA_ESTIMATION_METHOD = 'mean'  # 'mean' o 'robust'
ALPHA_TOP_Q_PERCENT = 30  # percentil superior a considerar en método robusto

# Parámetros modo sweep percent
TOP_Q_PERCENT_VALUES = [5, 20, 30, 50, 100]  # Porcentajes a evaluar (usado en modo 'sweep')

# Ruido gaussiano en features (0 = sin ruido; p.ej. 0.3 = 30% de la std por feature)
NOISE_LEVEL = 0

# Parámetros generales
TOP_K = 25  # Número de features top-K para evaluar overlap/AUC
EXPERIMENT_NAME = DATASET
RUN_NAME = "alphas pequeños TOP-25"  # Nombre del run (cambia esto para diferenciar los runs en mlflow)

# ── Paleta de colores canónica para todas las gráficas ────────────────────────
# Usada en _sweep_alpha, _sweep_percent y cualquier plot futuro.
# Importada automáticamente en main.py via `from src.config import *`.
COLORES_METODOS = {
    'PU_corregido':  '#3498DB',  # azul   – método propuesto
    'MI_naive':      '#FF8C42',  # naranja – baseline naive
    'MI_real':       '#2ECC71',  # verde   – cota superior (oráculo)
    'Varianza':      '#E74C3C',  # rojo    – baseline no supervisado
    'sin_seleccion': '#9B59B6',  # morado  – baseline sin selección de features
}
