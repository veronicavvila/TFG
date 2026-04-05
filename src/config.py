RANDOM_STATE = 42
RUN_MODE = 'sweep'  # 'single' o 'sweep'

# Parámetros modo single
ALPHA_TRUE = 0.2

# Parámetros modo sweep
SWEEP_ALPHAS = [0.5, 0.3, 0.2, 0.1, 0.05]
SWEEP_SEEDS = [0, 1, 2, 3, 4]

# Ruido gaussiano en features (0 = sin ruido; p.ej. 0.3 = 30% de la std por feature)
NOISE_LEVEL = 0

# Parámetros generales
TOP_K = 10
EXPERIMENT_NAME = "miniboone"
RUN_NAME = "v2.0.0"

# Método de estimación de alpha
# 'mean': media simple de scores en S=1 (método estándar)
# 'robust': usar top Q% de scores más altos en S=1 (positivos confiables A)
ALPHA_ESTIMATION_METHOD = 'robust'  # cambiar a 'mean' para usar método estándar

# Si ALPHA_ESTIMATION_METHOD == 'robust', qué percentil usar (0-100)
# 10 = usar top 10% más alto, 20 = top 20%, etc.
ALPHA_TOP_Q_PERCENT = 20
 