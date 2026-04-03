import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def entrenar_clasificador_pu(X, S, random_state=None):
    """
    Entrena un clasificador P vs U para estimar P(S=1 | x).

    Parámetros
    ----------
    X : array-like, shape (n_muestras, n_features)
        Matriz de características.
    S : array-like, shape (n_muestras,)
        Etiquetas PU:
            1 = positivo etiquetado
            0 = unlabeled
    random_state : int o None
        Semilla para reproducibilidad.

    Devuelve
    -------
    modelo : objeto entrenado
        Pipeline con estandarización + regresión logística.
    """

    modelo = Pipeline([
        ("escalado", StandardScaler()),
        ("clasificador", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state
        ))
    ])

    modelo.fit(X, S)

    return modelo


def obtener_scores(modelo, X):
    """
    Obtiene los scores continuos P(S=1 | x).

    Parámetros
    ----------
    modelo : objeto entrenado
        Clasificador PU entrenado.
    X : array-like
        Matriz de características.

    Devuelve
    -------
    scores : np.ndarray
        Probabilidad estimada de S=1 para cada muestra.
    """

    # Columna 1 corresponde a la probabilidad de clase 1
    scores = modelo.predict_proba(X)[:, 1]

    return scores


def estimar_alpha(scores, S):
    """
    Estima alpha = P(S=1 | Y=1)

    Parámetros
    ----------
    scores : array-like
        Probabilidades estimadas P(S=1 | x).
    S : array-like
        Etiquetas PU (1 = positivo etiquetado, 0 = unlabeled).

    Devuelve
    -------
    alpha_hat : float
        Estimación de alpha.
    """

    scores = np.asarray(scores)
    S = np.asarray(S)

    # Seleccionar solo los positivos etiquetados
    scores_positivos = scores[S == 1]

    # Media de los scores en P
    alpha_hat = np.mean(scores_positivos)

    return alpha_hat


def estimar_alpha_robusto(scores, S, top_q_percent=20):
    """
    Estima alpha usando solo los positivos etiquetados con scores más altos
    (positivos "más confiables"). Robusto contra subestimación.
    
    Parámetros
    ----------
    scores : array-like
        Probabilidades estimadas P(S=1 | x).
    S : array-like
        Etiquetas PU (1 = positivo etiquetado, 0 = unlabeled).
    top_q_percent : float
        Percentil (0-100) de scores más altos a usar. Default 20%.
        Ej: 20 = usar top 20% de los positivos etiquetados.
    
    Devuelve
    -------
    alpha_hat : float
        Estimación robusta de alpha (tipicamente más alta que la media).
    """
    scores = np.asarray(scores)
    S = np.asarray(S)
    
    # Seleccionar positivos etiquetados
    scores_positivos = scores[S == 1]
    
    if len(scores_positivos) == 0:
        return 0.0
    
    # Threshold es el percentil (100 - top_q_percent)
    # ej: si top_q_percent=20, queremos scores >= percentil 80
    threshold = np.percentile(scores_positivos, 100 - top_q_percent)
    
    # Seleccionar solo los scores por encima del threshold
    scores_confiables = scores_positivos[scores_positivos >= threshold]
    
    # Estimar alpha con los positivos confiables
    alpha_hat = np.mean(scores_confiables)
    
    return alpha_hat


def estimar_probabilidad_real(scores, alpha_hat):
    """
    Estima P(Y=1 | x) a partir de P(S=1 | x)

    Parámetros
    ----------
    scores : array-like
        Probabilidades estimadas P(S=1 | x).
    alpha_hat : float
        Estimación de alpha = P(S=1 | Y=1).

    Devuelve
    -------
    p_y : np.ndarray
        Estimación corregida de P(Y=1 | x).
    """

    scores = np.asarray(scores)

    if alpha_hat <= 0:
        raise ValueError("alpha_hat debe ser mayor que 0.")

    # Corrección teórica
    p_y = scores / alpha_hat

    # Recorte a rango [0, 1]
    p_y = np.clip(p_y, 0.0, 1.0)

    return p_y
