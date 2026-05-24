import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

def evaluar_clasificador_final(X_train, X_test, y_train, y_test, ranking, k):
    """
    Entrena un clasificador final usando las top-k características (ranking calculado
    exclusivamente sobre train) y evalúa el AUC sobre test.
    El split debe realizarse antes de llamar a esta función.
    """
    features_seleccionadas = ranking[:k]

    modelo = Pipeline([
        ("escalado", StandardScaler()),
        ("clasificador", LogisticRegression(max_iter=5000))
    ])

    modelo.fit(X_train[:, features_seleccionadas], y_train)
    probabilidades = modelo.predict_proba(X_test[:, features_seleccionadas])[:, 1]

    auc = roc_auc_score(y_test, probabilidades)

    return auc


def calcular_mi_naive(X, S, random_state=42):
    """
    Calcula MI usando S como si fuera la etiqueta real.
    """
    mi_scores = mutual_info_classif(
        X, S,
        random_state=random_state
    )

    ranking = np.argsort(mi_scores)[::-1]

    return mi_scores, ranking


def calcular_mi_real(X, y_real, random_state=42):
    """
    Calcula MI usando las etiquetas reales.
    """
    mi_scores = mutual_info_classif(
        X, y_real,
        random_state=random_state
    )

    ranking = np.argsort(mi_scores)[::-1]

    return mi_scores, ranking


def calcular_varianza(X):
    """
    Ranking basado en varianza (método no supervisado).
    """
    varianzas = np.var(X, axis=0)
    ranking = np.argsort(varianzas)[::-1]

    return varianzas, ranking


def comparar_metodos(
    X_train, X_test, y_train, y_test,
    ranking_pu,
    ranking_naive,
    ranking_real,
    ranking_varianza,
    k=10
):
    """
    Compara distintos métodos de selección usando AUC.
    Los rankings deben estar calculados exclusivamente sobre X_train.
    El AUC se evalúa sobre X_test.
    """

    resultados = {}

    resultados["PU_corregido"] = evaluar_clasificador_final(
        X_train, X_test, y_train, y_test, ranking_pu, k
    )

    resultados["MI_naive"] = evaluar_clasificador_final(
        X_train, X_test, y_train, y_test, ranking_naive, k
    )

    resultados["MI_real"] = evaluar_clasificador_final(
        X_train, X_test, y_train, y_test, ranking_real, k
    )

    resultados["Varianza"] = evaluar_clasificador_final(
        X_train, X_test, y_train, y_test, ranking_varianza, k
    )

    return resultados


def calcular_overlap(ranking_a, ranking_b, k):
    """
    Calcula el overlap entre dos rankings en posición top-k.
    
    Parámetros:
    -----------
    ranking_a, ranking_b: arrays de índices ordenados por importancia
    k: número de top features a comparar
    
    Devuelve:
    ---------
    dict con:
      - 'count': número de features en común
      - 'porcentaje': porcentaje de overlap (0-100%)
      - 'features_a_only': features solo en ranking_a
      - 'features_b_only': features solo en ranking_b
      - 'features_common': features en ambos rankings
    """
    top_k_a = set(ranking_a[:k])
    top_k_b = set(ranking_b[:k])
    
    common = top_k_a & top_k_b
    only_a = top_k_a - top_k_b
    only_b = top_k_b - top_k_a
    
    return {
        'count': len(common),
        'porcentaje': (len(common) / k) * 100,
        'features_a_only': only_a,
        'features_b_only': only_b,
        'features_common': common,
    }


def spearman_rankings(ranking_a, ranking_b):
    """
    Calcula la correlación de Spearman entre dos rankings completos.
    ranking_a, ranking_b: arrays de índices ordenados de mayor a menor importancia.
    Devuelve un float en [-1, 1]; 1 = rankings idénticos.
    """
    n = len(ranking_a)
    pos_a = np.empty(n, dtype=float)
    pos_b = np.empty(n, dtype=float)
    pos_a[ranking_a] = np.arange(n)
    pos_b[ranking_b] = np.arange(n)
    corr, _ = spearmanr(pos_a, pos_b)
    return float(corr)


def spearman_rankings_topk(ranking_a, ranking_b, k):
    """
    Calcula la correlación de Spearman solo sobre los top-k elementos de cada ranking.
    
    Parámetros:
    -----------
    ranking_a, ranking_b: arrays de índices ordenados por importancia (ranking completo)
    k: número de top elementos a considerar
    
    Devuelve:
    ---------
    float en [-1, 1]; 1 = rankings idénticos en top-k
    """
    # Obtener los top-k de cada ranking
    top_k_a = set(ranking_a[:k])
    top_k_b = set(ranking_b[:k])
    
    # Elementos comunes en top-k
    common = top_k_a & top_k_b
    
    if len(common) == 0:
        return np.nan
    
    # Calcular posiciones en top-k solo para elementos comunes
    pos_a = []
    pos_b = []
    
    for idx in common:
        # Posición en ranking_a (en el top-k)
        pos_a.append(list(ranking_a[:k]).index(idx))
        # Posición en ranking_b (en el top-k)
        pos_b.append(list(ranking_b[:k]).index(idx))
    
    corr, _ = spearmanr(pos_a, pos_b)
    return float(corr)