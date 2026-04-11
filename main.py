import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend sin ventana
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from src.data_utiles import generar_etiquetas_pu, añadir_ruido_gaussiano
from src.config import *
from src.datasets import load_dataset_from_config, load_dataset
from src.pu_model import entrenar_clasificador_pu, estimar_alpha, estimar_alpha_robusto, obtener_scores, estimar_probabilidad_real
from src.mi_utiles import calcular_mi_ranking, guardar_ranking
from src.evaluacion import comparar_metodos, calcular_mi_naive, calcular_mi_real, calcular_varianza, spearman_rankings


def _topk_overlap(ranking_a, ranking_b, k):
    return len(set(ranking_a[:k]) & set(ranking_b[:k]))


def _ranking_instability(rankings_list):
    """Devuelve la inestabilidad media del ranking entre semillas:
    std promedio de la posición de cada feature a lo largo de las semillas."""
    n = len(rankings_list[0])
    pos_matrix = np.zeros((len(rankings_list), n))
    for i, r in enumerate(rankings_list):
        pos_matrix[i, r] = np.arange(n)
    return float(np.mean(np.std(pos_matrix, axis=0)))


def _sweep_alpha(X, y, feature_names, estimation_method):
    """Sweep variando alpha (fijo top_q_percent)."""
    print(f"\n{'='*70}")
    print("EJECUTANDO SWEEP DE ALPHA")
    print(f"{'='*70}")
    
    alphas = SWEEP_ALPHAS
    seeds = SWEEP_SEEDS
    
    rows = []
    rankings_by_alpha = {}
    
    for alpha in alphas:
        for seed in seeds:
            X_noisy = añadir_ruido_gaussiano(X, NOISE_LEVEL, random_state=seed)
            S = generar_etiquetas_pu(y, alpha, random_state=seed)
            modelo = entrenar_clasificador_pu(X_noisy, S, random_state=seed)
            scores = obtener_scores(modelo, X_noisy)
            
            if estimation_method == 'robust':
                alpha_hat = estimar_alpha_robusto(scores, S, top_q_percent=ALPHA_TOP_Q_PERCENT)
            else:
                alpha_hat = estimar_alpha(scores, S)
            
            p_y = estimar_probabilidad_real(scores, alpha_hat)
            
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            fold_results = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_noisy, y)):
                X_train = X_noisy[train_idx]
                X_test = X_noisy[test_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]
                S_train = S[train_idx]
                S_test = S[test_idx]
                p_y_train = p_y[train_idx]
                
                mi_scores, ranking = calcular_mi_ranking(
                    X_train, p_y_train, metodo="regresion", random_state=seed
                )
                mi_naive_scores, ranking_naive = calcular_mi_naive(X_train, S_train)
                mi_real_scores,  ranking_real  = calcular_mi_real(X_train, y_train)
                var_scores,      ranking_var   = calcular_varianza(X_train)
                
                if RUN_MODE == 'single' and fold_idx == 0:
                    guardar_ranking("PU_corregido", ranking,       feature_names, TOP_K, mi_scores)
                    guardar_ranking("MI_naive",     ranking_naive, feature_names, TOP_K, mi_naive_scores)
                    guardar_ranking("MI_real",      ranking_real,  feature_names, TOP_K, mi_real_scores)
                    guardar_ranking("Varianza",     ranking_var,   feature_names, TOP_K, var_scores)
                
                aucs = comparar_metodos(
                    X_train, X_test, y_train, y_test,
                    ranking, ranking_naive, ranking_real, ranking_var, k=TOP_K
                )
                
                spearman_naive = spearman_rankings(ranking_naive, ranking_real)
                spearman_pu    = spearman_rankings(ranking,       ranking_real)
                spearman_var   = spearman_rankings(ranking_var,   ranking_real)
                spearman_real  = 1.0
                
                overlap_naive = _topk_overlap(ranking_naive, ranking_real, TOP_K)
                overlap_pu    = _topk_overlap(ranking, ranking_real, TOP_K)
                
                fold_results.append({
                    "alpha_hat": float(alpha_hat),
                    "overlap_naive": int(overlap_naive),
                    "overlap_pu": float(overlap_pu),
                    "spearman_naive": float(spearman_naive),
                    "spearman_pu": float(spearman_pu),
                    "spearman_var": float(spearman_var),
                    "spearman_real": float(spearman_real),
                    "auc_PU_corregido": float(aucs["PU_corregido"]),
                    "auc_MI_naive": float(aucs["MI_naive"]),
                    "auc_MI_real": float(aucs["MI_real"]),
                    "auc_Varianza": float(aucs["Varianza"]),
                    "ranking": ranking,
                    "ranking_naive": ranking_naive,
                    "ranking_real": ranking_real,
                    "ranking_var": ranking_var,
                })
            
            mean_alpha_hat = np.mean([r['alpha_hat'] for r in fold_results])
            mean_overlap_pu = np.mean([r['overlap_pu'] for r in fold_results])
            mean_overlap_naive = np.mean([r['overlap_naive'] for r in fold_results])
            mean_spearman_pu = np.mean([r['spearman_pu'] for r in fold_results])
            mean_spearman_naive = np.mean([r['spearman_naive'] for r in fold_results])
            mean_spearman_var = np.mean([r['spearman_var'] for r in fold_results])
            mean_auc_pu = np.mean([r['auc_PU_corregido'] for r in fold_results])
            mean_auc_naive = np.mean([r['auc_MI_naive'] for r in fold_results])
            mean_auc_real = np.mean([r['auc_MI_real'] for r in fold_results])
            mean_auc_var = np.mean([r['auc_Varianza'] for r in fold_results])
            
            if alpha not in rankings_by_alpha:
                rankings_by_alpha[alpha] = {'pu': [], 'naive': [], 'real': [], 'var': []}
            rankings_by_alpha[alpha]['pu'].append(fold_results[-1]['ranking'])
            rankings_by_alpha[alpha]['naive'].append(fold_results[-1]['ranking_naive'])
            rankings_by_alpha[alpha]['real'].append(fold_results[-1]['ranking_real'])
            rankings_by_alpha[alpha]['var'].append(fold_results[-1]['ranking_var'])
            
            rows.append({
                "alpha_true":         alpha,
                "seed":               seed,
                "alpha_hat":          float(mean_alpha_hat),
                "overlap_naive":      int(mean_overlap_naive),
                "overlap_pu":         float(mean_overlap_pu),
                "spearman_naive":     float(mean_spearman_naive),
                "spearman_pu":        float(mean_spearman_pu),
                "spearman_var":       float(mean_spearman_var),
                "spearman_real":      1.0,
                "auc_PU_corregido":   float(mean_auc_pu),
                "auc_MI_naive":       float(mean_auc_naive),
                "auc_MI_real":        float(mean_auc_real),
                "auc_Varianza":       float(mean_auc_var),
            })
    
    # Guardar y procesar resultados
    df = pd.DataFrame(rows)
    summary = df.groupby('alpha_true').agg(
        alpha_hat_mean=      ('alpha_hat',      'mean'),
        alpha_hat_std=       ('alpha_hat',      'std'),
        overlap_naive_mean=  ('overlap_naive',  'mean'),
        overlap_naive_std=   ('overlap_naive',  'std'),
        overlap_pu_mean=     ('overlap_pu',     'mean'),
        overlap_pu_std=      ('overlap_pu',     'std'),
        spearman_naive_mean= ('spearman_naive', 'mean'),
        spearman_naive_std=  ('spearman_naive', 'std'),
        spearman_pu_mean=    ('spearman_pu',    'mean'),
        spearman_pu_std=     ('spearman_pu',    'std'),
        spearman_var_mean=   ('spearman_var',   'mean'),
        spearman_var_std=    ('spearman_var',   'std'),
        spearman_real_mean=  ('spearman_real',  'mean'),
        spearman_real_std=   ('spearman_real',  'std'),
    ).reset_index()
    
    stability_rows = []
    for alpha_val, mrankings in sorted(rankings_by_alpha.items()):
        stability_rows.append({
            'alpha_true':      alpha_val,
            'stability_pu':    _ranking_instability(mrankings['pu']),
            'stability_naive': _ranking_instability(mrankings['naive']),
            'stability_real':  _ranking_instability(mrankings['real']),
            'stability_var':   _ranking_instability(mrankings['var']),
        })
    stability_df = pd.DataFrame(stability_rows)
    summary = summary.merge(stability_df, on='alpha_true')
    
    df.to_csv('alpha_sweep_runs.csv',    index=False)
    summary.to_csv('alpha_sweep_summary.csv', index=False)
    
    mlflow.log_param('sweep_alphas', str(alphas))
    mlflow.log_param('sweep_seeds',  str(seeds))
    mlflow.log_param('n_runs',       int(len(rows)))
    
    mlflow.log_artifact('alpha_sweep_runs.csv')
    mlflow.log_artifact('alpha_sweep_summary.csv')
    
    for _, row in summary.iterrows():
        a = row['alpha_true']
        mlflow.log_metric(f'alpha_{a}_hat_mean',             float(row['alpha_hat_mean']))
        mlflow.log_metric(f'alpha_{a}_hat_std',              float(row['alpha_hat_std']))
        mlflow.log_metric(f'alpha_{a}_overlap_naive_mean',   float(row['overlap_naive_mean']))
        mlflow.log_metric(f'alpha_{a}_overlap_pu_mean',      float(row['overlap_pu_mean']))
        mlflow.log_metric(f'alpha_{a}_spearman_naive_mean',  float(row['spearman_naive_mean']))
        mlflow.log_metric(f'alpha_{a}_spearman_pu_mean',     float(row['spearman_pu_mean']))
        mlflow.log_metric(f'alpha_{a}_stability_pu',         float(row['stability_pu']))
        mlflow.log_metric(f'alpha_{a}_stability_naive',      float(row['stability_naive']))
    
    # Gráfico 1: AUC vs alpha
    metodos = ['PU_corregido', 'MI_naive', 'MI_real', 'Varianza']
    auc_summary = df.groupby('alpha_true')[[f'auc_{m}' for m in metodos]].agg(['mean', 'std'])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for metodo in metodos:
        means = auc_summary[(f'auc_{metodo}', 'mean')].values
        stds  = auc_summary[(f'auc_{metodo}', 'std')].values
        x     = auc_summary.index.values
        ax.plot(x, means, marker='o', label=metodo)
        ax.fill_between(x, means - stds, means + stds, alpha=0.15)
    
    ax.set_xlabel('Alpha verdadero')
    ax.set_ylabel(f'AUC (top-{TOP_K} features)')
    ax.set_title('AUC vs Alpha por método de selección')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig('auc_vs_alpha.png', dpi=150)
    plt.close(fig)
    mlflow.log_artifact('auc_vs_alpha.png')
    
    # Gráfico 2: Spearman vs alpha
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for label, col_mean, col_std in [
        ('MI_real (ref.)', 'spearman_real_mean',  'spearman_real_std'),
        ('PU_corregido',   'spearman_pu_mean',    'spearman_pu_std'),
        ('MI_naive',       'spearman_naive_mean', 'spearman_naive_std'),
        ('Varianza',       'spearman_var_mean',   'spearman_var_std'),
    ]:
        means = summary[col_mean].values
        stds  = summary[col_std].values
        x     = summary['alpha_true'].values
        ax2.plot(x, means, marker='o', label=label)
        ax2.fill_between(x, means - stds, means + stds, alpha=0.15)
    
    ax2.set_xlabel('Alpha verdadero')
    ax2.set_ylabel('Correlación de Spearman con MI_real')
    ax2.set_title('Correlación de Spearman vs Alpha')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    fig2.tight_layout()
    fig2.savefig('spearman_vs_alpha.png', dpi=150)
    plt.close(fig2)
    mlflow.log_artifact('spearman_vs_alpha.png')
    
    # Gráfico 3: Estabilidad vs alpha
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for label, col in [
        ('PU_corregido', 'stability_pu'),
        ('MI_naive',     'stability_naive'),
        ('MI_real',      'stability_real'),
        ('Varianza',     'stability_var'),
    ]:
        ax3.plot(stability_df['alpha_true'].values,
                 stability_df[col].values,
                 marker='o', label=label)
    
    ax3.set_xlabel('Alpha verdadero')
    ax3.set_ylabel('Inestabilidad (std promedio de posición entre semillas)')
    ax3.set_title('Estabilidad del ranking vs Alpha')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)
    fig3.tight_layout()
    fig3.savefig('stability_vs_alpha.png', dpi=150)
    plt.close(fig3)
    mlflow.log_artifact('stability_vs_alpha.png')
    
    print('\nAlpha sweep - runs saved to alpha_sweep_runs.csv')
    print('Summary saved to alpha_sweep_summary.csv\n')
    print(summary.to_string(index=False))


def _sweep_percent(X, y, feature_names, estimation_method):
    """Sweep variando top_q_percent con método 'robust' (varía el percentil usado para estimar alpha)."""
    print(f"\n{'='*70}")
    print("EJECUTANDO SWEEP DE TOP_Q_PERCENT")
    print(f"{'='*70}")
    
    top_q_percents = TOP_Q_PERCENT_VALUES
    seeds = SWEEP_SEEDS
    alpha_true = ALPHA_TRUE
    
    rows = []
    
    for top_q_percent in top_q_percents:
        for seed in seeds:
            X_noisy = añadir_ruido_gaussiano(X, NOISE_LEVEL, random_state=seed)
            S = generar_etiquetas_pu(y, alpha_true, random_state=seed)
            
            X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(
                X_noisy, y, S, test_size=0.3, random_state=seed, stratify=y
            )
            
            modelo = entrenar_clasificador_pu(X_train, S_train, random_state=seed)
            scores_train = obtener_scores(modelo, X_train)
            
            # En _sweep_percent se varía el percentil con 'robust'
            alpha_hat = estimar_alpha_robusto(scores_train, S_train, top_q_percent=top_q_percent)
            
            p_y = estimar_probabilidad_real(scores_train, alpha_hat)
            _, ranking_real = calcular_mi_real(X_train, y_train)
            top_k_features = ranking_real[:TOP_K]
            
            X_test_selected = X_test[:, top_k_features]
            X_train_selected = X_train[:, top_k_features]
            
            lr = LogisticRegression(random_state=seed, max_iter=1000)
            lr.fit(X_train_selected, y_train)
            auc = roc_auc_score(y_test, lr.predict_proba(X_test_selected)[:, 1])
            
            spearman_corr, _ = spearmanr(p_y, y_train)
            
            rows.append({
                "top_q_percent": top_q_percent,
                "seed": seed,
                "alpha_hat": float(alpha_hat),
                "alpha_true": alpha_true,
                "auc": float(auc),
                "spearman": float(spearman_corr),
            })
    
    # Guardar y procesar resultados
    df = pd.DataFrame(rows)
    df.to_csv('top_q_percent_sweep_runs.csv', index=False)
    
    summary = df.groupby('top_q_percent').agg({
        'alpha_hat': ['mean', 'std'],
        'auc': ['mean', 'std'],
        'spearman': ['mean', 'std'],
    }).reset_index()
    summary.columns = ['top_q_percent', 'alpha_hat_mean', 'alpha_hat_std', 
                       'auc_mean', 'auc_std', 'spearman_mean', 'spearman_std']
    summary.to_csv('top_q_percent_sweep_summary.csv', index=False)
    
    mlflow.log_param("dataset", DATASET)
    mlflow.log_param("alpha_true", alpha_true)
    mlflow.log_param("alpha_estimation_method", estimation_method)
    mlflow.log_param("top_k_features", TOP_K)
    mlflow.log_param("top_q_percents", str(top_q_percents))
    mlflow.log_param("seeds", str(list(seeds)))
    mlflow.log_param("n_runs", len(rows))
    
    # Gráficas 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = summary['top_q_percent'].values
    
    # Gráfico 1: Alpha estimado vs Top Q Percent
    ax = axes[0, 0]
    ax.plot(x, summary['alpha_hat_mean'].values, marker='o', linewidth=2, markersize=8, color='blue')
    ax.fill_between(x, 
                    summary['alpha_hat_mean'].values - summary['alpha_hat_std'].values,
                    summary['alpha_hat_mean'].values + summary['alpha_hat_std'].values,
                    alpha=0.2, color='blue')
    ax.axhline(y=alpha_true, color='red', linestyle='--', linewidth=2, label=f'α_true={alpha_true}')
    ax.set_xlabel('Top Q Percent (%)', fontsize=11)
    ax.set_ylabel('Alpha Estimado', fontsize=11)
    ax.set_title('Alpha Estimado vs Top Q Percent', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(top_q_percents)
    
    # Gráfico 2: AUC vs Top Q Percent
    ax = axes[0, 1]
    ax.plot(x, summary['auc_mean'].values, marker='s', linewidth=2, markersize=8, color='green')
    ax.fill_between(x,
                    summary['auc_mean'].values - summary['auc_std'].values,
                    summary['auc_mean'].values + summary['auc_std'].values,
                    alpha=0.2, color='green')
    ax.set_xlabel('Top Q Percent (%)', fontsize=11)
    ax.set_ylabel(f'AUC (top-{TOP_K} features)', fontsize=11)
    ax.set_title('AUC vs Top Q Percent', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(top_q_percents)
    
    # Gráfico 3: Spearman vs Top Q Percent
    ax = axes[1, 0]
    ax.plot(x, summary['spearman_mean'].values, marker='^', linewidth=2, markersize=8, color='purple')
    ax.fill_between(x,
                    summary['spearman_mean'].values - summary['spearman_std'].values,
                    summary['spearman_mean'].values + summary['spearman_std'].values,
                    alpha=0.2, color='purple')
    ax.set_xlabel('Top Q Percent (%)', fontsize=11)
    ax.set_ylabel('Spearman Correlation', fontsize=11)
    ax.set_title('Spearman Correlation vs Top Q Percent', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(top_q_percents)
    
    # Gráfico 4: Error relativo de alpha
    ax = axes[1, 1]
    error_alpha = np.abs(summary['alpha_hat_mean'].values - alpha_true) / alpha_true * 100
    ax.bar(x, error_alpha, width=2, color='orange', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Top Q Percent (%)', fontsize=11)
    ax.set_ylabel('Error Relativo de Alpha (%)', fontsize=11)
    ax.set_title('Error Relativo de Alpha vs Top Q Percent', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(top_q_percents)
    
    plt.tight_layout()
    plt.savefig('top_q_percent_sweep.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Log de métricas
    for _, row in summary.iterrows():
        q = int(row['top_q_percent'])
        alpha_mean = row['alpha_hat_mean']
        alpha_std = row['alpha_hat_std']
        auc_mean = row['auc_mean']
        auc_std = row['auc_std']
        spearman_mean = row['spearman_mean']
        spearman_std = row['spearman_std']
        error_alpha_rel = np.abs(alpha_mean - alpha_true) / alpha_true * 100
        
        mlflow.log_metric(f"top_q_{q}_alpha_hat_mean", float(alpha_mean))
        mlflow.log_metric(f"top_q_{q}_alpha_hat_std", float(alpha_std))
        mlflow.log_metric(f"top_q_{q}_auc_mean", float(auc_mean))
        mlflow.log_metric(f"top_q_{q}_auc_std", float(auc_std))
        mlflow.log_metric(f"top_q_{q}_spearman_mean", float(spearman_mean))
        mlflow.log_metric(f"top_q_{q}_spearman_std", float(spearman_std))
        mlflow.log_metric(f"top_q_{q}_error_alpha_rel", float(error_alpha_rel))
    
    # Log de artefactos
    mlflow.log_artifact('top_q_percent_sweep_runs.csv')
    mlflow.log_artifact('top_q_percent_sweep_summary.csv')
    mlflow.log_artifact('top_q_percent_sweep.png')
    
    print("\n" + "="*70)
    print("RESUMEN DE ESTIMACIONES")
    print("="*70)
    print(f"Dataset: {DATASET}")
    print(f"Método: {estimation_method.upper()}")
    print(f"Alpha verdadero: {alpha_true}")
    print(f"Semillas: {len(seeds)}")
    print("="*70)
    
    print(f"\n{'Top_Q_%':<10} {'Alpha_Est':<12} {'Error_%':<12} {'AUC':<10} {'Spearman':<10}")
    print("-"*54)
    for _, row in summary.iterrows():
        q = int(row['top_q_percent'])
        alpha_mean = row['alpha_hat_mean']
        error = np.abs(alpha_mean - alpha_true) / alpha_true * 100
        auc = row['auc_mean']
        spear = row['spearman_mean']
        print(f"{q:<10} {alpha_mean:<12.4f} {error:<12.2f} {auc:<10.4f} {spear:<10.4f}")


def main():
    """Ejecuta experimento según el modo configurado (single o sweep)."""
    X, y, feature_names, ds_meta = load_dataset_from_config()

    # Validación: SWEEP_MODE='percent' REQUIERE ALPHA_ESTIMATION_METHOD='robust'
    # (de lo contrario top_q_percent no se usa y todo se repite)
    if RUN_MODE == 'sweep' and SWEEP_MODE == 'percent':
        if ALPHA_ESTIMATION_METHOD != 'robust':
            print(
                f"ERROR: SWEEP_MODE='percent' requiere ALPHA_ESTIMATION_METHOD='robust'\n"
                f"(de lo contrario el top_q_percent no se usa y los resultados se repiten)\n"
                f"Se aplica: ALPHA_ESTIMATION_METHOD = 'robust'"
            )
        estimation_method = 'robust'
    else:
        estimation_method = ALPHA_ESTIMATION_METHOD

    # En modo single se itera una sola vez; en sweep se recorre según SWEEP_MODE
    alphas = SWEEP_ALPHAS if RUN_MODE == 'sweep' else [ALPHA_TRUE]
    seeds  = SWEEP_SEEDS  if RUN_MODE == 'sweep' else [RANDOM_STATE]

    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        client.create_experiment(EXPERIMENT_NAME, tags={"mlflow.experimentType": "TRAINING"})
    elif "mlflow.experimentType" not in (exp.tags or {}):
        client.set_experiment_tag(exp.experiment_id, "mlflow.experimentType", "TRAINING")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):
        # Estructura del dataset
        n_samples, n_features = X.shape
        n_positives = int(y.sum())
        n_negatives = int((y == 0).sum())
        class_balance = round(n_positives / n_samples, 4)

        mlflow.log_param("dataset",        ds_meta.get("dataset_id", DATASET))
        mlflow.log_param("dataset_kind",   ds_meta.get("kind", ""))
        mlflow.log_param("positive_label", str(ds_meta.get("positive_label", "")))
        mlflow.log_param("n_samples",      n_samples)
        mlflow.log_param("n_features",     n_features)
        mlflow.log_param("n_positives",    n_positives)
        mlflow.log_param("n_negatives",    n_negatives)
        mlflow.log_param("class_balance",  class_balance)

        # Estadísticas descriptivas de features
        stats = pd.DataFrame({
            "feature":  feature_names,
            "mean":     np.mean(X, axis=0),
            "std":      np.std(X, axis=0),
            "min":      np.min(X, axis=0),
            "max":      np.max(X, axis=0),
            "median":   np.median(X, axis=0),
        })
        stats_path = "dataset_feature_stats.csv"
        stats.to_csv(stats_path, index=False)
        mlflow.log_artifact(stats_path)

        # Configuración general
        mlflow.log_param("run_mode",    RUN_MODE)
        mlflow.log_param("noise_level", NOISE_LEVEL)
        mlflow.log_param("top_k",       TOP_K)
        mlflow.log_param("alpha_estimation_method", estimation_method)
        
        if RUN_MODE == 'single':
            mlflow.log_param("alpha_true", ALPHA_TRUE)
            mlflow.log_param("random_state", RANDOM_STATE)
            
            # Modo single: ejecutar una sola vez con parámetros fijos
            X_noisy = añadir_ruido_gaussiano(X, NOISE_LEVEL, random_state=RANDOM_STATE)
            S = generar_etiquetas_pu(y, ALPHA_TRUE, random_state=RANDOM_STATE)
            mlflow.log_metric("n_labeled_positive", int(sum(S)))
            mlflow.log_metric("n_unlabeled", int(len(S) - sum(S)))

            modelo = entrenar_clasificador_pu(X_noisy, S, random_state=RANDOM_STATE)
            mlflow.sklearn.log_model(modelo, "pu_model")

            scores = obtener_scores(modelo, X_noisy)
            if estimation_method == 'robust':
                alpha_hat = estimar_alpha_robusto(scores, S, top_q_percent=ALPHA_TOP_Q_PERCENT)
            else:
                alpha_hat = estimar_alpha(scores, S)
            mlflow.log_metric("alpha_estimated", float(alpha_hat))

            p_y = estimar_probabilidad_real(scores, alpha_hat)
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            
            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_noisy, y)):
                X_train = X_noisy[train_idx]
                X_test = X_noisy[test_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]
                S_train = S[train_idx]
                p_y_train = p_y[train_idx]
                
                mi_scores, ranking = calcular_mi_ranking(
                    X_train, p_y_train, metodo="regresion", random_state=RANDOM_STATE
                )
                mi_naive_scores, ranking_naive = calcular_mi_naive(X_train, S_train)
                mi_real_scores,  ranking_real  = calcular_mi_real(X_train, y_train)
                var_scores,      ranking_var   = calcular_varianza(X_train)
                
                if fold_idx == 0:
                    guardar_ranking("PU_corregido", ranking,       feature_names, TOP_K, mi_scores)
                    guardar_ranking("MI_naive",     ranking_naive, feature_names, TOP_K, mi_naive_scores)
                    guardar_ranking("MI_real",      ranking_real,  feature_names, TOP_K, mi_real_scores)
                    guardar_ranking("Varianza",     ranking_var,   feature_names, TOP_K, var_scores)
                
                aucs = comparar_metodos(
                    X_train, X_test, y_train, y_test,
                    ranking, ranking_naive, ranking_real, ranking_var, k=TOP_K
                )
                
                print("\nResultados AUC (Fold 1, Top", TOP_K, "features):")
                print(f"  PU_corregido: {aucs['PU_corregido']:.4f}")
                print(f"  MI_naive: {aucs['MI_naive']:.4f}")
                print(f"  MI_real: {aucs['MI_real']:.4f}")
                print(f"  Varianza: {aucs['Varianza']:.4f}")
                break
        
        elif RUN_MODE == 'sweep':
            mlflow.log_param("sweep_mode", SWEEP_MODE)
            
            if SWEEP_MODE == 'alpha':
                _sweep_alpha(X, y, feature_names, estimation_method)
            elif SWEEP_MODE == 'percent':
                _sweep_percent(X, y, feature_names, estimation_method)
            else:
                raise ValueError(f"SWEEP_MODE inválido: {SWEEP_MODE}. Debe ser 'alpha' o 'percent'")

        print("\nRun guardado en MLflow")


if __name__ == "__main__":
    main()
    