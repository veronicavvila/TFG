import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend sin ventana
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from src.data_utiles import generar_etiquetas_pu, añadir_ruido_gaussiano
from src.config import *
from src.datasets import load_dataset_from_config, load_dataset, crear_splitter_cv
from src.pu_model import entrenar_clasificador_pu, estimar_alpha, estimar_alpha_robusto, obtener_scores, estimar_probabilidad_real
from src.mi_utiles import calcular_mi_ranking, guardar_ranking
from src.evaluacion import comparar_metodos, calcular_mi_naive, calcular_mi_real, calcular_varianza, spearman_rankings, spearman_rankings_topk


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


def _sweep_alpha(X, y, feature_names, estimation_method, dataset_kind, output_dir):
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
            
            kfold = crear_splitter_cv(dataset_kind, n_splits=5, random_state=seed)
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
                
                # Spearman completo (sobre todas las características)
                spearman_naive_full = spearman_rankings(ranking_naive, ranking_real)
                spearman_pu_full    = spearman_rankings(ranking,       ranking_real)
                spearman_var_full   = spearman_rankings(ranking_var,   ranking_real)
                
                # Spearman top-k 
                spearman_naive_topk = spearman_rankings_topk(ranking_naive, ranking_real, TOP_K)
                spearman_pu_topk    = spearman_rankings_topk(ranking,       ranking_real, TOP_K)
                spearman_var_topk   = spearman_rankings_topk(ranking_var,   ranking_real, TOP_K)
                
                overlap_naive = _topk_overlap(ranking_naive, ranking_real, TOP_K)
                overlap_pu    = _topk_overlap(ranking, ranking_real, TOP_K)
                
                fold_results.append({
                    "alpha_hat": float(alpha_hat),
                    "overlap_naive": int(overlap_naive),
                    "overlap_pu": float(overlap_pu),
                    "spearman_naive_full": float(spearman_naive_full),
                    "spearman_pu_full": float(spearman_pu_full),
                    "spearman_var_full": float(spearman_var_full),
                    "spearman_naive_topk": float(spearman_naive_topk),
                    "spearman_pu_topk": float(spearman_pu_topk),
                    "spearman_var_topk": float(spearman_var_topk),
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
            mean_spearman_pu_full = np.mean([r['spearman_pu_full'] for r in fold_results])
            mean_spearman_naive_full = np.mean([r['spearman_naive_full'] for r in fold_results])
            mean_spearman_var_full = np.mean([r['spearman_var_full'] for r in fold_results])
            mean_spearman_pu_topk = np.mean([r['spearman_pu_topk'] for r in fold_results])
            mean_spearman_naive_topk = np.mean([r['spearman_naive_topk'] for r in fold_results])
            mean_spearman_var_topk = np.mean([r['spearman_var_topk'] for r in fold_results])
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
                "spearman_naive_full": float(mean_spearman_naive_full),
                "spearman_pu_full": float(mean_spearman_pu_full),
                "spearman_var_full": float(mean_spearman_var_full),
                "spearman_naive_topk": float(mean_spearman_naive_topk),
                "spearman_pu_topk": float(mean_spearman_pu_topk),
                "spearman_var_topk": float(mean_spearman_var_topk),
                "auc_PU_corregido":   float(mean_auc_pu),
                "auc_MI_naive":       float(mean_auc_naive),
                "auc_MI_real":        float(mean_auc_real),
                "auc_Varianza":       float(mean_auc_var),
            })
    
    # Guardar y procesar resultados
    df = pd.DataFrame(rows)
    summary = df.groupby('alpha_true').agg(
        alpha_hat_mean=        ('alpha_hat',        'mean'),
        alpha_hat_std=         ('alpha_hat',        'std'),
        overlap_naive_mean=    ('overlap_naive',    'mean'),
        overlap_naive_std=     ('overlap_naive',    'std'),
        overlap_pu_mean=       ('overlap_pu',       'mean'),
        overlap_pu_std=        ('overlap_pu',       'std'),
        auc_PU_corregido_mean= ('auc_PU_corregido', 'mean'),
        auc_PU_corregido_std=  ('auc_PU_corregido', 'std'),
        auc_MI_naive_mean=     ('auc_MI_naive',     'mean'),
        auc_MI_naive_std=      ('auc_MI_naive',     'std'),
        auc_MI_real_mean=      ('auc_MI_real',      'mean'),
        auc_MI_real_std=       ('auc_MI_real',      'std'),
        auc_Varianza_mean=     ('auc_Varianza',     'mean'),
        auc_Varianza_std=      ('auc_Varianza',     'std'),
        spearman_naive_full_mean= ('spearman_naive_full', 'mean'),
        spearman_naive_full_std=  ('spearman_naive_full', 'std'),
        spearman_pu_full_mean=    ('spearman_pu_full',    'mean'),
        spearman_pu_full_std=     ('spearman_pu_full',    'std'),
        spearman_var_full_mean=   ('spearman_var_full',   'mean'),
        spearman_var_full_std=    ('spearman_var_full',   'std'),
        spearman_naive_topk_mean= ('spearman_naive_topk', 'mean'),
        spearman_naive_topk_std=  ('spearman_naive_topk', 'std'),
        spearman_pu_topk_mean=    ('spearman_pu_topk',    'mean'),
        spearman_pu_topk_std=     ('spearman_pu_topk',    'std'),
        spearman_var_topk_mean=   ('spearman_var_topk',   'mean'),
        spearman_var_topk_std=    ('spearman_var_topk',   'std'),
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
    
    summary_cols = [
        'alpha_true',
        'alpha_hat_mean', 'alpha_hat_std',
        'spearman_pu_topk_mean', 'spearman_pu_topk_std',
        'spearman_naive_topk_mean', 'spearman_naive_topk_std',
        'spearman_var_topk_mean', 'spearman_var_topk_std',
        'stability_pu', 'stability_naive', 'stability_real', 'stability_var',
    ]
    auc_cols = [
        'alpha_true',
        'auc_PU_corregido_mean', 'auc_PU_corregido_std',
        'auc_MI_naive_mean', 'auc_MI_naive_std',
        'auc_MI_real_mean', 'auc_MI_real_std',
        'auc_Varianza_mean', 'auc_Varianza_std',
    ]

    df.to_csv(os.path.join(output_dir, 'alpha_sweep_runs.csv'), index=False)
    summary[summary_cols].to_csv(os.path.join(output_dir, 'alpha_sweep_summary.csv'), index=False)
    summary[auc_cols].to_csv(os.path.join(output_dir, 'auc_summary.csv'), index=False)

    mlflow.log_param('sweep_alphas', str(alphas))
    mlflow.log_param('sweep_seeds',  str(seeds))
    mlflow.log_param('n_runs',       int(len(rows)))
    if estimation_method == 'robust':
        mlflow.log_param('alpha_top_q_percent', ALPHA_TOP_Q_PERCENT)

    mlflow.log_artifact(os.path.join(output_dir, 'alpha_sweep_runs.csv'))
    mlflow.log_artifact(os.path.join(output_dir, 'alpha_sweep_summary.csv'))
    mlflow.log_artifact(os.path.join(output_dir, 'auc_summary.csv'))
    
    for _, row in summary.iterrows():
        a = row['alpha_true']
        mlflow.log_metric(f'alpha_{a}_hat_mean',             float(row['alpha_hat_mean']))
        mlflow.log_metric(f'alpha_{a}_hat_std',              float(row['alpha_hat_std']))
        mlflow.log_metric(f'alpha_{a}_overlap_naive_mean',   float(row['overlap_naive_mean']))
        mlflow.log_metric(f'alpha_{a}_overlap_pu_mean',      float(row['overlap_pu_mean']))
        mlflow.log_metric(f'alpha_{a}_spearman_naive_full_mean',  float(row['spearman_naive_full_mean']))
        mlflow.log_metric(f'alpha_{a}_spearman_naive_topk_mean',  float(row['spearman_naive_topk_mean']))
        mlflow.log_metric(f'alpha_{a}_spearman_pu_full_mean',     float(row['spearman_pu_full_mean']))
        mlflow.log_metric(f'alpha_{a}_spearman_pu_topk_mean',     float(row['spearman_pu_topk_mean']))
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
    fig.savefig(os.path.join(output_dir, 'auc_vs_alpha.png'), dpi=150)
    plt.close(fig)
    mlflow.log_artifact(os.path.join(output_dir, 'auc_vs_alpha.png'))
    
    # Gráfico 2: Spearman vs Alpha (FULL y TOP-K comparadas)
    fig2, (ax2_full, ax2_topk) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Definir colores para cada método
    colores = {
        'PU_corregido': '#FF8C42',     # Naranja
        'MI_naive': '#2ECC71',          # Verde
        'Varianza': '#E74C3C',          # Rojo
        'MI_real': '#3498DB',           # Azul
    }
    
    # Gráfico 2a: Spearman FULL (todas las características)
    for label, col_mean, col_std in [
        ('PU_corregido',   'spearman_pu_full_mean',    'spearman_pu_full_std'),
        ('MI_naive',       'spearman_naive_full_mean', 'spearman_naive_full_std'),
        ('Varianza',       'spearman_var_full_mean',   'spearman_var_full_std'),
    ]:
        means = summary[col_mean].values
        stds  = summary[col_std].values
        x     = summary['alpha_true'].values
        ax2_full.plot(x, means, marker='o', label=label, color=colores[label], linewidth=2)
        ax2_full.fill_between(x, means - stds, means + stds, alpha=0.15, color=colores[label])
    
    # Línea de referencia: MI_real siempre tiene Spearman=1.0
    ax2_full.axhline(y=1.0, color=colores['MI_real'], linestyle='--', linewidth=2, label='MI_real (ref.)', alpha=0.7)
    
    ax2_full.set_xlabel('Alpha verdadero')
    ax2_full.set_ylabel('Correlación de Spearman')
    ax2_full.set_title(f'Spearman FULL')
    ax2_full.set_ylim(0, 1)
    ax2_full.legend()
    ax2_full.grid(True, linestyle='--', alpha=0.5)
    
    # Gráfico 2b: Spearman TOP-K (solo características relevantes)
    for label, col_mean, col_std in [
        ('PU_corregido',   'spearman_pu_topk_mean',    'spearman_pu_topk_std'),
        ('MI_naive',       'spearman_naive_topk_mean', 'spearman_naive_topk_std'),
        ('Varianza',       'spearman_var_topk_mean',   'spearman_var_topk_std'),
    ]:
        means = summary[col_mean].values
        stds  = summary[col_std].values
        x     = summary['alpha_true'].values
        ax2_topk.plot(x, means, marker='s', label=label, color=colores[label], linewidth=2)
        ax2_topk.fill_between(x, means - stds, means + stds, alpha=0.15, color=colores[label])
    
    # Línea de referencia: MI_real siempre tiene Spearman=1.0
    ax2_topk.axhline(y=1.0, color=colores['MI_real'], linestyle='--', linewidth=2, label='MI_real (ref.)', alpha=0.7)
    
    ax2_topk.set_xlabel('Alpha verdadero')
    ax2_topk.set_ylabel('Correlación de Spearman')
    ax2_topk.set_title(f'Spearman TOP-{TOP_K}')
    ax2_topk.set_ylim(0, 1)
    ax2_topk.legend()
    ax2_topk.grid(True, linestyle='--', alpha=0.5)
    
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'spearman_vs_alpha.png'), dpi=150)
    plt.close(fig2)
    mlflow.log_artifact(os.path.join(output_dir, 'spearman_vs_alpha.png'))
    
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
    fig3.savefig(os.path.join(output_dir, 'stability_vs_alpha.png'), dpi=150)
    plt.close(fig3)
    mlflow.log_artifact(os.path.join(output_dir, 'stability_vs_alpha.png'))
    
    print('\nAlpha sweep - runs saved to alpha_sweep_runs.csv')
    print('Summary saved to alpha_sweep_summary.csv\n')
    print(summary.to_string(index=False))


def _sweep_percent(X, y, feature_names, estimation_method, dataset_kind, output_dir):
    """Sweep variando top_q_percent con método 'robust'.

    Pipeline idéntico a _sweep_alpha:
    - PU model + alpha_hat + p_y calculados sobre datos completos (antes del KFold)
    - KFold: dentro de cada fold se calculan los 4 rankings sobre X_train
    - comparar_metodos evalúa AUC de los 4 métodos sobre X_test
    """
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

            # Modelo PU y alpha estimados sobre datos completos (único valor por seed/top_q_percent)
            modelo = entrenar_clasificador_pu(X_noisy, S, random_state=seed)
            scores = obtener_scores(modelo, X_noisy)
            alpha_hat = estimar_alpha_robusto(scores, S, top_q_percent=top_q_percent)
            p_y = estimar_probabilidad_real(scores, alpha_hat)

            kfold = crear_splitter_cv(dataset_kind, n_splits=5, random_state=seed)
            fold_results = []

            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_noisy, y)):
                X_train = X_noisy[train_idx]
                X_test  = X_noisy[test_idx]
                y_train = y[train_idx]
                y_test  = y[test_idx]
                S_train = S[train_idx]
                p_y_train = p_y[train_idx]

                # 4 rankings calculados exclusivamente sobre X_train
                mi_scores_pu,    ranking_pu    = calcular_mi_ranking(X_train, p_y_train, metodo="regresion", random_state=seed)
                mi_naive_scores, ranking_naive = calcular_mi_naive(X_train, S_train)
                mi_real_scores,  ranking_real  = calcular_mi_real(X_train, y_train)
                var_scores,      ranking_var   = calcular_varianza(X_train)

                aucs = comparar_metodos(
                    X_train, X_test, y_train, y_test,
                    ranking_pu, ranking_naive, ranking_real, ranking_var, k=TOP_K
                )

                spearman_pu_full    = spearman_rankings(ranking_pu,    ranking_real)
                spearman_naive_full = spearman_rankings(ranking_naive, ranking_real)
                spearman_var_full   = spearman_rankings(ranking_var,   ranking_real)

                spearman_pu_topk    = spearman_rankings_topk(ranking_pu,    ranking_real, TOP_K)
                spearman_naive_topk = spearman_rankings_topk(ranking_naive, ranking_real, TOP_K)
                spearman_var_topk   = spearman_rankings_topk(ranking_var,   ranking_real, TOP_K)

                fold_results.append({
                    "auc_PU_corregido":    float(aucs["PU_corregido"]),
                    "auc_MI_naive":        float(aucs["MI_naive"]),
                    "auc_MI_real":         float(aucs["MI_real"]),
                    "auc_Varianza":        float(aucs["Varianza"]),
                    "spearman_pu_full":    float(spearman_pu_full),
                    "spearman_naive_full": float(spearman_naive_full),
                    "spearman_var_full":   float(spearman_var_full),
                    "spearman_pu_topk":    float(spearman_pu_topk),
                    "spearman_naive_topk": float(spearman_naive_topk),
                    "spearman_var_topk":   float(spearman_var_topk),
                })

            rows.append({
                "top_q_percent":       top_q_percent,
                "seed":                seed,
                "alpha_true":          alpha_true,
                "alpha_hat":           float(alpha_hat),
                "auc_PU_corregido":    float(np.mean([r["auc_PU_corregido"]    for r in fold_results])),
                "auc_MI_naive":        float(np.mean([r["auc_MI_naive"]        for r in fold_results])),
                "auc_MI_real":         float(np.mean([r["auc_MI_real"]         for r in fold_results])),
                "auc_Varianza":        float(np.mean([r["auc_Varianza"]        for r in fold_results])),
                "spearman_pu_full":    float(np.mean([r["spearman_pu_full"]    for r in fold_results])),
                "spearman_naive_full": float(np.mean([r["spearman_naive_full"] for r in fold_results])),
                "spearman_var_full":   float(np.mean([r["spearman_var_full"]   for r in fold_results])),
                "spearman_pu_topk":    float(np.mean([r["spearman_pu_topk"]    for r in fold_results])),
                "spearman_naive_topk": float(np.mean([r["spearman_naive_topk"] for r in fold_results])),
                "spearman_var_topk":   float(np.mean([r["spearman_var_topk"]   for r in fold_results])),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'top_q_percent_sweep_runs.csv'), index=False)

    # Summary: alpha_hat, AUCs y Spearman top-k agrupados por top_q_percent
    metodos = ['PU_corregido', 'MI_naive', 'MI_real', 'Varianza']
    summary_rows = []
    for q in top_q_percents:
        subset = df[df['top_q_percent'] == q]
        row = {
            'top_q_percent':   q,
            'alpha_hat_mean':  float(subset['alpha_hat'].mean()),
            'alpha_hat_std':   float(subset['alpha_hat'].std()),
        }
        for m in metodos:
            row[f'auc_{m}_mean'] = float(subset[f'auc_{m}'].mean())
            row[f'auc_{m}_std']  = float(subset[f'auc_{m}'].std())
        for col in ['spearman_pu_topk', 'spearman_naive_topk', 'spearman_var_topk']:
            row[f'{col}_mean'] = float(subset[col].mean())
            row[f'{col}_std']  = float(subset[col].std())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    summary_cols = ['top_q_percent', 'alpha_hat_mean', 'alpha_hat_std',
                    'spearman_pu_topk_mean', 'spearman_pu_topk_std',
                    'spearman_naive_topk_mean', 'spearman_naive_topk_std',
                    'spearman_var_topk_mean', 'spearman_var_topk_std']
    auc_cols = ['top_q_percent'] + [c for m in metodos for c in (f'auc_{m}_mean', f'auc_{m}_std')]

    summary[summary_cols].to_csv(os.path.join(output_dir, 'top_q_percent_sweep_summary.csv'), index=False)
    summary[auc_cols].to_csv(os.path.join(output_dir, 'auc_summary.csv'), index=False)

    mlflow.log_param("dataset",                 DATASET)
    mlflow.log_param("alpha_true",              alpha_true)
    mlflow.log_param("alpha_estimation_method", estimation_method)
    mlflow.log_param("top_k_features",          TOP_K)
    mlflow.log_param("top_q_percents",          str(top_q_percents))
    mlflow.log_param("seeds",                   str(list(seeds)))
    mlflow.log_param("n_runs",                  len(rows))

    for _, row in summary.iterrows():
        q = int(row['top_q_percent'])
        mlflow.log_metric(f"top_q_{q}_alpha_hat_mean",           float(row['alpha_hat_mean']))
        mlflow.log_metric(f"top_q_{q}_alpha_hat_std",            float(row['alpha_hat_std']))
        mlflow.log_metric(f"top_q_{q}_error_alpha_rel",          float(np.abs(row['alpha_hat_mean'] - alpha_true) / alpha_true * 100))
        for m in metodos:
            mlflow.log_metric(f"top_q_{q}_auc_{m}_mean",         float(row[f'auc_{m}_mean']))
            mlflow.log_metric(f"top_q_{q}_auc_{m}_std",          float(row[f'auc_{m}_std']))
        mlflow.log_metric(f"top_q_{q}_spearman_pu_topk_mean",    float(row['spearman_pu_topk_mean']))
        mlflow.log_metric(f"top_q_{q}_spearman_naive_topk_mean", float(row['spearman_naive_topk_mean']))
        mlflow.log_metric(f"top_q_{q}_spearman_var_topk_mean",   float(row['spearman_var_topk_mean']))

    mlflow.log_artifact(os.path.join(output_dir, 'top_q_percent_sweep_runs.csv'))
    mlflow.log_artifact(os.path.join(output_dir, 'top_q_percent_sweep_summary.csv'))
    mlflow.log_artifact(os.path.join(output_dir, 'auc_summary.csv'))

    # Colores consistentes con _sweep_alpha
    colores = {
        'PU_corregido': '#FF8C42',
        'MI_naive':     '#2ECC71',
        'Varianza':     '#E74C3C',
        'MI_real':      '#3498DB',
    }
    x = summary['top_q_percent'].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(top_q_percents)

    # Gráfico 2: AUC vs Top Q Percent (4 métodos)
    ax = axes[0, 1]
    for metodo in metodos:
        means = summary[f'auc_{metodo}_mean'].values
        stds  = summary[f'auc_{metodo}_std'].values
        ax.plot(x, means, marker='o', label=metodo, color=colores[metodo], linewidth=2)
        ax.fill_between(x, means - stds, means + stds, alpha=0.15, color=colores[metodo])
    ax.set_xlabel('Top Q Percent (%)', fontsize=11)
    ax.set_ylabel(f'AUC (top-{TOP_K} features)', fontsize=11)
    ax.set_title('AUC vs Top Q Percent por método', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(top_q_percents)

    # Gráfico 3: Spearman FULL vs Top Q Percent (3 métodos, calculado del df completo)
    ax = axes[1, 0]
    for label, col in [
        ('PU_corregido', 'spearman_pu_full'),
        ('MI_naive',     'spearman_naive_full'),
        ('Varianza',     'spearman_var_full'),
    ]:
        means = df.groupby('top_q_percent')[col].mean().values
        stds  = df.groupby('top_q_percent')[col].std().values
        ax.plot(x, means, marker='o', label=label, color=colores[label], linewidth=2)
        ax.fill_between(x, means - stds, means + stds, alpha=0.15, color=colores[label])
    ax.axhline(y=1.0, color=colores['MI_real'], linestyle='--', linewidth=2, label='MI_real (ref.)', alpha=0.7)
    ax.set_xlabel('Top Q Percent (%)', fontsize=11)
    ax.set_ylabel('Correlación de Spearman', fontsize=11)
    ax.set_title('Spearman FULL vs Top Q Percent', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(top_q_percents)

    # Gráfico 4: Spearman TOP-K vs Top Q Percent (3 métodos)
    ax = axes[1, 1]
    for label, col_mean, col_std in [
        ('PU_corregido', 'spearman_pu_topk_mean',    'spearman_pu_topk_std'),
        ('MI_naive',     'spearman_naive_topk_mean',  'spearman_naive_topk_std'),
        ('Varianza',     'spearman_var_topk_mean',    'spearman_var_topk_std'),
    ]:
        means = summary[col_mean].values
        stds  = summary[col_std].values
        ax.plot(x, means, marker='s', label=label, color=colores[label], linewidth=2)
        ax.fill_between(x, means - stds, means + stds, alpha=0.15, color=colores[label])
    ax.axhline(y=1.0, color=colores['MI_real'], linestyle='--', linewidth=2, label='MI_real (ref.)', alpha=0.7)
    ax.set_xlabel('Top Q Percent (%)', fontsize=11)
    ax.set_ylabel('Correlación de Spearman', fontsize=11)
    ax.set_title(f'Spearman TOP-{TOP_K} vs Top Q Percent', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(top_q_percents)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'top_q_percent_sweep.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    mlflow.log_artifact(os.path.join(output_dir, 'top_q_percent_sweep.png'))

    print("\n" + "="*70)
    print("RESUMEN DE ESTIMACIONES")
    print("="*70)
    print(f"Dataset: {DATASET}")
    print(f"Método: {estimation_method.upper()}")
    print(f"Alpha verdadero: {alpha_true}")
    print(f"Semillas: {len(seeds)}")
    print("="*70)

    header = f"{'Top_Q_%':<10} {'Alpha_Est':<12} {'Error_%':<10} {'AUC_PU':<10} {'AUC_Naive':<11} {'AUC_Real':<10} {'AUC_Var':<10} {'Spear_PU_K':<12}"
    print(f"\n{header}")
    print("-"*85)
    for _, row in summary.iterrows():
        q = int(row['top_q_percent'])
        alpha_mean = row['alpha_hat_mean']
        error = np.abs(alpha_mean - alpha_true) / alpha_true * 100
        print(
            f"{q:<10} {alpha_mean:<12.4f} {error:<10.2f}"
            f" {row['auc_PU_corregido_mean']:<10.4f}"
            f" {row['auc_MI_naive_mean']:<11.4f}"
            f" {row['auc_MI_real_mean']:<10.4f}"
            f" {row['auc_Varianza_mean']:<10.4f}"
            f" {row['spearman_pu_topk_mean']:<12.4f}"
        )


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

    output_dir = os.path.join("resultados", DATASET, estimation_method)
    os.makedirs(output_dir, exist_ok=True)

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

        # Configuración general
        mlflow.log_param("run_mode",    RUN_MODE)
        mlflow.log_param("noise_level", NOISE_LEVEL)
        mlflow.log_param("top_k",       TOP_K)
        mlflow.log_param("alpha_estimation_method", estimation_method)
        if estimation_method == 'robust':
            mlflow.log_param("alpha_top_q_percent", ALPHA_TOP_Q_PERCENT)
        
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
            kfold = crear_splitter_cv(ds_meta.get("kind", ""), n_splits=3, random_state=RANDOM_STATE)
            
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
                _sweep_alpha(X, y, feature_names, estimation_method, ds_meta.get("kind", ""), output_dir)
            elif SWEEP_MODE == 'percent':
                _sweep_percent(X, y, feature_names, estimation_method, ds_meta.get("kind", ""), output_dir)
            else:
                raise ValueError(f"SWEEP_MODE inválido: {SWEEP_MODE}. Debe ser 'alpha' o 'percent'")

        print("\nRun guardado en MLflow")


if __name__ == "__main__":
    main()
    