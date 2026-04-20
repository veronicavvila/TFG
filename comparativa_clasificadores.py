#!/usr/bin/env python
"""
Comparativa de tres clasificadores: Real, PU Corregido, Naive
    
Para cambiar dataset:
    1. Editar src/config_clasificador.py
    2. Cambiar DATASET = "nombre"
    3. Ejecutar de nuevo
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

# Importar configuración y funciones
from src.config_clasificador import (
    DATASET, N_SEEDS, ALPHAS, N_KFOLDS, NOISE_LEVEL,
    ALPHA_ESTIMATION_METHOD, ALPHA_TOP_Q_PERCENT,
    EXPERIMENT_NAME, RUN_NAME
)
from src.datasets import load_dataset
from src.data_utiles import generar_etiquetas_pu, añadir_ruido_gaussiano
from src.pu_model import entrenar_clasificador_pu, estimar_alpha, estimar_alpha_robusto, obtener_scores


def crear_directorio_salida():
    """Crear directorio para guardar artefactos locales."""
    output_dir = Path("resultados_comparativa") / DATASET
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def crear_visualizaciones(df_resultados, df_resumen, output_dir):
    """Crea gráficos de comparación de modelos."""
    
    sns.set_style("whitegrid")
    
    # ===== GRÁFICO 1: Boxplot de AUC =====
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Preparar datos para boxplot
    auc_real = df_resultados['real_auc'].values
    auc_pu = df_resultados['pu_auc'].values
    auc_naive = df_resultados['naive_auc'].values
    
    data_boxplot = [auc_real, auc_pu, auc_naive]
    
    bp = ax.boxplot(data_boxplot, tick_labels=['Real', 'PU', 'Naive'], patch_artist=True)
    
    # Colorear los boxes
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title(f'Comparación de AUC - Dataset: {DATASET}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Agregar valores medios como texto
    for i, (data, label) in enumerate(zip(data_boxplot, ['Real', 'PU', 'Naive']), 1):
        mean_val = np.mean(data)
        ax.text(i, mean_val + 0.005, f'{mean_val:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    boxplot_path = output_dir / 'comparacion_auc_boxplot.png'
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== GRÁFICO 2: Barplot de todas las métricas =====
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metodos = ['Real', 'PU', 'Naive']
    metricas = ['auc', 'acc', 'prec', 'rec']
    
    # Preparar datos
    x = np.arange(len(metricas))
    width = 0.25
    
    datos_plot = {}
    for metodo in metodos:
        subset = df_resumen[df_resumen['metodo'] == metodo.lower()]
        datos_plot[metodo] = {
            'media': subset[subset['metrica'].isin(metricas)].set_index('metrica')['media'],
            'std': subset[subset['metrica'].isin(metricas)].set_index('metrica')['std']
        }
    
    # Plotear barras
    colors_dict = {'Real': '#2ecc71', 'PU': '#3498db', 'Naive': '#e74c3c'}
    for i, metodo in enumerate(metodos):
        offset = (i - 1) * width
        means = [datos_plot[metodo]['media'].get(m, 0) for m in metricas]
        stds = [datos_plot[metodo]['std'].get(m, 0) for m in metricas]
        
        ax.bar(x + offset, means, width, label=metodo, 
               color=colors_dict[metodo], alpha=0.8, capsize=5)
        ax.errorbar(x + offset, means, yerr=stds, fmt='none', 
                   color='black', capsize=3, alpha=0.5, elinewidth=1)
    
    ax.set_ylabel('Valor de Métrica', fontsize=12, fontweight='bold')
    ax.set_title(f'Comparación de Métricas - Dataset: {DATASET}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metricas])
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    barplot_path = output_dir / 'comparacion_metricas_barplot.png'
    plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return boxplot_path, barplot_path


def entrenar_y_evaluar_modelos(X_train, X_test, y_train, y_test, S_train, S_test, seed):
    """
    Entrena 3 modelos y retorna métricas.
    
    Args:
        X_train, X_test: features (ya preprocessadas)
        y_train, y_test: etiquetas verdaderas (0/1)
        S_train, S_test: etiquetas PU (P=1, U=-1)
        seed: para reproducibilidad
    
    Returns:
        dict con métricas: auc, acc, prec, rec para cada modelo
    """
    
    metricas = {}
    
    # 1. MODELO REAL (Supervisado con y verdadero)
    modelo_real = LogisticRegression(
        max_iter=500,
        solver='lbfgs',
        class_weight=None,
        random_state=seed
    )
    modelo_real.fit(X_train, y_train)
    
    proba_real = modelo_real.predict_proba(X_test)[:, 1]
    pred_real = modelo_real.predict(X_test)
    
    metricas['real'] = {
        'auc': roc_auc_score(y_test, proba_real),
        'acc': accuracy_score(y_test, pred_real),
        'prec': precision_score(y_test, pred_real, zero_division=0),
        'rec': recall_score(y_test, pred_real, zero_division=0),
    }
    
    # 2. MODELO NAIVE (Asume U como negativos) 
    S_train_naive = S_train.copy()
    S_train_naive[S_train_naive == -1] = 0  # Unlabeled = Negative
    
    modelo_naive = LogisticRegression(
        max_iter=500,
        solver='lbfgs',
        class_weight=None,
        random_state=seed
    )
    modelo_naive.fit(X_train, S_train_naive)
    
    proba_naive = modelo_naive.predict_proba(X_test)[:, 1]
    pred_naive = modelo_naive.predict(X_test)
    
    metricas['naive'] = {
        'auc': roc_auc_score(y_test, proba_naive),
        'acc': accuracy_score(y_test, pred_naive),
        'prec': precision_score(y_test, pred_naive, zero_division=0),
        'rec': recall_score(y_test, pred_naive, zero_division=0),
    }
    
    # 3. MODELO PU CORREGIDO (Estimación de alpha según configuración)
    modelo_pu = entrenar_clasificador_pu(X_train, S_train, random_state=seed)
    
    # Estimar alpha (mean o robust según configuración)
    scores_train = obtener_scores(modelo_pu, X_train)
    if ALPHA_ESTIMATION_METHOD == 'robust':
        alpha_hat = estimar_alpha_robusto(scores_train, S_train, 
                                         top_q_percent=ALPHA_TOP_Q_PERCENT)
    else:  # mean
        alpha_hat = estimar_alpha(scores_train, S_train)
    
    # Corregir probabilidades en test
    scores_test = obtener_scores(modelo_pu, X_test)
    proba_pu = np.clip(scores_test / alpha_hat, 0, 1)
    pred_pu = (proba_pu > 0.5).astype(int)
    
    metricas['pu'] = {
        'auc': roc_auc_score(y_test, proba_pu),
        'acc': accuracy_score(y_test, pred_pu),
        'prec': precision_score(y_test, pred_pu, zero_division=0),
        'rec': recall_score(y_test, pred_pu, zero_division=0),
        'alpha_hat': alpha_hat,
    }
    
    return metricas


def ejecutar_experimento():
    """Función principal: ejecuta el experimento completo."""
    
    print(f"\n{'='*70}")
    print(f"COMPARATIVA DE CLASIFICADORES")
    print(f"Dataset: {DATASET}")
    print(f"Método estimación alpha: {ALPHA_ESTIMATION_METHOD}", end="")
    if ALPHA_ESTIMATION_METHOD == 'robust':
        print(f" (top_q_percent={ALPHA_TOP_Q_PERCENT})")
    else:
        print()
    print(f"{'='*70}\n")
    
    # Crear directorio de salida
    output_dir = crear_directorio_salida()
    
    # Cargar dataset
    X, y, feature_names, meta = load_dataset(DATASET)
    X = StandardScaler().fit_transform(X)
    print(f"      Formato: X={X.shape}, y={y.shape}")
    print(f"      Features: {len(feature_names)}\n")
    
    # Almacenar todos los resultados
    resultados_lista = []
    
    # Configurar MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=RUN_NAME) as run:
        
        # Log configuración
        mlflow.log_param("dataset", DATASET)
        mlflow.log_param("n_seeds", len(N_SEEDS))
        mlflow.log_param("n_alphas", len(ALPHAS))
        mlflow.log_param("n_kfolds", N_KFOLDS)
        mlflow.log_param("alpha_estimation_method", ALPHA_ESTIMATION_METHOD)
        if ALPHA_ESTIMATION_METHOD == 'robust':
            mlflow.log_param("top_q_percent", ALPHA_TOP_Q_PERCENT)
        mlflow.log_param("noise_level", NOISE_LEVEL)
        
        
        for seed in N_SEEDS:
            for alpha in ALPHAS:
                
                # Generar etiquetas PU
                S = generar_etiquetas_pu(y, alpha, random_state=seed)
                X_noisy = añadir_ruido_gaussiano(X, NOISE_LEVEL, random_state=seed)
                
                # K-Fold
                kfold = StratifiedKFold(n_splits=N_KFOLDS, shuffle=True, 
                                       random_state=seed)
                
                for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_noisy, y)):
                    
                    X_train, X_test = X_noisy[train_idx], X_noisy[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    S_train, S_test = S[train_idx], S[test_idx]
                    
                    # Entrenar y evaluar
                    metricas = entrenar_y_evaluar_modelos(
                        X_train, X_test, y_train, y_test, 
                        S_train, S_test, seed
                    )
                    
                    # Guardar resultado
                    resultado = {
                        'seed': seed,
                        'alpha': alpha,
                        'fold': fold_idx,
                    }
                    
                    for metodo in ['real', 'pu', 'naive']:
                        for metrica in ['auc', 'acc', 'prec', 'rec']:
                            if metrica in metricas[metodo]:
                                resultado[f'{metodo}_{metrica}'] = metricas[metodo][metrica]
                    
                    if 'alpha_hat' in metricas['pu']:
                        resultado['alpha_hat'] = metricas['pu']['alpha_hat']
                    
                    resultados_lista.append(resultado)
                    
        print()
        
        # Resultados detallados        
        df_resultados = pd.DataFrame(resultados_lista)
        csv_path = output_dir / f"resultados_detallados_{DATASET}.csv"
        df_resultados.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))
        print(f"       {csv_path}\n")
        
        # Estadísticas        
        # Resumen por método
        resumen_metodo = []
        for metodo in ['real', 'pu', 'naive']:
            for metrica in ['auc', 'acc', 'prec', 'rec']:
                col = f'{metodo}_{metrica}'
                if col in df_resultados.columns:
                    datos = df_resultados[col].dropna()
                    resumen_metodo.append({
                        'metodo': metodo,
                        'metrica': metrica,
                        'media': datos.mean(),
                        'std': datos.std(),
                        'mediana': datos.median(),
                        'min': datos.min(),
                        'max': datos.max(),
                        'n': len(datos),
                        'ic95_lower': datos.mean() - 1.96 * datos.std() / np.sqrt(len(datos)),
                        'ic95_upper': datos.mean() + 1.96 * datos.std() / np.sqrt(len(datos)),
                    })
        
        df_resumen = pd.DataFrame(resumen_metodo)
        resumen_path = output_dir / f"resumen_{DATASET}.csv"
        df_resumen.to_csv(resumen_path, index=False)
        mlflow.log_artifact(str(resumen_path))
        
        print("RESUMEN POR MÉTODO:")
        print("-" * 70)
        for metodo in ['real', 'pu', 'naive']:
            print(f"\n{metodo.upper()}:")
            subset = df_resumen[df_resumen['metodo'] == metodo]
            for _, row in subset.iterrows():
                print(f"  {row['metrica']:4s}: {row['media']:.4f} ± {row['std']:.4f}")
        
        # Tests pareados AUC (principal métrica)
        print(f"\n\n{'='*70}")
        print("TESTS ESTADÍSTICOS (AUC - Paired t-test):")
        print(f"{'='*70}\n")
        
        auc_real = df_resultados['real_auc'].values
        auc_pu = df_resultados['pu_auc'].values
        auc_naive = df_resultados['naive_auc'].values
        
        tests_stats = []
        
        # Real vs PU
        t_stat, p_val = ttest_rel(auc_real, auc_pu)
        tests_stats.append({
            'dataset': DATASET,
            'comparison': 'Real = PU',
            'media_real': auc_real.mean(),
            'media_pu': auc_pu.mean(),
            'diferencia': auc_real.mean() - auc_pu.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
        })
        
        print(f"Comparación: Real = PU")
        print(f"Real: {auc_real.mean():.4f} ± {auc_real.std():.4f}")
        print(f"PU:   {auc_pu.mean():.4f} ± {auc_pu.std():.4f}")
        print(f"Diferencia: {auc_real.mean() - auc_pu.mean():.4f}")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}\n")


        # PU vs Naive
        t_stat, p_val = ttest_rel(auc_pu, auc_naive)
        tests_stats.append({
            'dataset': DATASET,
            'comparison': 'PU = Naive',
            'media_pu': auc_pu.mean(),
            'media_naive': auc_naive.mean(),
            'diferencia': auc_pu.mean() - auc_naive.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
        })
        
        print(f"Comparación: PU = Naive")
        print(f"PU: {auc_pu.mean():.4f} ± {auc_pu.std():.4f}")
        print(f"Naive: {auc_naive.mean():.4f} ± {auc_naive.std():.4f}")
        print(f"Diferencia: {auc_pu.mean() - auc_naive.mean():.4f}")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}\n")
        
        # PU > Naive (unilateral)
        t_stat, p_val = ttest_rel(auc_pu, auc_naive, alternative='greater')
        tests_stats.append({
            'dataset': DATASET,
            'comparison': 'PU > Naive',
            'media_pu': auc_pu.mean(),
            'media_naive': auc_naive.mean(),
            'diferencia': auc_pu.mean() - auc_naive.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
        })
        
        print(f"Comparación: PU > Naive (H0 = PU < Naive)")
        print(f"PU: {auc_pu.mean():.4f} ± {auc_pu.std():.4f}")
        print(f"Naive: {auc_naive.mean():.4f} ± {auc_naive.std():.4f}")
        print(f"Diferencia: {auc_pu.mean() - auc_naive.mean():.4f}")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}\n")


        df_tests = pd.DataFrame(tests_stats)
        tests_path = output_dir / f"tests_estadisticos_{DATASET}.csv"
        df_tests.to_csv(tests_path, index=False)
        mlflow.log_artifact(str(tests_path))
        
        # Log métricas resumen en MLflow
        mlflow.log_metric("auc_real_mean", auc_real.mean())
        mlflow.log_metric("auc_pu_mean", auc_pu.mean())
        mlflow.log_metric("auc_naive_mean", auc_naive.mean())
        mlflow.log_metric("auc_real_std", auc_real.std())
        mlflow.log_metric("auc_pu_std", auc_pu.std())
        mlflow.log_metric("auc_naive_std", auc_naive.std())
        
        # Crear visualizaciones
        print(f"\n{'='*70}")
        print("GENERANDO VISUALIZACIONES...")
        print(f"{'='*70}\n")
        
        try:
            boxplot_path, barplot_path = crear_visualizaciones(
                df_resultados, df_resumen, output_dir
            )
            
            # Log gráficos en MLflow
            mlflow.log_artifact(str(boxplot_path))
            mlflow.log_artifact(str(barplot_path))
            
            print(f"Gráficos guardados:")
            print(f"  - {boxplot_path.name}")
            print(f"  - {barplot_path.name}\n")
        except Exception as e:
            print(f"Error al generar visualizaciones: {e}\n")
        
        # Resumen final
        print(f"{'-'*70}")
        print(f"\nArtefactos guardados en: {output_dir}")
        print(f"  - resultados_detallados_{DATASET}.csv")
        print(f"  - resumen_{DATASET}.csv")
        print(f"  - tests_estadisticos_{DATASET}.csv")
        print(f"  - comparacion_auc_boxplot.png")
        print(f"  - comparacion_metricas_barplot.png")


if __name__ == "__main__":
    ejecutar_experimento()
