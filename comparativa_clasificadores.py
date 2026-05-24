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
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import traceback

# Importar configuración y funciones
from src.config_clasificador import (
    DATASET, N_SEEDS, ALPHAS, N_KFOLDS, NOISE_LEVEL,
    ALPHA_ESTIMATION_METHOD, ALPHA_TOP_Q_PERCENT,
    EXPERIMENT_NAME, RUN_NAME, TOP_K, USE_FEATURE_SELECTION
)
from src.datasets import load_dataset
from src.datasets import crear_splitter_cv
from src.data_utiles import generar_etiquetas_pu, añadir_ruido_gaussiano
from src.pu_model import entrenar_clasificador_pu, estimar_alpha, estimar_alpha_robusto, obtener_scores
from src.evaluacion import calcular_mi_real, calcular_mi_naive


def crear_directorio_salida():
    """Crear directorio para guardar artefactos locales."""
    output_dir = Path("resultados_comparativa") / DATASET / ALPHA_ESTIMATION_METHOD
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
    
    # Nota: df_resumen solo resume AUC por alpha; para el barplot de (auc, acc, prec, rec)
    # tomamos medias/std directamente de df_resultados, que sí contiene todas las métricas.
    prefix_by_method = {'Real': 'real', 'PU': 'pu', 'Naive': 'naive'}
    
    # Plotear barras
    colors_dict = {'Real': '#2ecc71', 'PU': '#3498db', 'Naive': '#e74c3c'}
    for i, metodo in enumerate(metodos):
        offset = (i - 1) * width
        prefix = prefix_by_method[metodo]
        means = []
        stds = []
        for m in metricas:
            col = f"{prefix}_{m}"
            if col in df_resultados.columns:
                values = df_resultados[col].dropna().values
                means.append(float(np.nanmean(values)) if len(values) else 0.0)
                stds.append(float(np.nanstd(values)) if len(values) else 0.0)
            else:
                means.append(0.0)
                stds.append(0.0)
        
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


def entrenar_y_evaluar_modelos(X_train, X_test, y_train, y_test, S_train, S_test, p_y_train, seed):
    """
    Entrena 3 modelos y retorna métricas con selección de top-K features.
    
    Args:
        X_train, X_test: features (ya preprocessadas)
        y_train, y_test: etiquetas verdaderas (0/1)
        S_train, S_test: etiquetas PU (P=1, U=-1)
        p_y_train: probabilidades estimadas para entrenamiento PU
        seed: para reproducibilidad
    
    Returns:
        dict con métricas: auc, acc, prec, rec para cada modelo
    """
    
    from src.mi_utiles import calcular_mi_ranking
    
    # Seleccionar features basado en MI ranking PU si está habilitado
    if USE_FEATURE_SELECTION:
        # Calcular MI ranking usando el modelo PU
        mi_scores, ranking_pu = calcular_mi_ranking(
            X_train, p_y_train, metodo="regresion", random_state=seed
        )
        
        # Seleccionar top K features del ranking PU
        selected_features = ranking_pu[:TOP_K]
        
        X_train_sel = X_train[:, selected_features]
        X_test_sel = X_test[:, selected_features]
    else:
        X_train_sel = X_train
        X_test_sel = X_test
    
    metricas = {}
    
    # 1. MODELO REAL (Supervisado con y verdadero)
    modelo_real = LogisticRegression(
        max_iter=500,
        solver='lbfgs',
        class_weight=None,
        random_state=seed
    )
    modelo_real.fit(X_train_sel, y_train)
    
    proba_real = modelo_real.predict_proba(X_test_sel)[:, 1]
    pred_real = modelo_real.predict(X_test_sel)
    
    metricas['real'] = {
        'auc': roc_auc_score(y_test, proba_real),
        'acc': accuracy_score(y_test, pred_real),
        'prec': precision_score(y_test, pred_real, zero_division=0),
        'rec': recall_score(y_test, pred_real, zero_division=0),
    }
    
    # 2. MODELO NAIVE (Asume U como negativos)
    S_train_naive = S_train.copy()
    S_train_naive[S_train_naive == -1] = 0  # Unlabeled = Negative
    
    # Verificar si hay ambas clases (importante para datasets pequeños)
    unique_classes = np.unique(S_train_naive)
    '''
    if len(unique_classes) == 1:
        # Si solo hay una clase, usar predicción basada en proporción de clases en test
        # Este caso ocurre en datasets pequeños donde un fold puede no tener positivos etiquetados
        class_prop = np.mean(y_test)  # Proporción de positivos en test
        proba_naive = np.full(len(y_test), class_prop)
        pred_naive = (proba_naive > 0.5).astype(int)
    else:
    '''
    # Entrenamiento normal si hay ambas clases
    modelo_naive = LogisticRegression(
        max_iter=500,
        solver='lbfgs',
        class_weight=None,
        random_state=seed
    )
    modelo_naive.fit(X_train_sel, S_train_naive)
    
    proba_naive = modelo_naive.predict_proba(X_test_sel)[:, 1]
    pred_naive = modelo_naive.predict(X_test_sel)

    metricas['naive'] = {
        'auc': roc_auc_score(y_test, proba_naive),
        'acc': accuracy_score(y_test, pred_naive),
        'prec': precision_score(y_test, pred_naive, zero_division=0),
        'rec': recall_score(y_test, pred_naive, zero_division=0),
    }
    
    # 3. MODELO PU CORREGIDO (Estimación de alpha según configuración)
    modelo_pu = entrenar_clasificador_pu(X_train_sel, S_train, random_state=seed)
    
    # Estimar alpha (mean o robust según configuración)
    scores_train = obtener_scores(modelo_pu, X_train_sel)
    if ALPHA_ESTIMATION_METHOD == 'robust':
        alpha_hat = estimar_alpha_robusto(scores_train, S_train, 
                                         top_q_percent=ALPHA_TOP_Q_PERCENT)
    else:  # mean
        alpha_hat = estimar_alpha(scores_train, S_train)
    
    # Corregir probabilidades en test
    scores_test = obtener_scores(modelo_pu, X_test_sel)
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
        mlflow.log_param("use_feature_selection", USE_FEATURE_SELECTION)
        if USE_FEATURE_SELECTION:
            mlflow.log_param("top_k_features", TOP_K)
        
        
        for seed in N_SEEDS:
            for alpha in ALPHAS:
                
                # Generar etiquetas PU
                S = generar_etiquetas_pu(y, alpha, random_state=seed)
                X_noisy = añadir_ruido_gaussiano(X, NOISE_LEVEL, random_state=seed)
                
                # Entrenar modelo PU inicial para estimar alpha (usado en feature selection)
                modelo_pu_init = entrenar_clasificador_pu(X_noisy, S, random_state=seed)
                scores_init = obtener_scores(modelo_pu_init, X_noisy)
                
                if ALPHA_ESTIMATION_METHOD == 'robust':
                    alpha_hat_init = estimar_alpha_robusto(scores_init, S, 
                                                          top_q_percent=ALPHA_TOP_Q_PERCENT)
                else:
                    alpha_hat_init = estimar_alpha(scores_init, S)
                
                # Estimar probabilidades reales (para MI ranking)
                from src.pu_model import estimar_probabilidad_real
                p_y_full = estimar_probabilidad_real(scores_init, alpha_hat_init)
                
                # K-Fold
                kfold = crear_splitter_cv(meta.get("kind", ""), n_splits=N_KFOLDS, random_state=seed)
                
                for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_noisy, y)):
                    
                    X_train, X_test = X_noisy[train_idx], X_noisy[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    S_train, S_test = S[train_idx], S[test_idx]
                    p_y_train = p_y_full[train_idx]
                    
                    # Entrenar y evaluar
                    metricas = entrenar_y_evaluar_modelos(
                        X_train, X_test, y_train, y_test, 
                        S_train, S_test, p_y_train, seed
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
        # Resumen por método Y por alpha (solo AUC)
        resumen_metodo = []
        for alpha in ALPHAS:
            for metodo in ['real', 'pu', 'naive']:
                for metrica in ['auc']:
                    col = f'{metodo}_{metrica}'
                    if col in df_resultados.columns:
                        # Filtrar por alpha
                        datos = df_resultados[df_resultados['alpha'] == alpha][col].dropna()
                        if len(datos) > 0:
                            resumen_metodo.append({
                                'alpha': alpha,
                                'metodo': metodo,
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
        metodo_order = pd.CategoricalDtype(categories=['real', 'pu', 'naive'], ordered=True)
        if 'metodo' in df_resumen.columns:
            df_resumen['metodo'] = df_resumen['metodo'].astype(metodo_order)
        df_resumen = df_resumen.sort_values(['alpha', 'metodo']).reset_index(drop=True)
        resumen_path = output_dir / f"resumen_{DATASET}.csv"
        df_resumen.to_csv(resumen_path, index=False)
        mlflow.log_artifact(str(resumen_path))
        
        print("RESUMEN AUC POR MÉTODO:")
        print("-" * 70)
        for metodo in ['real', 'pu', 'naive']:
            print(f"\n{metodo.upper()}:")
            subset = df_resumen[df_resumen['metodo'] == metodo]
            for _, row in subset.iterrows():
                print(f"  α={row['alpha']}: {row['media']:.4f} ± {row['std']:.4f}")
        
        # Log métricas resumen en MLflow
        auc_real = df_resultados['real_auc'].values
        auc_pu = df_resultados['pu_auc'].values
        auc_naive = df_resultados['naive_auc'].values
        
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
            print(traceback.format_exc())
        
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
