"""
Tests estadísticos agregados sobre todos los resultados de mean y robust.
Compara: PU = Real, PU > Naive, Mean vs Robust
Métrica: AUC (la más comparable entre métodos)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_rel

def cargar_resultados_dataset(dataset_name, metodo):
    """Carga resultados_detallados_*.csv para un dataset y método."""
    base_path = Path("resultados_comparativa") / dataset_name / metodo
    csv_path = base_path / f"resultados_detallados_{dataset_name}.csv"
    
    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        return None

def hacer_tests_dataset(dataset_name):
    """Realiza todos los tests para un dataset."""
    
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Cargar datos mean y robust
    df_mean = cargar_resultados_dataset(dataset_name, 'mean')
    df_robust = cargar_resultados_dataset(dataset_name, 'robust')
    
    if df_mean is None and df_robust is None:
        print(f" No hay resultados para {dataset_name}")
        return None
    
    resultados_tests = []
    
    # ========== TESTS MEAN ==========
    if df_mean is not None:
        print(f"\nMETODO: MEAN")
        print("-" * 80)
        
        auc_real_mean = df_mean['real_auc'].values
        auc_pu_mean = df_mean['pu_auc'].values
        auc_naive_mean = df_mean['naive_auc'].values
        
        n = len(auc_real_mean)
        
        # Test 1: PU = Real (bilateral)
        t_stat, p_val = ttest_rel(auc_pu_mean, auc_real_mean)
        resultado = {
            'dataset': dataset_name,
            'metodo': 'mean',
            'test': 'PU = Real',
            'tipo': 'bilateral',
            'n': n,
            'media_pu': auc_pu_mean.mean(),
            'media_real': auc_real_mean.mean(),
            'diferencia': auc_pu_mean.mean() - auc_real_mean.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
            'significativo': 'Sí' if p_val < 0.05 else 'No',
            'conclusion': 'PU ≠ Real' if p_val < 0.05 else 'PU = Real (equivalentes)'
        }
        resultados_tests.append(resultado)
        
        print(f"Test: PU = Real (bilateral)")
        print(f"  H0: PU AUC = Real AUC (método PU equivalente a supervisado)")
        print(f"  H1: PU AUC ≠ Real AUC (método PU produce ranking diferente)")
        print(f"  N: {n}")
        print(f"  PU AUC:   {auc_pu_mean.mean():.4f} ± {auc_pu_mean.std():.4f}")
        print(f"  Real AUC: {auc_real_mean.mean():.4f} ± {auc_real_mean.std():.4f}")
        print(f"  Diferencia: {auc_pu_mean.mean() - auc_real_mean.mean():.4f}")
        print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
        print(f"  Conclusión: {resultado['conclusion']}\n")
        
        # Test 2: PU > Naive (unilateral)
        t_stat, p_val = ttest_rel(auc_pu_mean, auc_naive_mean, alternative='greater')
        resultado = {
            'dataset': dataset_name,
            'metodo': 'mean',
            'test': 'PU > Naive',
            'tipo': 'unilateral',
            'n': n,
            'media_pu': auc_pu_mean.mean(),
            'media_naive': auc_naive_mean.mean(),
            'diferencia': auc_pu_mean.mean() - auc_naive_mean.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
            'significativo': 'Sí' if p_val < 0.05 else 'No',
            'conclusion': 'PU > Naive' if p_val < 0.05 else 'PU ≤ Naive'
        }
        resultados_tests.append(resultado)
        
        print(f"Test: PU > Naive (unilateral)")
        print(f"  H0: PU AUC ≤ Naive AUC (PU no mejora)")
        print(f"  H1: PU AUC > Naive AUC (PU mejora)")
        print(f"  N: {n}")
        print(f"  PU AUC:    {auc_pu_mean.mean():.4f} ± {auc_pu_mean.std():.4f}")
        print(f"  Naive AUC: {auc_naive_mean.mean():.4f} ± {auc_naive_mean.std():.4f}")
        print(f"  Diferencia: {auc_pu_mean.mean() - auc_naive_mean.mean():.4f}")
        print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
        print(f"  Conclusión: {resultado['conclusion']}\n")
    
    # ========== TESTS ROBUST ==========
    if df_robust is not None:
        print(f"METODO: ROBUST")
        print("-" * 80)
        
        auc_real_robust = df_robust['real_auc'].values
        auc_pu_robust = df_robust['pu_auc'].values
        auc_naive_robust = df_robust['naive_auc'].values
        
        n = len(auc_real_robust)
        
        # Test 3: PU = Real (bilateral)
        t_stat, p_val = ttest_rel(auc_pu_robust, auc_real_robust)
        resultado = {
            'dataset': dataset_name,
            'metodo': 'robust',
            'test': 'PU = Real',
            'tipo': 'bilateral',
            'n': n,
            'media_pu': auc_pu_robust.mean(),
            'media_real': auc_real_robust.mean(),
            'diferencia': auc_pu_robust.mean() - auc_real_robust.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
            'significativo': 'Sí' if p_val < 0.05 else 'No',
            'conclusion': 'PU ≠ Real' if p_val < 0.05 else 'PU = Real (equivalentes)'
        }
        resultados_tests.append(resultado)
        
        print(f"Test: PU = Real (bilateral)")
        print(f"  H0: PU AUC = Real AUC (método PU equivalente a supervisado)")
        print(f"  H1: PU AUC ≠ Real AUC (método PU produce ranking diferente)")
        print(f"  N: {n}")
        print(f"  PU AUC:   {auc_pu_robust.mean():.4f} ± {auc_pu_robust.std():.4f}")
        print(f"  Real AUC: {auc_real_robust.mean():.4f} ± {auc_real_robust.std():.4f}")
        print(f"  Diferencia: {auc_pu_robust.mean() - auc_real_robust.mean():.4f}")
        print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
        print(f"  Conclusión: {resultado['conclusion']}\n")
        
        # Test 4: PU > Naive (unilateral)
        t_stat, p_val = ttest_rel(auc_pu_robust, auc_naive_robust, alternative='greater')
        resultado = {
            'dataset': dataset_name,
            'metodo': 'robust',
            'test': 'PU > Naive',
            'tipo': 'unilateral',
            'n': n,
            'media_pu': auc_pu_robust.mean(),
            'media_naive': auc_naive_robust.mean(),
            'diferencia': auc_pu_robust.mean() - auc_naive_robust.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
            'significativo': 'Sí' if p_val < 0.05 else 'No',
            'conclusion': 'PU > Naive' if p_val < 0.05 else 'PU ≤ Naive'
        }
        resultados_tests.append(resultado)
        
        print(f"Test: PU > Naive (unilateral)")
        print(f"  H0: PU AUC ≤ Naive AUC (PU no mejora)")
        print(f"  H1: PU AUC > Naive AUC (PU mejora)")
        print(f"  N: {n}")
        print(f"  PU AUC:    {auc_pu_robust.mean():.4f} ± {auc_pu_robust.std():.4f}")
        print(f"  Naive AUC: {auc_naive_robust.mean():.4f} ± {auc_naive_robust.std():.4f}")
        print(f"  Diferencia: {auc_pu_robust.mean() - auc_naive_robust.mean():.4f}")
        print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
        print(f"  Conclusión: {resultado['conclusion']}\n")
    
    # ========== TEST COMPARATIVA: MEAN vs ROBUST ==========
    if df_mean is not None and df_robust is not None:
        print(f"COMPARATIVA: MEAN vs ROBUST")
        print("-" * 80)
        
        auc_pu_mean = df_mean['pu_auc'].values
        auc_pu_robust = df_robust['pu_auc'].values
        
        # Verificar que tienen el mismo tamaño
        if len(auc_pu_mean) == len(auc_pu_robust):
            n = len(auc_pu_mean)
            
            # Test 5: PU_robust > PU_mean (¿Robust mejora Mean?)
            t_stat, p_val = ttest_rel(auc_pu_robust, auc_pu_mean, alternative='greater')
            resultado = {
                'dataset': dataset_name,
                'metodo': 'mean_vs_robust',
                'test': 'Robust > Mean',
                'tipo': 'unilateral',
                'n': n,
                'media_mean': auc_pu_mean.mean(),
                'media_robust': auc_pu_robust.mean(),
                'diferencia': auc_pu_robust.mean() - auc_pu_mean.mean(),
                't_statistic': t_stat,
                'p_value': p_val,
                'significativo': 'Sí' if p_val < 0.05 else 'No',
                'conclusion': 'Robust mejora Mean' if p_val < 0.05 else 'Robust NO mejora Mean'
            }
            resultados_tests.append(resultado)
            
            print(f"Test: Robust > Mean (¿Robust mejora Mean?)")
            print(f"  H0: Robust AUC ≤ Mean AUC (Robust no mejora)")
            print(f"  H1: Robust AUC > Mean AUC (Robust mejora)")
            print(f"  N: {n}")
            print(f"  PU Mean AUC:   {auc_pu_mean.mean():.4f} ± {auc_pu_mean.std():.4f}")
            print(f"  PU Robust AUC: {auc_pu_robust.mean():.4f} ± {auc_pu_robust.std():.4f}")
            print(f"  Diferencia: {auc_pu_robust.mean() - auc_pu_mean.mean():.4f}")
            print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
            print(f"  Conclusión: {resultado['conclusion']}\n")
    
    return resultados_tests

def main():
    """Ejecuta tests estadísticos para todos los datasets."""
    
    todos_resultados = []
    
    # Buscar todos los datasets en resultados_comparativa
    resultados_dir = Path("resultados_comparativa")
    datasets = [d.name for d in resultados_dir.iterdir() if d.is_dir()]
    datasets.sort()
    
    for dataset in datasets:
        resultados = hacer_tests_dataset(dataset)
        if resultados:
            todos_resultados.extend(resultados)
    
    # Guardar resultados en CSV
    if todos_resultados:
        df_resultados = pd.DataFrame(todos_resultados)
        output_path = Path("tests_estadisticos_agregados.csv")
        df_resultados.to_csv(output_path, index=False)
        
        print(f"\n{'='*80}")
        print(f"RESUMEN FINAL")
        print(f"{'='*80}")
        print(f"Resultados guardados en: {output_path}\n")
        
        # Imprimir tabla resumen
        print(df_resultados[['dataset', 'metodo', 'test', 'p_value', 'significativo', 'conclusion']].to_string(index=False))

if __name__ == "__main__":
    main()
