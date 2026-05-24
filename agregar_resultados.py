#!/usr/bin/env python
"""
Script simple para agregar resultados_detallados.csv por alpha y método.
Recorre resultados_comparativa/{dataset}/{method}/ y genera resumen_{dataset}.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

def agregar_por_alpha(detallados_file):
    """Lee detallados y agrupa por alpha y método (solo AUC)."""
    
    df = pd.read_csv(detallados_file)
    
    resumen_lista = []
    
    for alpha in sorted(df['alpha'].unique()):
        df_alpha = df[df['alpha'] == alpha]
        
        for metodo in ['real', 'pu', 'naive']:
            col = f'{metodo}_auc'
            
            if col in df_alpha.columns:
                datos = df_alpha[col].dropna()
                
                if len(datos) > 0:
                    resumen_lista.append({
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
    
    return pd.DataFrame(resumen_lista)


def main():
    """Recorre y agrega todos los resultados."""
    
    resultados_base = Path("resultados_comparativa")
    
    if not resultados_base.exists():
        print(f"❌ Directorio {resultados_base} no existe")
        return
    
    print("\n" + "="*70)
    print("AGREGANDO RESULTADOS POR ALPHA (SOLO AUC)")
    print("="*70 + "\n")
    
    total = 0
    
    for dataset_dir in sorted(resultados_base.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue
        
        dataset_name = dataset_dir.name
        
        for method_dir in sorted(dataset_dir.iterdir()):
            if not method_dir.is_dir() or method_dir.name.startswith("."):
                continue
            
            method_name = method_dir.name
            detallados_file = method_dir / f"resultados_detallados_{dataset_name}.csv"
            
            if not detallados_file.exists():
                continue
            
            # Generar resumen
            df_resumen = agregar_por_alpha(detallados_file)
            
            # Guardar en misma carpeta
            resumen_file = method_dir / f"resumen_{dataset_name}.csv"
            df_resumen.to_csv(resumen_file, index=False)
            
            print(f"✓ {dataset_name:20s} / {method_name:8s}  →  {len(df_resumen)} filas")
            total += 1
    
    print("\n" + "="*70)
    print(f"✓ {total} resúmenes generados")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
