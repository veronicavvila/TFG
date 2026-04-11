"""
Script de prueba para verificar que los datasets de microarray se cargan/descargan correctamente.
"""

from src.datasets import load_dataset

# Probar cada dataset de microarray
datasets_to_test = ["colon_cancer", "prostate_cancer", "lung_cancer"]

for dataset_name in datasets_to_test:
    try:
        print(f"\n{'='*60}")
        print(f"Cargando {dataset_name}...")
        print(f"{'='*60}")
        X, y, feature_names, meta = load_dataset(dataset_name)
        
        print(f"✓ {dataset_name} cargado exitosamente")
        print(f"  Shape: {X.shape}")
        print(f"  Features: {len(feature_names)}")
        print(f"  Class distribution: {sum(y)} positivos, {len(y) - sum(y)} negativos")
        print(f"  Description: {meta['description']}")
        
    except Exception as e:
        print(f"✗ Error cargando {dataset_name}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print("Prueba completada")
print(f"{'='*60}")
