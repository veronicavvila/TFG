from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    kind: str  # 'openml' | 'sklearn' | 'libsvm'
    openml_name: Optional[str] = None
    openml_version: int = 1
    positive_label: Optional[Any] = None
    description: str = ""


_DATASETS: Dict[str, DatasetSpec] = {
    # OpenML (UCI / others)
    "magic_telescope": DatasetSpec(
        kind="openml",
        openml_name="MagicTelescope",
        openml_version=1,
        positive_label="g",
        description="MAGIC Gamma Telescope (gamma=g vs hadron=h)",
    ),
    "ionosphere": DatasetSpec(
        kind="openml",
        openml_name="ionosphere",
        openml_version=1,
        positive_label="g",
        description="Ionosphere (good=g vs bad=b)",
    ),
    "spambase": DatasetSpec(
        kind="openml",
        openml_name="spambase",
        openml_version=1,
        positive_label="1",
        description="Spambase (spam=1 vs non-spam=0)",
    ),
    "sonar": DatasetSpec(
        kind="openml",
        openml_name="sonar",
        openml_version=1,
        positive_label="Mine",
        description="Sonar (Mine vs Rock)",
    ),
    "miniboone": DatasetSpec(
        kind="openml",
        openml_name="MiniBooNE",
        openml_version=1,
        positive_label="True",
        description="MiniBooNE (signal=True vs background=False)",
    ),
    "phoneme": DatasetSpec(
        kind="openml",
        openml_name="phoneme",
        openml_version=1,
        positive_label="1",
        description="Phoneme (class 1 vs 2)",
    ),
    # Scikit-learn toy dataset
    "breast_cancer": DatasetSpec(
        kind="sklearn",
        positive_label=1,
        description="Sklearn breast cancer (target=1 as positive by default)",
    ),
    # Local libsvm-style dataset (multi-class originally) converted to binary (one-vs-rest)
    "gas_sensor_drift": DatasetSpec(
        kind="libsvm",
        positive_label=1,
        description="Gas Sensor Array Drift (one gas vs rest, configurable)",
    ),
    # Microarray datasets (Affymetrix expression data, GEO)
    "colon_cancer": DatasetSpec(
        kind="microarray",
        openml_name="colon_cancer",  # filename para .npz
        positive_label=1,
        description="Colon Cancer (Alon et al. 2000) - 62 samples, ~2000 genes (Tumor vs Normal)",
    ),
    "prostate_cancer": DatasetSpec(
        kind="microarray",
        openml_name="prostate_cancer",
        positive_label=1,
        description="Prostate Cancer (Singh et al. 2002) - 102 samples, ~6000 genes (Tumor vs Normal)",
    ),
    "lung_cancer": DatasetSpec(
        kind="microarray",
        openml_name="lung_cancer",
        positive_label=1,
        description="Lung Cancer (Bhattacharjee et al. 2001) - ~181 samples, ~12000 genes (Adenocarcinoma vs Others)",
    ),
}


def available_datasets() -> Tuple[str, ...]:
    return tuple(sorted(_DATASETS.keys()))


def _download_microarray_dataset(dataset_name: str, dataset_key: str, cache_dir: str) -> None:
    """Descarga automáticamente un dataset de microarray desde GEO Database."""
    try:
        import GEOparse
    except ImportError:
        raise ImportError(
            f"GEOparse no instalado. Para usar datasets microarray:\n"
            f"  pip install GEOparse\n"
            f"Luego intenta de nuevo."
        )
    
    geo_map = {
        "colon_cancer": ("GSE5847", "Colon Cancer"),
        "prostate_cancer": ("GSE3039", "Prostate Cancer"),
        "lung_cancer": ("GSE7670", "Lung Cancer"),
    }
    
    if dataset_key not in geo_map:
        raise ValueError(f"No GEO mapping for {dataset_key}")
    
    geo_id, display_name = geo_map[dataset_key]
    print(f"Descargando {display_name} desde GEO ({geo_id})...")
    
    try:
        gse = GEOparse.get_GEO(geo=geo_id, destdir=cache_dir, silent=False)
    except Exception as e:
        raise RuntimeError(f"Error descargando de GEO: {e}")
    
    X_list = []
    y_list = []
    feature_names = None
    
    for gsm_name, gsm in gse.gsms.items():
        try:
            values = gsm.table['VALUE'].values
            X_list.append(values)
            
            # Guardar nombres de features del primer sample
            if feature_names is None:
                feature_names = gsm.table.index.values
            
            # Extraer label del título
            title = gsm.metadata.get('title', [''])[0].lower()
            
            if dataset_key == "colon_cancer":
                y_list.append(1 if 'tumor' in title else 0)
            elif dataset_key == "prostate_cancer":
                y_list.append(1 if 'tumor' in title else 0)
            elif dataset_key == "lung_cancer":
                y_list.append(1 if 'adeno' in title else 0)
        except Exception as e:
            print(f"  Skipping sample {gsm_name}: {e}")
            continue
    
    if not X_list:
        raise RuntimeError(f"No samples extracted from {display_name}")
    
    X = np.array(X_list, dtype=np.float32)  # Shape: (n_samples, n_features)
    y = np.array(y_list, dtype=int)
    
    # Si no tenemos feature names, generarlas
    if feature_names is None or len(feature_names) == 0:
        feature_names = np.array([f"gene_{i}" for i in range(X.shape[1])])
    
    # Normalizar si es necesario
    if np.max(X) > 100:
        X = np.log2(X + 1)
    
    # Guardar
    import os
    npz_file = os.path.join(cache_dir, f"{dataset_key}.npz")
    np.savez(npz_file, X=X, y=y, feature_names=feature_names)
    print(f"✓ {display_name} guardado: {X.shape}")



def _binarize_labels(y_raw: np.ndarray, *, positive_label: Any) -> np.ndarray:
    if positive_label is None:
        raise ValueError("positive_label must be provided for binarization")

    mask = y_raw == positive_label
    if not np.any(mask):
        unique = np.unique(y_raw)
        raise ValueError(
            f"positive_label={positive_label!r} not present in y; unique labels: {unique[:20]!r}"
        )
    return mask.astype(int)


def load_dataset(
    dataset: str,
    *,
    positive_label_override: Any = None,
    gas_positive_class: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Carga el dataset y devuelve (X, y, feature_names, meta).

    - X: np.ndarray (n_samples, n_features)
    - y: np.ndarray (n_samples,) con etiquetas binarias {0,1}
    - feature_names: np.ndarray (n_features,) de strings
    - meta: dict con info (dataset_id, description, positive_label, etc.)

    `positive_label_override` fuerza la etiqueta considerada como clase positiva.
    """
    if dataset not in _DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Available: {', '.join(available_datasets())}"
        )

    spec = _DATASETS[dataset]
    positive_label = positive_label_override if positive_label_override is not None else spec.positive_label

    if spec.kind == "openml":
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(name=spec.openml_name, version=spec.openml_version, as_frame=False)
        X = np.asarray(bunch.data)
        y_raw = np.asarray(bunch.target)
        y = _binarize_labels(y_raw, positive_label=positive_label)
        feature_names = np.asarray(getattr(bunch, "feature_names", [f"x{i}" for i in range(X.shape[1])]))
        meta = {
            "dataset_id": dataset,
            "kind": spec.kind,
            "openml_name": spec.openml_name,
            "openml_version": spec.openml_version,
            "description": spec.description,
            "positive_label": positive_label,
        }
        return X, y, feature_names, meta

    if spec.kind == "sklearn":
        from sklearn.datasets import load_breast_cancer

        bunch = load_breast_cancer(as_frame=False)
        X = np.asarray(bunch.data)
        y_raw = np.asarray(bunch.target)
        # sklearn typically provides 0/1 already, but still honor positive_label override
        if positive_label in (0, 1):
            y = (y_raw == int(positive_label)).astype(int)
        else:
            y = _binarize_labels(y_raw, positive_label=positive_label)
        feature_names = np.asarray(getattr(bunch, "feature_names", [f"x{i}" for i in range(X.shape[1])]))
        meta = {
            "dataset_id": dataset,
            "kind": spec.kind,
            "description": spec.description,
            "positive_label": positive_label,
        }
        return X, y, feature_names, meta

    if spec.kind == "libsvm":
        import os
        from sklearn.datasets import load_svmlight_files
        from scipy.sparse import vstack

        cache_dir = os.path.join("data", "gas_sensor_drift")
        paths = [os.path.join(cache_dir, f"batch{i}.dat") for i in range(1, 11)]

        # Cargar batch a batch para forzar dimensionalidad consistente (n_features)
        X_parts = []
        y_parts = []
        for p in paths:
            X_b, y_b = load_svmlight_files([p], n_features=128)
            X_parts.append(X_b)
            y_parts.append(np.asarray(y_b, dtype=int))

        X_sparse = vstack(X_parts)
        y_raw = np.concatenate(y_parts, axis=0)
        X = X_sparse.toarray()

        # One-vs-rest: positive gas id vs others
        positive_gas = int(gas_positive_class)
        y = (y_raw == positive_gas).astype(int)

        feature_names = np.asarray([f"sensor{(i // 8) + 1}_feat{(i % 8) + 1}" for i in range(128)])
        meta = {
            "dataset_id": dataset,
            "kind": spec.kind,
            "description": spec.description,
            "positive_label": positive_gas,
            "gas_positive_class": positive_gas,
        }
        return X, y, feature_names, meta

    if spec.kind == "microarray":
        import os
        
        # Load from .npz file (previously downloaded from GEO)
        cache_dir = os.path.join("data", "microarray_datasets")
        os.makedirs(cache_dir, exist_ok=True)
        npz_file = os.path.join(cache_dir, f"{spec.openml_name}.npz")
        
        # Si no existe, intentar descargar automáticamente
        if not os.path.exists(npz_file):
            print(f"Descargando {dataset} desde GEO Database...")
            _download_microarray_dataset(dataset, spec.openml_name, cache_dir)
        
        data = np.load(npz_file, allow_pickle=True)
        X = np.asarray(data['X'], dtype=np.float32)
        y_raw = np.asarray(data['y'], dtype=int)
        
        # Binarize if needed
        y = _binarize_labels(y_raw, positive_label=positive_label)
        
        # Feature names stored in .npz
        feature_names = np.asarray(data['feature_names'])
        if feature_names.dtype == object:
            feature_names = feature_names.astype(str)
        
        meta = {
            "dataset_id": dataset,
            "kind": spec.kind,
            "description": spec.description,
            "positive_label": positive_label,
        }
        return X, y, feature_names, meta

    raise RuntimeError(f"Unsupported dataset kind: {spec.kind}")


def load_dataset_from_config() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Convenience wrapper: loads dataset using values in src.config."""
    from src import config

    return load_dataset(
        config.DATASET,
        positive_label_override=getattr(config, "DATASET_POSITIVE_LABEL", None),
        gas_positive_class=getattr(config, "GAS_POSITIVE_CLASS", 1),
    )
