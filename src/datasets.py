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


def _download_dataset(dataset_key: str, cache_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Descarga dataset de microarray desde scikit-feature repository.
    
    Fuente: https://github.com/jundongl/scikit-feature
    
    Datasets:
    - colon_cancer:     62 samples × 2000 features
    - prostate_cancer: 102 samples × 6000 features
    - lung_cancer:     181 samples × 12000 features
    """
    import os
    import urllib.request
    
    scikit_feature_urls = {
        "colon_cancer": "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/colon.mat",
        "prostate_cancer": "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/prostate.mat",
        "lung_cancer": "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/lung.mat",
    }
    
    if dataset_key not in scikit_feature_urls:
        raise ValueError(f"Dataset {dataset_key} no disponible")
    
    url = scikit_feature_urls[dataset_key]
    mat_file = os.path.join(cache_dir, f"{dataset_key}.mat")
    
    # Descargar si no existe
    if not os.path.exists(mat_file):
        print(f"Descargando {dataset_key} desde scikit-feature...")
        try:
            urllib.request.urlretrieve(url, mat_file)
            print(f" {dataset_key} descargado: {mat_file}")
        except Exception as e:
            raise RuntimeError(
                f"Error descargando {dataset_key}:\n{e}\n"
                f"Intenta descargar manualmente desde: {url}"
            )
    
    # Cargar archivo .mat usando scipy
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy no instalado. Ejecuta: pip install scipy")
    
    mat_data = loadmat(mat_file)
    
    # Extraer X, y según el formato de scikit-feature
    # Formato típico: 'X' para features, 'Y' o 'y' para labels
    X = None
    y = None
    
    for key in ['X', 'x']:
        if key in mat_data:
            X = np.asarray(mat_data[key], dtype=np.float32)
            break
    
    for key in ['Y', 'y']:
        if key in mat_data:
            y_raw = np.asarray(mat_data[key], dtype=int).ravel()
            # Convertir a binario si es necesario (1-indexado a 0-indexado)
            if np.min(y_raw) == 1:
                y = y_raw - 1  # Convertir [1,2] a [0,1]
            else:
                y = y_raw
            break
    
    if X is None or y is None:
        raise ValueError(
            f"No se pudieron extraer X e y del archivo {mat_file}.\n"
            f"Claves disponibles: {list(mat_data.keys())}"
        )
    
    # Generar nombres de features
    feature_names = np.array([f"gene_{i+1}" for i in range(X.shape[1])])
    
    # Validar dimensiones esperadas
    expected_dims = {
        "colon_cancer": (62, 2000),
        "prostate_cancer": (102, 6000),
        "lung_cancer": (181, 12000),
    }
    
    if dataset_key in expected_dims:
        expected_samples, expected_features = expected_dims[dataset_key]
        if X.shape != (expected_samples, expected_features):
            print(f"⚠ Warning: {dataset_key} tiene shape {X.shape}, "
                  f"se esperaba ({expected_samples}, {expected_features})")
    
    return X, y, feature_names



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
        
        cache_dir = os.path.join("data", "microarray_datasets")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Descargar desde scikit-feature (con caché local)
        X, y_raw, feature_names = _download_dataset(spec.openml_name, cache_dir)
        
        # Binarize if needed
        y = _binarize_labels(y_raw, positive_label=positive_label)
        
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
