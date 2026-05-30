from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


@dataclass(frozen=True)
class DatasetSpec:
    kind: str  # 'openml' | 'sklearn' | 'libsvm' | 'microarray' | 'mat' | 'synthetic'
    openml_name: Optional[str] = None
    openml_version: int = 1
    positive_label: Optional[Any] = None
    description: str = ""
    # Parámetros exclusivos para kind='synthetic' (make_classification)
    n_samples: int = 3000
    n_features: int = 100
    n_informative: int = 15
    n_redundant: int = 5
    class_sep: float = 1.5
    weights: Optional[tuple] = None   # e.g. (0.65, 0.35) → 35% positivos


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
    # ── Feature-selection benchmarks ──────────────────────────────────────────
    # Madelon: dataset del NIPS 2003 Feature Selection Challenge
    #   2000 muestras, 500 features: 5 key + 15 redundantes + 480 ruido puro
    #   Balance 50/50. Gold-standard para demostrar selección con señal dispersa.
    "madelon": DatasetSpec(
        kind="openml",
        openml_name="madelon",
        openml_version=1,
        positive_label="2",
        description="Madelon NIPS-2003 - 2000 muestras, 500 features (5 clave + 15 redundantes + 480 ruido)",
    ),
    # Waveform: señal basada en ondas + 19 features ruido puro
    #   5000 muestras, 40 features: 21 basadas en onda + 19 random noise
    #   3 clases → binarizado: clase 0 vs resto (≈33% positivo)
    "waveform": DatasetSpec(
        kind="openml",
        openml_name="waveform-5000",
        openml_version=1,
        positive_label="0",
        description="Waveform-5000 - 5000 muestras, 40 features (21 señal + 19 ruido), clase 0 vs resto",
    ),
    # Gisette: NIPS 2003 Feature Selection Challenge
    #   ~7000 muestras, 5000 features: 2500 reales + 2500 probe (ruido puro inyectado por diseño)
    #   Balance 50/50. Gold-standard para demostrar que naive falla con señal dispersa.
    #   Tarea: separar dígito 4 (clase "1") vs dígito 9 (clase "2")
    "gisette": DatasetSpec(
        kind="openml",
        openml_name="gisette",
        openml_version=2,
        positive_label="1",   # OpenML: 1=digito9 (positivo), -1=digito4 (negativo)
        description="Gisette NIPS-2003 - ~7000 muestras, 5000 features (2500 reales + 2500 probe ruido), digito 9 vs 4",
    ),
    # ── Sintético controlado (make_classification) ────────────────────────────
    # Garantiza los 3 comportamientos teóricos: naive falla, PU mejora, alpha converge
    "synthetic_pu": DatasetSpec(
        kind="synthetic",
        description="Sintético PU - 3000 muestras, 100 features (10 informativas + 5 redundantes + 85 ruido)",
        n_samples=3000,
        n_features=100,
        n_informative=10,   # era 15 → señal más dispersa, naive falla más
        n_redundant=5,
        class_sep=0.8,      # era 1.5 → rompe el efecto techo en AUC (~0.99→~0.94)
        weights=(0.65, 0.35),
    ),
    # ── MUSK Version 1 ───────────────────────────────────────────────────────
    # 476 conformaciones de 92 moléculas (47 musks + 45 non-musks), 166 features
    # Features: descriptores físicos conformacionales (distancias, ángulos, cargas)
    # Clase positiva: 1 (musk), negativa: 0 (non-musk)
    # Origen: UCI ML Repository (descarga directa, ID=74)
    # Formato: clean1.data.Z → 2 cols ID + 166 features numéricas + 1 clase (0/1)
    "musk_v1": DatasetSpec(
        kind="uci_musk",
        positive_label=1,
        description="MUSK v1 - 476 conformaciones, 166 features moleculares (musk=1 vs non-musk=0)",
    ),
    # Raw microarray MAT files
    "cns": DatasetSpec(
        kind="mat",
        openml_name="CNS_microarray",
        positive_label=1,
        description="CNS Microarray - 60 samples, 7130 genes (Tumor classification)",
    ),
    "dlbcl": DatasetSpec(
        kind="mat",
        openml_name="DLBCL_microarray",
        positive_label=1,
        description="DLBCL Microarray - 47 samples, 4027 genes (Lymphoma subtype classification)",
    ),
}


MICROARRAY_CV_N_SPLITS = 3
MICROARRAY_CV_N_REPEATS = 10
MICROARRAY_CV_RANDOM_STATE = 42


def available_datasets() -> Tuple[str, ...]:
    return tuple(sorted(_DATASETS.keys()))


def crear_splitter_cv(dataset_kind: str, n_splits: int, random_state: int):
    if dataset_kind in {"microarray", "mat"}:
        return RepeatedStratifiedKFold(
            n_splits=MICROARRAY_CV_N_SPLITS,
            n_repeats=MICROARRAY_CV_N_REPEATS,
            random_state=MICROARRAY_CV_RANDOM_STATE,
        )

    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


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
            print(f" Warning: {dataset_key} tiene shape {X.shape}, "
                  f"se esperaba ({expected_samples}, {expected_features})")
    
    return X, y, feature_names



def _load_musk_v1() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Descarga y carga MUSK Version 1 desde UCI ML Repository.

    Formato del fichero clean1.data (después de descomprimir .Z con LZW):
        molecule_name, conformation_name, feat_1, ..., feat_166, class
        → columna 0:   nombre molécula   (descartada)
        → columna 1:   nombre conformación (descartada)
        → columnas 2–167: 166 features numéricas
        → columna 168: clase (0 = non-musk, 1 = musk)

    El fichero se descarga como ZIP desde UCI y el .Z interno se descomprime
    con `unlzw3` (LZW puro Python). Si no está instalado se lanza un mensaje claro.

    Caché local: data/musk_v1/clean1.csv (generado la primera vez)
    """
    import os
    import zipfile
    import io
    import urllib.request

    cache_dir = os.path.join("data", "musk_v1")
    os.makedirs(cache_dir, exist_ok=True)
    csv_cache = os.path.join(cache_dir, "clean1.csv")

    # ── 1. Usar caché si ya existe ────────────────────────────────────────────
    if os.path.exists(csv_cache):
        import pandas as pd
        df = pd.read_csv(csv_cache, header=None)
    else:
        # ── 2. Descargar ZIP desde UCI ────────────────────────────────────────
        zip_url = "https://archive.ics.uci.edu/static/public/74/musk+version+1.zip"
        zip_path = os.path.join(cache_dir, "musk_v1.zip")

        if not os.path.exists(zip_path):
            print("Descargando MUSK v1 desde UCI ML Repository...")
            try:
                urllib.request.urlretrieve(zip_url, zip_path)
                print(f"  Descarga completada: {zip_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Error descargando MUSK v1:\n{e}\n"
                    f"Descarga manual desde: {zip_url}\n"
                    f"Guarda el ZIP en: {zip_path}"
                )

        # ── 3. Extraer clean1.data.Z del ZIP ─────────────────────────────────
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Buscar el fichero .Z dentro del ZIP
            z_candidates = [n for n in zf.namelist() if n.endswith("clean1.data.Z")]
            csv_candidates = [n for n in zf.namelist() if n.lower().endswith(".csv")]

            if csv_candidates:
                # Formato nuevo UCI (con CSV embebido)
                import pandas as pd
                with zf.open(csv_candidates[0]) as f:
                    df = pd.read_csv(f, header=None)
                df.to_csv(csv_cache, index=False, header=False)
                print(f"  MUSK v1 cargado desde CSV embebido: {csv_candidates[0]}")

            elif z_candidates:
                # Formato clásico UCI: descomprimir .Z (LZW)
                try:
                    from unlzw3 import unlzw
                except ImportError:
                    raise ImportError(
                        "El fichero MUSK v1 está en formato .Z (Unix compress/LZW).\n"
                        "Instala el descompresor: pip install unlzw3\n"
                        "Y vuelve a ejecutar."
                    )
                with zf.open(z_candidates[0]) as f:
                    compressed_bytes = f.read()
                raw_bytes = unlzw(compressed_bytes)
                text = raw_bytes.decode("ascii")

                import pandas as pd
                df = pd.read_csv(io.StringIO(text), header=None)
                df.to_csv(csv_cache, index=False, header=False)
                print(f"  MUSK v1 descomprimido (.Z->CSV) y guardado en cache: {csv_cache}")

            else:
                available = zf.namelist()
                raise RuntimeError(
                    f"No se encontró clean1.data.Z ni CSV en el ZIP.\n"
                    f"Ficheros disponibles: {available}"
                )

    # ── 4. Parsear: col0=mol_name, col1=conf_name, cols2-167=features, col168=class ─
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(csv_cache, header=None)

    # Sanidad: debería tener 169 columnas
    if df.shape[1] not in (168, 169):
        raise ValueError(
            f"Se esperaban 168 o 169 columnas en MUSK v1, se encontraron {df.shape[1]}.\n"
            f"Primeras columnas: {list(df.columns[:5])}\n"
            f"Borra la caché en {cache_dir} y vuelve a intentarlo."
        )

    n_cols = df.shape[1]
    # Descartar cols 0 y 1 (identificadores), última columna = clase
    X = df.iloc[:, 2:n_cols - 1].values.astype(np.float32)
    y_raw = df.iloc[:, n_cols - 1].values.astype(int)

    # Clase: 0 = non-musk, 1 = musk (ya en formato binario)
    y = y_raw.copy()

    feature_names = np.array([f"feat_{i + 1}" for i in range(X.shape[1])])

    print(
        f"  [musk_v1] {X.shape[0]} conformaciones | {X.shape[1]} features | "
        f"musk={int((y == 1).sum())} non-musk={int((y == 0).sum())}"
    )

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
        # Algunos datasets (p.ej. Gisette) vienen como scipy sparse matrix desde OpenML.
        # np.asarray(sparse) produce un array 0-D → usar .toarray() para convertir a denso.
        from scipy.sparse import issparse
        X = bunch.data.toarray() if issparse(bunch.data) else np.asarray(bunch.data)
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

    if spec.kind == "mat":
        import os
        from scipy.io import loadmat
        
        cache_dir = os.path.join("data", "microarray_datasets")
        mat_file = os.path.join(cache_dir, f"{spec.openml_name}.mat")
        
        if not os.path.exists(mat_file):
            raise FileNotFoundError(f"MAT file not found: {mat_file}")
        
        # Load MATLAB file
        mat_data = loadmat(mat_file)
        
        # Extract data matrix (should be 'data' key for CNS/DLBCL)
        if 'data' not in mat_data:
            raise ValueError(
                f"No 'data' key in {mat_file}. Available keys: {[k for k in mat_data.keys() if not k.startswith('__')]}"
            )
        
        data = np.asarray(mat_data['data'], dtype=np.float32)
        
        # Last column contains class labels (1 or 2)
        # Features = all columns except last
        X = data[:, :-1]
        y_raw = data[:, -1].astype(int)
        
        # Convert labels from {1,2} to {0,1}
        y = (y_raw == positive_label).astype(int) if positive_label in (1, 2) else y_raw - 1
        
        # Generate feature names
        feature_names = np.array([f"gene_{i+1}" for i in range(X.shape[1])])
        
        print(f"  Loaded {spec.openml_name}: X shape {X.shape}, y shape {y.shape}")
        print(f"  Class distribution: {np.bincount(y)}")
        
        meta = {
            "dataset_id": dataset,
            "kind": spec.kind,
            "description": spec.description,
            "positive_label": positive_label,
            "label_mapping": f"{{2->0, {positive_label}->1}}",
        }
        return X, y, feature_names, meta

    if spec.kind == "synthetic":
        from sklearn.datasets import make_classification

        weights = list(spec.weights) if spec.weights is not None else None
        X, y = make_classification(
            n_samples=spec.n_samples,
            n_features=spec.n_features,
            n_informative=spec.n_informative,
            n_redundant=spec.n_redundant,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            weights=weights,
            class_sep=spec.class_sep,
            flip_y=0.01,        # pequeño ruido de etiquetado (realismo)
            random_state=42,    # fijo para que el dataset sea reproducible entre runs
        )
        y = y.astype(int)

        feature_names = np.array(
            [f"info_{i+1}" for i in range(spec.n_informative)]
            + [f"redun_{i+1}" for i in range(spec.n_redundant)]
            + [f"noise_{i+1}" for i in range(spec.n_features - spec.n_informative - spec.n_redundant)]
        )

        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        print(
            f"  [synthetic_pu] {spec.n_samples} muestras | {spec.n_features} features "
            f"({spec.n_informative} inform. + {spec.n_redundant} redund. + "
            f"{spec.n_features - spec.n_informative - spec.n_redundant} ruido) | "
            f"pos={n_pos} neg={n_neg}"
        )

        meta = {
            "dataset_id": dataset,
            "kind": spec.kind,
            "description": spec.description,
            "positive_label": 1,
            "n_informative": spec.n_informative,
            "n_redundant": spec.n_redundant,
            "n_noise": spec.n_features - spec.n_informative - spec.n_redundant,
            "class_sep": spec.class_sep,
        }
        return X, y, feature_names, meta

    if spec.kind == "uci_musk":
        X, y, feature_names = _load_musk_v1()
        # Binarize: clase positiva = 1 (musk), negativa = 0 (non-musk)
        effective_positive_label = positive_label if positive_label is not None else 1
        if effective_positive_label != 1:
            # Permitir invertir la clase si se quiere
            y = (y == effective_positive_label).astype(int)
        meta = {
            "dataset_id": dataset,
            "kind": spec.kind,
            "description": spec.description,
            "positive_label": effective_positive_label,
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
