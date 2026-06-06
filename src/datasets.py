from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


@dataclass(frozen=True)
class DatasetSpec:
    kind: str  # 'openml' | 'sklearn' | 'libsvm' | 'microarray' | 'mat' | 'synthetic' | 'uci_nips2003'
    openml_name: Optional[str] = None
    openml_version: int = 1
    positive_label: Optional[Any] = None
    positive_labels: Optional[tuple] = None  # multi-class binarization: y in positive_labels -> 1
    description: str = ""
    # Parámetros exclusivos para kind='synthetic' (make_classification)
    n_samples: int = 3000
    n_features: int = 100
    n_informative: int = 15
    n_redundant: int = 5
    class_sep: float = 1.5
    weights: Optional[tuple] = None   # e.g. (0.65, 0.35) → 35% positivos
    # Parámetros exclusivos para kind='synthetic_isotropic'
    # X_i ~ N(d_i*(2y-1), 1) → MI(X_i;y) ≈ d_i²/2 para d_i pequeño
    d_informative_range: tuple = (0.40, 0.49)  # MI ≈ [0.08, 0.12]
    d_noise_range: tuple = (0.28, 0.40)         # MI ≈ [0.04, 0.08]
    # Parámetros exclusivos para kind='uci_nips2003'
    uci_id: int = 0
    nips_format: str = ""   # 'dense' | 'sparse_kv' | 'sparse_binary'
    uci_zip_name: str = ""  # nombre del ZIP en UCI (sin .zip); si vacío usa la clave del dataset
    # Parámetros exclusivos para kind='moleculenet_tox21'
    tox21_task: str = "NR-AR"  # tarea de toxicología: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase,
                                # NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53


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
    # ── AutoML Challenge 2 / UCI ─────────────────────────────────────────────
    # Jasmine: dataset tabular binario del AutoML Challenge 2 (OpenML #41143)
    #   ~2984 muestras, 144 features. n/p = 20.7. Balance ~50/50.
    #   positive_label: "pos" (si falla, np.unique(bunch.target) mostrara las etiquetas reales)
    "jasmine": DatasetSpec(
        kind="automl_challenge",
        positive_label=1,
        description="Jasmine (AutoML Challenge 2) - 2984 muestras, 144 features, binario",
    ),
    # Internet Advertisements (UCI #51)
    #   3279 muestras, 1558 features (3 cont. + 1555 binarias: URLs/patrones imagen).
    #   Binario: 'ad.' = anuncio (positivo), 'nonad.' = contenido legítimo.
    #   Alpha real ≈ 0.14 (14% anuncios). Features muy correlacionadas por diseño
    #   (patrones URL co-ocurrentes) → naive MI falla por correlación + alpha pequeño.
    #   Missing values en las 3 features continuas → imputados con mediana.
    "internet_ads": DatasetSpec(
        kind="uci_internet_ads",
        positive_label="ad.",
        description="Internet Advertisements (UCI #51) - 3279 muestras, 1558 features (3 cont. + 1555 bin.), ad=1 vs nonad=0, alpha≈0.14",
    ),
    # Epsilon (PASCAL Large Scale Learning Challenge 2008)
    #   400K muestras, 2000 features numéricas densas, balanceado 50/50.
    #   Descarga directa desde LIBSVM repository con streaming (solo descarga MAX_SAMPLES líneas).
    #   Etiquetas originales: +1 / -1 → convertidas a {1, 0}.
    "epsilon": DatasetSpec(
        kind="libsvm_epsilon",
        positive_label=1,
        description="Epsilon PASCAL 2008 - 400K muestras, 2000 features densas, balanceado 50/50 (LIBSVM streaming)",
    ),
    # Phishing Websites (UCI / OpenML #4534)
    #   11055 muestras, 30 features categoricas (-1/0/1: URL/pagina/dominio).
    #   Binario: -1 = phishing (positivo, clase de fraude), 1 = legitimo.
    #   n/p = 368. Balance ~55% phishing / 45% legitimo. LR no supera 95% facilmente.
    #   positive_label: "-1" (si falla ajustar segun np.unique(bunch.target))
    "phishing_websites": DatasetSpec(
        kind="uci_phishing",
        positive_label="-1",
        description="Phishing Websites (UCI #327) - 2456 muestras, 30 features, phishing=-1 vs legitimo=1",
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
    # ── Sintético isótropo: todas las features tienen MI parecida ────────────
    # 5000 muestras, 100 features: 10 informativas con MI(X;y) ∈ [0.08, 0.12]
    # y 90 de ruido con MI(X;y) ∈ [0.04, 0.08].
    # Con alpha pequeño, el ruido en MI(X;S) supera el gap (≈0.04) y naive
    # eleva features de ruido al top, mientras PU (corregido por alpha) lo evita.
    "synthetic_pu2": DatasetSpec(
        kind="synthetic_isotropic",
        description="Sintético isótropo - 5000 muestras, 100 features (10 inform. MI 0.08-0.12 + 90 ruido MI 0.04-0.08)",
        n_samples=5000,
        n_features=100,
        n_informative=10,
        d_informative_range=(0.40, 0.49),
        d_noise_range=(0.28, 0.40),
    ),
    # ── CorrAL100: benchmark de features correlacionadas ─────────────────────
    # 100000 muestras, 100 features: 5 informativas + 45 redundantes (combinaciones
    # lineales de las informativas) + 50 ruido puro. Balance 50/50.
    # Diseñado para que NB falle (viola independencia por correlación) y PU funcione
    # (muestra grande, clases separables con las features correctas).
    "corral100": DatasetSpec(
        kind="synthetic",
        description="CorrAL100 - 100000 muestras, 100 features (5 informativas + 45 redundantes correladas + 50 ruido), balanceado",
        n_samples=100000,
        n_features=100,
        n_informative=5,
        n_redundant=45,
        class_sep=1.5,
        weights=None,
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
    # ── MUSK Version 2 ───────────────────────────────────────────────────────
    # 6598 conformaciones de 102 moléculas, 166 features moleculares
    # ~40% positivos (musk=1), ~60% negativos (non-musk=0)
    # Origen: UCI ML Repository (ID=75)
    "musk_v2": DatasetSpec(
        kind="uci_musk_v2",
        positive_label=1,
        description="MUSK v2 - 6598 conformaciones, 166 features moleculares (musk=1 vs non-musk=0)",
    ),
    # ── NIPS 2003 Feature Selection Challenge (UCI directo) ───────────────────
    # Arcene: microarray, detección de cáncer (cancer vs normal)
    #   300 muestras (train=100 + valid=100 + test=100), 10000 features (masa/carga iónica)
    #   Labels: +1 (cancer) / -1 (normal). Formato: denso float.
    "arcene": DatasetSpec(
        kind="uci_nips2003",
        uci_id=167,
        n_features=10000,
        nips_format="dense",
        positive_label=1,
        description="Arcene NIPS-2003 - 300 muestras, 10000 features (cancer vs normal, microarray)",
    ),
    # Gisette extenso: reconocimiento de dígitos 4 vs 9 (versión completa UCI)
    #   13500 muestras (train=6000 + valid=1000 + test=6500), 5000 features
    #   2500 features reales + 2500 probe (ruido inyectado por diseño).
    #   Formato: denso int.
    "gisette_extenso": DatasetSpec(
        kind="uci_nips2003",
        uci_id=170,
        n_features=5000,
        nips_format="dense",
        uci_zip_name="gisette",
        positive_label=1,
        description="Gisette NIPS-2003 extenso - 13500 muestras, 5000 features (dígito 4 vs 9, UCI completo)",
    ),
    # Dorothea: drug discovery, features binarias (presencia de subestructuras químicas)
    #   1950 muestras (train=800 + valid=350 + test=800), 100000 features binarias
    #   Muy disperso: ~1% de features activas por muestra. Formato: sparse binary (índices 1-based).
    "dorothea": DatasetSpec(
        kind="uci_nips2003",
        uci_id=169,
        n_features=100000,
        nips_format="sparse_binary",
        positive_label=1,
        description="Dorothea NIPS-2003 - 1950 muestras, 100000 features binarias (drug discovery, sparse)",
    ),
    # Dexter: categorización de texto (Reuters), features TF-IDF
    #   2600 muestras (train=300 + valid=300 + test=2000), 20000 features
    #   Muy disperso. Formato: sparse key:value (índices 1-based).
    "dexter": DatasetSpec(
        kind="uci_nips2003",
        uci_id=168,
        n_features=20000,
        nips_format="sparse_kv",
        positive_label=1,
        description="Dexter NIPS-2003 - 2600 muestras, 20000 features (text categorization, TF-IDF sparse)",
    ),
    # Madelon extenso: dataset artificial del NIPS 2003 (versión completa UCI)
    #   4400 muestras (train=2000 + valid=600 + test=1800), 500 features
    #   5 features clave + 15 redundantes + 480 ruido puro. Formato: denso float.
    "madelon_extenso": DatasetSpec(
        kind="uci_nips2003",
        uci_id=171,
        n_features=500,
        nips_format="dense",
        uci_zip_name="madelon",
        positive_label=1,
        description="Madelon NIPS-2003 extenso - 4400 muestras, 500 features (5 clave + 15 redund. + 480 ruido, UCI completo)",
    ),
    # ── MoleculeNet Tox21 (ECFP4 fingerprints) ──────────────────────────────────
    # 7831 compuestos, 2048 features binarias (ECFP4 radius=2).
    # Cada bit = presencia/ausencia de una subestructura circular en la molécula.
    # Estructura idéntica a bag-of-words: sparse binario de alta dimensión.
    # 12 tareas de toxicología binarias independientes (receptores nucleares + estrés).
    # Origen: Tox21 Data Challenge 2014 / MoleculeNet benchmark (Wu et al. 2018).
    # Requiere: pip install rdkit
    "tox21": DatasetSpec(
        kind="moleculenet_tox21",
        tox21_task="NR-AR",
        positive_label=1,
        description="Tox21 MoleculeNet - ~7800 compuestos, 2048 features ECFP4 binarias (NR-AR: androgenic receptor)",
    ),
    "tox21_sr_mmp": DatasetSpec(
        kind="moleculenet_tox21",
        tox21_task="SR-MMP",
        positive_label=1,
        description="Tox21 SR-MMP - ~7800 compuestos, 2048 features ECFP4 binarias (stress response mitochondrial membrane potential)",
    ),
    "tox21_nr_ahr": DatasetSpec(
        kind="moleculenet_tox21",
        tox21_task="NR-AhR",
        positive_label=1,
        description="Tox21 NR-AhR - ~7800 compuestos, 2048 features ECFP4 binarias (aryl hydrocarbon receptor)",
    ),
    # ── DREBIN Android Malware ───────────────────────────────────────────────────
    # ~5560 malware + ~123K goodware (según subset disponible), ~500K features binarias.
    # Features: presencia/ausencia de API calls, permisos, intents, actividades en APK.
    # Estructura: cada muestra es un conjunto de strings de features → matriz binaria.
    # Origen: Arp et al. 2014, "DREBIN: Effective and Explainable Detection of Android Malware".
    # Descarga (requiere registro): https://drebin.mlsec.org/
    # Coloca los archivos en: data/drebin/malware/ y data/drebin/goodware/
    # (un archivo .txt por muestra, una feature por línea)
    "drebin": DatasetSpec(
        kind="drebin",
        positive_label=1,
        description="DREBIN Android Malware - ~5K malware vs goodware, ~500K features binarias (API calls, permisos)",
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


def _load_musk_v2() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Descarga y carga MUSK Version 2 desde UCI ML Repository.

    Formato de clean2.data (igual que v1):
        molecule_name, conformation_name, feat_1, ..., feat_166, class
        → columna 0:   nombre molécula   (descartada)
        → columna 1:   nombre conformación (descartada)
        → columnas 2–167: 166 features numéricas
        → columna 168: clase (0 = non-musk, 1 = musk)

    Caché local: data/musk_v2/clean2.csv
    """
    import os
    import zipfile
    import io
    import urllib.request

    cache_dir = os.path.join("data", "musk_v2")
    os.makedirs(cache_dir, exist_ok=True)
    csv_cache = os.path.join(cache_dir, "clean2.csv")

    if os.path.exists(csv_cache):
        import pandas as pd
        df = pd.read_csv(csv_cache, header=None)
    else:
        zip_url = "https://archive.ics.uci.edu/static/public/75/musk+version+2.zip"
        zip_path = os.path.join(cache_dir, "musk_v2.zip")

        if not os.path.exists(zip_path):
            print("Descargando MUSK v2 desde UCI ML Repository...")
            try:
                urllib.request.urlretrieve(zip_url, zip_path)
                print(f"  Descarga completada: {zip_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Error descargando MUSK v2:\n{e}\n"
                    f"Descarga manual desde: {zip_url}\n"
                    f"Guarda el ZIP en: {zip_path}"
                )

        with zipfile.ZipFile(zip_path, 'r') as zf:
            z_candidates = [n for n in zf.namelist() if n.endswith("clean2.data.Z")]
            csv_candidates = [n for n in zf.namelist() if n.lower().endswith(".csv")]

            if csv_candidates:
                import pandas as pd
                with zf.open(csv_candidates[0]) as f:
                    df = pd.read_csv(f, header=None)
                df.to_csv(csv_cache, index=False, header=False)
                print(f"  MUSK v2 cargado desde CSV embebido: {csv_candidates[0]}")

            elif z_candidates:
                try:
                    from unlzw3 import unlzw
                except ImportError:
                    raise ImportError(
                        "El fichero MUSK v2 está en formato .Z (Unix compress/LZW).\n"
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
                print(f"  MUSK v2 descomprimido (.Z->CSV) y guardado en cache: {csv_cache}")

            else:
                available = zf.namelist()
                raise RuntimeError(
                    f"No se encontró clean2.data.Z ni CSV en el ZIP.\n"
                    f"Ficheros disponibles: {available}"
                )

    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(csv_cache, header=None)

    if df.shape[1] not in (168, 169):
        raise ValueError(
            f"Se esperaban 168 o 169 columnas en MUSK v2, se encontraron {df.shape[1]}.\n"
            f"Borra la caché en {cache_dir} y vuelve a intentarlo."
        )

    n_cols = df.shape[1]
    X = df.iloc[:, 2:n_cols - 1].values.astype(np.float32)
    y_raw = df.iloc[:, n_cols - 1].values.astype(int)
    y = y_raw.copy()

    feature_names = np.array([f"feat_{i + 1}" for i in range(X.shape[1])])

    print(
        f"  [musk_v2] {X.shape[0]} conformaciones | {X.shape[1]} features | "
        f"musk={int((y == 1).sum())} non-musk={int((y == 0).sum())}"
    )

    return X, y, feature_names


def _load_phishing_websites() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Descarga y carga Phishing Websites desde UCI ML Repository (ID 379).

    30 features categoricas {-1, 0, 1}, target Result {-1=phishing, 1=legitimo}.
    Cache local: data/phishing_websites/Training Dataset.arff
    """
    import os
    import zipfile
    import urllib.request

    cache_dir = os.path.join("data", "phishing_websites")
    os.makedirs(cache_dir, exist_ok=True)
    arff_cache = os.path.join(cache_dir, "Training Dataset.arff")

    if not os.path.exists(arff_cache):
        zip_url = "https://archive.ics.uci.edu/static/public/327/phishing+websites.zip"
        zip_path = os.path.join(cache_dir, "phishing_websites.zip")

        if not os.path.exists(zip_path):
            print("Descargando Phishing Websites desde UCI ML Repository...")
            try:
                urllib.request.urlretrieve(zip_url, zip_path)
                print(f"  Descarga completada: {zip_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Error descargando Phishing Websites:\n{e}\n"
                    f"Descarga manual desde: https://archive.ics.uci.edu/dataset/327/phishing+websites\n"
                    f"Guarda el ZIP en: {zip_path}"
                )

        with zipfile.ZipFile(zip_path, "r") as zf:
            arff_names = [n for n in zf.namelist() if n.lower().endswith(".arff")]
            if not arff_names:
                raise RuntimeError(
                    f"No se encontro fichero ARFF en el ZIP.\n"
                    f"Contenido: {zf.namelist()}"
                )
            member = arff_names[0]
            zf.extract(member, cache_dir)
            extracted = os.path.join(cache_dir, member)
            if extracted != arff_cache:
                os.replace(extracted, arff_cache)
        print(f"  ARFF extraido en: {arff_cache}")

    from scipy.io.arff import loadarff
    data, meta_arff = loadarff(arff_cache)

    import pandas as pd
    df = pd.DataFrame(data)

    target_col = "Result"
    if target_col not in df.columns:
        target_col = df.columns[-1]

    y_raw = df[target_col].values
    if len(y_raw) > 0 and hasattr(y_raw[0], "decode"):
        y_raw = np.array([v.decode("utf-8") for v in y_raw])

    feature_cols = [c for c in df.columns if c != target_col]
    X_raw = df[feature_cols].values
    # Nominal attributes come as bytes; convert to float
    if len(X_raw) > 0 and hasattr(X_raw.flat[0], "decode"):
        X = np.array([[float(v.decode("utf-8")) for v in row] for row in X_raw], dtype=np.float32)
    else:
        X = X_raw.astype(np.float32)

    feature_names = np.asarray(feature_cols)

    print(
        f"  [phishing_websites] {X.shape[0]} muestras | {X.shape[1]} features | "
        f"phishing={int((y_raw == '-1').sum())} legitimo={int((y_raw == '1').sum())}"
    )
    return X, y_raw, feature_names


def _load_epsilon_libsvm(max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Descarga epsilon desde LIBSVM repository usando streaming bz2.

    Solo descarga las primeras `max_samples` líneas del stream comprimido,
    evitando bajar los ~2.3 GB completos cuando MAX_SAMPLES está configurado.
    Caché local: data/epsilon/epsilon_n{max_samples}.libsvm  (o epsilon_full.libsvm)

    Etiquetas originales +1/-1 → convertidas a {1, 0}.
    """
    import os
    import bz2
    import urllib.request
    from sklearn.datasets import load_svmlight_file
    from scipy.sparse import issparse

    cache_dir = os.path.join("data", "epsilon")
    os.makedirs(cache_dir, exist_ok=True)
    tag = f"n{max_samples}" if max_samples else "full"
    libsvm_cache = os.path.join(cache_dir, f"epsilon_{tag}.libsvm")

    if not os.path.exists(libsvm_cache):
        url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2"
        print(f"Descargando epsilon desde LIBSVM (streaming, max_samples={max_samples})...")
        req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
        import ssl
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        lines_collected = []
        try:
            with urllib.request.urlopen(req, timeout=120, context=ssl_ctx) as resp:
                decomp = bz2.BZ2Decompressor()
                partial = b""
                bytes_dl = 0
                while True:
                    raw = resp.read(1 << 16)   # 64 KB
                    if not raw:
                        break
                    bytes_dl += len(raw)
                    try:
                        data = decomp.decompress(raw)
                    except EOFError:
                        break
                    partial += data
                    while b"\n" in partial:
                        line, partial = partial.split(b"\n", 1)
                        if line.strip():
                            lines_collected.append(line)
                            if max_samples and len(lines_collected) >= max_samples:
                                break
                    if max_samples and len(lines_collected) >= max_samples:
                        print(f"  {len(lines_collected)} líneas / {bytes_dl / 2**20:.0f} MB descargados")
                        break
                    if lines_collected and len(lines_collected) % 50_000 == 0:
                        print(f"  {len(lines_collected)} líneas leídas...")
        except Exception as e:
            raise RuntimeError(
                f"Error descargando epsilon:\n{e}\n"
                f"Descarga manual desde: {url}\n"
                f"Descomprime y guarda como: {os.path.abspath(libsvm_cache)}"
            )
        if not lines_collected:
            raise RuntimeError("No se descargaron líneas de epsilon. Comprueba la conexión.")

        with open(libsvm_cache, "wb") as f:
            f.write(b"\n".join(lines_collected) + b"\n")
        print(f"  Caché guardado: {libsvm_cache}")

    X_sp, y_raw = load_svmlight_file(libsvm_cache, n_features=2000)
    X = X_sp.toarray().astype(np.float32) if issparse(X_sp) else np.asarray(X_sp, dtype=np.float32)
    y_raw = np.asarray(y_raw)
    y = (y_raw > 0).astype(int)   # +1 → 1, -1 → 0

    feature_names = np.array([f"feat_{i + 1}" for i in range(X.shape[1])])
    print(
        f"  [epsilon] {X.shape[0]} muestras | {X.shape[1]} features | "
        f"+1={int((y == 1).sum())} -1={int((y == 0).sum())}"
    )
    return X, y, feature_names


def _load_internet_ads() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Descarga y carga Internet Advertisements desde UCI ML Repository (ID 51).

    3279 muestras, 1558 features (3 continuas: height/width/aratio + 1555 binarias).
    Missing values en cols 0-2 marcados como '?' → imputados con mediana de columna.
    Clases: 'ad.' (positivo=1) vs 'nonad.' (negativo=0). Alpha real ≈ 0.14.
    Caché local: data/internet_ads/ad.data
    """
    import os
    import zipfile
    import urllib.request

    cache_dir = os.path.join("data", "internet_ads")
    os.makedirs(cache_dir, exist_ok=True)
    csv_cache = os.path.join(cache_dir, "ad.data")

    if not os.path.exists(csv_cache):
        zip_url = "https://archive.ics.uci.edu/static/public/51/internet+advertisements.zip"
        zip_path = os.path.join(cache_dir, "internet_ads.zip")

        if not os.path.exists(zip_path):
            print("Descargando Internet Advertisements desde UCI ML Repository...")
            try:
                urllib.request.urlretrieve(zip_url, zip_path)
                print(f"  Descarga completada: {zip_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Error descargando Internet Ads:\n{e}\n"
                    f"Descarga manual desde: https://archive.ics.uci.edu/dataset/51/internet+advertisements\n"
                    f"Guarda el ZIP en: {zip_path}"
                )

        with zipfile.ZipFile(zip_path, "r") as zf:
            data_candidates = [n for n in zf.namelist() if n.lower().endswith("ad.data")]
            if not data_candidates:
                raise RuntimeError(
                    f"No se encontró ad.data en el ZIP.\nContenido: {zf.namelist()}"
                )
            with zf.open(data_candidates[0]) as f:
                content = f.read().decode("utf-8", errors="replace")
        with open(csv_cache, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ad.data extraído en: {csv_cache}")

    import pandas as pd

    # '   ?' y variantes son missing values en las 3 primeras columnas (continuas)
    df = pd.read_csv(
        csv_cache, header=None,
        na_values=["   ?", "  ?", " ?", "?"],
        skipinitialspace=True,
    )

    y_raw = np.array([str(v).strip() for v in df.iloc[:, -1].values])
    X_df = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")

    # Imputa con mediana por columna (solo cols 0-2 tienen NaN en la práctica)
    for col in X_df.columns:
        if X_df[col].isna().any():
            X_df[col] = X_df[col].fillna(X_df[col].median())

    X = X_df.values.astype(np.float32)
    feature_names = np.array([f"feat_{i + 1}" for i in range(X.shape[1])])

    n_ad = int((y_raw == "ad.").sum())
    n_nonad = int((y_raw == "nonad.").sum())
    print(
        f"  [internet_ads] {X.shape[0]} muestras | {X.shape[1]} features | "
        f"ad={n_ad} nonad={n_nonad} alpha_real={n_ad / (n_ad + n_nonad):.2f}"
    )
    return X, y_raw, feature_names


def _generate_synthetic_isotropic(spec: "DatasetSpec"):
    """Genera dataset sintético isótropo con gap de MI controlado.

    Modelo de generación:
        X_i ~ N( d_i * (2*y - 1),  1 )

    donde d_i controla MI(X_i; y) ≈ d_i²/2 (aproximación válida para d<0.6).
    Las features informativas tienen d_i ∈ spec.d_informative_range y las de
    ruido d_j ∈ spec.d_noise_range. El gap entre rangos es ≈ 0.04 nats, justo
    suficiente para que con alpha pequeño MI(X_j; S) supere MI(X_i; S) por ruido.
    """
    rng = np.random.RandomState(42)
    n = spec.n_samples
    n_info = spec.n_informative
    n_noise = spec.n_features - n_info

    # Etiquetas balanceadas, barajadas
    y = np.zeros(n, dtype=int)
    y[: n // 2] = 1
    rng.shuffle(y)

    d_info  = rng.uniform(*spec.d_informative_range, size=n_info)
    d_noise = rng.uniform(*spec.d_noise_range,        size=n_noise)
    d_all   = np.concatenate([d_info, d_noise])

    signs = (2 * y - 1).astype(float)          # +1 clase positiva, -1 negativa
    X = rng.normal(0.0, 1.0, size=(n, len(d_all)))
    X += signs[:, None] * d_all[None, :]        # desplaza la media según la clase

    feature_names = np.array(
        [f"info_{i + 1}" for i in range(n_info)] +
        [f"noise_{i + 1}" for i in range(n_noise)]
    )
    return X.astype(np.float32), y, feature_names


def _load_nips2003(
    dataset_name: str,
    uci_id: int,
    n_features: int,
    fmt: str,
    uci_zip_name: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carga datasets NIPS-2003 Feature Selection Challenge desde UCI ML Repository.

    Formatos soportados:
      'dense'         – filas de floats separados por espacios (Arcene, Gisette, Madelon)
      'sparse_kv'     – pares índice:valor 1-based (Dexter)
      'sparse_binary' – lista de índices 1-based de features activas (Dorothea)

    Labels: +1 / -1 → convertidas a {1, 0}.
    Combina todos los splits (train/valid/test) que tengan etiquetas en el ZIP.
    Caché: data/{dataset_name}/{dataset_name}.zip
    """
    import os, zipfile, io, urllib.request

    cache_dir = os.path.join("data", dataset_name)
    os.makedirs(cache_dir, exist_ok=True)

    zip_file_name = uci_zip_name if uci_zip_name else dataset_name
    zip_url = f"https://archive.ics.uci.edu/static/public/{uci_id}/{zip_file_name}.zip"
    zip_path = os.path.join(cache_dir, f"{zip_file_name}.zip")

    if not os.path.exists(zip_path):
        print(f"Descargando {dataset_name} desde UCI (ID={uci_id})...")
        try:
            urllib.request.urlretrieve(zip_url, zip_path)
            print(f"  Descarga completada: {zip_path}")
        except Exception as e:
            raise RuntimeError(
                f"Error descargando {dataset_name} (UCI ID={uci_id}):\n{e}\n"
                f"Descarga manual desde: https://archive.ics.uci.edu/dataset/{uci_id}\n"
                f"Guarda el ZIP en: {os.path.abspath(zip_path)}"
            )

    X_parts: list = []
    y_parts: list = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_files = zf.namelist()

        for split in ("train", "valid", "test"):
            data_file = next(
                (f for f in all_files if f.lower().endswith(f"_{split}.data")), None
            )
            label_file = next(
                (f for f in all_files if f.lower().endswith(f"_{split}.labels")), None
            )

            if label_file is None:
                if split in ("train", "valid"):
                    raise RuntimeError(
                        f"No se encontró {split}.labels para {dataset_name}.\n"
                        f"Archivos en ZIP: {all_files}"
                    )
                continue  # test labels pueden no estar incluidas

            if data_file is None:
                if split in ("train", "valid"):
                    raise RuntimeError(
                        f"No se encontró {split}.data para {dataset_name}.\n"
                        f"Archivos en ZIP: {all_files}"
                    )
                continue

            with zf.open(label_file) as fh:
                y_split = np.array(
                    [int(v) for v in fh.read().decode("ascii", errors="replace").split()],
                    dtype=int,
                )
            n_split = len(y_split)

            with zf.open(data_file) as fh:
                data_text = fh.read().decode("ascii", errors="replace")

            lines = [ln for ln in data_text.splitlines() if ln.strip()]

            if fmt == "dense":
                rows = [np.fromstring(ln, dtype=np.float32, sep=" ") for ln in lines]
                X_split = np.stack(rows).astype(np.float32)

            elif fmt == "sparse_kv":
                X_split = np.zeros((n_split, n_features), dtype=np.float32)
                for i, ln in enumerate(lines[:n_split]):
                    for token in ln.split():
                        if ":" in token:
                            idx_s, val_s = token.split(":", 1)
                            idx = int(idx_s) - 1
                            if 0 <= idx < n_features:
                                X_split[i, idx] = float(val_s)

            elif fmt == "sparse_binary":
                X_split = np.zeros((n_split, n_features), dtype=np.float32)
                for i, ln in enumerate(lines[:n_split]):
                    for token in ln.split():
                        idx = int(token) - 1
                        if 0 <= idx < n_features:
                            X_split[i, idx] = 1.0

            else:
                raise ValueError(f"nips_format desconocido: {fmt!r}")

            X_parts.append(X_split)
            y_parts.append(y_split)
            print(f"  [{dataset_name}] {split}: {n_split} muestras")

    if not X_parts:
        raise RuntimeError(f"No se cargó ningún split para {dataset_name}")

    X = np.concatenate(X_parts, axis=0)
    y_raw = np.concatenate(y_parts, axis=0)
    y = (y_raw > 0).astype(int)  # ±1 → {1, 0}

    feature_names = np.array([f"feat_{i + 1}" for i in range(n_features)])
    print(
        f"  [{dataset_name}] total: {X.shape[0]} muestras | {X.shape[1]} features "
        f"| pos={int((y == 1).sum())} neg={int((y == 0).sum())}"
    )
    return X, y, feature_names


def _load_tox21(task: str = "NR-AR") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carga Tox21 con fingerprints ECFP4 (2048 bits binarios).

    Descarga el CSV desde DeepChem/AWS (caché en data/tox21/).
    7831 compuestos, 2048 features binarias (ECFP4 radius=2).
    Requiere: pip install rdkit
    """
    import os, gzip, csv, urllib.request

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError(
            "RDKit es necesario para cargar Tox21.\n"
            "Instala con: pip install rdkit\n"
            "O con conda: conda install -c conda-forge rdkit"
        )

    cache_dir = os.path.join("data", "tox21")
    os.makedirs(cache_dir, exist_ok=True)
    csv_path = os.path.join(cache_dir, "tox21.csv")

    if not os.path.exists(csv_path):
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
        gz_path = csv_path + ".gz"
        print(f"Descargando Tox21 desde DeepChem S3...")
        try:
            urllib.request.urlretrieve(url, gz_path)
        except Exception as e:
            raise RuntimeError(
                f"Error descargando Tox21:\n{e}\n"
                f"Descarga manual desde: {url}\n"
                f"Guarda el archivo descomprimido en: {os.path.abspath(csv_path)}"
            )
        with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
            f_out.write(f_in.read())
        os.remove(gz_path)
        print(f"  Descarga completada: {csv_path}")

    # Caché de fingerprints para no recomputar con RDKit cada vez
    cache_X = os.path.join(cache_dir, f"X_ecfp4_{task.replace('-', '_')}.npy")
    cache_y = os.path.join(cache_dir, f"y_{task.replace('-', '_')}.npy")

    if os.path.exists(cache_X) and os.path.exists(cache_y):
        print(f"  [tox21/{task}] Cargando fingerprints desde caché...")
        X = np.load(cache_X)
        y = np.load(cache_y)
        feature_names = np.array([f"ecfp4_{i}" for i in range(2048)])
        print(
            f"  [tox21/{task}] {X.shape[0]} compuestos | 2048 features ECFP4 "
            f"| pos={int((y == 1).sum())} neg={int((y == 0).sum())}"
        )
        return X, y, feature_names

    # Leer CSV
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        headers = list(reader.fieldnames or [])
        rows = list(reader)

    tox_tasks = [h for h in headers if h not in ("smiles", "mol_id")]
    if task not in tox_tasks:
        raise ValueError(
            f"Tarea '{task}' no disponible.\nTareas en Tox21: {tox_tasks}"
        )

    # Filtrar filas con label no vacío
    valid_rows = [r for r in rows if r[task].strip() not in ("", "nan")]

    fps, y_list = [], []
    for r in valid_rows:
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fps.append(np.array(fp, dtype=np.float32))
        y_list.append(int(float(r[task])))

    X = np.stack(fps).astype(np.float32)
    y = np.array(y_list, dtype=int)

    np.save(cache_X, X)
    np.save(cache_y, y)

    feature_names = np.array([f"ecfp4_{i}" for i in range(2048)])
    print(
        f"  [tox21/{task}] {X.shape[0]} compuestos | 2048 features ECFP4 "
        f"| pos={int((y == 1).sum())} neg={int((y == 0).sum())}"
    )
    return X, y, feature_names


def _load_drebin() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carga DREBIN: detección de malware Android con features binarias.

    Formato esperado en data/drebin/:
      malware/   → archivos .txt (uno por APK), cada línea es el nombre de una feature
      goodware/  → archivos .txt (goodware/benign), mismo formato

    La primera vez construye la matriz binaria y la guarda como caché NPZ.

    Descarga (requiere registro): https://drebin.mlsec.org/
    Alternativa pública (subset): https://github.com/MLDroid/drebin_work
    """
    import os

    cache_dir = os.path.join("data", "drebin")
    npz_cache = os.path.join(cache_dir, "drebin_features.npz")

    if os.path.exists(npz_cache):
        print(f"  [drebin] Cargando desde caché ({npz_cache})...")
        data = np.load(npz_cache, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(int)
        feature_names = data["feature_names"]
        print(
            f"  [drebin] {X.shape[0]} muestras | {X.shape[1]} features "
            f"| malware={int((y == 1).sum())} goodware={int((y == 0).sum())}"
        )
        return X, y, feature_names

    malware_dir = os.path.join(cache_dir, "malware")
    goodware_dir = os.path.join(cache_dir, "goodware")

    if not os.path.exists(malware_dir) or not os.path.exists(goodware_dir):
        raise FileNotFoundError(
            f"DREBIN dataset no encontrado en {os.path.abspath(cache_dir)}/\n"
            f"Descarga (requiere registro): https://drebin.mlsec.org/\n"
            f"Alternativa pública (subset): https://github.com/MLDroid/drebin_work\n\n"
            f"Coloca los archivos de features en:\n"
            f"  {os.path.abspath(malware_dir)}/   (un .txt por muestra malware)\n"
            f"  {os.path.abspath(goodware_dir)}/  (un .txt por muestra goodware)\n"
            f"Cada archivo: una feature por línea (nombre de API call / permiso / intent)."
        )

    def read_sample(fpath: str) -> set:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            return {line.strip() for line in f if line.strip()}

    malware_files = sorted(
        os.path.join(malware_dir, f) for f in os.listdir(malware_dir) if f.endswith(".txt")
    )
    goodware_files = sorted(
        os.path.join(goodware_dir, f) for f in os.listdir(goodware_dir) if f.endswith(".txt")
    )

    if not malware_files or not goodware_files:
        raise FileNotFoundError(
            f"No se encontraron archivos .txt en {malware_dir} o {goodware_dir}."
        )

    print(f"  [drebin] Leyendo {len(malware_files)} malware + {len(goodware_files)} goodware...")
    all_features: set = set()
    samples: list = []
    labels: list = []

    for fpath in malware_files:
        feats = read_sample(fpath)
        samples.append(feats)
        labels.append(1)
        all_features.update(feats)

    for fpath in goodware_files:
        feats = read_sample(fpath)
        samples.append(feats)
        labels.append(0)
        all_features.update(feats)

    feature_list = sorted(all_features)
    feat_idx = {f: i for i, f in enumerate(feature_list)}
    n_feat = len(feature_list)

    print(f"  [drebin] Construyendo matriz binaria ({len(samples)} x {n_feat})...")
    X = np.zeros((len(samples), n_feat), dtype=np.float32)
    for i, feats in enumerate(samples):
        for feat in feats:
            X[i, feat_idx[feat]] = 1.0

    y = np.array(labels, dtype=int)
    feature_names = np.array(feature_list)

    np.savez_compressed(npz_cache, X=X, y=y, feature_names=feature_names)
    print(f"  [drebin] Caché guardada → {npz_cache}")
    print(
        f"  [drebin] {X.shape[0]} muestras | {n_feat} features "
        f"| malware={int((y == 1).sum())} goodware={int((y == 0).sum())}"
    )
    return X, y, feature_names


def _binarize_labels(y_raw: np.ndarray, *,
                     positive_label: Any = None,
                     positive_labels: Any = None) -> np.ndarray:
    if positive_labels is not None:
        mask = np.isin(y_raw, list(positive_labels))
        if not np.any(mask):
            unique = np.unique(y_raw)
            raise ValueError(
                f"positive_labels={positive_labels!r} not found in y; "
                f"unique labels: {unique[:20]!r}"
            )
        return mask.astype(int)

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
        if spec.positive_labels is not None:
            y = _binarize_labels(y_raw, positive_labels=spec.positive_labels)
        else:
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

    if spec.kind in ("uci_musk", "uci_musk_v2"):
        loader = _load_musk_v2 if spec.kind == "uci_musk_v2" else _load_musk_v1
        X, y, feature_names = loader()
        effective_positive_label = positive_label if positive_label is not None else 1
        if effective_positive_label != 1:
            y = (y == effective_positive_label).astype(int)
        meta = {
            "dataset_id": dataset,
            "kind": spec.kind,
            "description": spec.description,
            "positive_label": effective_positive_label,
        }
        return X, y, feature_names, meta

    if spec.kind == "synthetic_isotropic":
        X, y, feature_names = _generate_synthetic_isotropic(spec)
        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        print(
            f"  [synthetic_isotropic] {spec.n_samples} muestras | {spec.n_features} features "
            f"({spec.n_informative} inform. d={spec.d_informative_range} MI~0.08-0.12 | "
            f"{spec.n_features - spec.n_informative} ruido d={spec.d_noise_range} MI~0.04-0.08) | "
            f"pos={n_pos} neg={n_neg}"
        )
        meta = {
            "dataset_id": dataset,
            "kind": spec.kind,
            "description": spec.description,
            "positive_label": 1,
            "n_informative": spec.n_informative,
            "n_noise": spec.n_features - spec.n_informative,
            "d_informative_range": str(spec.d_informative_range),
            "d_noise_range": str(spec.d_noise_range),
        }
        return X, y, feature_names, meta

    if spec.kind == "automl_challenge":
        import os

        data_dir = os.path.join("data", dataset)
        data_file = os.path.join(data_dir, f"{dataset}_train.data")
        labels_file = os.path.join(data_dir, f"{dataset}_train.solution")

        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Local data file not found: {data_file}\n"
                f"Download the dataset and place it in {data_dir}/"
            )

        X = np.loadtxt(data_file, dtype=np.float32)
        y = np.loadtxt(labels_file, dtype=int)
        feature_names = np.asarray([f"x{i}" for i in range(X.shape[1])])
        meta = {
            "dataset_id": dataset,
            "kind": spec.kind,
            "description": spec.description,
            "positive_label": 1,
        }
        return X, y, feature_names, meta

    if spec.kind == "uci_phishing":
        X, y_raw, feature_names = _load_phishing_websites()
        y = _binarize_labels(y_raw, positive_label=positive_label)
        meta = {
            "dataset_id": dataset,
            "kind": spec.kind,
            "description": spec.description,
            "positive_label": positive_label,
        }
        return X, y, feature_names, meta

    if spec.kind == "uci_internet_ads":
        X, y_raw, feature_names = _load_internet_ads()
        y = _binarize_labels(y_raw, positive_label=positive_label)
        meta = {
            "dataset_id": dataset,
            "kind": spec.kind,
            "description": spec.description,
            "positive_label": positive_label,
        }
        return X, y, feature_names, meta

    if spec.kind == "libsvm_epsilon":
        from src import config as _cfg
        max_samples = getattr(_cfg, "MAX_SAMPLES", None)
        X, y, feature_names = _load_epsilon_libsvm(max_samples=max_samples)
        meta = {
            "dataset_id": dataset,
            "kind": spec.kind,
            "description": spec.description,
            "positive_label": 1,
        }
        return X, y, feature_names, meta

    if spec.kind == "uci_nips2003":
        X, y, feature_names = _load_nips2003(
            dataset_name=dataset,
            uci_id=spec.uci_id,
            n_features=spec.n_features,
            fmt=spec.nips_format,
            uci_zip_name=spec.uci_zip_name,
        )
        meta = {
            "dataset_id": dataset,
            "kind": spec.kind,
            "description": spec.description,
            "positive_label": 1,
            "uci_id": spec.uci_id,
            "nips_format": spec.nips_format,
        }
        return X, y, feature_names, meta

    if spec.kind == "moleculenet_tox21":
        X, y, feature_names = _load_tox21(task=spec.tox21_task)
        meta = {
            "dataset_id": dataset,
            "kind": spec.kind,
            "description": spec.description,
            "positive_label": 1,
            "tox21_task": spec.tox21_task,
        }
        return X, y, feature_names, meta

    if spec.kind == "drebin":
        X, y, feature_names = _load_drebin()
        meta = {
            "dataset_id": dataset,
            "kind": spec.kind,
            "description": spec.description,
            "positive_label": 1,
        }
        return X, y, feature_names, meta

    raise RuntimeError(f"Unsupported dataset kind: {spec.kind}")


def load_dataset_from_config() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Convenience wrapper: loads dataset using values in src.config."""
    from src import config

    X, y, feature_names, meta = load_dataset(
        config.DATASET,
        positive_label_override=getattr(config, "DATASET_POSITIVE_LABEL", None),
        gas_positive_class=getattr(config, "GAS_POSITIVE_CLASS", 1),
    )

    max_samples = getattr(config, "MAX_SAMPLES", None)
    if max_samples is not None and X.shape[0] > max_samples:
        n_original = X.shape[0]
        rng = np.random.RandomState(getattr(config, "RANDOM_STATE", 42))
        idx = np.sort(rng.choice(n_original, size=max_samples, replace=False))
        X, y = X[idx], y[idx]
        meta["n_original"] = n_original
        print(f"  [subsample] {X.shape[0]} muestras de {n_original} (MAX_SAMPLES={max_samples})")

    return X, y, feature_names, meta
