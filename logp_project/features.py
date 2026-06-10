from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .constants import PROPERTY_NAME

matplotlib.use("Agg")
RDLogger.DisableLog("rdApp.*")

try:
    from umap import UMAP
except ImportError:
    UMAP = None


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", name).strip("_")


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def morgan_fingerprint(smiles_list: Sequence[str], n_bits: int, radius: int) -> np.ndarray:
    features = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smiles in enumerate(tqdm(smiles_list, desc="Morgan fingerprints", leave=False)):
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        features[i] = arr.astype(np.float32)
    return features


def maccs_fingerprint(smiles_list: Sequence[str]) -> np.ndarray:
    n_bits = 167
    features = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smiles in enumerate(tqdm(smiles_list, desc="MACCS fingerprints", leave=False)):
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        features[i] = arr.astype(np.float32)
    return features


def rdkit_2d_descriptors(smiles_list: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    desc_items = list(Descriptors._descList)
    desc_names = [name for name, _ in desc_items]
    features = np.full((len(smiles_list), len(desc_items)), np.nan, dtype=np.float32)
    for i, smiles in enumerate(tqdm(smiles_list, desc="RDKit 2D descriptors", leave=False)):
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        values = []
        for _, fn in desc_items:
            try:
                value = fn(mol)
                if value is None or not np.isfinite(value):
                    value = np.nan
            except Exception:
                value = np.nan
            values.append(value)
        features[i] = np.asarray(values, dtype=np.float32)
    return features, desc_names


def get_rdkit_leakage_rules() -> Tuple[set[str], Tuple[str, ...]]:
    return {"MolLogP"}, ("SlogP_VSA",)


def filter_descriptor_matrix(
    features: np.ndarray,
    desc_names: Sequence[str],
    drop_names: set[str],
    drop_prefixes: Tuple[str, ...],
) -> Tuple[np.ndarray, List[str]]:
    keep_indices = [
        idx
        for idx, name in enumerate(desc_names)
        if name not in drop_names and not any(name.startswith(prefix) for prefix in drop_prefixes)
    ]
    filtered = features[:, keep_indices].astype(np.float32)
    kept_names = [desc_names[idx] for idx in keep_indices]
    return filtered, kept_names


def write_rdkit_metadata(desc_names: List[str], rdkit_train: np.ndarray, metadata_dir: Path) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / "rdkit_2d_descriptor_names.json", "w", encoding="utf-8") as f:
        json.dump(desc_names, f, indent=2)

    drop_names, drop_prefixes = get_rdkit_leakage_rules()
    _, kept_names = filter_descriptor_matrix(rdkit_train, desc_names, drop_names, drop_prefixes)
    report = {
        PROPERTY_NAME: {
            "removed_count": len(desc_names) - len(kept_names),
            "removed_exact_names": sorted(drop_names),
            "removed_prefixes": list(drop_prefixes),
            "kept_count": len(kept_names),
        }
    }
    with open(metadata_dir / "rdkit_2d_fairness_rules.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def safe_standardize(features: np.ndarray) -> np.ndarray:
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    return pipe.fit_transform(features)


def plot_projection(
    features: np.ndarray,
    values: np.ndarray,
    title: str,
    out_path: Path,
    method: str,
    seed: int,
    max_points: int,
) -> None:
    if features.shape[0] < 3:
        return
    rng = np.random.default_rng(seed)
    indices = np.arange(features.shape[0])
    if max_points > 0 and features.shape[0] > max_points:
        indices = rng.choice(indices, size=max_points, replace=False)

    x = safe_standardize(features[indices])
    y = values[indices]
    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        z = reducer.fit_transform(x)
        xlabel, ylabel = "PC1", "PC2"
    elif method == "tsne":
        perplexity = min(30, max(5, (x.shape[0] - 1) // 3))
        reducer = TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        )
        z = reducer.fit_transform(x)
        xlabel, ylabel = "t-SNE 1", "t-SNE 2"
    elif method == "umap":
        if UMAP is None:
            return
        reducer = UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=min(15, max(2, x.shape[0] - 1)),
            min_dist=0.1,
        )
        z = reducer.fit_transform(x)
        xlabel, ylabel = "UMAP 1", "UMAP 2"
    else:
        raise ValueError(f"Unknown projection method: {method}")

    plt.figure(figsize=(7.5, 6.2), dpi=150)
    scatter = plt.scatter(z[:, 0], z[:, 1], c=y, s=10, alpha=0.75, cmap="viridis", linewidths=0)
    plt.colorbar(scatter, label=PROPERTY_NAME)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def make_visualizations(
    feature_sets_all: Dict[str, np.ndarray],
    logp_values: np.ndarray,
    figure_dir: Path,
    seed: int,
    max_points: int,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    pca_dir = figure_dir / "projection_pca"
    tsne_dir = figure_dir / "projection_tsne"
    umap_dir = figure_dir / "projection_umap"
    pca_dir.mkdir(parents=True, exist_ok=True)
    tsne_dir.mkdir(parents=True, exist_ok=True)
    umap_dir.mkdir(parents=True, exist_ok=True)
    for feature_name, features in feature_sets_all.items():
        safe_name = sanitize_name(feature_name)
        plot_projection(
            features,
            logp_values,
            f"{feature_name} PCA colored by {PROPERTY_NAME}",
            pca_dir / f"{safe_name}_pca_logp.png",
            method="pca",
            seed=seed,
            max_points=max_points,
        )
        if features.shape[0] >= 50:
            plot_projection(
                features,
                logp_values,
                f"{feature_name} t-SNE colored by {PROPERTY_NAME}",
                tsne_dir / f"{safe_name}_tsne_logp.png",
                method="tsne",
                seed=seed,
                max_points=max_points,
            )
            plot_projection(
                features,
                logp_values,
                f"{feature_name} UMAP colored by {PROPERTY_NAME}",
                umap_dir / f"{safe_name}_umap_logp.png",
                method="umap",
                seed=seed,
                max_points=max_points,
            )
