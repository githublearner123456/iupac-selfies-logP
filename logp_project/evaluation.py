from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .constants import PROPERTY_NAME
from .data import MoleculeRecord, get_logp_targets
from .features import filter_descriptor_matrix, get_rdkit_leakage_rules, sanitize_name

matplotlib.use("Agg")

#负责测试集最终结果
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_random_forest(x_train: np.ndarray, y_train: np.ndarray, seed: int, n_estimators: int) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=seed,
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def plot_property_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(6.5, 6.0), dpi=150)
    plt.scatter(y_true, y_pred, s=12, alpha=0.65, linewidths=0)
    low = float(min(np.min(y_true), np.min(y_pred)))
    high = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([low, high], [low, high], linestyle="--", linewidth=1.2, color="black")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_property_summary(result_df: pd.DataFrame, figure_dir: Path) -> None:
    if result_df.empty:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = figure_dir / "property_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    metric_specs = [("rmse", "Lower is Better"), ("r2", "Higher is Better")]
    feature_labels = result_df["feature_set"].drop_duplicates().tolist()
    result_df = result_df.copy()
    result_df["feature_set"] = pd.Categorical(result_df["feature_set"], categories=feature_labels, ordered=True)
    result_df = result_df.sort_values("feature_set")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    for ax, (metric, subtitle) in zip(axes, metric_specs):
        ax.bar(result_df["feature_set"], result_df[metric])
        ax.set_title(f"{PROPERTY_NAME} {metric.upper()} ({subtitle})")
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(summary_dir / f"property_summary_{PROPERTY_NAME}.png")
    plt.close(fig)


def evaluate_property_prediction(
    feature_sets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    train_records: Sequence[MoleculeRecord],
    valid_records: Sequence[MoleculeRecord],
    test_records: Sequence[MoleculeRecord],
    seed: int,
    rf_trees: int,
    evaluation_dir: Path,
    model_dir: Path,
    figure_dir: Path,
    prediction_dir: Path,
    rdkit_pack: Optional[Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]] = None,
) -> pd.DataFrame:
    results = []
    all_train_records = list(train_records) + list(valid_records)
    model_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    scatter_dir = figure_dir / "property_scatter"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    for feature_name, (base_x_train, base_x_valid, base_x_test) in feature_sets.items():
        x_train, x_valid, x_test = base_x_train, base_x_valid, base_x_test
        fairness_note = "standard"

        if rdkit_pack is not None and feature_name == "rdkit_2d":
            desc_names, rdkit_train, rdkit_valid, rdkit_test = rdkit_pack
            drop_names, drop_prefixes = get_rdkit_leakage_rules()
            x_train, kept_names = filter_descriptor_matrix(rdkit_train, desc_names, drop_names, drop_prefixes)
            x_valid, _ = filter_descriptor_matrix(rdkit_valid, desc_names, drop_names, drop_prefixes)
            x_test, _ = filter_descriptor_matrix(rdkit_test, desc_names, drop_names, drop_prefixes)
            fairness_note = f"rdkit_filtered_removed_{len(desc_names) - len(kept_names)}"

        x_fit = np.concatenate([x_train, x_valid], axis=0)
        print(
            f"\nProperty prediction with feature set: {feature_name} "
            f"target={PROPERTY_NAME} shape={x_fit.shape} fairness={fairness_note}"
        )

        y_fit = get_logp_targets(all_train_records)
        y_test = get_logp_targets(test_records)
        model = fit_random_forest(x_fit, y_fit, seed, rf_trees)
        pred = model.predict(x_test)
        metrics = regression_metrics(y_test, pred)
        feature_tag = sanitize_name(feature_name)
        results.append(
            {
                "feature_set": feature_name,
                "target": PROPERTY_NAME,
                "model": "RandomForestRegressor",
                "fairness_note": fairness_note,
                **metrics,
            }
        )

        with open(model_dir / f"{feature_tag}__{PROPERTY_NAME}.pkl", "wb") as f:
            pickle.dump(model, f)
        pd.DataFrame(
            {
                "row_id": [record.row_id for record in test_records],
                "smiles": [record.smiles for record in test_records],
                "iupac": [record.iupac for record in test_records],
                "target": y_test,
                "prediction": pred,
            }
        ).to_csv(prediction_dir / f"{feature_tag}__{PROPERTY_NAME}_test_predictions.csv", index=False)
        plot_property_scatter(
            y_test,
            pred,
            f"{feature_name} -> {PROPERTY_NAME}",
            scatter_dir / f"{feature_tag}__{PROPERTY_NAME}_scatter.png",
        )
        print(f"  {PROPERTY_NAME:>5s} | RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f}")

    result_df = pd.DataFrame(results).sort_values("rmse")
    result_df.to_csv(evaluation_dir / "property_prediction_results.csv", index=False)
    return result_df
