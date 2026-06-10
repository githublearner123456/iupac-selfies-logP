from pathlib import Path
from typing import Dict


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def build_results_layout(project_root: Path, results_dir: Path) -> Dict[str, Path]:
    figures_dir = results_dir / "figures"
    metadata_dir = results_dir / "metadata"
    evaluation_dir = results_dir / "test_evaluation"
    representations_dir = results_dir / "representations"
    dataset_dir = results_dir / "dataset"
    return {
        "root": project_root,
        "raw_data": project_root / "dataset" / "rawdata.xlsx",
        "results": results_dir,
        "models": results_dir / "models",
        "evaluation": evaluation_dir,
        "evaluation_seq2seq": evaluation_dir,
        "evaluation_property_prediction": evaluation_dir / "feature_comparison_results",
        "figures": figures_dir,
        "figures_training_curves": figures_dir / "training_curves",
        "figures_property_scatter": figures_dir / "property_scatter",
        "figures_property_summary": figures_dir / "property_summary",
        "figures_projection_pca": figures_dir / "projection_pca",
        "figures_projection_tsne": figures_dir / "projection_tsne",
        "figures_projection_umap": figures_dir / "projection_umap",
        "logs": results_dir / "logs",
        "predictions": evaluation_dir / "feature_prediction_comparison_results",
        "representations": representations_dir,
        "representations_embedding": representations_dir / "embedding",
        "representations_row_ids": representations_dir / "row_ids",
        "metadata": metadata_dir,
        "metadata_run": metadata_dir / "run_info",
        "metadata_vocab": metadata_dir / "vocab",
        "metadata_rdkit": metadata_dir / "rdkit",
        "dataset": dataset_dir,
    }


def get_results_layout(project_root: Path) -> Dict[str, Path]:
    return build_results_layout(project_root, project_root / "results")


def ensure_layout(layout: Dict[str, Path]) -> None:
    for name, path in layout.items():
        if name in {"root", "raw_data"}:
            continue
        path.mkdir(parents=True, exist_ok=True)
