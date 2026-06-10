from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import torch

from .data import (
    build_property_target_stats,
    build_vocabs,
    get_logp_targets,
    get_smiles,
    load_records,
    make_loader,
    save_records_split,
    set_seed,
    split_records,
)
from .evaluation import evaluate_property_prediction, plot_property_summary
from .features import (
    maccs_fingerprint,
    make_visualizations,
    morgan_fingerprint,
    rdkit_2d_descriptors,
    write_rdkit_metadata,
)
from .inference import run_single_iupac_prediction
from .model import extract_encoder_embeddings
from .paths import build_results_layout, ensure_layout, get_project_root, get_results_layout
from .training import evaluate_generation_metrics, evaluate_seq2seq, train_seq2seq

warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage.*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IUPAC -> SELFIES Transformer Seq2Seq for logP.")
    parser.add_argument("--excel", type=str, default="dataset/rawdata.xlsx", help="Path to input Excel file.")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--max_rows", type=int, default=0, help="0 means use all rows.")
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_src_len", type=int, default=220)
    parser.add_argument("--max_tgt_len", type=int, default=220)
    parser.add_argument("--max_src_vocab", type=int, default=0)
    parser.add_argument("--max_tgt_vocab", type=int, default=0)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--encoder_layers", type=int, default=5)
    parser.add_argument("--decoder_layers", type=int, default=5)
    parser.add_argument("--dim_feedforward", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--property_loss_weight", type=float, default=0.5)
    parser.add_argument(
        "--exact_match_batches",
        type=int,
        default=1,
        help="Set >0 to compute valid exact-match metrics on the full validation set; 0 disables them.",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--morgan_bits", type=int, default=2048)
    parser.add_argument("--morgan_radius", type=int, default=2)
    parser.add_argument("--rf_trees", type=int, default=300)
    parser.add_argument("--viz_max_points", type=int, default=3000)
    parser.add_argument("--skip_visualization", action="store_true")
    parser.add_argument("--skip_rdkit_2d", action="store_true")
    parser.add_argument("--skip_maccs", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--predict_iupac", type=str, default="")
    parser.add_argument("--predict_feature_set", type=str, default="encoder_memory")
    args = parser.parse_args()
    args.max_src_vocab = None if args.max_src_vocab <= 0 else args.max_src_vocab
    args.max_tgt_vocab = None if args.max_tgt_vocab <= 0 else args.max_tgt_vocab
    return args


def write_config(args: argparse.Namespace, metadata_dir: Path) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    project_root = get_project_root()
    layout = get_results_layout(project_root)
    if args.out_dir:
        results_root = Path(args.out_dir)
        if not results_root.is_absolute():
            results_root = project_root / results_root
        layout = build_results_layout(project_root, results_root)
    ensure_layout(layout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if device.type == "cpu":
        print("CPU training can be slow. For the full file, CUDA is strongly recommended.")

    if args.predict_iupac:
        run_single_iupac_prediction(args, layout, device)
        return

    write_config(args, layout["metadata_run"])
    with open(layout["metadata_run"] / "environment.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "device": str(device),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                "python_entry": str(Path(__file__).resolve()),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    excel_path = Path(args.excel)
    if not excel_path.exists() and not excel_path.is_absolute():
        excel_path = project_root / excel_path
    if not excel_path.exists() and args.excel == "dataset/rawdata.xlsx":
        legacy_excel_path = project_root / "dataset" / "raw" / "rawdata.xlsx"
        if legacy_excel_path.exists():
            excel_path = legacy_excel_path
    if not excel_path.exists():
        raise FileNotFoundError(f"Cannot find Excel file: {excel_path}")

    records = load_records(excel_path, args.max_rows)
    train_records, valid_records, test_records = split_records(
        records,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"Loaded {len(records)} molecules | train={len(train_records)} valid={len(valid_records)} test={len(test_records)}")
    save_records_split(train_records, valid_records, test_records, layout["dataset"] / "data_splits.csv")

    src_vocab, tgt_vocab = build_vocabs(train_records, args.max_src_vocab, args.max_tgt_vocab)
    print(f"IUPAC vocab size: {len(src_vocab)}")
    print(f"SELFIES vocab size: {len(tgt_vocab)}")
    with open(layout["metadata_vocab"] / "src_vocab.json", "w", encoding="utf-8") as f:
        json.dump(src_vocab.to_dict(), f, indent=2, ensure_ascii=False)
    with open(layout["metadata_vocab"] / "tgt_vocab.json", "w", encoding="utf-8") as f:
        json.dump(tgt_vocab.to_dict(), f, indent=2, ensure_ascii=False)

    model = train_seq2seq(
        args,
        train_records,
        valid_records,
        src_vocab,
        tgt_vocab,
        layout["models"],
        layout["logs"],
        layout["figures"],
        device,
    )

    property_means_np, property_stds_np = build_property_target_stats(train_records)
    property_means = torch.tensor(property_means_np, dtype=torch.float32, device=device)
    property_stds = torch.tensor(property_stds_np, dtype=torch.float32, device=device)
    test_loader = make_loader(
        test_records,
        src_vocab,
        tgt_vocab,
        args.max_src_len,
        args.max_tgt_len,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_metrics = evaluate_seq2seq(
        model,
        test_loader,
        device,
        tgt_vocab.pad_id,
        args.label_smoothing,
        property_means,
        property_stds,
        args.property_loss_weight,
        epoch=1,
        total_epochs=1,
    )
    test_exact = evaluate_generation_metrics(
        model,
        test_loader,
        tgt_vocab,
        args.max_tgt_len,
        device,
        max_batches=max(1, len(test_loader)),
    )
    with open(layout["evaluation_seq2seq"] / "seq2seq_transformer_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_loss": test_metrics["loss"],
                "test_property_loss": test_metrics["property_loss"],
                "test_joint_loss": test_metrics["joint_loss"],
                "test_token_acc": test_metrics["token_acc"],
                "test_exact_match": test_exact["exact_match"],
                "test_avg_edit_distance": test_exact["avg_edit_distance"],
                "test_normalized_edit_distance": test_exact["normalized_edit_distance"],
            },
            f,
            indent=2,
        )
    print(
        "\nSeq2Seq test metrics | "
        f"loss={test_metrics['loss']:.4f} prop_loss={test_metrics['property_loss']:.4f} "
        f"token_acc={test_metrics['token_acc']:.4f} exact_match={test_exact['exact_match']:.4f} "
        f"norm_edit={test_exact['normalized_edit_distance']:.4f}"
    )

    print("\nExtracting encoder memory embeddings...")
    emb_train, train_row_ids = extract_encoder_embeddings(
        model,
        train_records,
        src_vocab,
        tgt_vocab,
        args.max_src_len,
        args.max_tgt_len,
        args.batch_size,
        args.num_workers,
        device,
        desc="Encoder embeddings train",
    )
    emb_valid, valid_row_ids = extract_encoder_embeddings(
        model,
        valid_records,
        src_vocab,
        tgt_vocab,
        args.max_src_len,
        args.max_tgt_len,
        args.batch_size,
        args.num_workers,
        device,
        desc="Encoder embeddings valid",
    )
    emb_test, test_row_ids = extract_encoder_embeddings(
        model,
        test_records,
        src_vocab,
        tgt_vocab,
        args.max_src_len,
        args.max_tgt_len,
        args.batch_size,
        args.num_workers,
        device,
        desc="Encoder embeddings test",
    )
    np.save(layout["representations_embedding"] / "encoder_embedding_train.npy", emb_train)
    np.save(layout["representations_embedding"] / "encoder_embedding_valid.npy", emb_valid)
    np.save(layout["representations_embedding"] / "encoder_embedding_test.npy", emb_test)
    np.save(layout["representations_row_ids"] / "row_ids_train.npy", train_row_ids)
    np.save(layout["representations_row_ids"] / "row_ids_valid.npy", valid_row_ids)
    np.save(layout["representations_row_ids"] / "row_ids_test.npy", test_row_ids)

    print("\nComputing traditional molecular descriptors...")
    smiles_train = get_smiles(train_records)
    smiles_valid = get_smiles(valid_records)
    smiles_test = get_smiles(test_records)
    morgan_train = morgan_fingerprint(smiles_train, args.morgan_bits, args.morgan_radius)
    morgan_valid = morgan_fingerprint(smiles_valid, args.morgan_bits, args.morgan_radius)
    morgan_test = morgan_fingerprint(smiles_test, args.morgan_bits, args.morgan_radius)

    feature_sets = {
        "encoder_memory": (emb_train, emb_valid, emb_test),
        "morgan_fp": (morgan_train, morgan_valid, morgan_test),
    }
    feature_sets_all = {
        "encoder_memory": np.concatenate([emb_train, emb_valid, emb_test], axis=0),
        "morgan_fp": np.concatenate([morgan_train, morgan_valid, morgan_test], axis=0),
    }

    if not args.skip_maccs:
        maccs_train = maccs_fingerprint(smiles_train)
        maccs_valid = maccs_fingerprint(smiles_valid)
        maccs_test = maccs_fingerprint(smiles_test)
        feature_sets["maccs_keys"] = (maccs_train, maccs_valid, maccs_test)
        feature_sets_all["maccs_keys"] = np.concatenate([maccs_train, maccs_valid, maccs_test], axis=0)

    rdkit_pack = None
    if not args.skip_rdkit_2d:
        rdkit_train, desc_names = rdkit_2d_descriptors(smiles_train)
        rdkit_valid, _ = rdkit_2d_descriptors(smiles_valid)
        rdkit_test, _ = rdkit_2d_descriptors(smiles_test)
        write_rdkit_metadata(desc_names, rdkit_train, layout["metadata_rdkit"])
        feature_sets["rdkit_2d"] = (rdkit_train, rdkit_valid, rdkit_test)
        feature_sets_all["rdkit_2d"] = np.concatenate([rdkit_train, rdkit_valid, rdkit_test], axis=0)
        rdkit_pack = (desc_names, rdkit_train, rdkit_valid, rdkit_test)

    print("\nEvaluating property prediction...")
    result_df = evaluate_property_prediction(
        feature_sets,
        train_records,
        valid_records,
        test_records,
        args.seed,
        args.rf_trees,
        layout["evaluation_property_prediction"],
        layout["models"],
        layout["figures"],
        layout["predictions"],
        rdkit_pack=rdkit_pack,
    )
    plot_property_summary(result_df, layout["figures"])
    best_rows = result_df.head(1).copy()
    best_rows.to_csv(layout["evaluation_property_prediction"] / "best_property_results.csv", index=False)
    print(f"\nBest result for logP:\n{best_rows.to_string(index=False)}")

    if not args.skip_visualization:
        print("\nCreating PCA/t-SNE/UMAP visualizations...")
        logp_all = np.concatenate(
            [
                get_logp_targets(train_records),
                get_logp_targets(valid_records),
                get_logp_targets(test_records),
            ],
            axis=0,
        )
        make_visualizations(feature_sets_all, logp_all, layout["figures"], seed=args.seed, max_points=args.viz_max_points)

    print(f"\nFinished. Outputs saved to: {layout['results'].resolve()}")
    print("Key folders:")
    print(f"  {layout['models']}")
    print(f"  {layout['logs']}")
    print(f"  {layout['evaluation']}")
    print(f"  {layout['figures']}")
    print(f"  {layout['predictions']}")
    print(f"  {layout['representations']}")
    print(f"  {layout['metadata']}")
