from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from .data import Vocab, build_property_target_stats, make_loader, normalize_property_targets
from .model import TransformerSeq2Seq, compute_token_accuracy, greedy_decode

matplotlib.use("Agg")


def levenshtein_distance(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    previous = list(range(len(seq_b) + 1))
    for i, token_a in enumerate(seq_a, start=1):
        current = [i]
        for j, token_b in enumerate(seq_b, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + int(token_a != token_b),
                )
            )
        previous = current
    return previous[-1]


def train_one_epoch(
    model: TransformerSeq2Seq,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    tgt_pad_id: int,
    grad_clip: float,
    label_smoothing: float,
    property_means: torch.Tensor,
    property_stds: torch.Tensor,
    property_loss_weight: float,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.train()
    total_seq_loss = 0.0
    total_prop_loss = 0.0
    total_joint_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} train", leave=False, dynamic_ncols=True)
    for batch in progress:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        property_targets = batch["property_targets"].to(device)
        normalized_targets = normalize_property_targets(property_targets, property_means, property_stds)

        optimizer.zero_grad(set_to_none=True)
        seq_logits, prop_logits = model.forward_multitask(src, tgt_in)
        seq_loss = F.cross_entropy(
            seq_logits.reshape(-1, seq_logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=tgt_pad_id,
            label_smoothing=label_smoothing,
        )
        prop_loss = F.mse_loss(prop_logits, normalized_targets)
        loss = seq_loss + property_loss_weight * prop_loss
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_tokens = tgt_out.ne(tgt_pad_id).sum().item()
        batch_size = src.size(0)
        correct, _ = compute_token_accuracy(seq_logits.detach(), tgt_out, tgt_pad_id)
        total_seq_loss += seq_loss.item() * batch_tokens
        total_prop_loss += prop_loss.item() * batch_size
        total_joint_loss += loss.item() * batch_size
        total_tokens += batch_tokens
        total_correct += correct
        total_samples += batch_size
        progress.set_postfix(
            seq=f"{total_seq_loss / max(total_tokens, 1):.4f}",
            prop=f"{total_prop_loss / max(total_samples, 1):.4f}",
            acc=f"{total_correct / max(total_tokens, 1):.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return {
        "loss": total_seq_loss / max(total_tokens, 1),
        "property_loss": total_prop_loss / max(total_samples, 1),
        "joint_loss": total_joint_loss / max(total_samples, 1),
        "token_acc": total_correct / max(total_tokens, 1),
    }


@torch.no_grad()
def evaluate_seq2seq(#在验证集上评估，使用了 teacher forcing。
    model: TransformerSeq2Seq,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    tgt_pad_id: int,
    label_smoothing: float,
    property_means: torch.Tensor,
    property_stds: torch.Tensor,
    property_loss_weight: float,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.eval()
    total_seq_loss = 0.0
    total_prop_loss = 0.0
    total_joint_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} valid", leave=False, dynamic_ncols=True)
    for batch in progress:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        property_targets = batch["property_targets"].to(device)
        normalized_targets = normalize_property_targets(property_targets, property_means, property_stds)

        seq_logits, prop_logits = model.forward_multitask(src, tgt_in)
        seq_loss = F.cross_entropy(
            seq_logits.reshape(-1, seq_logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=tgt_pad_id,
            label_smoothing=label_smoothing,
        )
        prop_loss = F.mse_loss(prop_logits, normalized_targets)
        loss = seq_loss + property_loss_weight * prop_loss
        batch_tokens = tgt_out.ne(tgt_pad_id).sum().item()
        batch_size = src.size(0)
        correct, _ = compute_token_accuracy(seq_logits, tgt_out, tgt_pad_id)
        total_seq_loss += seq_loss.item() * batch_tokens
        total_prop_loss += prop_loss.item() * batch_size
        total_joint_loss += loss.item() * batch_size
        total_tokens += batch_tokens
        total_correct += correct
        total_samples += batch_size
        progress.set_postfix(
            seq=f"{total_seq_loss / max(total_tokens, 1):.4f}",
            prop=f"{total_prop_loss / max(total_samples, 1):.4f}",
            acc=f"{total_correct / max(total_tokens, 1):.4f}",
        )

    return {
        "loss": total_seq_loss / max(total_tokens, 1),
        "property_loss": total_prop_loss / max(total_samples, 1),
        "joint_loss": total_joint_loss / max(total_samples, 1),
        "token_acc": total_correct / max(total_tokens, 1),
    }


@torch.no_grad()
def evaluate_generation_metrics(#在验证集上评估，使用了greedy coding。
    model: TransformerSeq2Seq,
    loader: torch.utils.data.DataLoader,
    tgt_vocab: Vocab,
    max_tgt_len: int,
    device: torch.device,
    max_batches: int,
) -> Dict[str, float]:
    model.eval()
    exact = 0
    total = 0
    edit_distance_sum = 0.0
    normalized_edit_distance_sum = 0.0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        src = batch["src"].to(device)
        tgt = batch["tgt"].cpu().numpy().tolist()
        pred = greedy_decode(model, src, tgt_vocab, max_tgt_len, device).cpu().numpy().tolist()
        for pred_ids, tgt_ids in zip(pred, tgt):
            pred_tokens = tgt_vocab.decode(pred_ids)
            tgt_tokens = tgt_vocab.decode(tgt_ids)
            edit_distance = levenshtein_distance(pred_tokens, tgt_tokens)
            if pred_tokens == tgt_tokens:
                exact += 1
            edit_distance_sum += edit_distance
            normalized_edit_distance_sum += edit_distance / max(len(tgt_tokens), 1)
            total += 1
    return {
        "exact_match": exact / max(total, 1),
        "avg_edit_distance": edit_distance_sum / max(total, 1),
        "normalized_edit_distance": normalized_edit_distance_sum / max(total, 1),
    }


def plot_training_history(history: Sequence[Dict[str, float]], figure_dir: Path) -> None:
    if not history:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    training_dir = figure_dir / "training_curves"
    training_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(df["epoch"], df["train_loss"], label="train_loss", linewidth=2)
    plt.plot(df["epoch"], df["valid_loss"], label="valid_loss", linewidth=2)
    if "train_property_loss" in df.columns and "valid_property_loss" in df.columns:
        plt.plot(df["epoch"], df["train_property_loss"], label="train_property_loss", linewidth=2)
        plt.plot(df["epoch"], df["valid_property_loss"], label="valid_property_loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(training_dir / "seq2seq_loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(df["epoch"], df["train_token_acc"], label="train_token_acc", linewidth=2)
    plt.plot(df["epoch"], df["valid_token_acc"], label="valid_token_acc", linewidth=2)
    if "exact_match" in df.columns:
        plt.plot(df["epoch"], df["exact_match"], label="valid_exact_match", linewidth=2)
    if "normalized_edit_distance" in df.columns:
        plt.plot(df["epoch"], df["normalized_edit_distance"], label="valid_normalized_edit_distance", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Seq2Seq Accuracy Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(training_dir / "seq2seq_accuracy_curve.png")
    plt.close()

    if "lr" in df.columns:
        plt.figure(figsize=(8, 5), dpi=150)
        plt.plot(df["epoch"], df["lr"], label="learning_rate", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.tight_layout()
        plt.savefig(training_dir / "seq2seq_lr_curve.png")
        plt.close()


def save_training_summary(history: Sequence[Dict[str, float]], summary_path: Path) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    best_metric_name = "valid_joint_loss" if "valid_joint_loss" in df.columns else "valid_loss"
    best_idx = int(df[best_metric_name].idxmin())
    best_row = df.iloc[best_idx].to_dict()
    final_row = df.iloc[-1].to_dict()
    summary = {
        "epochs_ran": int(len(df)),
        "best_epoch": int(best_row["epoch"]),
        "best_valid_loss": float(best_row["valid_loss"]),
        "best_valid_joint_loss": float(best_row.get("valid_joint_loss", best_row["valid_loss"])),
        "best_valid_property_loss": float(best_row.get("valid_property_loss", np.nan)),
        "best_valid_token_acc": float(best_row["valid_token_acc"]),
        "final_epoch": int(final_row["epoch"]),
        "final_train_loss": float(final_row["train_loss"]),
        "final_valid_loss": float(final_row["valid_loss"]),
        "final_valid_joint_loss": float(final_row.get("valid_joint_loss", final_row["valid_loss"])),
        "final_valid_property_loss": float(final_row.get("valid_property_loss", np.nan)),
        "final_valid_token_acc": float(final_row["valid_token_acc"]),
    }
    if "exact_match" in df.columns and not pd.isna(best_row.get("exact_match", np.nan)):
        summary["best_valid_exact_match"] = float(best_row["exact_match"])
    if "avg_edit_distance" in df.columns and not pd.isna(best_row.get("avg_edit_distance", np.nan)):
        summary["best_valid_avg_edit_distance"] = float(best_row["avg_edit_distance"])
    if "normalized_edit_distance" in df.columns and not pd.isna(best_row.get("normalized_edit_distance", np.nan)):
        summary["best_valid_normalized_edit_distance"] = float(best_row["normalized_edit_distance"])
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def train_seq2seq(
    args,
    train_records,
    valid_records,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    model_dir: Path,
    log_dir: Path,
    figure_dir: Path,
    device: torch.device,
) -> TransformerSeq2Seq:
    property_means_np, property_stds_np = build_property_target_stats(train_records)
    property_means = torch.tensor(property_means_np, dtype=torch.float32, device=device)
    property_stds = torch.tensor(property_stds_np, dtype=torch.float32, device=device)
    train_loader = make_loader(train_records, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len, args.batch_size, True, args.num_workers)
    valid_loader = make_loader(valid_records, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len, args.batch_size, False, args.num_workers)

    model = TransformerSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best_model.pt"
    best_valid_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    print(
        f"Training setup | epochs={args.epochs} batch_size={args.batch_size} "
        f"lr={args.lr} patience={args.early_stopping_patience} property_weight={args.property_loss_weight}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            tgt_vocab.pad_id,
            args.grad_clip,
            args.label_smoothing,
            property_means,
            property_stds,
            args.property_loss_weight,
            epoch,
            args.epochs,
        )
        valid_metrics = evaluate_seq2seq(
            model,
            valid_loader,
            device,
            tgt_vocab.pad_id,
            args.label_smoothing,
            property_means,
            property_stds,
            args.property_loss_weight,
            epoch,
            args.epochs,
        )

        exact_metrics = {}
        if args.exact_match_batches > 0:
            exact_metrics = evaluate_generation_metrics(model, valid_loader, tgt_vocab, args.max_tgt_len, device, max(1, len(valid_loader)))

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_property_loss": train_metrics["property_loss"],
                "train_joint_loss": train_metrics["joint_loss"],
                "train_token_acc": train_metrics["token_acc"],
                "valid_loss": valid_metrics["loss"],
                "valid_property_loss": valid_metrics["property_loss"],
                "valid_joint_loss": valid_metrics["joint_loss"],
                "valid_token_acc": valid_metrics["token_acc"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                **exact_metrics,
            }
        )

        exact_text = ""
        if exact_metrics:
            exact_text = f" valid_exact={exact_metrics['exact_match']:.4f} valid_ned={exact_metrics['normalized_edit_distance']:.4f}"
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_prop={train_metrics['property_loss']:.4f} "
            f"train_acc={train_metrics['token_acc']:.4f} | "
            f"valid_loss={valid_metrics['loss']:.4f} valid_prop={valid_metrics['property_loss']:.4f} "
            f"valid_acc={valid_metrics['token_acc']:.4f} lr={optimizer.param_groups[0]['lr']:.6g}"
            f"{exact_text}"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "args": vars(args),
            "src_vocab": src_vocab.to_dict(),
            "tgt_vocab": tgt_vocab.to_dict(),
            "property_target_means": property_means_np.tolist(),
            "property_target_stds": property_stds_np.tolist(),
            "epoch": epoch,
            "valid_loss": valid_metrics["loss"],
            "valid_property_loss": valid_metrics["property_loss"],
            "valid_joint_loss": valid_metrics["joint_loss"],
        }

        if valid_metrics["joint_loss"] < best_valid_loss - args.early_stopping_min_delta:
            best_valid_loss = valid_metrics["joint_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, best_path)
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch:03d} | best_epoch={best_epoch:03d} best_valid_joint_loss={best_valid_loss:.4f}")
            break

    pd.DataFrame(history).to_csv(log_dir / "seq2seq_history.csv", index=False)
    plot_training_history(history, figure_dir)
    save_training_summary(history, log_dir / "training_summary.json")
    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    print(f"Best model selected from epoch {best_checkpoint['epoch']:03d} with valid_joint_loss={best_checkpoint['valid_joint_loss']:.4f}")
    return model
