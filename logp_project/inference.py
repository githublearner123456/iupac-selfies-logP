from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from .constants import PROPERTY_NAME
from .data import Vocab, tokenize_iupac
from .features import sanitize_name
from .model import TransformerSeq2Seq, greedy_decode


def load_seq2seq_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[TransformerSeq2Seq, Vocab, Vocab, Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint["args"]
    src_vocab = Vocab.from_dict(checkpoint["src_vocab"])
    tgt_vocab = Vocab.from_dict(checkpoint["tgt_vocab"])
    model = TransformerSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        d_model=int(saved_args["d_model"]),
        nhead=int(saved_args["nhead"]),
        num_encoder_layers=int(saved_args["encoder_layers"]),
        num_decoder_layers=int(saved_args["decoder_layers"]),
        dim_feedforward=int(saved_args["dim_feedforward"]),
        dropout=float(saved_args["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model, src_vocab, tgt_vocab, saved_args


def predict_selfies_text(
    model: TransformerSeq2Seq,
    iupac: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    max_src_len: int,
    max_tgt_len: int,
    device: torch.device,
) -> str:
    src_ids = src_vocab.encode(tokenize_iupac(iupac), max_src_len)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    pred_ids = greedy_decode(model, src, tgt_vocab, max_tgt_len, device)[0].cpu().tolist()
    return "".join(tgt_vocab.decode(pred_ids))


def encode_single_iupac_embedding(
    model: TransformerSeq2Seq,
    iupac: str,
    src_vocab: Vocab,
    max_src_len: int,
    device: torch.device,
) -> np.ndarray:
    src_ids = src_vocab.encode(tokenize_iupac(iupac), max_src_len)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        pooled = model.encode_pooled(src)
    return pooled.squeeze(0).cpu().numpy().astype(np.float32)


def run_single_iupac_prediction(args, layout: Dict[str, Path], device: torch.device) -> None:#用于单分子预测 不是用于验证集或者测试集，给定一个 IUPAC 名称，同时预测其 SELFIES 字符串和 logP 属性值。
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else layout["models"] / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Cannot find seq2seq checkpoint: {checkpoint_path}")

    feature_name = args.predict_feature_set
    if feature_name != "encoder_memory":
        raise ValueError(
            "Single-IUPAC prediction currently supports only feature_set=encoder_memory "
            "because no SMILES-based descriptors are available."
        )

    model, src_vocab, tgt_vocab, saved_args = load_seq2seq_checkpoint(checkpoint_path, device)
    embedding = encode_single_iupac_embedding(
        model,
        args.predict_iupac,
        src_vocab,
        int(saved_args["max_src_len"]),
        device,
    ).reshape(1, -1)
    pred_selfies = predict_selfies_text(
        model,
        args.predict_iupac,
        src_vocab,
        tgt_vocab,
        int(saved_args["max_src_len"]),
        int(saved_args["max_tgt_len"]),
        device,
    )

    model_path = layout["models"] / f"{sanitize_name(feature_name)}__{PROPERTY_NAME}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find property model: {model_path}. Run training/evaluation first.")
    with open(model_path, "rb") as f:
        reg_model = pickle.load(f)
    pred_logp = float(reg_model.predict(embedding)[0])

    print(f"Input IUPAC: {args.predict_iupac}")
    print(f"Predicted SELFIES: {pred_selfies}")
    print(f"Predicted {PROPERTY_NAME}: {pred_logp:.6f}")
