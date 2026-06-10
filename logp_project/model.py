from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .data import Vocab, make_loader


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 4096) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def masked_mean_max_pool(memory: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    valid_mask = (~padding_mask).unsqueeze(-1)
    valid_mask_f = valid_mask.float()
    mean_pooled = (memory * valid_mask_f).sum(dim=1) / valid_mask_f.sum(dim=1).clamp(min=1.0)
    masked_memory = memory.masked_fill(~valid_mask, float("-inf"))
    max_pooled = masked_memory.max(dim=1).values
    max_pooled = torch.where(torch.isfinite(max_pooled), max_pooled, torch.zeros_like(max_pooled))
    return torch.cat([mean_pooled, max_pooled], dim=1)


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_id: int,
        tgt_pad_id: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_id)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        pooled_dim = d_model * 2
        self.property_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, pooled_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, 1),
        )

    def encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_key_padding_mask = src.eq(self.src_pad_id)
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def decode(self, tgt_in: torch.Tensor, memory: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        tgt_key_padding_mask = tgt_in.eq(self.tgt_pad_id)
        tgt_mask = torch.triu(
            torch.ones(tgt_in.size(1), tgt_in.size(1), device=tgt_in.device, dtype=torch.bool),
            diagonal=1,
        )
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt_in) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.generator(out)

    def encode_pooled(self, src: torch.Tensor) -> torch.Tensor:
        memory, src_key_padding_mask = self.encode(src)
        return masked_mean_max_pool(memory, src_key_padding_mask)

    def forward_multitask(self, src: torch.Tensor, tgt_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        memory, src_key_padding_mask = self.encode(src)
        seq_logits = self.decode(tgt_in, memory, src_key_padding_mask)
        prop_logits = self.property_head(masked_mean_max_pool(memory, src_key_padding_mask))
        return seq_logits, prop_logits


def compute_token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> Tuple[int, int]:
    preds = logits.argmax(dim=-1)
    mask = targets.ne(pad_id)
    correct = preds.eq(targets).logical_and(mask).sum().item()
    total = mask.sum().item()
    return int(correct), int(total)


@torch.no_grad()
def greedy_decode(
    model: TransformerSeq2Seq,
    src: torch.Tensor,
    tgt_vocab: Vocab,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    src = src.to(device)
    memory, src_key_padding_mask = model.encode(src)
    ys = torch.full((src.size(0), 1), tgt_vocab.bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=device)
    for _ in range(max_len - 1):
        logits = model.decode(ys, memory, src_key_padding_mask)
        next_token = logits[:, -1, :].argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, tgt_vocab.pad_id), next_token)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        finished = finished | next_token.eq(tgt_vocab.eos_id)
        if finished.all():
            break
    return ys


@torch.no_grad()
def extract_encoder_embeddings(
    model: TransformerSeq2Seq,
    records: Sequence,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    max_src_len: int,
    max_tgt_len: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = make_loader(records, src_vocab, tgt_vocab, max_src_len, max_tgt_len, batch_size, False, num_workers)
    model.eval()
    embeddings = []
    row_ids = []
    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in progress:
        src = batch["src"].to(device)
        pooled = model.encode_pooled(src)
        embeddings.append(pooled.cpu().numpy())
        row_ids.append(batch["row_ids"].numpy())
    return np.concatenate(embeddings, axis=0), np.concatenate(row_ids, axis=0)
