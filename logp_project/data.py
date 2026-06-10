from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .constants import BOS, EOS, PAD, PROPERTY_NAME, SPECIAL_TOKENS, UNK


SELFIES_PATTERN = re.compile(r"\[[^\[\]]+\]")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\n", "").replace("\r", "").strip()


def tokenize_iupac(text: str) -> List[str]:
    return list(normalize_text(text))


def tokenize_selfies(text: str) -> List[str]:
    text = normalize_text(text)
    tokens = SELFIES_PATTERN.findall(text)
    if tokens:
        return tokens
    return list(text)


class Vocab:
    def __init__(
        self,
        token_lists: Sequence[Sequence[str]],
        min_freq: int = 1,
        max_size: Optional[int] = None,
    ) -> None:
        counts: Dict[str, int] = {}
        for tokens in token_lists:
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1

        sorted_tokens = sorted(
            [token for token, count in counts.items() if count >= min_freq],
            key=lambda token: (-counts[token], token),
        )
        if max_size is not None:
            sorted_tokens = sorted_tokens[: max(0, max_size - len(SPECIAL_TOKENS))]

        self.itos = list(SPECIAL_TOKENS) + sorted_tokens
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Sequence[str], max_len: int) -> List[int]:
        ids = [self.bos_id]
        ids.extend(self.stoi.get(token, self.unk_id) for token in tokens[: max_len - 2])
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int], remove_special: bool = True) -> List[str]:
        tokens = []
        for idx in ids:
            token = self.itos[idx] if 0 <= idx < len(self.itos) else UNK
            if remove_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return tokens

    def to_dict(self) -> Dict[str, object]:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Vocab":
        vocab = cls([])
        vocab.itos = list(data["itos"])
        vocab.stoi = {token: idx for idx, token in enumerate(vocab.itos)}
        return vocab


@dataclass
class MoleculeRecord:
    row_id: int
    smiles: str
    iupac: str
    selfies: str
    logp: float


class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: Sequence[MoleculeRecord],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_src_len: int,
        max_tgt_len: int,
    ) -> None:
        self.records = list(records)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self.records[idx]
        src_ids = self.src_vocab.encode(tokenize_iupac(record.iupac), self.max_src_len)
        tgt_ids = self.tgt_vocab.encode(tokenize_selfies(record.selfies), self.max_tgt_len)
        return {
            "row_id": record.row_id,
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "property_targets": torch.tensor([record.logp], dtype=torch.float32),
        }


def pad_1d(sequences: Sequence[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(seq.numel() for seq in sequences)
    out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        out[i, : seq.numel()] = seq
    return out


def make_collate_fn(src_pad_id: int, tgt_pad_id: int) -> Callable:
    def collate(batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        src = pad_1d([item["src_ids"] for item in batch], src_pad_id)
        tgt = pad_1d([item["tgt_ids"] for item in batch], tgt_pad_id)
        row_ids = torch.tensor([item["row_id"] for item in batch], dtype=torch.long)
        return {
            "row_ids": row_ids,
            "src": src,
            "tgt": tgt,
            "property_targets": torch.stack([item["property_targets"] for item in batch]),
        }

    return collate


def load_records(excel_path: Path, max_rows: int = 0) -> List[MoleculeRecord]:
    df = pd.read_excel(excel_path)
    df.columns = [str(col).strip() for col in df.columns]

    required = ["smiles", "selfies", "iupac", PROPERTY_NAME]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {excel_path}: {missing}")

    if max_rows > 0:
        df = df.head(max_rows).copy()

    records: List[MoleculeRecord] = []
    for _, row in df.iterrows():
        smiles = normalize_text(row["smiles"])
        selfies = normalize_text(row["selfies"])
        iupac = normalize_text(row["iupac"])
        logp_value = row[PROPERTY_NAME]
        if not smiles or not selfies or not iupac or pd.isna(logp_value):
            continue
        records.append(MoleculeRecord(len(records), smiles, iupac, selfies, float(logp_value)))

    if len(records) < 10:
        raise ValueError("Too few usable records after filtering.")
    return records


def split_records(
    records: Sequence[MoleculeRecord],
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[MoleculeRecord], List[MoleculeRecord], List[MoleculeRecord]]:
    indices = np.arange(len(records))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_total = len(indices)
    n_test = int(round(n_total * test_ratio))
    n_valid = int(round(n_total * valid_ratio))
    test_idx = set(indices[:n_test].tolist())
    valid_idx = set(indices[n_test : n_test + n_valid].tolist())

    train_records, valid_records, test_records = [], [], []
    for idx, record in enumerate(records):
        if idx in test_idx:
            test_records.append(record)
        elif idx in valid_idx:
            valid_records.append(record)
        else:
            train_records.append(record)
    return train_records, valid_records, test_records


def build_vocabs(
    train_records: Sequence[MoleculeRecord],
    max_src_vocab: Optional[int],
    max_tgt_vocab: Optional[int],
) -> Tuple[Vocab, Vocab]:
    src_tokens = [tokenize_iupac(record.iupac) for record in train_records]
    tgt_tokens = [tokenize_selfies(record.selfies) for record in train_records]
    return Vocab(src_tokens, max_size=max_src_vocab), Vocab(tgt_tokens, max_size=max_tgt_vocab)


def make_loader(
    records: Sequence[MoleculeRecord],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    max_src_len: int,
    max_tgt_len: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    dataset = Seq2SeqDataset(records, src_vocab, tgt_vocab, max_src_len, max_tgt_len)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate_fn(src_vocab.pad_id, tgt_vocab.pad_id),
        pin_memory=torch.cuda.is_available(),
    )


def build_property_target_stats(records: Sequence[MoleculeRecord]) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray([[record.logp] for record in records], dtype=np.float32)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return means.astype(np.float32), stds.astype(np.float32)


def normalize_property_targets(values: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    return (values - means) / stds


def get_logp_targets(records: Sequence[MoleculeRecord]) -> np.ndarray:
    return np.asarray([record.logp for record in records], dtype=np.float32)


def get_smiles(records: Sequence[MoleculeRecord]) -> List[str]:
    return [record.smiles for record in records]


def save_records_split(
    train_records: Sequence[MoleculeRecord],
    valid_records: Sequence[MoleculeRecord],
    test_records: Sequence[MoleculeRecord],
    output_path: Path,
) -> None:
    rows = []
    for split_name, split_rows in [("train", train_records), ("valid", valid_records), ("test", test_records)]:
        for record in split_rows:
            rows.append(
                {
                    "split": split_name,
                    "row_id": record.row_id,
                    "smiles": record.smiles,
                    "iupac": record.iupac,
                    "selfies": record.selfies,
                    PROPERTY_NAME: record.logp,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
