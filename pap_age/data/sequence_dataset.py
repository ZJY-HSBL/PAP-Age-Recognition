from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureSequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        ages: np.ndarray,
        person_ids: np.ndarray,
        splits: np.ndarray,
        split: str,
        min_sequence_length: int = 2,
    ):
        self.samples = []
        person_ids = person_ids.astype(str)
        splits = splits.astype(str)
        for person in sorted(np.unique(person_ids)):
            idx = np.where((person_ids == person) & (splits == split))[0]
            if len(idx) < min_sequence_length:
                continue
            idx = idx[np.argsort(ages[idx], kind="stable")]
            self.samples.append((person, features[idx].astype(np.float32), ages[idx].astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        person, x, y = self.samples[idx]
        return {
            "person_id": person,
            "features": torch.from_numpy(x),
            "ages": torch.from_numpy(y),
            "length": len(y),
        }


def collate_sequences(batch):
    if not batch:
        raise ValueError("Empty batch")
    lengths = torch.tensor([x["length"] for x in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    feat_dim = batch[0]["features"].shape[-1]
    x = torch.zeros((len(batch), max_len, feat_dim), dtype=torch.float32)
    y = torch.zeros((len(batch), max_len), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    people = []
    for i, sample in enumerate(batch):
        n = sample["length"]
        x[i, :n] = sample["features"]
        y[i, :n] = sample["ages"]
        mask[i, :n] = True
        people.append(sample["person_id"])
    return {"features": x, "ages": y, "mask": mask, "lengths": lengths, "person_ids": people}
