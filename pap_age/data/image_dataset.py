from __future__ import annotations

from pathlib import Path
from typing import Callable
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def default_transform(image_size: int, train: bool = False):
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops += [transforms.RandomHorizontalFlip(p=0.5)]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(ops)


class ImageAgeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 299,
        train: bool = False,
        transform: Callable | None = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform or default_transform(image_size, train=train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = Path(row["path"])
        with Image.open(path) as im:
            image = self.transform(im.convert("RGB"))
        age = float(row["age"])
        return {
            "image": image,
            "age": torch.tensor(age, dtype=torch.float32),
            "person_id": str(row["person_id"]),
            "path": str(path),
        }
