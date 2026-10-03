from __future__ import annotations

import torch
from torch import nn


class TinyBackbone(nn.Module):
    def __init__(self, num_classes: int = 101, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.num_features = 64
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.num_features, num_classes)

    def forward_features_vector(self, x):
        return self.encoder(x).flatten(1)

    def forward(self, x):
        return self.classifier(self.dropout(self.forward_features_vector(x)))


class InceptionAgeModel(nn.Module):
    def __init__(
        self,
        backbone: str = "inception_v4",
        num_classes: int = 101,
        pretrained: bool = True,
        dropout: float = 0.8,
    ):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "tiny":
            self.model = TinyBackbone(num_classes=num_classes, dropout=dropout)
            self.num_features = self.model.num_features
            self._tiny = True
        else:
            try:
                import timm
            except ImportError as e:
                raise ImportError("timm is required for Inception-v4: pip install timm") from e
            self.model = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout,
            )
            self.num_features = int(getattr(self.model, "num_features", 1536))
            self._tiny = False

    def forward(self, x):
        return self.model(x)

    def extract_features(self, x):
        if self._tiny:
            return self.model.forward_features_vector(x)
        f = self.model.forward_features(x)
        if hasattr(self.model, "forward_head"):
            return self.model.forward_head(f, pre_logits=True)
        if f.ndim == 4:
            f = f.mean(dim=(-2, -1))
        return f


def expected_age_from_logits(logits: torch.Tensor, age_min: int = 0) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    ages = torch.arange(
        age_min,
        age_min + logits.shape[-1],
        device=logits.device,
        dtype=logits.dtype,
    )
    return (probs * ages).sum(dim=-1)
