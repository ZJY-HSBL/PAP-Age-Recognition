from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {"path", "person_id", "age"}


def load_metadata(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")
    df = df.copy()
    df["person_id"] = df["person_id"].astype(str)
    df["age"] = pd.to_numeric(df["age"], errors="raise").astype(float)
    return df


def split_by_identity(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    people = sorted(df["person_id"].astype(str).unique().tolist())
    if len(people) < 2:
        raise ValueError("Need at least two identities to split")
    train_people, remain = train_test_split(
        people, train_size=train_ratio, random_state=seed, shuffle=True
    )
    mapping = {p: "train" for p in train_people}
    if val_ratio > 0 and remain:
        relative_val = val_ratio / max(1e-12, 1.0 - train_ratio)
        if len(remain) >= 2 and 0 < relative_val < 1:
            val_people, test_people = train_test_split(
                remain, train_size=relative_val, random_state=seed + 1, shuffle=True
            )
        else:
            val_people, test_people = remain, []
        mapping.update({p: "val" for p in val_people})
        mapping.update({p: "test" for p in test_people})
    else:
        mapping.update({p: "test" for p in remain})
    out = df.copy()
    out["split"] = out["person_id"].astype(str).map(mapping)
    return out
