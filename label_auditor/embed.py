"""Turn an image folder into feature vectors with a frozen pretrained CNN (needs torch + torchvision)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def embed_folder(root: Path, batch_size: int = 32):
    import torch
    from PIL import Image
    from torchvision.models import ResNet18_Weights, resnet18

    from .data import Dataset

    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    files, y = [], []
    for c, name in enumerate(classes):
        for f in sorted((root / name).rglob("*")):
            if f.suffix.lower() in EXTS:
                files.append(f)
                y.append(c)
    if not files:
        raise ValueError(f"no images under {root}")

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()  # keep the 512-d embedding
    model.eval()
    prep = weights.transforms()
    feats = []
    with torch.no_grad():
        for i in range(0, len(files), batch_size):
            batch = torch.stack([prep(Image.open(f).convert("RGB")) for f in files[i:i + batch_size]])
            feats.append(model(batch).numpy())
    ids = [str(f.relative_to(root)) for f in files]
    return Dataset(root.name, np.concatenate(feats), np.array(y), classes, ids)
