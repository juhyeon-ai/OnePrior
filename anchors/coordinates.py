from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


ANCHOR_ROOT = Path(__file__).resolve().parent


NUM_ANCHORS = 768


def anchor_image_paths(root=ANCHOR_ROOT):
    paths = sorted((Path(root) / "images").glob("*.png"))
    if len(paths) != NUM_ANCHORS:
        raise FileNotFoundError(f"found {len(paths)} of {NUM_ANCHORS} anchor images in {Path(root) / 'images'}; see anchors/README.md")
    return paths


@torch.no_grad()
def encode_anchors(encoder, transform, root=ANCHOR_ROOT, batch_size=64, device="cpu"):
    paths = anchor_image_paths(root)
    precision = getattr(encoder, "precision", torch.float32)
    features = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[start: start + batch_size]]).to(device)
        with torch.autocast(device_type=str(device).split(":")[0], dtype=precision, enabled=precision != torch.float32):
            features.append(encoder(batch).float().cpu())
    return F.normalize(torch.cat(features), dim=-1)


def anchor_coordinates(features, anchor_features):
    anchors = F.normalize(anchor_features.float(), dim=-1).to(features.device)
    return F.normalize(features.float(), dim=-1) @ anchors.t()
