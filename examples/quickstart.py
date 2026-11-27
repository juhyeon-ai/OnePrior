import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from anchors import anchor_coordinates
from evaluation import lesion_masks, localization_metrics, mean_localization
from mil_models import build_model, load_prior

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"


def read_slide(path):
    with h5py.File(path, "r") as f:
        features = torch.from_numpy(np.asarray(f["features"], dtype=np.float32))
        coords = np.asarray(f["coords"])
        fraction = np.asarray(f["lesion_fraction"], dtype=np.float32) if "lesion_fraction" in f else None
    return features, coords, fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", default=sorted(str(p) for p in DATA.glob("camelyon16_*_hibou_b.h5")))
    parser.add_argument("--anchor-features", default=str(DATA / "anchor_features_hibou_b.pt"))
    parser.add_argument("--arch", default="abmil")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--out", default="attention")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    anchor_features = torch.load(args.anchor_features, map_location=args.device, weights_only=True)
    model = build_model(args.arch, in_dim=anchor_features.shape[0], num_classes=2).eval().to(args.device)
    load_prior(model, args.weights or ROOT / "weights" / f"{args.arch}_scout.pt")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"{'slide':<32}{'patches':>8}{'scored':>8}{'lesion':>8}{'AUROC':>8}{'R-prec':>8}{'random':>8}")
    for path in args.features:
        features, coords, fraction = read_slide(path)
        coordinates = anchor_coordinates(features.to(args.device), anchor_features)
        with torch.no_grad():
            attention = torch.softmax(model(coordinates)["scores"].float(), dim=0).cpu().numpy()
        name = Path(path).stem
        np.save(out_dir / f"{name}.npy", attention)
        metrics = localization_metrics(attention, *lesion_masks(fraction)) if fraction is not None else None
        if metrics is None:
            top = int(np.argmax(attention))
            print(f"{name:<32}{len(attention):>8}{'-':>8}{'-':>8}{'-':>8}{'-':>8}{'-':>8}  top patch at {tuple(int(v) for v in coords[top])}")
            continue
        results.append(metrics)
        print(f"{name:<32}{len(attention):>8}{metrics['num_patches']:>8}{metrics['num_lesion']:>8}{metrics['auroc']:>8.3f}{metrics['r_precision']:>8.3f}{metrics['lesion_share']:>8.3f}")
    if results:
        mean = mean_localization(results)
        print(f"{'mean':<32}{'':>8}{'':>8}{'':>8}{mean['auroc']:>8.3f}{mean['r_precision']:>8.3f}{mean['lesion_share']:>8.3f}")
    print(f"attention saved to {out_dir}/")


if __name__ == "__main__":
    main()
