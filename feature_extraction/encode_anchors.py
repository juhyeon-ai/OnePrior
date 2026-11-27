import argparse

import torch
from trident.patch_encoder_models import encoder_factory

from anchors import encode_anchors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    encoder = encoder_factory(args.encoder).eval().to(args.device)
    features = encode_anchors(encoder, encoder.eval_transforms, batch_size=args.batch_size, device=args.device)
    torch.save(features, args.out)
    print(f"{args.encoder}: {tuple(features.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
