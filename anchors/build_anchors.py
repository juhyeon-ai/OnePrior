import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
NUM_ANCHORS = 768


def read_table():
    with (ROOT / "anchors.csv").open() as f:
        return list(csv.DictReader(f))


def pixel_hash(image):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(image.convert("RGB"))).tobytes()).hexdigest()


def crop_histai(row, histai_root):
    from trident import load_wsi

    slide = Path(histai_root) / row["dataset"] / row["slide_file"]
    wsi = load_wsi(slide_path=str(slide), lazy_init=False, mpp=0.5)
    coords = np.array([[int(row["x_level0"]), int(row["y_level0"])]], dtype=np.int64)
    patcher = wsi.create_patcher(patch_size=256, src_mag=int(row["level0_magnification"]), dst_mag=10,
                                 custom_coords=coords, coords_only=False, pil=True)
    return patcher[0][0].convert("RGB")


def stitch_spider(row, spider_dirs):
    organ = row["dataset"].split("-")[1]
    canvas = Image.new("RGB", (672, 672))
    for k, name in enumerate(row["spider_grid"].split()):
        tile = Image.open(Path(spider_dirs[organ]) / name).convert("RGB")
        canvas.paste(tile, ((k % 3) * 224, (k // 3) * 224))
    return canvas.crop((80, 80, 592, 592)).resize((256, 256), Image.LANCZOS)


def print_downloads(rows):
    files = defaultdict(set)
    for row in rows:
        if row["source"] == "HISTAI":
            files[row["dataset"]].add(row["slide_file"])
    for dataset in sorted(files):
        include = " ".join(f'--include "{name}"' for name in sorted(files[dataset]))
        print(f"hf download histai/{dataset} --repo-type dataset --local-dir histai/{dataset} {include}")
    for organ in sorted({row["dataset"] for row in rows if row["source"] == "SPIDER"}):
        print(f"hf download histai/{organ} --repo-type dataset --local-dir spider/{organ}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--histai-root", default=None)
    parser.add_argument("--spider", nargs="*", default=[], metavar="ORGAN=DIR")
    parser.add_argument("--print-downloads", action="store_true")
    args = parser.parse_args()
    rows = read_table()
    if args.print_downloads:
        print_downloads(rows)
        return
    spider_dirs = dict(item.split("=", 1) for item in args.spider)
    (ROOT / "images").mkdir(exist_ok=True)
    built = 0
    for row in rows:
        target = ROOT / row["file"]
        if target.exists():
            continue
        if row["source"] == "HISTAI" and args.histai_root:
            crop_histai(row, args.histai_root).save(target)
            built += 1
        elif row["source"] == "SPIDER" and row["dataset"].split("-")[1] in spider_dirs:
            stitch_spider(row, spider_dirs).save(target)
            built += 1
    missing, wrong = [], []
    for row in rows:
        target = ROOT / row["file"]
        if not target.exists():
            missing.append(row["file"])
        elif pixel_hash(Image.open(target)) != row["sha256"]:
            wrong.append(row["file"])
    print(f"built {built}, present {NUM_ANCHORS - len(missing)} of {NUM_ANCHORS}, checksum mismatches {len(wrong)}")
    for name in wrong:
        print(f"mismatch {name}")
    if missing:
        sources = sorted({row["source"] for row in rows if row["file"] in set(missing)})
        print(f"missing sources: {', '.join(sources)}")
    sys.exit(1 if missing or wrong else 0)


if __name__ == "__main__":
    main()
