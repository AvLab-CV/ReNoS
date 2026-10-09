#python visualize_cls_distribution.py --source ../assets/btad --out cls_distribution_btad.png --foundad_config configs/config.yaml --device cuda --batch_size 1
from pathlib import Path
import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt

from src.foundad import VisionModule
from src.datasets.dataset import build_base_transform


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_foundad_config(config_path):
    config_path = Path(config_path).resolve()
    base_cfg = OmegaConf.load(config_path)

    app_name = None
    for item in OmegaConf.to_container(base_cfg.get("defaults", []), resolve=False):
        if isinstance(item, dict) and "app" in item:
            app_name = item["app"]
            break

    app_cfg = OmegaConf.load(config_path.parent / "app" / f"{app_name}.yaml")
    cfg = OmegaConf.to_container(OmegaConf.merge(base_cfg, {"app": app_cfg}), resolve=True)

    params = dict(cfg["app"])
    params.update(cfg)
    return params


def build_model(params, device):
    m = params["meta"]
    model = VisionModule(
        m["model"],
        m["pred_depth"],
        m["pred_emb_dim"],
        use_cuda=(device.type == "cuda"),
        if_pe=m.get("if_pred_pe", True),
        feat_normed=m.get("feat_normed", False),
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model, int(m["crop_size"])


def image_files(path, recursive=False):
    iterator = path.rglob("*") if recursive else path.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def add_class_images(data, cls_name, img_dir, recursive=False):
    if not img_dir.is_dir():
        return False

    imgs = image_files(img_dir, recursive=recursive)
    if not imgs:
        return False

    data[cls_name] = imgs
    return True


def collect_realiad_meta(root, split="train", normal_only=True):
    meta_path = root / "meta.json"
    if not meta_path.exists():
        return {}

    meta = json.loads(meta_path.read_text())
    data = {}
    splits = ["train", "test"] if split == "all" else [split]
    for split_name in splits:
        for cls_name, entries in sorted(meta.get(split_name, {}).items()):
            imgs = data.setdefault(cls_name, [])
            for entry in sorted(entries, key=lambda x: x["img_path"]):
                if normal_only and int(entry.get("anomaly", 0)) != 0:
                    continue
                img_path = root / entry["img_path"]
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
                    imgs.append(img_path)
    data = {cls_name: imgs for cls_name, imgs in data.items() if imgs}
    return data


def collect_imagefolder(root):
    data = {}
    for cls_dir in sorted(root.iterdir()):
        if cls_dir.is_dir():
            add_class_images(data, cls_dir.name, cls_dir, recursive=True)
    return data


def collect_foundad_anomaly_train(root, dataset):
    normal_dirs = {
        "mvtec": ["train/good"],
        "visa": ["train/ok"],
        "realiad": ["train/good", "train/ok", "test/good", "OK"],
        "auto": ["train/good", "train/ok", "test/good", "OK"],
    }[dataset]
    data = {}

    for cls_dir in sorted(Path(root).iterdir()):
        if not cls_dir.is_dir():
            continue

        for sub in normal_dirs:
            if add_class_images(data, cls_dir.name, cls_dir / sub, recursive=(sub == "OK")):
                break

    return data


def collect_images(root, dataset="auto", realiad_split="train", normal_only=True):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"source directory not found: {root}")

    if dataset == "realiad":
        collectors = [
            lambda: collect_realiad_meta(root, realiad_split, normal_only),
            lambda: collect_imagefolder(root / "train") if (root / "train").is_dir() else {},
            lambda: collect_imagefolder(root) if root.name == "train" else {},
            lambda: collect_foundad_anomaly_train(root, "realiad"),
        ]
    elif dataset in {"mvtec", "visa"}:
        collectors = [lambda: collect_foundad_anomaly_train(root, dataset)]
    else:
        collectors = [
            lambda: collect_foundad_anomaly_train(root, "auto"),
            lambda: collect_realiad_meta(root, realiad_split, normal_only),
            lambda: collect_imagefolder(root / "train") if (root / "train").is_dir() else {},
            lambda: collect_imagefolder(root) if root.name == "train" else {},
        ]

    for collect in collectors:
        data = collect()
        if data:
            return data

    return {}


def parse_cls(out):
    if isinstance(out, tuple):
        cls = out[1]
    elif isinstance(out, dict):
        cls = out.get("cls_token", out.get("x_norm_clstoken"))
    elif torch.is_tensor(out):
        cls = out[:, 0, :] if out.ndim == 3 else out
    else:
        raise RuntimeError(type(out))

    return F.normalize(cls.float(), p=2, dim=-1)


@torch.no_grad()
def extract_cls(paths, model, transform, device, batch_size, cls_n_layer):
    all_cls = []
    model_name = str(model.model_name).lower()

    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i:i + batch_size]:
            img = Image.open(p).convert("RGB")
            batch.append(transform(img))

        x = torch.stack(batch).to(device)

        if model_name in {"dinov2", "dinov3"}:
            outs = model.encoder.get_intermediate_layers(
                x,
                n=cls_n_layer,
                return_class_token=True,
            )
        else:
            outs = model.encoder.get_intermediate_layers(x, n=cls_n_layer)

        all_cls.append(parse_cls(outs[0]).cpu().numpy())

    x = np.concatenate(all_cls, axis=0).astype(np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def pca_2d(x):
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="cls_distribution.png")
    ap.add_argument("--foundad_config", default="configs/config.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--cls_n_layer", type=int, default=1)
    ap.add_argument("--dataset", choices=["auto", "mvtec", "visa", "realiad"], default="auto")
    ap.add_argument("--realiad_split", choices=["train", "test", "all"], default="train")
    ap.add_argument("--include_anomaly", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    params = load_foundad_config(args.foundad_config)
    model, crop_size = build_model(params, device)
    transform = transforms.Compose(build_base_transform(crop_size))

    per_class = collect_images(
        args.source,
        args.dataset,
        realiad_split=args.realiad_split,
        normal_only=not args.include_anomaly,
    )
    if not per_class:
        raise RuntimeError(f"no images found under {args.source} for dataset={args.dataset}")

    xs, ys = [], []
    for cls_name, paths in per_class.items():
        print(f"{cls_name}: {len(paths)} images")
        emb = extract_cls(paths, model, transform, device, args.batch_size, args.cls_n_layer)
        xs.append(emb)
        ys.extend([cls_name] * len(emb))

    xs = np.concatenate(xs, axis=0)
    z = pca_2d(xs)

    plt.figure(figsize=(10, 8))
    ys = np.array(ys)

    classes = sorted(set(ys))

    # tab20 最多提供 20 種較容易區分的顏色
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(classes))]

    # 搭配不同 marker，避免只靠顏色辨識
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "H", "8", "p", "d"]

    for i, cls_name in enumerate(classes):
        idx = ys == cls_name
        plt.scatter(
            z[idx, 0],
            z[idx, 1],
            s=18,
            alpha=0.78,
            color=colors[i],
            marker=markers[i % len(markers)],
            edgecolors="black",
            linewidths=0.25,
            label=cls_name,
        )

    plt.title("CLS Token Distribution by Class")
    plt.legend(
        fontsize=12,          # class 文字大小
        title_fontsize=11,    # 標題大小
        ncol=1,               # 分成幾欄
        frameon=True,         # 顯示外框
        markerscale=1.2,      # legend 裡 marker 大小
        loc="center left",    # 位置
        bbox_to_anchor=(1.02, 0.5),
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()


## comments:python visualize_cls_distribution.py --source ../assets/btad --out cls_distribution_btad.png --foundad_config configs/config.yaml --device cuda --batch_size 1
