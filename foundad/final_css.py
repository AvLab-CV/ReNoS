from __future__ import annotations

import sys
import csv
import json
import random
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import cv2
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms

from src.foundad import VisionModule
from src.datasets.dataset import build_base_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------
# Config / FoundAD utilities
# ---------------------------------------------------------------------

def load_config_file(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
    return json.loads(path.read_text())


def load_foundad_hydra_config(config_path: str) -> Dict[str, Any]:
    """
    Load FoundAD config without calling hydra.initialize(),
    avoiding GlobalHydra re-initialization issues.
    """
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent

    if not config_path.is_file():
        raise FileNotFoundError(f"FoundAD config does not exist: {config_path}")

    base_cfg = OmegaConf.load(config_path)
    defaults = OmegaConf.to_container(base_cfg.get("defaults", []), resolve=False)

    app_name = None
    for item in defaults:
        if isinstance(item, dict) and "app" in item:
            app_name = item["app"]
            break

    if app_name is None:
        raise RuntimeError(f"Cannot find `app: ...` in defaults of {config_path}")

    app_cfg_path = config_dir / "app" / f"{app_name}.yaml"
    if not app_cfg_path.is_file():
        raise FileNotFoundError(f"FoundAD app config does not exist: {app_cfg_path}")

    app_cfg = OmegaConf.load(app_cfg_path)
    full_cfg = OmegaConf.merge(base_cfg, {"app": app_cfg})
    full_dict = OmegaConf.to_container(full_cfg, resolve=True)

    app_dict = dict(full_dict.get("app", {}))
    params = dict(app_dict)
    params.update(full_dict)

    if "meta" not in params:
        raise RuntimeError(f"Loaded FoundAD config does not contain `meta`: {app_cfg_path}")

    return params


def build_css_transform(crop_size: int) -> transforms.Compose:
    """
    Deterministic CSS transform:
    Resize / ToTensor / ImageNet Normalize only.
    """
    return transforms.Compose(build_base_transform(crop_size))


def build_foundad_model(
    params: Dict[str, Any],
    device: torch.device,
) -> Tuple[VisionModule, int, int]:
    mcfg = params["meta"]

    model = VisionModule(
        mcfg["model"],
        mcfg["pred_depth"],
        mcfg["pred_emb_dim"],
        use_cuda=(device.type == "cuda"),
        if_pe=mcfg.get("if_pred_pe", True),
        feat_normed=mcfg.get("feat_normed", False),
    )

    model.eval()
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False

    n_layer = int(mcfg.get("n_layer", 3))
    crop_size = int(mcfg["crop_size"])

    return model, n_layer, crop_size


# ---------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------

def collect_category_images(
    category_dir: Path,
    train_subpaths: Tuple[str, ...],
    allowed_exts: Tuple[str, ...],
    recursive_images: bool = False,
) -> Tuple[List[Path], str | None]:
    for sub in train_subpaths:
        candidate = category_dir / sub
        if candidate.is_dir():
            iterator = candidate.rglob("*") if recursive_images else candidate.iterdir()
            found = sorted(
                p for p in iterator
                if p.is_file() and p.suffix.lower() in allowed_exts
            )
            if found:
                return found, sub

    return [], None


def collect_all_train_images(
    source_root: Path,
    train_subpaths: Tuple[str, ...],
    allowed_exts: Tuple[str, ...],
    recursive_images: bool = False,
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}

    for category_dir in sorted(filter(Path.is_dir, source_root.iterdir())):
        if category_dir.name.startswith("."):
            continue

        images, chosen_subpath = collect_category_images(
            category_dir=category_dir,
            train_subpaths=train_subpaths,
            allowed_exts=allowed_exts,
            recursive_images=recursive_images,
        )

        if not images:
            print(f"[skip] {category_dir.name}: no train images found")
            continue

        result[category_dir.name] = {
            "images": images,
            "chosen_subpath": chosen_subpath,
        }

    return result


def clear_directory_images(dest_dir: Path, allowed_exts: Tuple[str, ...]) -> None:
    if not dest_dir.exists():
        return

    for p in dest_dir.iterdir():
        if p.is_file() and p.suffix.lower() in allowed_exts:
            p.unlink()


def copy_selected_images(
    selected_paths: List[Path],
    dest_dir: Path,
    allowed_exts: Tuple[str, ...],
    rename_images: bool,
    clear_target: bool,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)

    if clear_target:
        clear_directory_images(dest_dir, allowed_exts)

    existing_files = [
        p for p in dest_dir.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_exts
    ]
    start_idx = len(existing_files)

    for i, src in enumerate(selected_paths):
        if rename_images:
            new_name = f"{start_idx + i:03d}{src.suffix.lower()}"
        else:
            new_name = src.name

        shutil.copy2(src, dest_dir / new_name)


# ---------------------------------------------------------------------
# Image quality metrics
# ---------------------------------------------------------------------

def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def laplacian_variance(img_rgb: np.ndarray) -> float:
    """
    Sharpness proxy.
    Higher value usually means sharper image.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def highlight_ratio(img_rgb: np.ndarray, threshold: int = 250) -> float:
    """
    Ratio of pixels where any RGB channel exceeds threshold.
    Used as a soft highlight / saturation penalty.
    """
    return float(np.any(img_rgb > threshold, axis=-1).mean())


def compute_image_quality(
    image_paths: List[Path],
    highlight_threshold: int,
) -> Dict[str, np.ndarray]:
    blur = []
    highlight = []

    for p in image_paths:
        img = load_rgb(p)
        blur.append(laplacian_variance(img))
        highlight.append(highlight_ratio(img, threshold=highlight_threshold))

    return {
        "blur": np.asarray(blur, dtype=np.float32),
        "highlight": np.asarray(highlight, dtype=np.float32),
    }


# ---------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------

def l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def minmax_norm_np(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        return values

    v_min = float(np.min(values))
    v_max = float(np.max(values))

    if abs(v_max - v_min) < eps:
        return np.zeros_like(values, dtype=np.float32)

    return ((values - v_min) / (v_max - v_min + eps)).astype(np.float32)


# ---------------------------------------------------------------------
# CLS extraction
# ---------------------------------------------------------------------

def parse_cls_from_intermediate_output(out: Any) -> torch.Tensor:
    if isinstance(out, tuple):
        if len(out) >= 2:
            cls = out[1]
        else:
            raise RuntimeError(f"Unexpected tuple output length: {len(out)}")

    elif isinstance(out, dict):
        if "cls_token" in out:
            cls = out["cls_token"]
        elif "x_norm_clstoken" in out:
            cls = out["x_norm_clstoken"]
        else:
            raise RuntimeError(f"Cannot find CLS token in dict keys: {list(out.keys())}")

    elif torch.is_tensor(out):
        if out.ndim == 3:
            cls = out[:, 0, :]
        elif out.ndim == 2:
            cls = out
        else:
            raise RuntimeError(f"Unexpected tensor shape: {tuple(out.shape)}")

    else:
        raise RuntimeError(f"Unsupported intermediate output type: {type(out)}")

    if cls.ndim != 2:
        raise RuntimeError(f"Expected CLS shape [B, C], got {tuple(cls.shape)}")

    return F.normalize(cls.float(), p=2, dim=-1)


@torch.no_grad()
def extract_cls_embeddings_foundad(
    image_paths: List[Path],
    model: VisionModule,
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int,
    cls_n_layer: int,
) -> np.ndarray:
    """
    Extract CLS embeddings from FoundAD encoder.

    Supported:
        dinov2 / dinov3 / dino
    """
    model.eval()
    model_name = str(model.model_name).lower()

    if model_name not in {"dinov2", "dinov3", "dino"}:
        raise NotImplementedError(
            f"CLS extraction currently supports dinov2 / dinov3 / dino only. "
            f"Current model_name={model.model_name}"
        )

    all_cls = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]

        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))

        imgs = torch.stack(imgs, dim=0).to(device, non_blocking=True)

        if model_name in {"dinov2", "dinov3"}:
            outs = model.encoder.get_intermediate_layers(
                imgs,
                n=int(cls_n_layer),
                return_class_token=True,
            )
        else:
            outs = model.encoder.get_intermediate_layers(
                imgs,
                n=int(cls_n_layer),
            )

        cls = parse_cls_from_intermediate_output(outs[0])
        all_cls.append(cls.detach().cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(all_cls, axis=0)
    return l2_normalize_np(embeddings).astype(np.float32)


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------

VALID_METRICS = {
    "std_score",
    "mean_score",
    "blur_score",
    "highlight_score",

    "topk_mean_similarity",
    "top1_similarity",
    "topk_std_similarity",
    "all_mean_similarity",

    "raw_cls_css_score",
    "cls_css_score",

    "cross_topk_mean_similarity",
    "inter_class_margin",
    "inter_class_margin_score",

    "top1_gap",
    "top1_gap_score",
}


def sort_rows(
    rows: List[Dict[str, Any]],
    metric: str,
    order: str = "desc",
) -> List[Dict[str, Any]]:
    if metric not in VALID_METRICS:
        raise ValueError(f"Unsupported metric={metric}. Use one of {sorted(VALID_METRICS)}.")

    reverse = str(order).lower() == "desc"
    return sorted(rows, key=lambda r: float(r[metric]), reverse=reverse)


def directional_norm(values: np.ndarray, order: str) -> np.ndarray:
    score = minmax_norm_np(values)
    if str(order).lower() == "asc":
        score = 1.0 - score
    return score.astype(np.float32)


def add_cls_css_scores(
    rows: List[Dict[str, Any]],
    w_std: float,
    w_mean: float,
    w_blur: float,
    w_highlight: float,
) -> List[Dict[str, Any]]:
    if not rows:
        return rows

    topk_std = np.asarray([float(r["topk_std_similarity"]) for r in rows], dtype=np.float32)
    topk_mean = np.asarray([float(r["topk_mean_similarity"]) for r in rows], dtype=np.float32)
    blur = np.asarray([float(r["blur"]) for r in rows], dtype=np.float32)
    highlight = np.asarray([float(r["highlight"]) for r in rows], dtype=np.float32)

    std_score = minmax_norm_np(topk_std)
    mean_score = minmax_norm_np(topk_mean)
    blur_score = minmax_norm_np(blur)
    highlight_score = minmax_norm_np(highlight)

    raw_score = (
        float(w_std) * std_score
        + float(w_mean) * mean_score
        + float(w_blur) * blur_score
        - float(w_highlight) * highlight_score
    ).astype(np.float32)

    raw_min = -float(w_highlight)
    raw_max = float(w_std) + float(w_mean) + float(w_blur)
    denom = raw_max - raw_min

    if abs(denom) < 1e-8:
        cls_css_score = np.zeros_like(raw_score, dtype=np.float32)
    else:
        cls_css_score = ((raw_score - raw_min) / denom).astype(np.float32)

    cls_css_score = np.clip(cls_css_score, 0.0, 1.0).astype(np.float32)

    for i, r in enumerate(rows):
        r["std_score"] = float(std_score[i])
        r["mean_score"] = float(mean_score[i])
        r["blur_score"] = float(blur_score[i])
        r["highlight_score"] = float(highlight_score[i])
        r["raw_cls_css_score"] = float(raw_score[i])
        r["cls_css_score"] = float(cls_css_score[i])

    return rows

def add_inter_class_margin_scores(
    rows: List[Dict[str, Any]],
    embeddings: np.ndarray,
    other_embeddings: np.ndarray,
    cross_topk: int,
) -> List[Dict[str, Any]]:
    """
    Add inter-class margin scores.

    cross_topk_mean_similarity:
        mean Top-K cosine similarity between this image and images from other classes.

    inter_class_margin:
        topk_mean_similarity - cross_topk_mean_similarity

    inter_class_margin_score:
        class-wise min-max normalization of inter_class_margin.
    """
    if not rows:
        return rows

    if other_embeddings.size == 0:
        for r in rows:
            r["cross_topk_mean_similarity"] = 0.0
            r["inter_class_margin"] = float(r["topk_mean_similarity"])
            r["inter_class_margin_score"] = 0.0
        return rows

    emb = l2_normalize_np(embeddings)
    other = l2_normalize_np(other_embeddings)

    sim_cross = emb @ other.T

    k = max(1, min(int(cross_topk), sim_cross.shape[1]))

    cross_means = []
    margins = []

    for i, r in enumerate(rows):
        order = np.argsort(-sim_cross[i])
        top_idx = order[:k]
        cross_topk_mean = float(np.mean(sim_cross[i, top_idx]))

        margin = float(r["topk_mean_similarity"]) - cross_topk_mean

        cross_means.append(cross_topk_mean)
        margins.append(margin)

    margin_score = minmax_norm_np(np.asarray(margins, dtype=np.float32))

    for i, r in enumerate(rows):
        r["cross_topk_mean_similarity"] = float(cross_means[i])
        r["inter_class_margin"] = float(margins[i])
        r["inter_class_margin_score"] = float(margin_score[i])

    return rows

def add_top1_gap_scores(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add top1-gap stability scores.

    top1_gap:
        top1_similarity - topk_mean_similarity

    Smaller top1_gap means the image is not only close to one nearest neighbor,
    but is stably supported by its Top-K normal neighborhood.

    top1_gap_score:
        class-wise normalized score where larger is better.
    """
    if not rows:
        return rows

    gaps = []

    for r in rows:
        gap = float(r["top1_similarity"]) - float(r["topk_mean_similarity"])
        r["top1_gap"] = gap
        gaps.append(gap)

    gap_arr = np.asarray(gaps, dtype=np.float32)

    # smaller gap is better, so reverse normalized score
    gap_score = 1.0 - minmax_norm_np(gap_arr)

    for i, r in enumerate(rows):
        r["top1_gap_score"] = float(gap_score[i])

    return rows

def build_rows_for_category(
    category: str,
    image_paths: List[Path],
    embeddings: np.ndarray,
    other_embeddings: np.ndarray,
    top_k: int,
    cross_topk: int,
    highlight_threshold: int,
    score_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    n = len(image_paths)

    if n == 0:
        return []

    quality = compute_image_quality(
        image_paths=image_paths,
        highlight_threshold=highlight_threshold,
    )

    if n == 1:
        row = {
            "_idx": 0,
            "_path": image_paths[0],
            "category": category,
            "rank_in_class": 1,
            "filename": image_paths[0].name,
            "num_images": 1,
            "mean_score": 0.0,
            "blur_score": 0.0,
            "highlight_score": 0.0,
            "raw_cls_css_score": 0.0,
            "cls_css_score": 0.0,
            "std_score": 0.0,
            "blur": float(quality["blur"][0]),
            "highlight": float(quality["highlight"][0]),
            "topk_mean_similarity": 0.0,
            "top1_similarity": 0.0,
            "topk_std_similarity": 0.0,
            "all_mean_similarity": 0.0,
            "cross_topk_mean_similarity": 0.0,
            "inter_class_margin": 0.0,
            "inter_class_margin_score": 0.0,
            "top1_gap": 0.0,
            "top1_gap_score": 0.0,
        }
        return [row]

    k = max(1, min(int(top_k), n - 1))

    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -np.inf)

    rows: List[Dict[str, Any]] = []

    for i, p in enumerate(image_paths):
        order = np.argsort(-sim[i])
        top_idx = order[:k]
        top_sims = sim[i, top_idx]
        valid_all = sim[i][np.isfinite(sim[i])]

        rows.append({
            "_idx": i,
            "_path": p,
            "category": category,
            "rank_in_class": -1,
            "filename": p.name,
            "num_images": int(n),
            "blur": float(quality["blur"][i]),
            "highlight": float(quality["highlight"][i]),
            "topk_mean_similarity": float(np.mean(top_sims)),
            "top1_similarity": float(top_sims[0]),
            "topk_std_similarity": float(np.std(top_sims)),
            "all_mean_similarity": float(np.mean(valid_all)),
        })

    rows = add_cls_css_scores(
        rows=rows,
        w_std=float(score_cfg.get("w_std", 0.0)),
        w_mean=float(score_cfg.get("w_mean", 0.4)),
        w_blur=float(score_cfg.get("w_blur", 0.85)),
        w_highlight=float(score_cfg.get("w_highlight", 0.2)),
    )

    rows = add_inter_class_margin_scores(
        rows=rows,
        embeddings=embeddings,
        other_embeddings=other_embeddings,
        cross_topk=int(cross_topk),
    )

    rows = add_top1_gap_scores(rows)

    return rows


# ---------------------------------------------------------------------
# Final selection
# ---------------------------------------------------------------------

def select_final_rows(
    rows: List[Dict[str, Any]],
    embeddings: np.ndarray,
    num_samples: int,
    selection_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Flexible K-shot selector.

    Supported modes:
        two_stage:
            Stage 1: keep top-K by candidate_metric
            Stage 2: select by selector_metric

        three_stage:
            Stage 1 -> Stage 2 -> Stage 3

        four_stage:
            Stage 1 -> Stage 2 -> Stage 3 -> Stage 4

    For K-shot:
        metric_order:
            select top num_samples by final stage metric.

        greedy_diverse:
            select first by final stage metric, then greedily add diverse samples.
    """
    if not rows:
        return rows

    k_select = max(1, min(int(num_samples), len(rows)))

    enabled = bool(selection_cfg.get("enabled", True))
    mode = str(selection_cfg.get("mode", "two_stage"))

    kshot_strategy = str(selection_cfg.get("kshot_strategy", "metric_order"))
    kshot_score_weight = float(selection_cfg.get("kshot_score_weight", 0.6))
    kshot_diversity_weight = float(selection_cfg.get("kshot_diversity_weight", 0.4))

    if not enabled:
        metric = str(selection_cfg.get("candidate_metric", "cls_css_score"))
        order = str(selection_cfg.get("candidate_order", "desc"))
        final_rows = sort_rows(rows, metric, order)

        for rank, row in enumerate(final_rows, start=1):
            row["rank_in_class"] = rank

        return final_rows

    # ------------------------------------------------------------
    # Build candidate pool
    # ------------------------------------------------------------
    if mode == "four_stage":
        stage1_metric = str(selection_cfg.get("stage1_metric", "std_score"))
        stage1_order = str(selection_cfg.get("stage1_order", "desc"))
        stage1_topk = int(selection_cfg.get("stage1_topk", 15))

        stage2_metric = str(selection_cfg.get("stage2_metric", "all_mean_similarity"))
        stage2_order = str(selection_cfg.get("stage2_order", "desc"))
        stage2_topk = int(selection_cfg.get("stage2_topk", 10))

        stage3_metric = str(selection_cfg.get("stage3_metric", "inter_class_margin_score"))
        stage3_order = str(selection_cfg.get("stage3_order", "desc"))
        stage3_topk = int(selection_cfg.get("stage3_topk", 3))

        stage4_metric = str(selection_cfg.get("stage4_metric", "top1_gap_score"))
        stage4_order = str(selection_cfg.get("stage4_order", "desc"))

        stage1_sorted = sort_rows(rows, stage1_metric, stage1_order)
        stage1_candidates = stage1_sorted[:max(k_select, min(stage1_topk, len(stage1_sorted)))]

        stage2_sorted = sort_rows(stage1_candidates, stage2_metric, stage2_order)
        stage2_candidates = stage2_sorted[:max(k_select, min(stage2_topk, len(stage2_sorted)))]

        stage3_sorted = sort_rows(stage2_candidates, stage3_metric, stage3_order)
        candidates = stage3_sorted[:max(k_select, min(stage3_topk, len(stage3_sorted)))]

        metric_sorted = sort_rows(candidates, stage4_metric, stage4_order)

        final_metric = stage4_metric
        final_order = stage4_order

    elif mode == "three_stage":
        stage1_metric = str(selection_cfg.get("stage1_metric", "topk_std_similarity"))
        stage1_order = str(selection_cfg.get("stage1_order", "desc"))
        stage1_topk = int(selection_cfg.get("stage1_topk", 15))

        stage2_metric = str(selection_cfg.get("stage2_metric", "inter_class_margin_score"))
        stage2_order = str(selection_cfg.get("stage2_order", "desc"))
        stage2_topk = int(selection_cfg.get("stage2_topk", 4))

        stage3_metric = str(selection_cfg.get("stage3_metric", "top1_gap"))
        stage3_order = str(selection_cfg.get("stage3_order", "asc"))

        stage1_sorted = sort_rows(rows, stage1_metric, stage1_order)
        stage1_candidates = stage1_sorted[:max(k_select, min(stage1_topk, len(stage1_sorted)))]

        stage2_sorted = sort_rows(stage1_candidates, stage2_metric, stage2_order)
        candidates = stage2_sorted[:max(k_select, min(stage2_topk, len(stage2_sorted)))]

        metric_sorted = sort_rows(candidates, stage3_metric, stage3_order)

        final_metric = stage3_metric
        final_order = stage3_order

    elif mode == "two_stage":
        candidate_metric = str(selection_cfg.get("candidate_metric", "cls_css_score"))
        candidate_order = str(selection_cfg.get("candidate_order", "desc"))
        candidate_topk = int(selection_cfg.get("candidate_topk", 5))

        selector_metric = str(selection_cfg.get("selector_metric", "top1_similarity"))
        selector_order = str(selection_cfg.get("selector_order", "asc"))

        base_sorted = sort_rows(rows, candidate_metric, candidate_order)
        candidates = base_sorted[:max(k_select, min(candidate_topk, len(base_sorted)))]

        metric_sorted = sort_rows(candidates, selector_metric, selector_order)

        final_metric = selector_metric
        final_order = selector_order

    else:
        raise ValueError(
            f"Unsupported selection.mode={mode}. "
            "Use 'two_stage', 'three_stage', or 'four_stage'."
        )

    # ------------------------------------------------------------
    # Select final supports
    # ------------------------------------------------------------
    if k_select == 1 or kshot_strategy == "metric_order":
        selected = metric_sorted[:k_select]

    elif kshot_strategy == "greedy_diverse":
        selected = [metric_sorted[0]]

        candidate_indices = np.asarray([int(r["_idx"]) for r in candidates], dtype=np.int64)
        candidate_emb = embeddings[candidate_indices]
        candidate_emb = l2_normalize_np(candidate_emb)

        metric_values = np.asarray([float(r[final_metric]) for r in candidates], dtype=np.float32)
        candidate_score = directional_norm(metric_values, final_order)

        selected_ids = {id(selected[0])}

        while len(selected) < k_select:
            selected_local_indices = [
                candidates.index(r) for r in selected
                if r in candidates
            ]
            selected_emb = candidate_emb[selected_local_indices]

            best_row = None
            best_score = -1e18

            for local_i, row in enumerate(candidates):
                if id(row) in selected_ids:
                    continue

                sim_to_selected = candidate_emb[local_i][None, :] @ selected_emb.T
                diversity = float(1.0 - np.max(sim_to_selected))

                score = (
                    kshot_score_weight * float(candidate_score[local_i])
                    + kshot_diversity_weight * diversity
                )

                if score > best_score:
                    best_score = score
                    best_row = row

            selected.append(best_row)
            selected_ids.add(id(best_row))

    else:
        raise ValueError(
            f"Unsupported kshot_strategy={kshot_strategy}. "
            "Use 'metric_order' or 'greedy_diverse'."
        )

    selected_ids = {id(r) for r in selected}

    # selected rows first; remaining rows sorted by std_score for readability
    remainder = [
        r for r in sort_rows(rows, "std_score", "desc")
        if id(r) not in selected_ids
    ]

    final_rows = selected + remainder

    for rank, row in enumerate(final_rows, start=1):
        row["rank_in_class"] = rank

    return final_rows


# ---------------------------------------------------------------------
# CSV / output
# ---------------------------------------------------------------------

CSV_FIELDS = [
    "category",
    "rank_in_class",
    "filename",
    "num_images",
    "mean_score",
    "blur_score",
    "highlight_score",
    "raw_cls_css_score",
    "cls_css_score",
    "std_score",
    "blur",
    "highlight",
    "topk_mean_similarity",
    "top1_similarity",
    "topk_std_similarity",
    "all_mean_similarity",
    "cross_topk_mean_similarity",
    "inter_class_margin",
    "inter_class_margin_score",
    "top1_gap",
    "top1_gap_score",
]


def save_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def run_final_css(cfgd: Dict[str, Any]) -> None:
    seed = cfgd.get("seed", 42)
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

    device_str = str(cfgd["device"])
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable. Use CPU instead.")
        device_str = "cpu"

    device = torch.device(device_str)

    foundad_params = load_foundad_hydra_config(cfgd["foundad_config"])
    model, foundad_n_layer, crop_size = build_foundad_model(foundad_params, device)
    transform = build_css_transform(crop_size)

    print(
        f"[Final-CSS] encoder={foundad_params['meta']['model']} "
        f"crop_size={crop_size} foundad_n_layer={foundad_n_layer} "
        f"cls_n_layer={cfgd['cls_n_layer']}"
    )

    source_root = Path(cfgd["source"])
    target_root = Path(cfgd["target"])
    allowed_exts = tuple(ext.lower() for ext in cfgd["allowed_exts"])

    target_train_root = target_root / "train"
    target_train_root.mkdir(parents=True, exist_ok=True)

    per_class = collect_all_train_images(
        source_root=source_root,
        train_subpaths=tuple(cfgd["train_subpaths"]),
        allowed_exts=allowed_exts,
        recursive_images=bool(cfgd.get("recursive_images", False)),
    )

    embeddings_by_class: Dict[str, np.ndarray] = {}

    for category, info in per_class.items():
        image_paths: List[Path] = info["images"]
        print(f"[Final-CSS] {category}: extracting CLS for {len(image_paths)} images")

        embeddings_by_class[category] = extract_cls_embeddings_foundad(
            image_paths=image_paths,
            model=model,
            transform=transform,
            device=device,
            batch_size=int(cfgd["batch_size"]),
            cls_n_layer=int(cfgd["cls_n_layer"]),
        )

    all_rows: List[Dict[str, Any]] = []

    for category, info in per_class.items():
        image_paths: List[Path] = info["images"]
        embeddings = embeddings_by_class[category]

        other_embeddings_list = [
            emb for cls_name, emb in embeddings_by_class.items()
            if cls_name != category
        ]

        if other_embeddings_list:
            other_embeddings = np.concatenate(other_embeddings_list, axis=0)
        else:
            other_embeddings = np.empty((0, embeddings.shape[1]), dtype=np.float32)

        rows = build_rows_for_category(
            category=category,
            image_paths=image_paths,
            embeddings=embeddings,
            other_embeddings=other_embeddings,
            top_k=int(cfgd["top_k"]),
            cross_topk=int(cfgd.get("cross_topk", cfgd["top_k"])),
            highlight_threshold=int(cfgd["highlight_threshold"]),
            score_cfg=cfgd["score"],
        )

        rows = select_final_rows(
            rows=rows,
            embeddings=embeddings,
            num_samples=int(cfgd["num_samples"]),
            selection_cfg=cfgd["selection"],
        )

        selected_rows = rows[:min(int(cfgd["num_samples"]), len(rows))]
        selected_paths = [Path(r["_path"]) for r in selected_rows]

        copy_selected_images(
            selected_paths=selected_paths,
            dest_dir=target_train_root / category,
            allowed_exts=allowed_exts,
            rename_images=bool(cfgd["rename_images"]),
            clear_target=bool(cfgd["clear_target"]),
        )

        all_rows.extend(rows)

        selected_names = ", ".join(p.name for p in selected_paths)
        print(f"[✓] {category}: selected {len(selected_paths)} -> {selected_names}")

    output_csv = Path(cfgd["output_csv"])
    if str(output_csv).strip() == "":
        output_csv = target_root / "final_css_report.csv"

    save_csv(output_csv, all_rows)
    print(f"[done] saved CSV to {output_csv}")
    print(f"[done] copied selected images to {target_train_root}")


# ---------------------------------------------------------------------
# Hydra config
# ---------------------------------------------------------------------

@dataclass
class FinalCSSCfg:
    source: str = ""
    target: str = ""
    output_csv: str = ""

    num_samples: int = 1

    train_subpaths: List[str] = field(default_factory=lambda: ["train/good", "train/ok"])
    allowed_exts: List[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg"])
    recursive_images: bool = False

    rename_images: bool = False
    clear_target: bool = True

    foundad_config: str = "configs/config.yaml"
    device: str = "cuda"
    batch_size: int = 8

    cls_n_layer: int = 1
    top_k: int = 30
    cross_topk: int = 5

    highlight_threshold: int = 250

    score: Dict[str, Any] = field(default_factory=lambda: {
        "w_std": 0.00,
        "w_mean": 0.40,
        "w_blur": 0.85,
        "w_highlight": 0.20,
    })

    selection: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,

        # two_stage / three_stage / four_stage
        "mode": "four_stage",

        # Stage 1: normalized normal variation
        "stage1_metric": "std_score",
        "stage1_order": "desc",
        "stage1_topk": 15,

        # Stage 2: global same-class representativeness
        "stage2_metric": "all_mean_similarity",
        "stage2_order": "desc",
        "stage2_topk": 10,

        # Stage 3: multi-class separation
        "stage3_metric": "inter_class_margin_score",
        "stage3_order": "desc",
        "stage3_topk": 3,

        # Stage 4: neighborhood stability
        "stage4_metric": "top1_gap_score",
        "stage4_order": "desc",

        # K-shot options:
        # metric_order or greedy_diverse
        "kshot_strategy": "metric_order",
        "kshot_score_weight": 0.60,
        "kshot_diversity_weight": 0.40,
    })

    seed: int | None = 42
    user_config: str = ""


def _finalize_cfg(cfg: DictConfig) -> Dict[str, Any]:
    if cfg.user_config:
        ext = load_config_file(Path(cfg.user_config))
        cfg = OmegaConf.merge(OmegaConf.structured(FinalCSSCfg), ext, cfg)

    source = cfg.get("source")
    target = cfg.get("target")

    if not source or not target:
        sys.exit(
            "Both source and target must be specified. "
            "Example: python final_css.py source=../assets/mvtec target=../assets/mvtec_css"
        )

    return OmegaConf.to_container(cfg, resolve=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="final_css")
def main(cfg: DictConfig) -> None:
    cfgd = _finalize_cfg(cfg)
    run_final_css(cfgd)


if __name__ == "__main__":
    main()
