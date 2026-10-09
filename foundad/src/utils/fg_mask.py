# src/utils/fg_mask.py
from pathlib import Path

import cv2
import numpy as np
import torch

from scipy.ndimage import distance_transform_edt # [新增] 用於計算背景到前景的距離


def keep_large_components(mask: np.ndarray, min_component_area_ratio: float = 0.00001) -> np.ndarray:
    h, w = mask.shape[:2]
    min_area = h * w * min_component_area_ratio

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return mask.astype(bool)

    out = np.zeros_like(mask, dtype=bool)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == label] = True

    return out


def morphology_close_open(
    mask: np.ndarray,
    close_kernel_size: int = 70,
    open_kernel_size: int = 1,
) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255

    close_kernel_size = max(1, int(close_kernel_size))
    open_kernel_size = max(1, int(open_kernel_size))

    close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)

    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)

    return mask_u8 > 0


def fill_internal_holes(
    mask: np.ndarray,
    max_hole_area_ratio: float = 0.12,
    max_hole_total_ratio: float = 0.25,
) -> np.ndarray:
    h, w = mask.shape[:2]
    image_area = h * w

    mask_bool = mask.astype(bool)
    bg = ~mask_bool

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bg.astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return mask_bool

    filled = mask_bool.copy()
    filled_area = 0

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        ww = stats[label, cv2.CC_STAT_WIDTH]
        hh = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        touches_border = (
            x == 0 or y == 0 or
            x + ww >= w or
            y + hh >= h
        )

        if touches_border:
            continue

        if area / image_area > max_hole_area_ratio:
            continue

        if (filled_area + area) / image_area > max_hole_total_ratio:
            continue

        filled[labels == label] = True
        filled_area += area

    return filled


def expand_mask(mask: np.ndarray, expand_kernel_size: int = 15) -> np.ndarray:
    expand_kernel_size = int(expand_kernel_size)

    if expand_kernel_size <= 1:
        return mask.astype(bool)

    kernel = np.ones((expand_kernel_size, expand_kernel_size), np.uint8)
    expanded = cv2.dilate(mask.astype(np.uint8) * 255, kernel, iterations=1)

    return expanded > 0


def build_fg_mask(
    img_rgb: np.ndarray,
    border_ratio: float = 0.05,
    min_area: float = 0.10,
    max_area: float = 0.95,
    morph_close_kernel: int = 70,
    morph_open_kernel: int = 1,
    mask_expand_kernel: int = 15,
    blur_sigma: float = 1.0,
    min_component_area_ratio: float = 0.00001,
    lab_l_weight: float = 0.05,
    lab_ab_weight: float = 1.0,
    min_weight: float = 0.0,
    fill_holes: bool = True,
    max_hole_area_ratio: float = 0.12,
    max_hole_total_ratio: float = 0.25,
) -> np.ndarray:
    """
    Input:
        img_rgb: RGB image, uint8, [H, W, 3]

    Output:
        soft foreground mask, float32, [H, W], range [0, 1]
    """
    h, w = img_rgb.shape[:2]
    b = max(1, int(min(h, w) * border_ratio))

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    border_pixels = np.concatenate(
        [
            lab[:b, :, :].reshape(-1, 3),
            lab[-b:, :, :].reshape(-1, 3),
            lab[:, :b, :].reshape(-1, 3),
            lab[:, -b:, :].reshape(-1, 3),
        ],
        axis=0,
    )

    mu_bg = np.median(border_pixels, axis=0)

    dL = lab[:, :, 0] - mu_bg[0]
    da = lab[:, :, 1] - mu_bg[1]
    db = lab[:, :, 2] - mu_bg[2]

    dist = np.sqrt(
        lab_l_weight * (dL ** 2)
        + lab_ab_weight * (da ** 2 + db ** 2)
    )

    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask_u8 = cv2.threshold(
        dist_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    mask = mask_u8 > 0

    mask = keep_large_components(
        mask,
        min_component_area_ratio=min_component_area_ratio,
    )

    mask = morphology_close_open(
        mask,
        close_kernel_size=morph_close_kernel,
        open_kernel_size=morph_open_kernel,
    )

    if fill_holes:
        mask = fill_internal_holes(
            mask,
            max_hole_area_ratio=max_hole_area_ratio,
            max_hole_total_ratio=max_hole_total_ratio,
        )

    mask = expand_mask(
        mask,
        expand_kernel_size=mask_expand_kernel,
    )

    area_ratio = mask.mean()
    if area_ratio < min_area or area_ratio > max_area:
        mask = np.ones((h, w), dtype=bool)

    soft = cv2.GaussianBlur(
        mask.astype(np.float32),
        ksize=(0, 0),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma,
    )

    soft = np.clip(soft, 0.0, 1.0).astype(np.float32)

    if min_weight > 0:
        min_weight = float(np.clip(min_weight, 0.0, 1.0))
        soft = min_weight + (1.0 - min_weight) * soft
        soft = np.clip(soft, 0.0, 1.0).astype(np.float32)

    return soft


def _read_rgb_resized(path: str | Path, crop_size: int) -> np.ndarray | None:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if crop_size > 0:
        img_rgb = cv2.resize(
            img_rgb,
            (crop_size, crop_size),
            interpolation=cv2.INTER_LINEAR,
        )

    return img_rgb


def build_patch_fg_mask(
    image_paths,
    crop_size: int,
    patch_h: int,
    patch_w: int,
    device,
    **fg_kwargs,
) -> torch.Tensor:
    """
    Build patch-level foreground mask for FoundAD.

    Input:
        image_paths: list[str]
        crop_size: int
        patch_h, patch_w: FoundAD patch grid size

    Output:
        patch_fg_mask: torch.Tensor, [B, patch_h * patch_w]
    """
    patch_masks = []

    for path in image_paths:
        img_rgb = _read_rgb_resized(path, crop_size)

        if img_rgb is None:
            patch_mask = np.ones((patch_h, patch_w), dtype=np.float32)
            patch_masks.append(patch_mask.reshape(-1))
            continue

        fg_mask = build_fg_mask(img_rgb, **fg_kwargs)

        patch_mask = cv2.resize(
            fg_mask,
            (patch_w, patch_h),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)

        patch_masks.append(patch_mask.reshape(-1))

    patch_masks = np.stack(patch_masks, axis=0)

    return torch.from_numpy(patch_masks).to(
        device=device,
        dtype=torch.float32,
    )


def save_mask(mask: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mask_u8 = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), mask_u8)


def save_fg_masks_for_paths(
    image_paths,
    save_root: str | Path,
    test_root: str | Path,
    crop_size: int,
    suffix: str = "_fg_mask",
    **fg_kwargs,
) -> None:
    """
    Save full-resolution foreground masks while preserving dataset relative folders.

    Example output:
        save_root / class / test / defect_type / xxx_fg_mask.png
    """
    save_root = Path(save_root)
    test_root = Path(test_root)

    for path in image_paths:
        path = Path(path)
        img_rgb = _read_rgb_resized(path, crop_size)

        if img_rgb is None:
            continue

        fg_mask = build_fg_mask(img_rgb, **fg_kwargs)

        try:
            rel = path.relative_to(test_root)
        except ValueError:
            rel = Path(path.name)

        save_path = save_root / rel.parent / f"{path.stem}{suffix}.png"
        save_mask(fg_mask, save_path)


def apply_fgs_to_patch_score(
    patch_score: torch.Tensor,
    image_paths,
    crop_size: int,
    device,
    fgs_rescue_mode: str = "score",
    fg_rescue_score_thresh: float = 0.6,
    fg_rescue_weight: float = 1.0,
    fg_rescue_dilate_patches: int = 1,
    fg_rescue_near_fg_only: bool = False,
    max_dist_patches: int = 3,
    return_debug: bool = False,
    rescue_score_thresh: float | None = None,
    **fg_kwargs,
) -> torch.Tensor:
    h = w = int(np.sqrt(patch_score.size(1)))

    # 取得原始 FGS Mask: [B, P]
    patch_fg_mask = build_patch_fg_mask(
        image_paths=image_paths,
        crop_size=crop_size,
        patch_h=h,
        patch_w=w,
        device=device,
        **fg_kwargs,
    )

    final_mask = patch_fg_mask.clone()
    pure_fgs_score = patch_score * patch_fg_mask

    if rescue_score_thresh is not None:
        fg_rescue_score_thresh = rescue_score_thresh

    if fgs_rescue_mode not in {"none", "score"}:
        raise ValueError(f"Unsupported fgs_rescue_mode: {fgs_rescue_mode}")

    if fgs_rescue_mode == "score":
        min_weight = float(fg_kwargs.get("min_weight", 0.1))
        rescue_weight = float(np.clip(fg_rescue_weight, 0.0, 1.0))
        rescue_thresh = float(fg_rescue_score_thresh)
        dilate_patches = max(0, int(fg_rescue_dilate_patches))
        max_dist_patches = max(0, int(max_dist_patches))

        for b in range(patch_score.size(0)):
            score_img = patch_score[b]
            score_norm = (score_img - score_img.min()) / (score_img.max() - score_img.min() + 1e-8)
            rescue_area = (score_norm >= rescue_thresh).view(h, w).detach().cpu().numpy()

            if dilate_patches > 0:
                kernel_size = 2 * dilate_patches + 1
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                rescue_area = cv2.dilate(rescue_area.astype(np.uint8), kernel, iterations=1) > 0

            if fg_rescue_near_fg_only:
                fg_bin = patch_fg_mask[b].view(h, w).detach().cpu().numpy() > min_weight
                if fg_bin.sum() == 0 or fg_bin.sum() == h * w:
                    continue
                dist_to_fg = distance_transform_edt(~fg_bin)
                rescue_area = rescue_area & (dist_to_fg <= max_dist_patches)

            rescue_tensor = torch.from_numpy(rescue_area).to(device=device, dtype=torch.bool).view(-1)
            final_mask[b, rescue_tensor] = torch.maximum(
                final_mask[b, rescue_tensor],
                final_mask.new_full((int(rescue_tensor.sum().item()),), rescue_weight),
            )

    rescued_score = patch_score * final_mask

    if return_debug:
        return rescued_score, pure_fgs_score, final_mask, patch_fg_mask

    return rescued_score
