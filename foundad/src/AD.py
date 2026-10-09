import os
import logging
import math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch, cv2
import torch.nn.functional as F
from matplotlib import cm, pyplot as plt
from PIL import Image
from omegaconf import OmegaConf

import shutil
from tempfile import mkdtemp
from scipy import ndimage
from sklearn.metrics import auc

from src.datasets.dataset import build_dataloader
from src.utils.metrics import (
    calculate_pro,
    compute_imagewise_retrieval_metrics,
    compute_pixelwise_retrieval_metrics,
)
from src.helper import save_segmentation_grid
from src.utils.logging import CSVLogger
from src.foundad import VisionModule

from src.utils.fg_mask import (
    apply_fgs_to_patch_score,
    save_fg_masks_for_paths,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluator")


def _build_model(meta: Dict[str, Any]) -> VisionModule:
    return VisionModule(
        model_name=meta["model"],
        pred_depth=meta["pred_depth"],
        pred_emb_dim=meta["pred_emb_dim"],
        if_pe=meta.get("if_pred_pe", True),
        feat_normed=meta.get("feat_normed", False),
    )

def _flush_segmentation_vis(
    cfg,
    cls,
    std,
    mean,
    vis_img_buf,
    vis_mask_buf,
    vis_pix_buf,
    vis_name_buf,
    vis_score_buf,
):
    if len(vis_name_buf) == 0:
        return

    std_cpu, mean_cpu = std.cpu(), mean.cpu()

    imgs = torch.cat(vis_img_buf)
    imgs_un = (imgs * std_cpu + mean_cpu).permute(0, 2, 3, 1).numpy()

    pix_tmp = torch.cat(vis_pix_buf)
    pmin, pmax = pix_tmp.min(), pix_tmp.max()
    pix_norm_tmp = ((pix_tmp - pmin) / (pmax - pmin + 1e-8)).numpy()

    mask_tmp = torch.cat(vis_mask_buf).squeeze(1).numpy()
    score_tmp = np.asarray(vis_score_buf, dtype=np.float32)

    out_dir = Path(cfg["logging"]["folder"]) / "segmentation" / cls
    save_segmentation_grid(out_dir, vis_name_buf, imgs_un, mask_tmp, pix_norm_tmp, score_tmp)

    vis_img_buf.clear()
    vis_mask_buf.clear()
    vis_pix_buf.clear()
    vis_name_buf.clear()
    vis_score_buf.clear()

def _stream_pixel_metrics_from_memmap(score_mm, mask_mm, num_images, height, width):
    """
    Exact AUROC/AUPR under float16 score precision.
    This matches the old code path that stored pix_buf as .half().
    score_mm: np.memmap, shape [N, H, W], dtype float16
    mask_mm:  np.memmap, shape [N, H, W], dtype uint8
    """

    pos_counts = {}
    neg_counts = {}

    total_pos = 0
    total_neg = 0

    for i in range(num_images):
        scores = np.asarray(score_mm[i]).reshape(-1)
        masks = np.asarray(mask_mm[i]).reshape(-1) > 0

        total_pos += int(masks.sum())
        total_neg += int((~masks).sum())

        # float16 scores have finite discrete values.
        s_pos = scores[masks]
        s_neg = scores[~masks]

        vals, cnts = np.unique(s_pos, return_counts=True)
        for v, c in zip(vals, cnts):
            pos_counts[v.item()] = pos_counts.get(v.item(), 0) + int(c)

        vals, cnts = np.unique(s_neg, return_counts=True)
        for v, c in zip(vals, cnts):
            neg_counts[v.item()] = neg_counts.get(v.item(), 0) + int(c)

    if total_pos == 0 or total_neg == 0:
        return {"auroc": float("nan"), "aupr": float("nan")}

    all_scores = sorted(set(pos_counts.keys()) | set(neg_counts.keys()))

    # AUROC: probability positive score > negative score + 0.5 ties.
    neg_less = 0
    concordant = 0.0
    for s in all_scores:
        p = pos_counts.get(s, 0)
        n = neg_counts.get(s, 0)
        concordant += p * (neg_less + 0.5 * n)
        neg_less += n
    auroc = concordant / (total_pos * total_neg)

    # AUPR / average precision: thresholds descending.
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0

    for s in reversed(all_scores):
        tp += pos_counts.get(s, 0)
        fp += neg_counts.get(s, 0)

        recall = tp / total_pos
        precision = tp / max(tp + fp, 1)

        ap += (recall - prev_recall) * precision
        prev_recall = recall

    return {"auroc": auroc, "aupr": ap}

def _stream_calculate_pro_from_memmap(
    mask_mm,
    score_mm,
    num_images,
    height,
    width,
    max_steps=200,
    expect_fpr=0.3,
):
    score_min = float(np.min(score_mm[:num_images]))
    score_max = float(np.max(score_mm[:num_images]))

    if score_max <= score_min:
        return 0.0

    thresholds = np.linspace(0.0, 1.0, max_steps)

    pros = []
    fprs = []

    for th in thresholds:
        pro_sum = 0.0
        pro_count = 0

        fp_total = 0
        normal_total = 0

        for i in range(num_images):
            score = np.asarray(score_mm[i], dtype=np.float32)
            score = (score - score_min) / (score_max - score_min + 1e-8)

            mask = np.asarray(mask_mm[i]) > 0
            pred = score > th

            normal = ~mask
            fp_total += int((pred & normal).sum())
            normal_total += int(normal.sum())

            labeled, num = ndimage.label(mask, structure=np.ones((3, 3), dtype=int)) #11111111111111111111111111
            for rid in range(1, num + 1):
                region = labeled == rid
                region_area = int(region.sum())
                if region_area == 0:
                    continue
                overlap = int((pred & region).sum()) / region_area
                pro_sum += overlap
                pro_count += 1

        fpr = fp_total / max(normal_total, 1)
        pro = pro_sum / max(pro_count, 1)

        fprs.append(fpr)
        pros.append(pro)

    fprs = np.asarray(fprs)
    pros = np.asarray(pros)

    keep = fprs <= expect_fpr
    if keep.sum() < 2:
        return 0.0

    fprs = fprs[keep]
    pros = pros[keep]

    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]

    span = fprs.max() - fprs.min()
    if span <= 0:
        return 0.0
    fprs = (fprs - fprs.min()) / span
    return auc(fprs, pros)

@torch.inference_mode()
def _evaluate_single_ckpt(ckpt: Path, cfg: Dict[str, Any]) -> None:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(cfg["meta"])
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.predictor.load_state_dict(state["predictor"])
    if model.projector is not None:
        model.projector.load_state_dict(state["projector"])
    model.to(device)
    model.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)

    # error = cfg["meta"].get("loss_mode", "l2")

    dataset_name = cfg["data"].get("dataset", "mvtec")
    if dataset_name == 'mvtec':
        classnames = cfg["data"]["mvtec_classnames"] 
        K = cfg["testing"]["K_top_mvtec"]
    elif dataset_name == 'visa':
        classnames = cfg["data"]["visa_classnames"]
        K = cfg["testing"]["K_top_visa"]
    elif dataset_name == 'btad':
        classnames = cfg["data"]["btad_classnames"]
        K = cfg["testing"]["K_top_btad"]
    elif dataset_name == 'mpdd':
        classnames = cfg["data"]["mpdd_classnames"]
        K = cfg["testing"]["K_top_mpdd"]
    elif dataset_name == 'realiad':
        classnames = cfg["data"]["realiad_classnames"]
        K = cfg["testing"]["K_top_realiad"]
    else:
        raise NotImplementedError
    assert dataset_name in cfg["data"]["test_root"].lower() # check if eval on the same dataset the ckpt trained on

    
    logger.info(f"Evaluating {ckpt.name} on {dataset_name}")
    
    os.makedirs(Path(cfg["logging"]["folder"]), exist_ok=True)
    csv_path = Path(cfg["logging"]["folder"]) / f"{cfg['logging']['write_tag']}_eval.csv"
    csv_logger = CSVLogger(
        csv_path,
        ("%s", "checkpoint"), ("%s", "class"),
        ("%.2f", "inst_auroc"), ("%.2f", "inst_aupr"),
        ("%.2f", "pix_auroc"),  ("%.2f", "pro_auc"),
    )

    inst_auc, inst_aupr, pix_auc, pro_auc = [], [], [], []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    # print("------------------------------------------------------------")
    # # print(cfg)
    # print("meta.n_layer:", cfg.get("meta", {}).get("n_layer"))
    # print("------------------------------------------------------------")

    for cls in classnames:
        _, loader, _ = build_dataloader(
            mode="test",
            root=cfg["data"]["test_root"],
            batch_size=1,
            classname=cls,
            resize=crop,
            datasetname=dataset_name,
        )

        print(f"Evaluating {cls}...")

        # [新增] 統計用變數
        suppressed_pure_count = 0     # 紀錄加 Threshold 前被遮擋的數量
        suppressed_rescued_count = 0  # 紀錄加 Threshold 後被遮擋的數量
        total_anomaly_count = 0

        patch_scores, labels = [], []
        name_buf = []

        # For visualization only. Do not use these for full-class buffering.
        vis_img_buf, vis_mask_buf, vis_pix_buf, vis_name_buf, vis_score_buf = [], [], [], [], []

        tmp_dir = Path(mkdtemp(prefix=f"foundad_eval_{dataset_name}_{cls}_"))
        score_mm = None
        mask_mm = None

        try:
            mm_index = 0
            metric_h = metric_w = None

            seg_vis_fraction = float(cfg["testing"].get("segmentation_vis_fraction", 1.0 / 3.0))
            seg_vis_chunk_size = int(cfg["testing"].get("segmentation_vis_chunk_size", 16))

            num_samples = len(loader.dataset)

            fg_mask_dir = Path(cfg["logging"]["folder"]) / "fg_mask"
            fg_mask_dir.mkdir(parents=True, exist_ok=True)

            for batch in loader:
                img = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                paths = batch["image_path"]; labels.extend(batch["is_anomaly"]); name_buf.extend(batch["image_name"])

                enc = model.target_features(img, paths, n_layer=n_layer)
                pred = model.predict(enc)

                l = F.mse_loss(enc, pred, reduction="none").mean(dim=2)
                #print("l shape", l.size())            

                #print("crop", crop)

                # print("h", h) # 32

                # [新增] 備份套用 FGS 前的原始分數
                l_orig = l.clone()

                # ------------------------------------------------------------
                # FGS: build foreground mask and apply it before Top-K
                # ------------------------------------------------------------

                fg_kwargs = dict(
                    border_ratio=cfg["testing"].get("fg_border_ratio", 0.05),
                    min_area=cfg["testing"].get("fg_min_area", 0.10),
                    max_area=cfg["testing"].get("fg_max_area", 0.95),
                    morph_close_kernel=cfg["testing"].get("fg_morph_close_kernel", 70),
                    morph_open_kernel=cfg["testing"].get("fg_morph_open_kernel", 1),
                    mask_expand_kernel=cfg["testing"].get("fg_mask_expand_kernel", 15),
                    blur_sigma=cfg["testing"].get("fg_blur_sigma", 1.0),
                    min_component_area_ratio=cfg["testing"].get("fg_min_component_area_ratio", 0.00001),
                    lab_l_weight=cfg["testing"].get("fg_lab_l_weight", 0.05),
                    lab_ab_weight=cfg["testing"].get("fg_lab_ab_weight", 1.0),
                    min_weight=cfg["testing"].get("fg_min_weight", 0.0),
                    fill_holes=cfg["testing"].get("fg_fill_holes", True),
                    max_hole_area_ratio=cfg["testing"].get("fg_max_hole_area_ratio", 0.12),
                    max_hole_total_ratio=cfg["testing"].get("fg_max_hole_total_ratio", 0.25),
                )

                if cfg["testing"].get("use_fgs", True):
                    # [修改] 用 l 接補救後的分數，l_pure 接補救前的分數
                    l, l_pure, _, _ = apply_fgs_to_patch_score(
                        patch_score=l,
                        image_paths=paths,
                        crop_size=crop,
                        device=device,
                        fgs_rescue_mode=cfg["testing"].get("fgs_rescue_mode", "score"),
                        fg_rescue_score_thresh=cfg["testing"].get("fg_rescue_score_thresh", 0.6),
                        fg_rescue_weight=cfg["testing"].get("fg_rescue_weight", 1.0),
                        fg_rescue_dilate_patches=cfg["testing"].get("fg_rescue_dilate_patches", 1),
                        fg_rescue_near_fg_only=cfg["testing"].get("fg_rescue_near_fg_only", False),
                        max_dist_patches=cfg["testing"].get("max_dist_patches", 3),
                        return_debug=True,
                        **fg_kwargs,
                    )
                else:
                    l_pure = l.clone() # 防呆：若沒開 FGS，兩者一致

                # [新增] 統計「原本分數高，但被 FGS 壓低，且屬於 GT 瑕疵區域」的情況
                # 這裡將 GT mask 從 [B, 1, H, W] 縮小到 Patch size [B, h, w] 方便比對
                b_size, _, h_orig, w_orig = mask.shape
                h_patch = w_patch = int(math.sqrt(l.size(1)))
                mask_patch = F.interpolate(mask.float(), size=(h_patch, w_patch), mode="nearest").view(b_size, -1)
                
                for b_idx in range(b_size):
                    if labels[-1 - (b_size - 1 - b_idx)] == 1: # 確保是異常樣本
                        total_anomaly_count += 1
                        
                        # 找出 GT 瑕疵區域
                        gt_defect = (mask_patch[b_idx] > 0)
                        if gt_defect.sum() == 0:
                            continue
                            
                        # 若原始分數高，但 FGS 後分數掉了超過 20% (可根據需求微調這項定義)
                        # [修改] 1. 計算加 Threshold 前 (Pure FGS) 的遮擋情況
                        score_drop_pure = (l_orig[b_idx] - l_pure[b_idx]) / (l_orig[b_idx] + 1e-8)
                        is_suppressed_pure = (gt_defect & (score_drop_pure > 0.2)).any()
                        if is_suppressed_pure:
                            suppressed_pure_count += 1

                        # [修改] 2. 計算加 Threshold 後 (Rescued FGS) 的遮擋情況
                        score_drop_rescued = (l_orig[b_idx] - l[b_idx]) / (l_orig[b_idx] + 1e-8)
                        is_suppressed_rescued = (gt_defect & (score_drop_rescued > 0.2)).any()
                        if is_suppressed_rescued:
                            suppressed_rescued_count += 1

                if cfg["testing"].get("save_fg_mask", True):
                    fg_mask_root = Path(cfg["logging"]["folder"]) / "fg_mask"

                    save_fg_masks_for_paths(
                        image_paths=paths,
                        save_root=fg_mask_root,
                        test_root=cfg["data"]["test_root"],
                        crop_size=crop,
                        suffix="_fg_mask",
                        **fg_kwargs,
                    )
                
                # ------------------------------------------------------------

                topk = torch.topk(l, K, dim=1).values.mean(dim=1)
                patch_scores.extend(topk.cpu())
                h = w = int(math.sqrt(l.size(1)))
                pix = F.interpolate(l.view(-1,1,h,w), size=img.shape[2:], mode="bilinear", align_corners=False)
                # Allocate full-resolution memmap lazily after we know H/W.
                b, _, ph, pw = pix.shape
                if score_mm is None:
                    metric_h, metric_w = ph, pw
                    score_mm = np.memmap(
                        tmp_dir / "score_float32.dat",
                        dtype=np.float32,
                        mode="w+",
                        shape=(num_samples, metric_h, metric_w),
                    )
                    mask_mm = np.memmap(
                        tmp_dir / "mask_uint8.dat",
                        dtype=np.uint8,
                        mode="w+",
                        shape=(num_samples, metric_h, metric_w),
                    )

                score_np = pix.squeeze(1).detach().cpu().numpy().astype(np.float32)
                mask_np_batch = mask.detach().cpu().squeeze(1).numpy().astype(np.uint8)

                score_mm[mm_index:mm_index + b] = score_np
                mask_mm[mm_index:mm_index + b] = mask_np_batch
                mm_index += b
                # pix_buf.append(pix.squeeze(1).cpu().half())
                # mask_buf.append(mask.cpu().to(torch.uint8))
                if cfg["testing"].get("segmentation_vis", False) and seg_vis_fraction > 0:
                    batch_names = batch["image_name"]

                    for bi in range(b):
                        global_i = mm_index - b + bi

                        # Deterministic 1/3 sampling by default.
                        # If fraction = 1/3, save indices 0,3,6,...
                        if seg_vis_fraction >= 1.0:
                            keep_vis = True
                        else:
                            stride = max(1, int(round(1.0 / seg_vis_fraction)))
                            keep_vis = (global_i % stride == 0)

                        if not keep_vis:
                            continue

                        vis_img_buf.append(img[bi:bi+1].detach().cpu())
                        vis_mask_buf.append(mask[bi:bi+1].detach().cpu().to(torch.uint8))
                        vis_pix_buf.append(pix[bi:bi+1].squeeze(1).detach().cpu().half())
                        vis_name_buf.append(batch_names[bi])
                        vis_score_buf.append(float(topk[bi].detach().cpu()))

                    if len(vis_name_buf) >= seg_vis_chunk_size:
                        _flush_segmentation_vis(
                            cfg=cfg,
                            cls=cls,
                            std=std,
                            mean=mean,
                            vis_img_buf=vis_img_buf,
                            vis_mask_buf=vis_mask_buf,
                            vis_pix_buf=vis_pix_buf,
                            vis_name_buf=vis_name_buf,
                            vis_score_buf=vis_score_buf,
                        )

            p_np = torch.tensor(patch_scores).numpy()
            p_np = (p_np - p_np.min()) / (p_np.max() - p_np.min() + 1e-8) # normed

            #print("p_np shape", p_np)

            # 畫 p_np 分布圖
            labels_np = np.array(labels)
            for_hist = p_np * 100
            x = np.arange(len(for_hist))

            normal_mask = (labels_np == 0)
            anomaly_mask = (labels_np == 1)

            # 依 label 指定顏色：1 -> 藍色, 0 -> 橘色
            colors = np.where(labels_np == 1, "blue", "orange")

            # 正常樣本(label=0)的最大分數
            normal_scores = for_hist[normal_mask]
            boundary_val = normal_scores.max()
            num_anomaly_below_boundary = np.sum(anomaly_mask & (for_hist < boundary_val))
            num_anomaly_total = np.sum(anomaly_mask)

            plt.figure(figsize=(10, 5))
            plt.scatter(x, for_hist, c=colors, s=12)
            plt.axhline(boundary_val, linestyle="--", linewidth=2, color="red",
                        label=f"max normal = {boundary_val:.2f}")

            plt.xlabel("Index")
            plt.ylabel("Score")
            plt.title(f"p_np values - {cls}\n"
                      f"anomaly below max-normal: {num_anomaly_below_boundary}/{num_anomaly_total}"
                      )
            plt.legend()
            plt.grid(alpha=0.3)

            dist_dir = Path(cfg["logging"]["folder"]) / "pnp_distribution"
            dist_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(dist_dir / f"{cls}_pnp_distribution.png", dpi=200, bbox_inches="tight")
            plt.close()

            score_mm.flush()
            mask_mm.flush()

            inst = compute_imagewise_retrieval_metrics(p_np, np.array(labels))

            pix = _stream_pixel_metrics_from_memmap(
                score_mm=score_mm,
                mask_mm=mask_mm,
                num_images=mm_index,
                height=metric_h,
                width=metric_w,
            )

            pro = _stream_calculate_pro_from_memmap(
                mask_mm=mask_mm,
                score_mm=score_mm,
                num_images=mm_index,
                height=metric_h,
                width=metric_w,
                max_steps=cfg["testing"]["max_steps"],
                expect_fpr=cfg["testing"]["expect_fpr"],
            )

            # [新增] 在計算 AUROC 前，印出該類別的統計比例
            if total_anomaly_count > 0:
                pure_ratio = suppressed_pure_count / total_anomaly_count
                rescued_ratio = suppressed_rescued_count / total_anomaly_count
                logger.info(
                    f"[{cls}] Pure FGS Suppressed: {suppressed_pure_count}/{total_anomaly_count} ({pure_ratio:.2%}) "
                    f"| Rescued FGS Suppressed: {suppressed_rescued_count}/{total_anomaly_count} ({rescued_ratio:.2%})"
                )

            logger.info("%s | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                        cls, inst["auroc"], inst["aupr"], pix["auroc"], pro)
            csv_logger.log(ckpt.name, cls, inst["auroc"]*100, inst["aupr"]*100, pix["auroc"]*100, pro*100)

            inst_auc.append(inst["auroc"]); inst_aupr.append(inst["aupr"])
            pix_auc.append(pix["auroc"]);   pro_auc.append(pro)

            #print("pix norm", len(pix_norm))

            # Generate visualizations
            if cfg["testing"].get("segmentation_vis", False) and len(vis_name_buf) > 0:
                _flush_segmentation_vis(
                    cfg=cfg,
                    cls=cls,
                    std=std,
                    mean=mean,
                    vis_img_buf=vis_img_buf,
                    vis_mask_buf=vis_mask_buf,
                    vis_pix_buf=vis_pix_buf,
                    vis_name_buf=vis_name_buf,
                    vis_score_buf=vis_score_buf,
                )

        finally:
            # Close memmaps before deleting their backing files.
            for mm_name in ("score_mm", "mask_mm"):
                mm = locals().get(mm_name)
                if mm is None:
                    continue

                try:
                    mm.flush()
                except Exception as exc:
                    logger.warning("Failed to flush %s for %s: %s", mm_name, cls, exc)

                try:
                    mmap_obj = getattr(mm, "_mmap", None)
                    if mmap_obj is not None:
                        mmap_obj.close()
                except Exception as exc:
                    logger.warning("Failed to close %s for %s: %s", mm_name, cls, exc)

            score_mm = None
            mask_mm = None

            # Remove this class's temporary score/mask files.
            try:
                shutil.rmtree(tmp_dir)
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning("Failed to remove temporary directory %s: %s", tmp_dir, exc)
 

    logger.info("Mean | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                np.mean(inst_auc), np.mean(inst_aupr), np.mean(pix_auc), np.mean(pro_auc))
    csv_logger.log(ckpt.name, "Mean", np.mean(inst_auc)*100, np.mean(inst_aupr)*100,
                   np.mean(pix_auc)*100, np.mean(pro_auc)*100)
    

@torch.inference_mode()
def _demo(ckpt: Path, cfg: Dict[str, Any]) -> None:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(cfg["meta"])
    state = torch.load(ckpt, map_location="cpu")
    model.predictor.load_state_dict(state["predictor"])
    if model.projector is not None:
        model.projector.load_state_dict(state["projector"])
    model.to(device)
    model.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)
    out_root = Path(cfg["logging"]["folder"]) / "heatmaps"
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg["data"].get("dataset", "mvtec")
    assert dataset_name in cfg["data"]["test_root"] # check if eval on the same dataset the ckpt trained on
    
    test_root = Path(cfg["data"]["test_root"])
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF", "*.WEBP")
    img_paths: List[Path] = []
    for ext in exts:
        img_paths += list(test_root.rglob(ext))
    img_paths = sorted(set(img_paths))
    if not img_paths:
        raise FileNotFoundError(f"No images found under: {test_root}")
    print(f"[INFO] Found {len(img_paths)} images under {test_root}")
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    def _load_and_preprocess(path: Path):
        pil = Image.open(path).convert("RGB")

        W0, H0 = pil.size

        pil_resized = pil.resize((crop, crop), Image.BILINEAR)

        img = torch.from_numpy(np.array(pil_resized)).float() / 255.0   # [H,W,3], 0~1
        img = img.permute(2, 0, 1).unsqueeze(0).to(device)              # [1,3,H,W]
        img = (img - mean) / std

        return pil, (W0, H0), img
    
    def _to_numpy_image(t_img: torch.Tensor):
        # t_img: [1,3,H,W]
        x = (t_img * std + mean).clamp(0, 1)
        x = x[0].permute(1, 2, 0).detach().cpu().numpy()  # [H,W,3]
        return (x * 255.0).astype(np.uint8)
    
    def _save_overlay_heatmap(rgb_uint8: np.ndarray, heat: np.ndarray, save_path: Path, alpha: float = 0.5):
        """
        rgb_uint8: [H,W,3] 0~255
        heat:      [H,W]   0~1
        """
        H, W = heat.shape

        heat_255 = (heat * 255.0).clip(0, 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_255, cv2.COLORMAP_JET)      # BGR
        rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)            # RGB->BGR
        overlay = cv2.addWeighted(heat_color, alpha, rgb_bgr, 1 - alpha, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        Image.fromarray(overlay_rgb).save(save_path)

    for i, path in enumerate(img_paths, 1):
        pil_orig, (W0, H0), img = _load_and_preprocess(path)

        enc = model.target_features(img, [str(path)], n_layer=n_layer)  # [1, P, D]
        pred = model.predict(enc)                                       # [1, P, D]

        l = F.mse_loss(enc, pred, reduction="none").mean(dim=2)         # [1, P]

        h = w = int(math.sqrt(l.size(1)))

        fg_kwargs = dict(
            border_ratio=cfg["testing"].get("fg_border_ratio", 0.05),
            min_area=cfg["testing"].get("fg_min_area", 0.10),
            max_area=cfg["testing"].get("fg_max_area", 0.95),
            morph_close_kernel=cfg["testing"].get("fg_morph_close_kernel", 70),
            morph_open_kernel=cfg["testing"].get("fg_morph_open_kernel", 1),
            mask_expand_kernel=cfg["testing"].get("fg_mask_expand_kernel", 15),
            blur_sigma=cfg["testing"].get("fg_blur_sigma", 1.0),
            min_component_area_ratio=cfg["testing"].get("fg_min_component_area_ratio", 0.00001),
            lab_l_weight=cfg["testing"].get("fg_lab_l_weight", 0.05),
            lab_ab_weight=cfg["testing"].get("fg_lab_ab_weight", 1.0),
            min_weight=cfg["testing"].get("fg_min_weight", 0.0),
            fill_holes=cfg["testing"].get("fg_fill_holes", True),
            max_hole_area_ratio=cfg["testing"].get("fg_max_hole_area_ratio", 0.12),
            max_hole_total_ratio=cfg["testing"].get("fg_max_hole_total_ratio", 0.25),
        )

        if cfg["testing"].get("use_fgs", True):
            l = apply_fgs_to_patch_score(
                patch_score=l,
                image_paths=[str(path)],
                crop_size=crop,
                device=device,
                fgs_rescue_mode=cfg["testing"].get("fgs_rescue_mode", "score"),
                fg_rescue_score_thresh=cfg["testing"].get("fg_rescue_score_thresh", 0.6),
                fg_rescue_weight=cfg["testing"].get("fg_rescue_weight", 1.0),
                fg_rescue_dilate_patches=cfg["testing"].get("fg_rescue_dilate_patches", 1),
                fg_rescue_near_fg_only=cfg["testing"].get("fg_rescue_near_fg_only", False),
                max_dist_patches=cfg["testing"].get("max_dist_patches", 3),
                **fg_kwargs,
            )

        pix = F.interpolate(l.view(1, 1, h, w), size=img.shape[2:], mode="bilinear", align_corners=False)  # [1,1,H,W]
        pix = pix.squeeze(0).squeeze(0)  # [H,W]

        pmin, pmax = pix.min(), pix.max()
        pix_norm = (pix - pmin) / (pmax - pmin + 1e-8)                  # [H,W], 0~1

        img_uint8 = _to_numpy_image(img)                                 # [H,W,3] @ crop

        rel = path.relative_to(test_root)
        #save_dir = (out_root / rel.parent)
        save_dir = (Path("./outputs") / rel.parent)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{path.stem}_heatmap.png"

        _save_overlay_heatmap(img_uint8, pix_norm.detach().cpu().numpy(), save_path)
        print(f"[{i}/{len(img_paths)}] Saved: {save_path}")


def main(args: Dict[str, Any]) -> None:
    ckpt = Path(args["ckpt_path"])
    print(f"loading {ckpt}...")
    _evaluate_single_ckpt(ckpt, args)
    logger.info("Finished. Metrics appended to CSV.")

if __name__ == "__main__":
    main()
