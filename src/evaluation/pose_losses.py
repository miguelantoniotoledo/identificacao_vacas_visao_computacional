"""
Funções de loss para avaliação do modelo de pose (keypoints).
Usadas para avaliar predições vs. ground truth em termos de:
- Cross entropy (visibilidade/confiança)
- IoU loss (caixas delimitadoras)
- Focal loss (confiança/desbalanceamento)
- MSE e L1 (coordenadas dos keypoints)
- Heatmap loss (mapas de calor Gaussianos)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional


def _clip_eps(x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    return np.clip(x, eps, 1.0 - eps)


def cross_entropy_loss(
    pred_conf: np.ndarray,
    gt_visible: np.ndarray,
) -> float:
    """
    Cross entropy binária: pred_conf como probabilidade de keypoint visível,
    gt_visible em {0, 1} (0 = não visível, 1 = visível).
    pred_conf e gt_visible: shape (N,) ou (n_samples, n_kp) → média sobre elementos.
    """
    p = _clip_eps(np.asarray(pred_conf, dtype=np.float64).ravel())
    y = np.asarray(gt_visible, dtype=np.float64).ravel()
    y = (y >= 1).astype(np.float64)  # 2 (visible) e 1 (occluded) → 1
    ce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.mean(ce))


def cross_entropy_loss_multiclass(
    pred_logits: np.ndarray,
    gt_class: np.ndarray,
    num_classes: int = 3,
) -> float:
    """
    Cross entropy multiclasse (ex.: visibilidade 0/1/2).
    pred_logits: (N, num_classes); gt_class: (N,) inteiros em [0, num_classes-1].
    """
    pred_logits = np.asarray(pred_logits, dtype=np.float64)
    gt_class = np.asarray(gt_class, dtype=np.int64).ravel()
    if pred_logits.ndim == 1:
        pred_logits = pred_logits[:, None]
    if pred_logits.shape[1] < num_classes:
        pred_logits = np.pad(pred_logits, ((0, 0), (0, num_classes - pred_logits.shape[1])))
    log_probs = pred_logits - np.log(np.sum(np.exp(pred_logits), axis=1, keepdims=True) + 1e-9)
    n = len(gt_class)
    ce = -log_probs[np.arange(n), np.clip(gt_class, 0, num_classes - 1)]
    return float(np.mean(ce))


def iou_box_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU entre duas caixas no formato [x1, y1, x2, y2]. Cada box shape (4,)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / (union + 1e-9)


def box_cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """Converte (center_x, center_y, width, height) para (x1, y1, x2, y2)."""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return (x1, y1, x2, y2)


def load_yolo_pose_label(
    label_path: str,
    img_w: int,
    img_h: int,
    n_keypoints: int = 8,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Carrega um arquivo de label YOLO pose (uma linha por objeto).
    Retorna lista de (box_xyxy, kp_xy, vis) por objeto.
    box_xyxy: (4,) em pixels; kp_xy: (n_keypoints, 2) em pixels; vis: (n_keypoints,) em {0,1,2}.
    """
    from pathlib import Path
    path = Path(label_path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    n_vals = 5 + n_keypoints * 3
    out = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < n_vals:
            continue
        vals = [float(x) for x in parts[:n_vals]]
        cx_n, cy_n, w_n, h_n = vals[1], vals[2], vals[3], vals[4]
        cx = cx_n * img_w
        cy = cy_n * img_h
        w = w_n * img_w
        h = h_n * img_h
        x1, y1, x2, y2 = box_cxcywh_to_xyxy(cx, cy, w, h)
        box_xyxy = np.array([x1, y1, x2, y2], dtype=np.float64)
        kp_xy = np.zeros((n_keypoints, 2), dtype=np.float64)
        vis = np.zeros(n_keypoints, dtype=np.float64)
        for j in range(n_keypoints):
            kp_xy[j, 0] = vals[5 + j * 3] * img_w
            kp_xy[j, 1] = vals[5 + j * 3 + 1] * img_h
            vis[j] = int(vals[5 + j * 3 + 2])
        out.append((box_xyxy, kp_xy, vis))
    return out


def iou_loss(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    format: str = "xyxy",
) -> float:
    """
    IoU loss = 1 - IoU (média sobre pares).
    pred_boxes, gt_boxes: (N, 4). format 'xyxy' ou 'cxcywh'.
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    if pred_boxes.ndim == 1:
        pred_boxes = pred_boxes[None, :]
    if gt_boxes.ndim == 1:
        gt_boxes = gt_boxes[None, :]
    n = min(len(pred_boxes), len(gt_boxes))
    if n == 0:
        return float("nan")
    ious = []
    for i in range(n):
        pb = pred_boxes[i]
        gb = gt_boxes[i]
        if format == "cxcywh":
            pb = np.array(box_cxcywh_to_xyxy(pb[0], pb[1], pb[2], pb[3]))
            gb = np.array(box_cxcywh_to_xyxy(gb[0], gb[1], gb[2], gb[3]))
        ious.append(iou_box_xyxy(pb, gb))
    return float(1.0 - np.mean(ious))


def focal_loss(
    pred_conf: np.ndarray,
    gt_visible: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> float:
    """
    Focal loss para confiança vs. visibilidade (binário).
    FL(p) = -alpha * (1-p)^gamma * log(p) para positivo, -(1-alpha) * p^gamma * log(1-p) para negativo.
    """
    p = _clip_eps(np.asarray(pred_conf, dtype=np.float64).ravel())
    y = np.asarray(gt_visible, dtype=np.float64).ravel()
    y = (y >= 1).astype(np.float64)
    pt = np.where(y == 1, p, 1 - p)
    alpha_t = np.where(y == 1, alpha, 1 - alpha)
    fl = -alpha_t * (1 - pt) ** gamma * np.log(pt + 1e-9)
    return float(np.mean(fl))


def mse_loss(
    pred_kp: np.ndarray,
    gt_kp: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    MSE (L2) nas coordenadas dos keypoints.
    pred_kp, gt_kp: (N, 2) ou (n_samples, n_kp, 2). mask: opcional, 1 onde válido.
    """
    pred_kp = np.asarray(pred_kp, dtype=np.float64)
    gt_kp = np.asarray(gt_kp, dtype=np.float64)
    diff = pred_kp - gt_kp
    sq = (diff ** 2).reshape(-1, 2).sum(axis=1)
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64).ravel()
        if mask.size != sq.size:
            mask = np.broadcast_to(mask, sq.shape)
        sq = sq * mask
        count = max(np.sum(mask), 1)
        return float(np.sum(sq) / count)
    return float(np.mean(sq))


def l1_loss(
    pred_kp: np.ndarray,
    gt_kp: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    L1 loss nas coordenadas dos keypoints.
    pred_kp, gt_kp: (N, 2) ou (n_samples, n_kp, 2). mask opcional.
    """
    pred_kp = np.asarray(pred_kp, dtype=np.float64)
    gt_kp = np.asarray(gt_kp, dtype=np.float64)
    diff = np.abs(pred_kp - gt_kp)
    l1 = diff.reshape(-1, 2).sum(axis=1)
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64).ravel()
        if mask.size != l1.size:
            mask = np.broadcast_to(mask, l1.shape)
        l1 = l1 * mask
        count = max(np.sum(mask), 1)
        return float(np.sum(l1) / count)
    return float(np.mean(l1))


def gaussian_heatmap(
    shape: Tuple[int, int],
    x: float,
    y: float,
    sigma: float = 2.0,
) -> np.ndarray:
    """Gera um heatmap 2D com um único pico Gaussiano em (x, y). Coordenadas em pixels."""
    h, w = shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))


def keypoints_to_heatmap(
    shape: Tuple[int, int],
    keypoints_xy: np.ndarray,
    vis: Optional[np.ndarray] = None,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Soma de Gaussianas centradas em cada keypoint.
    keypoints_xy: (n_kp, 2) em pixels. vis: (n_kp,) 0/1/2; se None, todos visíveis.
    """
    h, w = shape
    out = np.zeros((h, w), dtype=np.float64)
    n_kp = len(keypoints_xy)
    for i in range(n_kp):
        if vis is not None and vis[i] == 0:
            continue
        x, y = keypoints_xy[i, 0], keypoints_xy[i, 1]
        if not (0 <= x < w and 0 <= y < h):
            continue
        out += gaussian_heatmap((h, w), x, y, sigma)
    if out.max() > 0:
        out /= out.max()
    return out


def heatmap_loss(
    pred_kp_xy: np.ndarray,
    gt_kp_xy: np.ndarray,
    heatmap_shape: Tuple[int, int] = (64, 64),
    img_shape: Tuple[int, int] = (640, 640),
    gt_vis: Optional[np.ndarray] = None,
    pred_conf: Optional[np.ndarray] = None,
    sigma: float = 2.0,
) -> float:
    """
    Gera heatmaps a partir de pred e GT (escalando para heatmap_shape), depois MSE entre os dois.
    pred_kp_xy, gt_kp_xy: (n_kp, 2) em coordenadas de imagem (img_shape).
    Escala para heatmap_shape para gerar os mapas.
    """
    pred_kp_xy = np.asarray(pred_kp_xy, dtype=np.float64)
    gt_kp_xy = np.asarray(gt_kp_xy, dtype=np.float64)
    hh, hw = heatmap_shape
    ih, iw = img_shape
    scale_x = (hw - 1) / max(iw - 1, 1)
    scale_y = (hh - 1) / max(ih - 1, 1)

    def to_heatmap_coords(xy: np.ndarray) -> np.ndarray:
        out = xy.copy()
        out[:, 0] *= scale_x
        out[:, 1] *= scale_y
        return out

    pred_hm = keypoints_to_heatmap(
        heatmap_shape,
        to_heatmap_coords(pred_kp_xy),
        vis=pred_conf if pred_conf is not None else None,
        sigma=sigma,
    )
    gt_hm = keypoints_to_heatmap(
        heatmap_shape,
        to_heatmap_coords(gt_kp_xy),
        vis=gt_vis,
        sigma=sigma,
    )
    return float(np.mean((pred_hm - gt_hm) ** 2))


def keypoint_mean_distance_px(
    pred_kp: np.ndarray,
    gt_kp: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Distância média em pixels (Euclidiana) entre pred e GT por keypoint.
    pred_kp, gt_kp: (N, 2) ou (n_samples, n_kp, 2). mask: 1 onde keypoint visível.
    """
    pred_kp = np.asarray(pred_kp, dtype=np.float64)
    gt_kp = np.asarray(gt_kp, dtype=np.float64)
    diff = pred_kp - gt_kp
    dist = np.sqrt((diff ** 2).reshape(-1, 2).sum(axis=1))
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64).ravel()
        if mask.size != dist.size:
            mask = np.broadcast_to(mask, dist.shape)
        dist = dist * mask
        count = max(np.sum(mask), 1)
        return float(np.sum(dist) / count)
    return float(np.mean(dist))


def keypoint_pck(
    pred_kp: np.ndarray,
    gt_kp: np.ndarray,
    threshold_px: float,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    PCK (Percentage of Correct Keypoints): fração de keypoints com distância <= threshold_px.
    Retorna valor entre 0 e 1 (ex.: 0.6 = 60% dos pontos dentro do limiar).
    pred_kp, gt_kp: (N, 2) ou (n_samples, n_kp, 2). mask: 1 onde keypoint visível.
    """
    pred_kp = np.asarray(pred_kp, dtype=np.float64)
    gt_kp = np.asarray(gt_kp, dtype=np.float64)
    diff = pred_kp - gt_kp
    dist = np.sqrt((diff ** 2).reshape(-1, 2).sum(axis=1))
    correct = (dist <= threshold_px).astype(np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64).ravel()
        if mask.size != correct.size:
            mask = np.broadcast_to(mask, correct.shape)
        correct = correct * mask
        count = max(np.sum(mask), 1)
        return float(np.sum(correct) / count)
    return float(np.mean(correct))


def compute_all_pose_losses(
    pred_boxes: np.ndarray,
    pred_kp: np.ndarray,
    pred_conf: np.ndarray,
    gt_boxes: np.ndarray,
    gt_kp: np.ndarray,
    gt_vis: np.ndarray,
    box_format: str = "xyxy",
    img_shape: Tuple[int, int] = (640, 640),
    heatmap_size: Tuple[int, int] = (64, 64),
) -> dict:
    """
    Calcula todas as losses para um lote (ou amostra agregada).
    pred_boxes, gt_boxes: (N, 4). pred_kp, gt_kp: (N, n_kp, 2) em pixels.
    pred_conf, gt_vis: (N, n_kp). gt_vis em {0,1,2}.
    Retorna dicionário com as chaves: iou_loss, mse_loss, l1_loss, cross_entropy, focal_loss, heatmap_loss.
    """
    pred_kp = np.asarray(pred_kp, dtype=np.float64)
    gt_kp = np.asarray(gt_kp, dtype=np.float64)
    pred_conf = np.asarray(pred_conf, dtype=np.float64)
    gt_vis = np.asarray(gt_vis, dtype=np.float64)
    mask = (gt_vis >= 1).astype(np.float64)  # visível ou oculto = considerar na coord

    n = pred_kp.shape[0]
    if n == 0:
        return {
            "iou_loss": float("nan"),
            "mse_loss": float("nan"),
            "l1_loss": float("nan"),
            "cross_entropy": float("nan"),
            "focal_loss": float("nan"),
            "heatmap_loss": float("nan"),
            "mean_distance_px": float("nan"),
            "pck_20px": float("nan"),
            "pck_30px": float("nan"),
        }

    iou = iou_loss(pred_boxes, gt_boxes, format=box_format)
    pred_flat = pred_kp.reshape(-1, 2)
    gt_flat = gt_kp.reshape(-1, 2)
    mask_flat = mask.reshape(-1)

    mse = mse_loss(pred_flat, gt_flat, mask=mask_flat)
    l1 = l1_loss(pred_flat, gt_flat, mask=mask_flat)
    ce = cross_entropy_loss(pred_conf.ravel(), gt_vis.ravel())
    fl = focal_loss(pred_conf.ravel(), gt_vis.ravel())

    hm_losses = []
    for i in range(n):
        hl = heatmap_loss(
            pred_kp[i],
            gt_kp[i],
            heatmap_shape=heatmap_size,
            img_shape=img_shape,
            gt_vis=gt_vis[i],
            pred_conf=pred_conf[i],
            sigma=2.0,
        )
        hm_losses.append(hl)
    heatmap = float(np.mean(hm_losses))

    mean_dist_px = keypoint_mean_distance_px(pred_flat, gt_flat, mask=mask_flat)
    pck_20 = keypoint_pck(pred_flat, gt_flat, 20.0, mask=mask_flat)
    pck_30 = keypoint_pck(pred_flat, gt_flat, 30.0, mask=mask_flat)

    return {
        "iou_loss": iou,
        "mse_loss": mse,
        "l1_loss": l1,
        "cross_entropy": ce,
        "focal_loss": fl,
        "heatmap_loss": heatmap,
        "mean_distance_px": mean_dist_px,
        "pck_20px": pck_20,
        "pck_30px": pck_30,
    }
