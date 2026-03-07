#!/usr/bin/env python
"""
Gera imagens com keypoints e retas entre pontos a partir dos labels YOLO pose.

Uso:
  python scripts/visualize_keypoints.py
"""

import sys
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_full_config, get_keypoint_names  # noqa: E402
from src.utils.metrics_logger import create_step_logger  # noqa: E402


DEFAULT_KP_NAMES: List[str] = [
    "withers",
    "back",
    "hook_up",
    "hook_down",
    "hip",
    "tail_head",
    "pin_up",
    "pin_down",
]

DEFAULT_SEGMENTS: List[tuple[str, str]] = [
    ("withers", "back"),
    ("back", "hip"),
    ("hip", "tail_head"),
    ("tail_head", "pin_up"),
    ("pin_up", "pin_down"),
]


def _get_kp_names() -> List[str]:
    names = get_keypoint_names()
    if names:
        return list(names)
    return list(DEFAULT_KP_NAMES)


def _load_yolo_pose_label(label_path: Path, n_kp: int) -> np.ndarray:
    """
    Lê um arquivo de label YOLO pose e retorna array (n_kp, 3) com (x, y, v).
    """
    line = label_path.read_text(encoding="utf-8").strip()
    if not line:
        raise ValueError(f"Label vazio: {label_path}")
    parts = line.split()
    if len(parts) < 5 + n_kp * 3:
        raise ValueError(f"Label incompleto: {label_path}")
    vals = [float(x) for x in parts[5 : 5 + n_kp * 3]]
    kps = np.array(vals, dtype=float).reshape(n_kp, 3)
    return kps


def draw_keypoints_and_segments(
    img_path: Path,
    label_path: Path,
    out_path: Path,
    kp_names: Sequence[str],
    segments: Sequence[tuple[str, str]],
) -> bool:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return False

    h, w = img.shape[:2]
    n_kp = len(kp_names)
    try:
        kps = _load_yolo_pose_label(label_path, n_kp)
    except Exception:
        return False

    name_to_idx = {name: i for i, name in enumerate(kp_names)}

    # Desenhar pontos
    for name, (x_norm, y_norm, v) in zip(kp_names, kps):
        if v <= 0:
            continue
        x = int(round(float(x_norm) * w))
        y = int(round(float(y_norm) * h))
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)  # vermelho
        cv2.putText(
            img,
            name,
            (x + 3, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Desenhar retas
    for a, b in segments:
        if a not in name_to_idx or b not in name_to_idx:
            continue
        ia, ib = name_to_idx[a], name_to_idx[b]
        xa, ya, va = kps[ia]
        xb, yb, vb = kps[ib]
        if va <= 0 or vb <= 0:
            continue
        xa_pix, ya_pix = int(round(float(xa) * w)), int(round(float(ya) * h))
        xb_pix, yb_pix = int(round(float(xb) * w)), int(round(float(yb) * h))
        cv2.line(img, (xa_pix, ya_pix), (xb_pix, yb_pix), (0, 255, 0), 2)  # verde

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def main() -> None:
    cfg = get_full_config()
    paths_cfg = cfg.get("paths", {})
    root = Path(__file__).resolve().parents[1]

    unified = root / paths_cfg.get("unified_dir", "data/unified")
    images_dir = unified / "keypoints" / "images"
    labels_dir = unified / "keypoints" / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"Pastas {images_dir} ou {labels_dir} não encontradas. Rode antes:")
        print("  python scripts/unify_and_convert.py")
        print("  python scripts/prepare_dataset.py")
        return

    logger = create_step_logger("visualize_keypoints", root)
    outputs_dir = root / paths_cfg.get("outputs_dir", "outputs")
    out_vis_dir = outputs_dir / "vis" / "keypoints"
    out_vis_dir.mkdir(parents=True, exist_ok=True)
    logger.log(f"Diretório de saída: {out_vis_dir}")

    kp_names = _get_kp_names()
    segments = list(DEFAULT_SEGMENTS)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    count = 0
    for img in images_dir.iterdir():
        if not img.is_file() or img.suffix.lower() not in exts:
            continue
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        out = out_vis_dir / img.name
        ok = draw_keypoints_and_segments(img, lbl, out, kp_names, segments)
        if ok:
            count += 1
            logger.log(f"Visualizando imagem {img.name}. Resultado: sucesso")
        else:
            logger.log(f"Visualizando imagem {img.name}. Resultado: falha")

    metrics = {"n_images_generated": count, "output_dir": str(out_vis_dir)}
    log_path = logger.finalize(metrics)
    print(f"Log e métricas em: {log_path}")
    print(f"Geradas {count} imagens com keypoints em: {out_vis_dir}")


if __name__ == "__main__":
    main()

