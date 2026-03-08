#!/usr/bin/env python
"""
Detecta e desenha os keypoints de vacas em imagens não anotadas,
usando o modelo YOLO de pose treinado.

Uso:
  python scripts/predict_keypoints.py --image caminho/para/imagem.jpg
  python scripts/predict_keypoints.py --input-dir caminho/para/pasta
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_full_config, get_keypoint_names  # noqa: E402
from src.utils.metrics_logger import log_script_run  # noqa: E402

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: Path) -> Path:
    """Resolve path; se for relativo e não existir no cwd, tenta em relação à raiz do projeto."""
    if path.is_absolute() or path.exists():
        return path
    root = _project_root()
    candidate = root / path
    if candidate.exists():
        return candidate
    return path


def _collect_images(path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    path = _resolve_path(path)
    if not path.exists():
        print(f"Erro: caminho não encontrado: {path}")
        return []
    if path.is_file() and path.suffix.lower() in exts:
        return [path]
    if path.is_dir():
        return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return []


BOX_LABEL = "cow"


def _draw_result(
    img,
    boxes_xyxy,
    keypoints_xy,
    keypoints_conf: Optional["np.ndarray"],
    kp_names: List[str],
    min_conf: float = 0.25,
) -> None:
    """Desenha na imagem (in-place) as caixas com label e os keypoints com nomes."""
    if not _HAS_CV2:
        return
    if boxes_xyxy is not None:
        xyxy_np = boxes_xyxy.cpu().numpy() if hasattr(boxes_xyxy, "cpu") else np.asarray(boxes_xyxy)
        for xyxy in xyxy_np:
            x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy[:4]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img, BOX_LABEL, (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )
    if keypoints_xy is None:
        return
    kp_xy = keypoints_xy.cpu().numpy() if hasattr(keypoints_xy, "cpu") else np.asarray(keypoints_xy)
    kp_conf = keypoints_conf.cpu().numpy() if keypoints_conf is not None and hasattr(keypoints_conf, "cpu") else None
    if kp_conf is None and keypoints_conf is not None:
        kp_conf = np.asarray(keypoints_conf)
    n_kp = len(kp_names)
    for det_idx in range(kp_xy.shape[0]):
        for kp_idx in range(min(n_kp, kp_xy.shape[1])):
            if kp_conf is not None and kp_conf.shape[0] > det_idx and kp_conf.shape[1] > kp_idx:
                if float(kp_conf[det_idx, kp_idx]) < min_conf:
                    continue
            x, y = kp_xy[det_idx, kp_idx, 0], kp_xy[det_idx, kp_idx, 1]
            xi, yi = int(round(float(x))), int(round(float(y)))
            cv2.circle(img, (xi, yi), 5, (0, 0, 255), -1)
            name = kp_names[kp_idx] if kp_idx < len(kp_names) else str(kp_idx)
            cv2.putText(
                img, name, (xi + 4, yi - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
            )


def _load_model() -> "tuple[object, dict]":
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Instale ultralytics: pip install ultralytics")
        sys.exit(1)

    cfg = get_full_config()
    paths = cfg.get("paths", {})
    root = _project_root()
    model_path = (
        root
        / paths.get("outputs_dir", "outputs")
        / "keypoints"
        / "train"
        / "weights"
        / "best.pt"
    )
    if not model_path.exists():
        print(f"Modelo de keypoints não encontrado em: {model_path}")
        print("Rode antes: python scripts/train_keypoints.py")
        sys.exit(1)

    model = YOLO(str(model_path))
    return model, cfg


def predict(images: Iterable[Path]) -> None:
    images = list(images)
    if not images:
        print("Nenhuma imagem válida encontrada.")
        return

    model, cfg = _load_model()
    device = cfg.get("training", {}).get("device", "0")
    imgsz = cfg.get("data", {}).get("image_size", 640)
    paths_cfg = cfg.get("paths", {})
    root = _project_root()
    outputs_dir = root / paths_cfg.get("outputs_dir", "outputs") / "inference" / "keypoints"
    pred_dir = outputs_dir / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)

    kp_names = get_keypoint_names() or [
        "withers", "back", "hook_up", "hook_down", "hip", "tail_head", "pin_up", "pin_down",
    ]

    n_with_kp = 0
    print(f"Rodando detecção de keypoints em {len(images)} imagem(ns)...")

    use_custom_draw = _HAS_CV2
    results = model.predict(
        source=[str(p) for p in images],
        device=device,
        imgsz=imgsz,
        save=not use_custom_draw,
        project=str(outputs_dir) if not use_custom_draw else None,
        name="pred" if not use_custom_draw else None,
        exist_ok=True,
        verbose=False,
    )

    for img_path, res in zip(images, results):
        print(f"\nImagem: {img_path}")
        keypoints = getattr(res, "keypoints", None)
        if keypoints is not None and hasattr(keypoints, "xy"):
            try:
                kps_xy = keypoints.xy[0]
                n_with_kp += 1
            except Exception:
                pass
            else:
                for idx, (x, y) in enumerate(kps_xy):
                    name = kp_names[idx] if idx < len(kp_names) else str(idx)
                    print(f"  {name}: x={float(x):.1f}, y={float(y):.1f}")

        if not use_custom_draw:
            continue
        img = getattr(res, "orig_img", None)
        if img is None:
            continue
        if hasattr(img, "numpy"):
            img = img.numpy()
        img = np.asarray(img).copy()
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        boxes = getattr(res, "boxes", None)
        boxes_xyxy = boxes.xyxy if boxes is not None else None
        kp_xy = keypoints.xy if keypoints is not None else None
        kp_conf = getattr(keypoints, "conf", None) if keypoints is not None else None
        _draw_result(img, boxes_xyxy, kp_xy, kp_conf, kp_names)

        out_name = img_path.name
        out_path = pred_dir / out_name
        if out_path.exists():
            base, ext = out_path.stem, out_path.suffix
            for i in range(1, 1000):
                out_path = pred_dir / f"{base}_{i}{ext}"
                if not out_path.exists():
                    break
        to_save = img
        if img.ndim == 3 and img.shape[2] == 3:
            to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), to_save)

    metrics = {"n_images": len(images), "n_with_keypoints": n_with_kp}
    log_script_run("predict_keypoints", [f"Keypoints detectados em {n_with_kp}/{len(images)} imagens"], metrics, root)
    if use_custom_draw:
        print(f"\nImagens com keypoints e box (nomes) salvas em: {pred_dir}")
    else:
        print(f"\nImagens salvas em: {pred_dir} (instale opencv-python para desenhar nomes dos keypoints e da box).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detectar keypoints em imagens não anotadas.")
    parser.add_argument("--image", type=str, help="Caminho para uma imagem única.")
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Diretório contendo imagens para detecção de keypoints.",
    )
    args = parser.parse_args()

    imgs: List[Path] = []
    if args.image:
        imgs.extend(_collect_images(Path(args.image)))
    if args.input_dir:
        imgs.extend(_collect_images(Path(args.input_dir)))

    if not imgs:
        print("Nenhuma imagem válida encontrada. Verifique se o arquivo ou diretório existe.")
        parser.print_help()
        sys.exit(1)

    predict(imgs)


if __name__ == "__main__":
    main()

