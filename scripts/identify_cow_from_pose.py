#!/usr/bin/env python
"""
Identifica qual vaca está na imagem usando o modelo de keypoints para localizar o animal
e o modelo de classificação para reconhecer a vaca (classes do dataset de classificação).

Pipeline: imagem → modelo keypoints (bbox da vaca) → recorte da região → modelo classificador → ID da vaca.

Não altera scripts de train, evaluate ou predict existentes. Saída em terminal e opcionalmente
em outputs/inference/identify_cow_from_pose/.

Uso:
  python scripts/identify_cow_from_pose.py --image caminho/para/imagem.jpg
  python scripts/identify_cow_from_pose.py --input-dir caminho/para/pasta
  python scripts/identify_cow_from_pose.py --image foto.jpg --top-k 3 --save-crops
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_full_config  # noqa: E402

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: Path) -> Path:
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
        return []
    if path.is_file() and path.suffix.lower() in exts:
        return [path]
    if path.is_dir():
        return [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    return []


def _draw_prediction_on_image(
    img: "np.ndarray",
    boxes_xyxy,
    all_top_classes: List[List[str]],
    all_top_scores: List[List[float]],
    used_fallback: bool,
) -> "np.ndarray":
    """
    Desenha na imagem as bboxes (se houver) e o texto top_k por detecção.
    img em RGB; retorna cópia em RGB. Requer cv2.
    """
    if not _HAS_CV2 or not _HAS_NUMPY:
        return img
    out = img.copy()
    if out.dtype != np.uint8:
        out = (np.clip(out, 0, 1) * 255).astype(np.uint8) if out.max() <= 1.0 else out.astype(np.uint8)
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(w, h) / 700)
    thickness = max(1, int(font_scale * 2))
    line_margin = 6  # espaço vertical entre linhas

    def _put_text_lines(im, x, y, lines, color=(0, 255, 0), bg_color=(0, 0, 0)):
        y_cur = y
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            line_h = th + line_margin
            pad_x, pad_y = 4, 3
            x1 = x
            y1 = y_cur - th - pad_y
            x2 = x + tw + pad_x * 2
            y2 = y_cur + pad_y
            cv2.rectangle(im, (x1, y1), (x2, y2), bg_color, -1)
            cv2.putText(im, line, (x + pad_x, y_cur), font, font_scale, color, thickness, cv2.LINE_AA)
            y_cur += line_h

    if boxes_xyxy is not None and not used_fallback:
        xyxy_np = boxes_xyxy.cpu().numpy() if hasattr(boxes_xyxy, "cpu") else np.asarray(boxes_xyxy)
        for idx, xyxy in enumerate(xyxy_np):
            x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy[:4]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if idx < len(all_top_classes) and idx < len(all_top_scores):
                classes = all_top_classes[idx]
                scores = all_top_scores[idx]
                lines = [f"{c} ({s:.2f})" for c, s in zip(classes, scores)]
                tx = x1
                ty = max(30, y1 - 4)
                _put_text_lines(out, tx, ty, lines)
    else:
        # Fallback: texto no canto superior esquerdo
        if all_top_classes and all_top_scores:
            classes = all_top_classes[0]
            scores = all_top_scores[0]
            lines = [f"vaca: {c} ({s:.2f})" for c, s in zip(classes, scores)]
            _put_text_lines(out, 8, 8, lines)

    return out


def _crop_to_boxes(
    img: "np.ndarray",
    boxes_xyxy,
    padding: float = 0.1,
) -> List["np.ndarray"]:
    """
    Recorta a imagem para cada bbox. padding: fração (0.1 = 10%) para expandir cada lado.
    Retorna lista de arrays (crops) em RGB.
    """
    if not _HAS_NUMPY or boxes_xyxy is None:
        return []
    h, w = img.shape[:2]
    xyxy_np = boxes_xyxy.cpu().numpy() if hasattr(boxes_xyxy, "cpu") else np.asarray(boxes_xyxy)
    crops = []
    for xyxy in xyxy_np:
        x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
        bw, bh = x2 - x1, y2 - y1
        pad_w = max(0, bw * padding)
        pad_h = max(0, bh * padding)
        x1 = max(0, int(x1 - pad_w))
        y1 = max(0, int(y1 - pad_h))
        x2 = min(w, int(x2 + pad_w))
        y2 = min(h, int(y2 + pad_h))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        crops.append(crop)
    return crops


def _load_pose_model(cfg: dict, root: Path, weights_pose: Optional[Path]) -> object:
    from ultralytics import YOLO

    if weights_pose is None:
        weights_pose = (
            root
            / cfg.get("paths", {}).get("outputs_dir", "outputs")
            / "keypoints"
            / "train"
            / "weights"
            / "best.pt"
        )
    else:
        weights_pose = _resolve_path(weights_pose)
    if not weights_pose.exists():
        outputs = weights_pose.parents[2]
        for i in range(1, 10):
            candidate = outputs / f"fold_{i}" / "weights" / "best.pt"
            if candidate.exists():
                weights_pose = candidate
                break
        else:
            raise FileNotFoundError(
                f"Modelo de keypoints não encontrado. Rode train_keypoints.py. Procurado: {weights_pose}"
            )
    return YOLO(str(weights_pose))


def _load_classifier_model(cfg: dict, root: Path, weights_cls: Optional[Path]) -> object:
    from ultralytics import YOLO

    if weights_cls is None:
        weights_cls = (
            root
            / cfg.get("paths", {}).get("outputs_dir", "outputs")
            / "classifier"
            / "train"
            / "weights"
            / "best.pt"
        )
    else:
        weights_cls = _resolve_path(weights_cls)
    if not weights_cls.exists():
        raise FileNotFoundError(
            f"Modelo de classificador não encontrado. Rode train_classifier.py. Procurado: {weights_cls}"
        )
    return YOLO(str(weights_cls))


def _run_identify(
    images: Iterable[Path],
    pose_model: object,
    cls_model: object,
    cfg: dict,
    top_k: int = 5,
    padding: float = 0.1,
    fallback_full_image: bool = True,
    save_crops: bool = False,
    out_dir: Optional[Path] = None,
    save_pred: bool = True,
    pred_dir: Optional[Path] = None,
) -> None:
    device = cfg.get("training", {}).get("device", "0")
    imgsz_pose = cfg.get("data", {}).get("image_size", 640)
    imgsz_cls = cfg.get("training", {}).get("classifier_imgsz", 224)
    root = _project_root()

    images = list(images)
    if not images:
        print("Nenhuma imagem fornecida.")
        return

    if save_crops and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    if save_pred and pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"Identificando vaca em {len(images)} imagem(ns) (pose → recorte → classificador, top_k={top_k})...")
    print()

    # 1) Pose em lote
    pose_results = pose_model.predict(
        source=[str(p) for p in images],
        device=device,
        imgsz=imgsz_pose,
        verbose=False,
    )

    for img_path, res in zip(images, pose_results):
        orig_img = getattr(res, "orig_img", None)
        if orig_img is None:
            print(f"{img_path}: sem imagem no resultado do pose.")
            continue
        if hasattr(orig_img, "numpy"):
            orig_img = orig_img.numpy()
        img = np.asarray(orig_img).copy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        boxes = getattr(res, "boxes", None)
        boxes_xyxy = boxes.xyxy if (boxes is not None and hasattr(boxes, "xyxy")) else None

        # 2) Recortes a partir das bboxes
        crops = _crop_to_boxes(img, boxes_xyxy, padding=padding)

        if not crops and fallback_full_image:
            crops = [img]
            used_fallback = True
        else:
            used_fallback = False

        if not crops:
            print(f"{img_path}: nenhum animal detectado pelo pose (use --fallback-full para classificar a imagem inteira).")
            continue

        # 3) Classificar cada recorte (o classificador aceita lista de arrays)
        cls_results = cls_model.predict(
            source=crops,
            device=device,
            imgsz=imgsz_cls,
            verbose=False,
        )

        names = getattr(cls_model, "names", {}) or {}
        all_top_classes: List[List[str]] = []
        all_top_scores: List[List[float]] = []

        for idx, (crop, cres) in enumerate(zip(crops, cls_results)):
            if not hasattr(cres, "probs") or cres.probs is None:
                all_top_classes.append([])
                all_top_scores.append([])
                continue
            probs = cres.probs
            prob_data = getattr(probs, "data", None)
            if prob_data is not None:
                if hasattr(prob_data, "cpu"):
                    p = prob_data.cpu().numpy()
                else:
                    p = np.asarray(prob_data)
            else:
                p = None
            if p is not None and len(p) >= 1:
                k = min(top_k, len(p))
                top_indices = p.argsort()[::-1][:k]
                top_scores = [float(p[i]) for i in top_indices]
            else:
                top_indices = [int(getattr(probs, "top1", 0))]
                top_scores = [float(getattr(probs, "top1conf", 0.0))]

            top_classes = [names.get(int(i), f"class_{i}") for i in top_indices]
            all_top_classes.append(top_classes)
            all_top_scores.append(top_scores)

            label = " (imagem inteira)" if (used_fallback and len(crops) == 1) else f" (recorte {idx + 1})"
            print(f"{img_path}{label}")
            print(f"  → vaca: {top_classes[0]} (confiança={top_scores[0]:.3f})")
            if top_k > 1:
                for c, s in zip(top_classes[1:], top_scores[1:]):
                    print(f"     top: {c} ({s:.3f})")
            print()

            if save_crops and out_dir is not None:
                stem = img_path.stem
                suffix = img_path.suffix
                crop_name = f"{stem}_crop{idx}{suffix}" if len(crops) > 1 else f"{stem}{suffix}"
                try:
                    if _HAS_CV2:
                        to_save = crop
                        if to_save.ndim == 3 and to_save.shape[2] == 3:
                            to_save = cv2.cvtColor(to_save, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(out_dir / crop_name), to_save)
                except Exception:
                    pass

        # 4) Salvar imagem com predição top_k desenhada
        if save_pred and pred_dir is not None and all_top_classes and _HAS_CV2:
            vis = _draw_prediction_on_image(img, boxes_xyxy, all_top_classes, all_top_scores, used_fallback)
            out_path = pred_dir / img_path.name
            if out_path.exists():
                base, ext = out_path.stem, out_path.suffix
                for i in range(1, 1000):
                    out_path = pred_dir / f"{base}_{i}{ext}"
                    if not out_path.exists():
                        break
            to_save = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), to_save)

    if save_crops and out_dir is not None:
        print(f"Recortes salvos em: {out_dir}")
    if save_pred and pred_dir is not None:
        print(f"Imagens com predição top_k salvas em: {pred_dir}")


def main() -> None:
    if not _HAS_NUMPY:
        print("Instale numpy: pip install numpy")
        sys.exit(1)

    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        print("Instale ultralytics: pip install ultralytics")
        sys.exit(1)

    cfg = get_full_config()
    root = _project_root()
    paths_cfg = cfg.get("paths", {})
    default_top_k = cfg.get("app", {}).get("top_k", 5)

    parser = argparse.ArgumentParser(
        description="Identificar vaca na imagem: keypoints (localizar) → classificador (ID da vaca)."
    )
    parser.add_argument("--image", type=str, help="Caminho para uma imagem.")
    parser.add_argument("--input-dir", type=str, help="Diretório com imagens.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=default_top_k,
        metavar="K",
        help=f"Mostrar top-K classes por recorte (default: {default_top_k}).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.1,
        metavar="P",
        help="Margem em torno da bbox do pose, como fração (default: 0.1 = 10%%).",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Não classificar a imagem inteira quando o pose não detectar nenhum animal.",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Salvar recortes em outputs/inference/identify_cow_from_pose/crops.",
    )
    parser.add_argument(
        "--no-save-pred",
        action="store_true",
        help="Não salvar imagem com predição top_k desenhada (por padrão salva em outputs/.../identify_cow_from_pose/pred).",
    )
    parser.add_argument(
        "--keypoints-weights",
        type=str,
        default=None,
        help="Caminho do modelo de keypoints (default: outputs/keypoints/train/weights/best.pt).",
    )
    parser.add_argument(
        "--classifier-weights",
        type=str,
        default=None,
        help="Caminho do modelo de classificador (default: outputs/classifier/train/weights/best.pt).",
    )
    args = parser.parse_args()

    imgs: List[Path] = []
    if args.image:
        imgs.extend(_collect_images(Path(args.image)))
    if args.input_dir:
        imgs.extend(_collect_images(Path(args.input_dir)))
    if not imgs:
        print("Nenhuma imagem encontrada. Use --image ou --input-dir.")
        parser.print_help()
        sys.exit(1)

    weights_pose = Path(args.keypoints_weights) if args.keypoints_weights else None
    weights_cls = Path(args.classifier_weights) if args.classifier_weights else None

    try:
        pose_model = _load_pose_model(cfg, root, weights_pose)
        cls_model = _load_classifier_model(cfg, root, weights_cls)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    base_out = root / paths_cfg.get("outputs_dir", "outputs") / "inference" / "identify_cow_from_pose"
    out_dir = base_out / "crops" if args.save_crops else None
    pred_dir = base_out / "pred" if not args.no_save_pred else None

    _run_identify(
        imgs,
        pose_model,
        cls_model,
        cfg,
        top_k=args.top_k,
        padding=args.padding,
        fallback_full_image=not args.no_fallback,
        save_crops=args.save_crops,
        out_dir=out_dir,
        save_pred=not args.no_save_pred,
        pred_dir=pred_dir,
    )


if __name__ == "__main__":
    main()
