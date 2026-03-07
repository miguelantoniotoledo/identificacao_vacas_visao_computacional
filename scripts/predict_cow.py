#!/usr/bin/env python
"""
Classifica qual vaca está presente em uma ou mais imagens,
usando o modelo YOLO de classificação treinado.

Uso:
  python scripts/predict_cow.py --image caminho/para/imagem.jpg
  python scripts/predict_cow.py --input-dir caminho/para/pasta
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_full_config  # noqa: E402
from src.utils.metrics_logger import create_step_logger  # noqa: E402


def _collect_images(path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if path.is_file() and path.suffix.lower() in exts:
        return [path]
    if path.is_dir():
        return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return []


def _load_model() -> "tuple[object, dict]":
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Instale ultralytics: pip install ultralytics")
        sys.exit(1)

    cfg = get_full_config()
    paths = cfg.get("paths", {})
    root = Path(__file__).resolve().parents[1]
    model_path = (
        root
        / paths.get("outputs_dir", "outputs")
        / "classifier"
        / "train"
        / "weights"
        / "best.pt"
    )
    if not model_path.exists():
        print(f"Modelo de classificador não encontrado em: {model_path}")
        print("Rode antes: python scripts/train_classifier.py")
        sys.exit(1)

    model = YOLO(str(model_path))
    return model, cfg


def predict(images: Iterable[Path], top_k: int = 1) -> None:
    images = list(images)
    if not images:
        print("Nenhuma imagem válida encontrada.")
        return

    model, cfg = _load_model()
    device = cfg.get("training", {}).get("device", "0")
    imgsz = cfg.get("data", {}).get("image_size", 640)
    paths_cfg = cfg.get("paths", {})
    root = Path(__file__).resolve().parents[1]
    logger = create_step_logger("predict_cow", root)
    outputs_dir = root / paths_cfg.get("outputs_dir", "outputs") / "inference" / "classifier"
    logger.log(f"Classificando {len(images)} imagem(ns). top_k={top_k}. Resultado: iniciado")

    print(f"Rodando classificação em {len(images)} imagem(ns) (top_k={top_k})...")
    results = model.predict(
        source=[str(p) for p in images],
        device=device,
        imgsz=imgsz,
        save=True,
        project=str(outputs_dir),
        name="pred",
        exist_ok=True,
        verbose=False,
    )

    predictions = []
    sum_conf = 0.0
    n_ok = 0
    for img_path, res in zip(images, results):
        if not hasattr(res, "probs") or res.probs is None:
            logger.log(f"Classificando imagem {img_path.name}. Resultado: sem predição")
            print(f"{img_path}: sem predição.")
            continue
        probs = res.probs
        names = res.names or {}
        # Top-k: obter índices ordenados por probabilidade decrescente
        prob_data = getattr(probs, "data", None)
        if prob_data is not None:
            if hasattr(prob_data, "cpu"):
                try:
                    p = prob_data.cpu().numpy()
                except Exception:
                    import numpy as np
                    p = np.asarray(prob_data)
            else:
                import numpy as np
                p = np.asarray(prob_data)
        else:
            p = None
        if p is not None and len(p) >= 1:
            k = min(top_k, len(p))
            top_indices = p.argsort()[::-1][:k]
            top_scores = [float(p[i]) for i in top_indices]
        else:
            top_indices = [int(probs.top1)]
            top_scores = [float(probs.top1conf)]

        top_classes = [names.get(int(i), f"class_{i}") for i in top_indices]
        pred_entry = {
            "image": str(img_path),
            "class": top_classes[0],
            "confidence": top_scores[0],
            "top_k": [{"class": c, "confidence": s} for c, s in zip(top_classes, top_scores)],
        }
        predictions.append(pred_entry)
        sum_conf += top_scores[0]
        n_ok += 1
        if top_k <= 1:
            logger.log(f"Classificando imagem {img_path.name}. Resultado: vaca={top_classes[0]} (confiança={top_scores[0]:.3f})")
            print(f"{img_path}: vaca={top_classes[0]} (confiança={top_scores[0]:.3f})")
        else:
            parts = [f"{c}={s:.3f}" for c, s in zip(top_classes, top_scores)]
            logger.log(f"Classificando imagem {img_path.name}. Resultado: top-{top_k} → {', '.join(parts)}")
            print(f"{img_path}: top-{top_k} → {', '.join(parts)}")

    metrics = {
        "n_images": len(images),
        "n_with_prediction": n_ok,
        "mean_confidence": sum_conf / n_ok if n_ok else 0,
        "top_k": top_k,
    }
    logger.finalize(metrics)
    print(f"Imagens anotadas salvas em: {outputs_dir / 'pred'}")


def main() -> None:
    cfg = get_full_config()
    default_top_k = cfg.get("app", {}).get("top_k", 1)

    parser = argparse.ArgumentParser(description="Classificar vacas em imagens.")
    parser.add_argument("--image", type=str, help="Caminho para uma imagem única.")
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Diretório contendo imagens para classificação.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="K",
        help=f"Mostrar top-K classes por imagem (default: config app.top_k={default_top_k}).",
    )
    args = parser.parse_args()

    top_k = args.top_k if args.top_k is not None else default_top_k
    top_k = max(1, top_k)

    imgs: List[Path] = []
    if args.image:
        imgs.extend(_collect_images(Path(args.image)))
    if args.input_dir:
        imgs.extend(_collect_images(Path(args.input_dir)))

    if not imgs:
        parser.print_help()
        sys.exit(1)

    predict(imgs, top_k=top_k)


if __name__ == "__main__":
    main()

