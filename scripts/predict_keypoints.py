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
from typing import Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_full_config  # noqa: E402
from src.utils.metrics_logger import log_script_run  # noqa: E402


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
    root = Path(__file__).resolve().parents[1]
    outputs_dir = root / paths_cfg.get("outputs_dir", "outputs") / "inference" / "keypoints"

    n_with_kp = 0
    print(f"Rodando detecção de keypoints em {len(images)} imagem(ns)...")

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

    # Também imprime coordenadas dos keypoints no console
    for img_path, res in zip(images, results):
        print(f"\nImagem: {img_path}")
        if not hasattr(res, "keypoints") or res.keypoints is None:
            print("  Nenhum keypoint retornado.")
            continue
        try:
            kps_xy = res.keypoints.xy[0]
            n_with_kp += 1
        except Exception:
            print("  Formato de keypoints inesperado.")
            continue
        for idx, (x, y) in enumerate(kps_xy):
            print(f"  kp[{idx}]: x={float(x):.1f}, y={float(y):.1f}")

    metrics = {"n_images": len(images), "n_with_keypoints": n_with_kp}
    log_script_run("predict_keypoints", [f"Keypoints detectados em {n_with_kp}/{len(images)} imagens"], metrics, root)
    print(f"\nImagens anotadas com keypoints salvas em: {outputs_dir / 'pred'}")


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
        parser.print_help()
        sys.exit(1)

    predict(imgs)


if __name__ == "__main__":
    main()

