#!/usr/bin/env python
"""
Avalia o modelo de keypoints no conjunto de TESTE (data/unified/yolo_pose/test).
Use após o treino para obter métricas no hold-out de teste (nunca visto no treino).

Uso:
  python scripts/evaluate_keypoints.py
  python scripts/evaluate_keypoints.py --weights outputs/keypoints/train/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_full_config


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Instale ultralytics: pip install ultralytics")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Avaliar modelo de keypoints no conjunto de TESTE.")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Caminho para o modelo (ex.: outputs/keypoints/train/weights/best.pt). Padrão: config + outputs/keypoints/train/weights/best.pt",
    )
    args = parser.parse_args()

    cfg = get_full_config()
    paths = cfg.get("paths", {})
    root = Path(__file__).resolve().parents[1]
    unified = root / paths.get("unified_dir", "data/unified")
    yolo_pose_dir = unified / "yolo_pose"
    data_yaml = yolo_pose_dir / "data.yaml"

    if not data_yaml.exists():
        print("data.yaml não encontrado. Rode antes: python scripts/prepare_dataset.py")
        sys.exit(1)

    weights = args.weights
    if not weights:
        weights = root / paths.get("outputs_dir", "outputs") / "keypoints" / "train" / "weights" / "best.pt"
    else:
        weights = Path(weights)
        if not weights.is_absolute():
            weights = root / weights

    if not weights.exists():
        print(f"Modelo não encontrado: {weights}")
        print("Rode antes: python scripts/train_keypoints.py")
        sys.exit(1)

    test_images = yolo_pose_dir / "test" / "images"
    if not test_images.exists() or not list(test_images.glob("*.*")):
        print("Pasta de teste vazia ou inexistente:", test_images)
        print("Confira que prepare_dataset foi rodado com split 80/10/10 (train/val/test).")
        sys.exit(1)

    print("Avaliando no conjunto de TESTE (hold-out, não usado no treino).")
    print(f"  Modelo: {weights}")
    print(f"  Data:   {data_yaml}")
    print(f"  Split:  test")
    print()

    model = YOLO(str(weights))
    results = model.val(data=str(data_yaml), split="test")

    if results is not None:
        if hasattr(results, "results_dict") and results.results_dict:
            d = results.results_dict
            print("Métricas no TESTE:")
            for k, v in sorted(d.items()):
                if v is not None and isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        elif hasattr(results, "metrics"):
            print("Métricas no TESTE:", results.metrics)
    print()
    print("Concluído. Use essas métricas como desempenho final do modelo no conjunto de teste.")


if __name__ == "__main__":
    main()
