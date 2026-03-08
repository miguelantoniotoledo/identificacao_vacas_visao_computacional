#!/usr/bin/env python
"""
Avalia o modelo classificador (validador): compara predições com rótulos e calcula acurácia.
Se existir data/unified/classification_split/ (split 80-10-10), use --split val ou --split test.
Métricas: top-1 accuracy, top-5 accuracy.

Uso:
  python scripts/evaluate_classifier.py
  python scripts/evaluate_classifier.py --split val
  python scripts/evaluate_classifier.py --split test
  python scripts/evaluate_classifier.py --weights outputs/classifier/train/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_full_config
from src.utils.metrics_logger import create_step_logger


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Instale ultralytics: pip install ultralytics")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Validar classificador (top-1 e top-5 accuracy no conjunto val ou test)."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Caminho para o modelo (ex.: outputs/classifier/train/weights/best.pt).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("val", "test"),
        default="val",
        help="Pasta a usar: val (validação, 10%% dos dados) ou test (teste hold-out, 10%%). Padrão: val.",
    )
    args = parser.parse_args()

    cfg = get_full_config()
    paths = cfg.get("paths", {})
    training = cfg.get("training", {})
    root = Path(__file__).resolve().parents[1]
    unified = root / paths.get("unified_dir", "data/unified")
    class_dir = unified / "classification"
    classification_split = unified / "classification_split"

    # Definir pasta de avaliação: classification_split/val ou .../test, ou classification/ (split interno)
    if classification_split.exists() and (classification_split / args.split).exists():
        data_dir = classification_split / args.split
    elif class_dir.exists():
        data_dir = class_dir
    else:
        print("Pasta de classificação não encontrada:", class_dir)
        print("Execute antes: python scripts/unify_and_convert.py e python scripts/prepare_dataset.py")
        sys.exit(1)

    weights = args.weights
    if not weights:
        weights = (
            root
            / paths.get("outputs_dir", "outputs")
            / "classifier"
            / "train"
            / "weights"
            / "best.pt"
        )
    else:
        weights = Path(weights)
        if not weights.is_absolute():
            weights = root / weights

    if not weights.exists():
        print(f"Modelo não encontrado: {weights}")
        print("Rode antes: python scripts/train_classifier.py")
        sys.exit(1)

    imgsz = training.get("classifier_imgsz", 224)
    device = training.get("device", "0")

    logger = create_step_logger("evaluate_classifier", root)
    logger.log(f"Modelo: {weights}")
    logger.log(f"Data: {data_dir} (split={args.split})")
    logger.log(f"imgsz: {imgsz}")

    print("Validando classificador (validador: compara predição com rótulo e calcula acurácia).")
    print(f"  Modelo: {weights}")
    print(f"  Pasta:  {data_dir} (split={args.split})")
    print()

    model = YOLO(str(weights))
    results = model.val(data=str(data_dir), imgsz=imgsz, device=device)

    metrics = {}
    if results is not None:
        if hasattr(results, "results_dict") and results.results_dict:
            d = results.results_dict
            for k, v in d.items():
                if v is not None and isinstance(v, (int, float)):
                    metrics[k] = float(v) if isinstance(v, (int, float)) else v
        if hasattr(results, "metrics") and results.metrics:
            m = results.metrics
            if isinstance(m, dict):
                metrics.update(m)
            elif hasattr(m, "__dict__"):
                metrics.update({k: v for k, v in m.__dict__.items() if v is not None and isinstance(v, (int, float))})
        # Normalizar nomes (Ultralytics pode usar top1_acc ou accuracy_top1)
        top1 = metrics.get("top1_acc") or metrics.get("accuracy_top1") or metrics.get("metrics/accuracy_top1")
        top5 = metrics.get("top5_acc") or metrics.get("accuracy_top5") or metrics.get("metrics/accuracy_top5")
        if top1 is not None:
            metrics["top1_acc"] = float(top1)
        if top5 is not None:
            metrics["top5_acc"] = float(top5)
        print(f"Métricas no conjunto '{args.split}':")
        if "top1_acc" in metrics:
            print(f"  top1_acc (acurácia): {metrics['top1_acc']:.4f}")
        if "top5_acc" in metrics:
            print(f"  top5_acc:            {metrics['top5_acc']:.4f}")
        for k, v in sorted(metrics.items()):
            if k not in ("top1_acc", "top5_acc") and isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    logger.log(f"top1_acc: {metrics.get('top1_acc', 'N/A')}")
    logger.log(f"top5_acc: {metrics.get('top5_acc', 'N/A')}")
    log_path = logger.finalize(metrics)
    print()
    print(f"Log em: {log_path}")
    print("Concluído. Use top1_acc como acurácia principal. val = validação durante treino; test = hold-out final.")


if __name__ == "__main__":
    main()
