#!/usr/bin/env python
"""
Treina classificador YOLO para identificação de vacas (uma classe por vaca).

Usa a pasta data/unified/classification/<nome_vaca>/*.jpg como dados de treino.

Suporta GPU via config. Use: python scripts/train_classifier.py

Requisitos: pip install ultralytics
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_full_config
from src.utils.metrics_logger import (
    create_step_logger,
    extract_yolo_metrics_and_plot,
)


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Instale ultralytics: pip install ultralytics")
        sys.exit(1)

    cfg = get_full_config()
    paths = cfg.get("paths", {})
    training = cfg.get("training", {})

    root = Path(__file__).resolve().parents[1]
    logger = create_step_logger("train_classifier", root)
    class_dir = root / paths.get("unified_dir", "data/unified") / "classification"
    if not class_dir.exists():
        logger.log("Pasta classification não encontrada. Execute unify_and_convert. Resultado: falha")
        logger.finalize({"error": "classification_dir_not_found"})
        print("Execute primeiro: python scripts/unify_and_convert.py")
        print("Certifique-se de que raw/classificacao contém pastas com fotos por vaca.")
        sys.exit(1)

    cow_folders = [d for d in class_dir.iterdir() if d.is_dir()]
    if not cow_folders:
        logger.log(f"Nenhuma pasta de vaca encontrada em {class_dir}. Resultado: falha")
        logger.finalize({"error": "no_cow_folders"})
        print("Nenhuma pasta de vaca encontrada em", class_dir)
        sys.exit(1)

    logger.log(f"Iniciando treino classificador. Classes: {len(cow_folders)}. Resultado: iniciado")
    device = training.get("device", "0")
    epochs = training.get("epochs", 100)
    batch = training.get("batch_size", 16)
    patience = training.get("patience", 20)  # early stopping
    imgsz = cfg.get("data", {}).get("image_size", 640)
    logger.log(f"Early stopping: patience={patience} épocas (treino interrompe se val não melhorar)")

    run_dir = root / paths.get("outputs_dir", "outputs") / "classifier" / "train"
    model = YOLO("yolov8n-cls.pt")
    model.train(
        data=str(class_dir),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        patience=patience,
        project=str(root / paths.get("outputs_dir", "outputs") / "classifier"),
        name="train",
        exist_ok=True,
    )
    logger.log("Treino classificador concluído. Resultado: sucesso")

    metrics = {"epochs": epochs, "batch_size": batch, "imgsz": imgsz, "device": device, "n_classes": len(cow_folders)}
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        yolo_metrics = extract_yolo_metrics_and_plot(results_csv, "train_classifier", root)
        if yolo_metrics:
            metrics.update(yolo_metrics)
            for key in ["metrics/accuracy_top1", "metrics/accuracy_top5"]:
                if key in yolo_metrics and yolo_metrics[key] is not None:
                    metrics["accuracy_top1" if "top1" in key else "accuracy_top5"] = yolo_metrics[key]
    log_path = logger.finalize(metrics)
    print(f"Log e métricas em: {log_path}")
    print("Treino do classificador concluído. Modelo em outputs/classifier/train/")


if __name__ == "__main__":
    main()
