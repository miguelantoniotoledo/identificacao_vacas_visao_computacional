#!/usr/bin/env python
"""
Treina modelo YOLO pose (keypoints) para detecção de pontos anatômicos em vacas.
Suporta k-fold (group k-fold): treina um modelo por fold e escolhe o melhor por mAP50-95.

Requisitos: pip install ultralytics
"""

import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_full_config
from src.utils.metrics_logger import (
    create_step_logger,
    extract_yolo_metrics_and_plot,
)


def _get_best_map50_95(results_csv: Path) -> float:
    """Lê a última linha do results.csv e retorna metrics/pose_mAP50-95 ou 0."""
    if not results_csv.exists():
        return 0.0
    try:
        with results_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return 0.0
        last = rows[-1]
        key = "metrics/pose_mAP50-95"
        val = last.get(key) or last.get("Pose_mAP50-95")
        return float(val) if val else 0.0
    except Exception:
        return 0.0


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Instale ultralytics: pip install ultralytics")
        sys.exit(1)

    cfg = get_full_config()
    paths = cfg.get("paths", {})
    training = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    pose_cfg = cfg.get("pose", {})

    root = Path(__file__).resolve().parents[1]
    logger = create_step_logger("train_keypoints", root)
    unified = root / paths.get("unified_dir", "data/unified")
    yolo_pose_dir = unified / "yolo_pose"
    keypoints_out = root / paths.get("outputs_dir", "outputs") / "keypoints"

    device = training.get("device", "0")
    epochs = training.get("epochs", 100)
    batch = training.get("batch_size", 16)
    patience = training.get("patience", 20)  # early stopping: para se val não melhorar em N épocas
    workers = training.get("workers", 4)
    logger.log(f"Early stopping: patience={patience} épocas (treino interrompe se val não melhorar)")
    imgsz = data_cfg.get("image_size", 640)
    k_folds = pose_cfg.get("k_folds", 1)

    # Verificar se existe estrutura de folds
    fold_data_yamls = []
    if k_folds > 1:
        for i in range(1, k_folds + 1):
            fy = yolo_pose_dir / f"fold_{i}" / "data.yaml"
            if fy.exists():
                fold_data_yamls.append((i, fy))
        if not fold_data_yamls:
            fold_data_yamls = []

    if fold_data_yamls:
        logger.log("Modo k-fold. Iniciando treino por fold.")
        model = YOLO("yolov8n-pose.pt")
        best_fold = None
        best_mAP = 0.0
        fold_metrics = []
        for fold_i, data_yaml in fold_data_yamls:
            logger.log(f"Treinando fold {fold_i}/{len(fold_data_yamls)}. Resultado: iniciado")
            model.train(
                data=str(data_yaml),
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                device=device,
                patience=patience,
                workers=workers,
                project=str(keypoints_out),
                name=f"fold_{fold_i}",
                exist_ok=True,
            )
            run_dir = keypoints_out / f"fold_{fold_i}"
            results_csv = run_dir / "results.csv"
            mAP = _get_best_map50_95(results_csv)
            fold_metrics.append({"fold": fold_i, "mAP50_95": mAP})
            logger.log(f"Treino fold {fold_i} concluído. mAP50-95: {mAP:.4f}. Resultado: sucesso")
            if mAP > best_mAP:
                best_mAP = mAP
                best_fold = fold_i
        train_weights = keypoints_out / "train" / "weights"
        train_weights.mkdir(parents=True, exist_ok=True)
        if best_fold is not None:
            src_best = keypoints_out / f"fold_{best_fold}" / "weights" / "best.pt"
            if src_best.exists():
                shutil.copy2(src_best, train_weights / "best.pt")
                logger.log(f"Melhor fold: {best_fold} (mAP50-95={best_mAP:.4f}). Copiando best.pt para train/weights. Resultado: sucesso")
        metrics = {"epochs": epochs, "batch_size": batch, "imgsz": imgsz, "device": device, "k_folds": len(fold_data_yamls), "best_fold": best_fold, "best_mAP50_95": best_mAP, "fold_metrics": fold_metrics}
    else:
        data_yaml = yolo_pose_dir / "data.yaml"
        if not data_yaml.exists():
            logger.log("data.yaml não encontrado. Execute unify_and_convert e prepare_dataset. Resultado: falha")
            logger.finalize({"error": "data_yaml_not_found"})
            print("Execute primeiro: python scripts/unify_and_convert.py")
            print("Depois: python scripts/prepare_dataset.py")
            sys.exit(1)
        logger.log("Treino único. Iniciando treino keypoints.")
        run_dir = keypoints_out / "train"
        model = YOLO("yolov8n-pose.pt")
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            patience=patience,
            workers=workers,
            project=str(keypoints_out),
            name="train",
            exist_ok=True,
        )
        logger.log("Treino keypoints concluído. Resultado: sucesso")
        metrics = {"epochs": epochs, "batch_size": batch, "imgsz": imgsz, "device": device}
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            yolo_metrics = extract_yolo_metrics_and_plot(results_csv, "train_keypoints", root)
            if yolo_metrics:
                metrics.update(yolo_metrics)
                for key in ["metrics/pose_P", "metrics/pose_R", "metrics/pose_mAP50", "metrics/pose_mAP50-95"]:
                    if key in yolo_metrics and yolo_metrics[key] is not None:
                        short = key.replace("metrics/", "")
                        metrics[f"precision_recall_mAP_{short}"] = yolo_metrics[key]

    log_path = logger.finalize(metrics)
    print(f"Log e métricas em: {log_path}")
    print("Treino concluído. Modelo salvo em outputs/keypoints/train/")


if __name__ == "__main__":
    main()
