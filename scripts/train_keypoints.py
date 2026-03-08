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

import numpy as np

from src.config import get_full_config
from src.utils.metrics_logger import (
    create_step_logger,
    extract_yolo_metrics_and_plot,
    get_statistics_dir,
)


POSE_WEIGHTS = "yolov8n-pose.pt"


def _ensure_model_override(model, weights: str = POSE_WEIGHTS) -> None:
    """Garante que model.overrides tenha 'model' (exigido pelo Ultralytics após o 1º fold)."""
    if "model" not in getattr(model, "overrides", {}):
        model.overrides["model"] = weights


def _get_best_map50_95(results_csv: Path) -> float:
    """Lê a última linha do results.csv e retorna mAP50-95 da pose (última época)."""
    if not results_csv.exists():
        return 0.0
    try:
        with results_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return 0.0
        last = rows[-1]
        # Ultralytics pode usar nomes diferentes; (P) = Pose
        for key in (
            "metrics/mAP50-95(P)",
            "metrics/pose_mAP50-95",
            "Pose_mAP50-95",
            "metrics/mAP50-95",
        ):
            val = last.get(key)
            if val is not None and str(val).strip() != "":
                return float(val)
        return 0.0
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
    # Mostrar no console se está usando GPU ou CPU
    try:
        import torch
        if str(device).lower() == "cpu" or not torch.cuda.is_available():
            _dev_str = "cpu (CUDA não disponível ou device=cpu no config)"
        else:
            _dev_str = f"cuda:{device}" if str(device).isdigit() else str(device)
        print(f"Device: {_dev_str}", flush=True)
        logger.log(f"Device de treino: {_dev_str}")
    except Exception:
        print(f"Device (config): {device}", flush=True)
        logger.log(f"Device de treino (config): {device}")
    epochs = training.get("epochs", 100)
    batch = training.get("batch_size", 16)
    lr0 = training.get("lr0", 0.01)
    lrf = training.get("lrf", 0.01)
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
        best_fold = None
        best_mAP = 0.0
        fold_metrics = []
        for fold_i, data_yaml in fold_data_yamls:
            logger.log(f"Treinando fold {fold_i}/{len(fold_data_yamls)}. Resultado: iniciado")
            # Novo modelo a cada fold para evitar estado corrompido após train() (KeyError 'model' ou falha no 2º fold)
            model = YOLO(POSE_WEIGHTS)
            _ensure_model_override(model)
            model.train(
                data=str(data_yaml),
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                device=device,
                lr0=lr0,
                lrf=lrf,
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
        # Escolher qual fold copiar: o de maior mAP ou, se todos 0, o primeiro com best.pt
        fold_to_copy = best_fold
        if fold_to_copy is None:
            for m in fold_metrics:
                fi = m["fold"]
                if (keypoints_out / f"fold_{fi}" / "weights" / "best.pt").exists():
                    fold_to_copy = fi
                    logger.log(f"Nenhum mAP50-95 > 0 lido do CSV; usando fold {fi} como modelo em train/weights.")
                    break
        if fold_to_copy is not None:
            src_best = keypoints_out / f"fold_{fold_to_copy}" / "weights" / "best.pt"
            if src_best.exists():
                shutil.copy2(src_best, train_weights / "best.pt")
                logger.log(f"Melhor fold: {fold_to_copy} (mAP50-95={best_mAP:.4f}). Copiando best.pt para train/weights. Resultado: sucesso")
                # Copiar também results.csv e last.pt do fold escolhido para ter dados em train/
                run_fold = keypoints_out / f"fold_{fold_to_copy}"
                train_dir = keypoints_out / "train"
                for fname in ("results.csv", "args.yaml"):
                    src_f = run_fold / fname
                    if src_f.exists():
                        shutil.copy2(src_f, train_dir / fname)
                last_pt = run_fold / "weights" / "last.pt"
                if last_pt.exists():
                    shutil.copy2(last_pt, train_weights / "last.pt")
                if best_fold is None:
                    best_fold = fold_to_copy

        # Estatísticas comparativas entre folds
        maps = [m["mAP50_95"] for m in fold_metrics]
        fold_comparison = {
            "mAP50_95_mean": float(np.mean(maps)),
            "mAP50_95_std": float(np.std(maps)) if len(maps) > 1 else 0.0,
            "mAP50_95_min": float(np.min(maps)),
            "mAP50_95_max": float(np.max(maps)),
            "fold_summary": [{"fold": m["fold"], "mAP50_95": m["mAP50_95"]} for m in fold_metrics],
        }
        logger.log(
            f"Folds: mAP50-95 média={fold_comparison['mAP50_95_mean']:.4f}, "
            f"std={fold_comparison['mAP50_95_std']:.4f}, min={fold_comparison['mAP50_95_min']:.4f}, max={fold_comparison['mAP50_95_max']:.4f}"
        )
        stats_dir = get_statistics_dir(root)
        stats_dir.mkdir(parents=True, exist_ok=True)
        report_path = stats_dir / "train_keypoints_folds.md"
        report_lines = [
            "# Comparativo entre folds (treino keypoints)",
            "",
            f"- **Melhor fold:** {best_fold} (mAP50-95 = {best_mAP:.4f})",
            f"- **Média:** {fold_comparison['mAP50_95_mean']:.4f} | **Desvio:** {fold_comparison['mAP50_95_std']:.4f} | **Mín:** {fold_comparison['mAP50_95_min']:.4f} | **Máx:** {fold_comparison['mAP50_95_max']:.4f}",
            "",
            "| Fold | mAP50-95 |",
            "|------|---------|",
        ]
        for m in fold_metrics:
            report_lines.append(f"| {m['fold']} | {m['mAP50_95']:.4f} |")
        report_lines.append("")
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        logger.log(f"Relatório comparativo dos folds salvo em: {report_path.relative_to(root)}")

        metrics = {
            "epochs": epochs,
            "batch_size": batch,
            "imgsz": imgsz,
            "device": device,
            "k_folds": len(fold_data_yamls),
            "best_fold": best_fold,
            "best_mAP50_95": best_mAP,
            "fold_metrics": fold_metrics,
            "fold_comparison": fold_comparison,
        }
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
        model = YOLO(POSE_WEIGHTS)
        _ensure_model_override(model)
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            lr0=lr0,
            lrf=lrf,
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
