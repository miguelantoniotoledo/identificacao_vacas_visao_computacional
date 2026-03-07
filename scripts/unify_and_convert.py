#!/usr/bin/env python
"""
Unifica pastas raw e converte anotações Label Studio para YOLO pose.
Log: uma linha por passo com data e hora; passos detalhados por imagem na conversão.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.unify import unify_raw_folders
from src.data.convert_labelstudio import convert_labelstudio_to_yolo_pose
from src.utils.metrics_logger import create_step_logger


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    logger = create_step_logger("unify_and_convert", root)

    logger.log("Unificando pastas raw.")
    counts = unify_raw_folders()
    n_kp = counts.get("keypoints_annotations", 0)
    n_cls = counts.get("classification_images", 0)
    logger.log(f"Unify concluído: {n_kp} anotações keypoints, {n_cls} imagens classificação.")

    logger.log("Convertendo Label Studio -> YOLO pose.")
    result = convert_labelstudio_to_yolo_pose(step_log=logger.log)
    logger.log(f"Conversão concluída: {result['converted']} convertidos, {result['failed']} falhas.")

    metrics = {
        "keypoints_annotations": n_kp,
        "classification_images": n_cls,
        "converted": result["converted"],
        "failed": result["failed"],
    }
    log_path = logger.finalize(metrics)
    print(f"Log e métricas salvos em: {log_path}")
    print("Concluído.")


if __name__ == "__main__":
    main()
