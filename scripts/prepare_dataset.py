#!/usr/bin/env python
"""
Prepara splits train/val/test para treino YOLO.
Log: uma linha por passo com data/hora; passos detalhados por imagem (cópia, augmentation).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.prepare_dataset import prepare_pose_dataset
from src.utils.metrics_logger import create_step_logger


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    logger = create_step_logger("prepare_dataset", root)

    logger.log("Preparando dataset de pose...")
    train, val, test, counts = prepare_pose_dataset(step_log=logger.log)
    logger.log(f"Train: {train}")
    logger.log(f"Val:   {val}")
    logger.log(f"Test:  {test}")

    metrics = dict(counts)
    log_path = logger.finalize(metrics)
    print(f"Log e métricas salvos em: {log_path}")
    print("Concluído. data.yaml gerado em data/unified/yolo_pose/")


if __name__ == "__main__":
    main()
