#!/usr/bin/env python
"""
Prepara splits train/val/test para treino YOLO.
Log: uma linha por passo com data/hora; passos detalhados por imagem (cópia, augmentation).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.prepare_dataset import prepare_classification_split, prepare_pose_dataset
from src.utils.metrics_logger import create_step_logger


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    logger = create_step_logger("prepare_dataset", root)

    def log_and_print(msg: str) -> None:
        logger.log(msg)
        print(msg, flush=True)

    log_and_print("Preparando dataset de pose...")
    train, val, test, counts = prepare_pose_dataset(step_log=log_and_print)
    log_and_print(f"Train: {train}")
    log_and_print(f"Val:   {val}")
    log_and_print(f"Test:  {test}")

    result = prepare_classification_split(step_log=log_and_print)
    if result is not None:
        class_split_path, class_counts = result
        log_and_print(f"Classificação split: {class_split_path} (train={class_counts.get('n_train', 0)}, val={class_counts.get('n_val', 0)}, test={class_counts.get('n_test', 0)})")
        counts["classification_n_train"] = class_counts.get("n_train", 0)
        counts["classification_n_val"] = class_counts.get("n_val", 0)
        counts["classification_n_test"] = class_counts.get("n_test", 0)
    else:
        log_and_print("Pasta data/unified/classification não encontrada; split de classificação omitido.")

    metrics = dict(counts)
    log_path = logger.finalize(metrics)
    print(f"Log e métricas salvos em: {log_path}")
    print("Concluído. Pose: data/unified/yolo_pose/. Classificação: data/unified/classification_split/ (train/val/test).")


if __name__ == "__main__":
    main()
