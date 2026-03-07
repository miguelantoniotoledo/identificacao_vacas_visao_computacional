#!/usr/bin/env python
"""
Unifica pastas raw e converte anotações Label Studio para YOLO pose.
Log: uma linha por passo com data e hora; passos detalhados por imagem na conversão.

Uso:
  python scripts/unify_and_convert.py
  python scripts/unify_and_convert.py --debug   # imprime resumo e primeiros motivos de falha
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.unify import unify_raw_folders
from src.data.convert_labelstudio import convert_labelstudio_to_yolo_pose
from src.utils.metrics_logger import create_step_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Unificar raw e converter Label Studio -> YOLO pose.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Imprimir resumo no terminal e as primeiras 15 falhas com motivo (para debugar).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    logger = create_step_logger("unify_and_convert", root)

    print("Unificando pastas raw...")
    logger.log("Unificando pastas raw.")
    counts = unify_raw_folders()
    n_kp = counts.get("keypoints_annotations", 0)
    n_cls = counts.get("classification_images", 0)
    logger.log(f"Unify concluído: {n_kp} anotações keypoints, {n_cls} imagens classificação.")
    print(f"  Anotações keypoints: {n_kp}. Imagens classificação: {n_cls}.")

    print("Convertendo Label Studio -> YOLO pose...")
    logger.log("Convertendo Label Studio -> YOLO pose.")
    result = convert_labelstudio_to_yolo_pose(
        step_log=logger.log,
        collect_failure_reasons=15 if args.debug else None,
    )
    logger.log(f"Conversão concluída: {result['converted']} convertidos, {result['failed']} falhas.")

    metrics = {
        "keypoints_annotations": n_kp,
        "classification_images": n_cls,
        "converted": result["converted"],
        "failed": result["failed"],
    }
    log_path = logger.finalize(metrics)

    print(f"  Convertidos: {result['converted']}. Falhas: {result['failed']}.")
    if args.debug and result.get("failure_reasons"):
        print("\n--- Debug: primeiras falhas e motivo ---")
        for rel, reason in result["failure_reasons"]:
            print(f"  {rel}")
            print(f"    -> {reason}")
        print("---\nDica: o JSON tem task.data.img com o nome do arquivo (ex.: /data/upload/1/xxx.jpg).")
        print("Coloque as imagens na pasta da vaca com o mesmo nome (xxx.jpg) ou ajuste o código para buscar em outro diretório.")

    print(f"Log e métricas salvos em: {log_path}")
    print("Concluído.")


if __name__ == "__main__":
    main()
