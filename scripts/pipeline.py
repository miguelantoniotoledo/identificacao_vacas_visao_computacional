#!/usr/bin/env python
"""
Pipeline completo para identificação de vacas:

1) Unificar dados brutos e converter Label Studio -> YOLO pose
2) Preparar splits train/val/test
3) Análise exploratória (EDA) das features de keypoints
4) Treinar modelo de keypoints (YOLO pose)
5) Treinar classificador de vacas (YOLO cls)
6) Gerar visualizações de keypoints e retas

Grava log consolidado em outputs/logs/pipeline_run_<timestamp>.log com
passo a passo e estatísticas finais (F1, acurácia, mAP, etc.) de cada processo.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.metrics_logger import get_logs_dir, read_latest_metrics  # noqa: E402


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_step(description: str, script: str, pipeline_log_path: Path) -> bool:
    """Retorna True se sucesso. Cada evento grava uma linha com data/hora."""
    with pipeline_log_path.open("a", encoding="utf-8") as f:
        f.write(f"{_ts()} - {description}\n")
    print(f"\n=== {description} ===")
    cmd = [sys.executable, str(ROOT / "scripts" / script)]
    result = subprocess.run(cmd)
    with pipeline_log_path.open("a", encoding="utf-8") as f:
        status = "OK" if result.returncode == 0 else f"FALHOU (código {result.returncode})"
        f.write(f"{_ts()} - Status: {status}\n")
    if result.returncode != 0:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline completo de treinamento.")
    parser.add_argument(
        "--skip-train-keypoints",
        action="store_true",
        help="Pular treino do modelo de keypoints (YOLO pose).",
    )
    parser.add_argument(
        "--skip-train-classifier",
        action="store_true",
        help="Pular treino do classificador de vacas (YOLO cls).",
    )
    parser.add_argument(
        "--skip-visualize",
        action="store_true",
        help="Pular geração de visualizações de keypoints.",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Pular análise exploratória (EDA) das features.",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = get_logs_dir(ROOT)
    pipeline_log_path = logs_dir / f"pipeline_run_{ts}.log"

    with pipeline_log_path.open("w", encoding="utf-8") as f:
        f.write(f"{_ts()} - Pipeline de identificação de vacas iniciado\n")

    try:
        ok = run_step("1/6 - Unificar dados e converter Label Studio -> YOLO pose", "unify_and_convert.py", pipeline_log_path)
        if not ok:
            sys.exit(1)

        ok = run_step("2/6 - Preparar splits train/val/test", "prepare_dataset.py", pipeline_log_path)
        if not ok:
            sys.exit(1)

        if not args.skip_eda:
            ok = run_step("3/6 - Análise exploratória (EDA) das features", "analisar_features.py", pipeline_log_path)
            if not ok:
                sys.exit(1)
        else:
            with pipeline_log_path.open("a", encoding="utf-8") as f:
                f.write(f"{_ts()} - 3/6 - EDA PULADO (--skip-eda)\n")

        if not args.skip_train_keypoints:
            ok = run_step("4/6 - Treinar modelo de keypoints (YOLO pose)", "train_keypoints.py", pipeline_log_path)
            if not ok:
                sys.exit(1)
        else:
            with pipeline_log_path.open("a", encoding="utf-8") as f:
                f.write(f"{_ts()} - 4/6 - Treino keypoints PULADO (--skip-train-keypoints)\n")

        if not args.skip_train_classifier:
            ok = run_step("5/6 - Treinar classificador de vacas (YOLO cls)", "train_classifier.py", pipeline_log_path)
            if not ok:
                sys.exit(1)
        else:
            with pipeline_log_path.open("a", encoding="utf-8") as f:
                f.write(f"{_ts()} - 5/6 - Treino classificador PULADO (--skip-train-classifier)\n")

        if not args.skip_visualize:
            ok = run_step("6/6 - Gerar visualizações de keypoints", "visualize_keypoints.py", pipeline_log_path)
            if not ok:
                sys.exit(1)
        else:
            with pipeline_log_path.open("a", encoding="utf-8") as f:
                f.write(f"{_ts()} - 6/6 - Visualizações PULADAS (--skip-visualize)\n")

        # Estatísticas finais de cada processo (uma linha por evento com data/hora)
        with pipeline_log_path.open("a", encoding="utf-8") as f:
            f.write(f"{_ts()} - --- ESTATÍSTICAS FINAIS POR PROCESSO ---\n")
            for script_name in ["unify_and_convert", "prepare_dataset", "analisar_features", "train_keypoints", "train_classifier", "visualize_keypoints", "predict_cow", "predict_keypoints"]:
                m = read_latest_metrics(script_name, ROOT)
                if not m:
                    continue
                f.write(f"{_ts()} - --- {script_name} ---\n")
                for k, v in m.items():
                    if v is None or (isinstance(v, float) and (v != v)):
                        continue
                    if isinstance(v, float):
                        f.write(f"{_ts()} -   {k}: {v:.4f}\n")
                    else:
                        f.write(f"{_ts()} -   {k}: {v}\n")

        print(f"\nPipeline concluído com sucesso.")
        print(f"Log consolidado: {pipeline_log_path}")
    except KeyboardInterrupt:
        with pipeline_log_path.open("a", encoding="utf-8") as f:
            f.write(f"{_ts()} - INTERROMPIDO PELO USUÁRIO\n")
        print("\nPipeline interrompido pelo usuário.")
        sys.exit(1)


if __name__ == "__main__":
    main()

