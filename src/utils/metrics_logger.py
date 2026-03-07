"""
Gravação de métricas, logs de execução e gráficos em pasta de logs e estatísticas.
Suporta log uma linha por passo com data/hora (StepLogger).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import get_full_config


def _ts() -> str:
    """Timestamp no formato dia/hora para cada linha de log."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_logs_dir(root: Optional[Path] = None) -> Path:
    """Retorna o diretório de logs (cria se não existir)."""
    cfg = get_full_config()
    base = root or Path(__file__).resolve().parents[2]
    logs = base / cfg.get("paths", {}).get("logs_dir", "outputs/logs")
    logs.mkdir(parents=True, exist_ok=True)
    return logs


def get_statistics_dir(root: Optional[Path] = None) -> Path:
    """Retorna o diretório de estatísticas/gráficos (cria se não existir)."""
    cfg = get_full_config()
    base = root or Path(__file__).resolve().parents[2]
    stats = base / cfg.get("paths", {}).get("statistics_dir", "outputs/statistics")
    stats.mkdir(parents=True, exist_ok=True)
    return stats


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class StepLogger:
    """
    Logger que grava uma linha por passo com data e hora.
    Uso: logger = create_step_logger('meu_script', root); logger.log('Passo X'); logger.finalize(metrics)
    """

    def __init__(self, script_name: str, root: Path, log_path: Path, file_handle):
        self.script_name = script_name
        self.root = root
        self.log_path = log_path
        self._f = file_handle

    def log(self, message: str) -> None:
        """Grava uma linha: YYYY-MM-DD HH:MM:SS - message"""
        self._f.write(f"{_ts()} - {message}\n")
        self._f.flush()

    def finalize(self, metrics: Optional[Dict[str, Any]] = None) -> Path:
        """Appenda métricas ao log, grava JSON e fecha o arquivo."""
        self._f.write(f"{_ts()} - --- Métricas ---\n")
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, (int, float, str, bool)):
                    self._f.write(f"{_ts()} -   {k}: {v}\n")
                else:
                    self._f.write(f"{_ts()} -   {k}: {json.dumps(v, ensure_ascii=False)}\n")
        self._f.close()
        if metrics is not None:
            stats_dir = get_statistics_dir(self.root)
            ts = _timestamp()
            latest = stats_dir / f"{self.script_name}_latest.json"
            with latest.open("w", encoding="utf-8") as f:
                json.dump(
                    {"script": self.script_name, "timestamp": ts, "metrics": metrics},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        return self.log_path


def create_step_logger(script_name: str, root: Optional[Path] = None) -> StepLogger:
    """
    Cria um log por script com uma linha por passo (data/hora + mensagem).
    Retorna StepLogger: use .log('mensagem') e ao final .finalize(metrics).
    """
    base = root or Path(__file__).resolve().parents[2]
    logs_dir = get_logs_dir(base)
    ts = _timestamp()
    log_path = logs_dir / f"{script_name}_{ts}.log"
    f = log_path.open("w", encoding="utf-8")
    logger = StepLogger(script_name, base, log_path, f)
    logger.log(f"Início {script_name}")
    return logger


def log_script_run(
    script_name: str,
    messages: List[str],
    metrics: Optional[Dict[str, Any]] = None,
    root: Optional[Path] = None,
) -> Path:
    """
    Grava log de execução em arquivo .log e opcionalmente métricas em .json.
    Retorna o path do arquivo de log gerado.
    """
    logs_dir = get_logs_dir(root)
    stats_dir = get_statistics_dir(root)
    ts = _timestamp()
    log_file = logs_dir / f"{script_name}_{ts}.log"
    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"{_ts()} - Início {script_name}\n")
        for msg in messages:
            f.write(f"{_ts()} - {msg}\n")
        if metrics:
            f.write(f"{_ts()} - --- Métricas ---\n")
            for k, v in metrics.items():
                if isinstance(v, (int, float, str, bool)):
                    f.write(f"{_ts()} -   {k}: {v}\n")
                else:
                    f.write(f"{_ts()} -   {k}: {json.dumps(v, ensure_ascii=False)}\n")

    if metrics is not None:
        json_file = stats_dir / f"{script_name}_{ts}.json"
        with json_file.open("w", encoding="utf-8") as f:
            json.dump(
                {"script": script_name, "timestamp": ts, "metrics": metrics},
                f,
                ensure_ascii=False,
                indent=2,
            )
        # Também grava "último" para o pipeline ler
        latest = stats_dir / f"{script_name}_latest.json"
        with latest.open("w", encoding="utf-8") as f:
            json.dump(
                {"script": script_name, "timestamp": ts, "metrics": metrics},
                f,
                ensure_ascii=False,
                indent=2,
            )

    return log_file


def save_metrics_json(
    script_name: str,
    metrics: Dict[str, Any],
    root: Optional[Path] = None,
) -> Path:
    """Grava apenas o JSON de métricas (e _latest). Retorna path do JSON."""
    stats_dir = get_statistics_dir(root)
    ts = _timestamp()
    json_file = stats_dir / f"{script_name}_{ts}.json"
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(
            {"script": script_name, "timestamp": ts, "metrics": metrics},
            f,
            ensure_ascii=False,
            indent=2,
        )
    latest = stats_dir / f"{script_name}_latest.json"
    with latest.open("w", encoding="utf-8") as f:
        json.dump(
            {"script": script_name, "timestamp": ts, "metrics": metrics},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return json_file


def save_plot_png(
    fig: Any,
    name: str,
    root: Optional[Path] = None,
) -> Path:
    """Salva figura matplotlib como PNG na pasta de estatísticas."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return Path()
    stats_dir = get_statistics_dir(root)
    path = stats_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def read_latest_metrics(script_name: str, root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Lê as métricas do último run do script (arquivo _latest.json)."""
    stats_dir = get_statistics_dir(root)
    latest = stats_dir / f"{script_name}_latest.json"
    if not latest.exists():
        return None
    try:
        with latest.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("metrics")
    except Exception:
        return None


def extract_yolo_metrics_and_plot(
    results_csv: Path,
    plot_prefix: str,
    root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Lê results.csv do YOLO, extrai métricas da última época e gera gráficos de curva.
    Retorna dict com métricas finais (precision, recall, mAP50, mAP50-95, etc.) ou None.
    """
    if not results_csv.exists():
        return None
    try:
        import csv
        with results_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None
        last = rows[-1]
        # Normalizar chaves (YOLO pode usar / ou -)
        metrics = {}
        for k, v in last.items():
            try:
                metrics[k.strip()] = float(v) if v else None
            except ValueError:
                metrics[k.strip()] = v
        # Gerar gráfico de curvas se houver colunas numéricas
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            numeric_cols = [c for c in rows[0].keys() if c and rows[0].get(c) and _is_num(rows[0].get(c))]
            if len(numeric_cols) > 1:
                epochs = [int(r.get("epoch", i)) if _is_num(r.get("epoch")) else i for i, r in enumerate(rows)]
                for col in numeric_cols[:8]:
                    vals = [_safe_float(r.get(col)) for r in rows]
                    if any(v is not None for v in vals):
                        ax.plot(epochs, vals, label=col[:20], alpha=0.8)
                ax.set_xlabel("Epoch")
                ax.legend(loc="best", fontsize=7)
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                fig.tight_layout()
                save_plot_png(fig, f"{plot_prefix}_curves", root)
        except Exception:
            pass
        return metrics
    except Exception:
        return None


def _is_num(s: Any) -> bool:
    if s is None:
        return False
    try:
        float(str(s).strip())
        return True
    except ValueError:
        return False


def _safe_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None
