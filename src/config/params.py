from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class AppParams:
    """
    Parâmetros principais da aplicação, carregados de `config.yaml`.
    """

    model_name: str
    image_size: int
    confidence_threshold: float
    data_dir: str
    models_dir: str
    outputs_dir: str


def get_full_config() -> Dict[str, Any]:
    """
    Retorna o conteúdo completo do `config.yaml` como dicionário.
    Útil para acessar seções como data, augmentation, training, keypoints.
    """
    return _load_yaml_config()


def get_keypoint_names() -> List[str]:
    """Retorna a lista de nomes dos keypoints definidos no config."""
    cfg = get_full_config()
    names = cfg.get("keypoints", {}).get("names", [])
    return names if isinstance(names, list) else []


def _default_params() -> AppParams:
    return AppParams(
        model_name="baseline",
        image_size=640,
        confidence_threshold=0.5,
        data_dir="data",
        models_dir="models",
        outputs_dir="outputs",
    )


def _load_yaml_config() -> Dict[str, Any]:
    """
    Carrega o arquivo `config.yaml` na raiz do projeto.
    Se o arquivo não existir ou estiver inválido, retorna um dicionário vazio.
    """
    # src/config/params.py -> raiz do projeto (2 níveis acima de 'src')
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config.yaml"

    if not config_path.exists():
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    return data


def get_params() -> AppParams:
    """
    Retorna os parâmetros da aplicação a partir de `config.yaml`.
    Se algum valor não estiver definido, usa os padrões de `_default_params`.
    """
    base = _default_params()
    data = _load_yaml_config()

    app_cfg = data.get("app", {}) if isinstance(data.get("app", {}), dict) else {}
    paths_cfg = data.get("paths", {}) if isinstance(data.get("paths", {}), dict) else {}

    return AppParams(
        model_name=str(app_cfg.get("model_name", base.model_name)),
        image_size=int(app_cfg.get("image_size", base.image_size)),
        confidence_threshold=float(
            app_cfg.get("confidence_threshold", base.confidence_threshold)
        ),
        data_dir=str(paths_cfg.get("data_dir", base.data_dir)),
        models_dir=str(paths_cfg.get("models_dir", base.models_dir)),
        outputs_dir=str(paths_cfg.get("outputs_dir", base.outputs_dir)),
    )

