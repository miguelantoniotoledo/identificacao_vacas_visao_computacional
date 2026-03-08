"""
Unifica pastas do raw em estrutura padronizada.

Estrutura esperada em raw:
  raw/
    catalogo/
      <nome_vaca>/
        *.jpg, *.png, ...
        Key_points/
          <id>  (JSON sem extensão)
    classificacao/
      <nome_vaca>/
        *.jpg, *.png, ...

Saída unificada:
  data/unified/
    classification/
      <nome_vaca>/
        *.jpg, *.png, ...

A conversão de keypoints é feita por convert_labelstudio (escreve em keypoints/images e keypoints/labels).
"""

import shutil
from pathlib import Path
from typing import Dict, Optional

from src.config import get_full_config


def unify_raw_folders(
    raw_dir: Optional[Path] = None,
    unified_dir: Optional[Path] = None,
    catalogo_subdir: str = "catalogo",
    classificacao_subdir: str = "classificacao",
    keypoints_subdir: str = "Key_points",
) -> Dict[str, int]:
    """
    Unifica pastas do raw em data/unified.

    Retorna contagem de arquivos processados.
    """
    cfg = get_full_config()
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})

    root = Path(__file__).resolve().parents[2]
    raw = raw_dir or root / paths.get("raw_dir", "raw")
    unified = unified_dir or root / paths.get("unified_dir", "data/unified")
    catalogo = raw / (data_cfg.get("catalogo_subdir", catalogo_subdir))
    classificacao = raw / (data_cfg.get("classificacao_subdir", classificacao_subdir))
    kp_dir_name = data_cfg.get("keypoints_subdir", keypoints_subdir)

    counts = {"keypoints_annotations": 0, "classification_images": 0}

    # Contar anotações de keypoints (a conversão é feita em convert_labelstudio)
    if catalogo.exists():
        for cow_folder in catalogo.iterdir():
            if not cow_folder.is_dir():
                continue
            kp_folder = cow_folder / kp_dir_name
            if kp_folder.exists():
                counts["keypoints_annotations"] += sum(
                    1 for f in kp_folder.iterdir() if f.is_file()
                )

    # --- Classificação: copiar imagens por vaca ---
    class_out = unified / "classification"
    if classificacao.exists():
        for cow_folder in classificacao.iterdir():
            if not cow_folder.is_dir():
                continue
            dest_cow = class_out / cow_folder.name
            dest_cow.mkdir(parents=True, exist_ok=True)
            for img in cow_folder.rglob("*"):
                if img.is_file() and img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    dest_img = dest_cow / img.name
                    if not dest_img.exists() or dest_img.stat().st_mtime < img.stat().st_mtime:
                        shutil.copy2(img, dest_img)
                    counts["classification_images"] += 1

    return counts
