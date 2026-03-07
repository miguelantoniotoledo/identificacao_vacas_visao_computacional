"""
Prepara splits train/val/test e estrutura YOLO para treino.
Suporta group k-fold (por pasta) para reduzir vazamento e augmentation (contraste + ruído).
"""

import random
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import yaml

from src.config import get_full_config


def _get_image_files(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in exts]


def _group_from_stem(stem: str) -> str:
    """Extrai grupo do nome do arquivo (prefixo antes de __) para group k-fold."""
    if "__" in stem:
        return stem.split("__")[0]
    return stem


def _apply_train_augmentation(
    img_path: Path,
    lbl_path: Path,
    out_images: Path,
    out_labels: Path,
    n_copies: int,
    contrast_limit: float,
    gaussian_noise_std: float,
) -> int:
    """Gera n_copies imagens com contraste e ruído gaussiano; copia label. Retorna quantas foram salvas."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return 0
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    label_content = lbl_path.read_text(encoding="utf-8").strip()
    stem = img_path.stem
    ext = img_path.suffix
    count = 0
    for i in range(1, n_copies + 1):
        out_img = img.astype(np.float64)
        # Contraste/brilho
        alpha = 1.0 + random.uniform(-contrast_limit, contrast_limit)
        beta = random.uniform(-20, 20)
        out_img = np.clip(alpha * out_img + beta, 0, 255).astype(np.uint8)
        # Ruído gaussiano
        noise = np.random.normal(0, gaussian_noise_std, out_img.shape).astype(np.float64)
        out_img = np.clip(out_img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        out_name = f"{stem}_aug{i}{ext}"
        out_path = out_images / out_name
        cv2.imwrite(str(out_path), out_img)
        (out_labels / f"{stem}_aug{i}.txt").write_text(label_content, encoding="utf-8")
        count += 1
    return count


def prepare_pose_dataset(
    unified_dir: Optional[Path] = None,
    step_log: Optional[Callable[[str], None]] = None,
) -> Tuple[Path, Path, Path, Dict[str, int]]:
    """
    Cria splits train/val/test (e folds quando pose.k_folds > 1) para dataset de pose.
    Usa split_test_ratio para 90/10; group k-fold por pasta para reduzir vazamento.
    step_log(mensagem) grava uma linha por passo (ex.: cópia de imagem, augmentation).
    Retorna (train_path, val_path, test_path, counts_dict).
    """
    cfg = get_full_config()
    root = Path(__file__).resolve().parents[2]
    unified = unified_dir or root / cfg.get("paths", {}).get("unified_dir", "data/unified")
    data_cfg = cfg.get("data", {})
    pose_cfg = cfg.get("pose", {})
    aug_cfg = cfg.get("augmentation", {})

    seed = data_cfg.get("random_seed", 42)
    split_test_ratio = data_cfg.get("split_test_ratio", data_cfg.get("test_ratio", 0.1))
    train_ratio = data_cfg.get("train_ratio", 1.0 - split_test_ratio)
    val_ratio = data_cfg.get("val_ratio", 0.0)
    test_ratio = data_cfg.get("test_ratio", split_test_ratio)

    k_folds = pose_cfg.get("k_folds", 1)
    strategy = pose_cfg.get("strategy", "kfold_misturado")

    contrast_limit = aug_cfg.get("contrast_limit", 0.25)
    gaussian_noise_std = aug_cfg.get("gaussian_noise_std", 10.0)
    train_augment_copies = aug_cfg.get("train_augment_copies", 0)

    images_dir = unified / "keypoints" / "images"
    labels_dir = unified / "keypoints" / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Pastas {images_dir} ou {labels_dir} não encontradas.")

    files: List[Tuple[Path, Path]] = []
    for img in _get_image_files(images_dir):
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.exists():
            files.append((img, lbl))

    if step_log:
        step_log(f"Preparando dataset de pose. Total de pares imagem/label: {len(files)}")

    random.seed(seed)
    n_total = len(files)

    # Split teste (ex.: 10%)
    n_test = int(n_total * test_ratio)
    random.shuffle(files)
    test_f = files[:n_test]
    train_val_f = files[n_test:]

    # Grupos para group k-fold (por prefixo no nome)
    groups = [_group_from_stem(img.stem) for img, _ in train_val_f]
    unique_groups = list(dict.fromkeys(groups))
    n_groups = len(unique_groups)

    kp_names = cfg.get("keypoints", {}).get("names", [])
    base_yaml = {
        "names": {0: "cow"},
        "kpt_shape": [len(kp_names), 3],
        "flip_idx": list(range(len(kp_names))),
    }

    out = unified / "yolo_pose"
    out.mkdir(parents=True, exist_ok=True)

    train_f: List[Tuple[Path, Path]] = []
    val_f: List[Tuple[Path, Path]] = []

    if k_folds > 1 and strategy == "group_kfold" and n_groups >= k_folds:
        try:
            from sklearn.model_selection import GroupKFold
        except ImportError:
            k_folds = 1
        if k_folds > 1:
            if step_log:
                step_log(f"Usando group k-fold. Grupos: {n_groups}, Folds: {k_folds}")
            gkf = GroupKFold(n_splits=k_folds)
            group_arr = [unique_groups.index(g) for g in groups]
            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(train_val_f, groups=group_arr)):
                fold_name = f"fold_{fold_idx + 1}"
                fold_dir = out / fold_name
                train_fold = [train_val_f[i] for i in train_idx]
                val_fold = [train_val_f[i] for i in val_idx]
                for split_list, name in [(train_fold, "train"), (val_fold, "val")]:
                    img_out = fold_dir / name / "images"
                    lbl_out = fold_dir / name / "labels"
                    img_out.mkdir(parents=True, exist_ok=True)
                    lbl_out.mkdir(parents=True, exist_ok=True)
                    for img, lbl in split_list:
                        shutil.copy2(img, img_out / img.name)
                        shutil.copy2(lbl, lbl_out / lbl.name)
                        if step_log:
                            step_log(f"Copiando imagem {img.name} para {name}. Resultado: sucesso")
                    if name == "train" and train_augment_copies > 0:
                        for img, lbl in split_list:
                            n_copies = _apply_train_augmentation(
                                img_out / img.name,
                                lbl_out / lbl.name,
                                img_out,
                                lbl_out,
                                train_augment_copies,
                                contrast_limit,
                                gaussian_noise_std,
                            )
                            if step_log:
                                step_log(f"Aplicando augmentation em {img.name}. Resultado: {n_copies} cópias geradas")
                test_dir = fold_dir / "test"
                (test_dir / "images").mkdir(parents=True, exist_ok=True)
                (test_dir / "labels").mkdir(parents=True, exist_ok=True)
                for img, lbl in test_f:
                    shutil.copy2(img, test_dir / "images" / img.name)
                    shutil.copy2(lbl, test_dir / "labels" / lbl.name)
                data_yaml = {
                    **base_yaml,
                    "path": str(fold_dir.resolve()),
                    "train": "train/images",
                    "val": "val/images",
                    "test": "test/images",
                }
                (fold_dir / "data.yaml").write_text(
                    yaml.dump(data_yaml, default_flow_style=False, allow_unicode=True),
                    encoding="utf-8",
                )
                if step_log:
                    step_log(f"Fold {fold_idx + 1}: data.yaml gerado em {fold_dir}")
            counts = {
                "n_train": len(train_val_f) * (k_folds - 1) // k_folds,
                "n_val": len(train_val_f) // k_folds,
                "n_test": len(test_f),
                "n_total": n_total,
                "k_folds": k_folds,
                "n_groups": n_groups,
            }
            return out / "fold_1" / "train", out / "fold_1" / "val", out / "fold_1" / "test", counts

    # Split único (sem k-fold)
    n_train = int(len(train_val_f) * train_ratio)
    n_val = len(train_val_f) - n_train
    train_f = train_val_f[:n_train]
    val_f = train_val_f[n_train:]
    if k_folds <= 1 or n_groups < k_folds:
        for split_list, name in [(train_f, "train"), (val_f, "val"), (test_f, "test")]:
            img_out = out / name / "images"
            lbl_out = out / name / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            for img, lbl in split_list:
                shutil.copy2(img, img_out / img.name)
                shutil.copy2(lbl, lbl_out / lbl.name)
                if step_log:
                    step_log(f"Copiando imagem {img.name} para {name}. Resultado: sucesso")
            if name == "train" and train_augment_copies > 0:
                for img, lbl in split_list:
                    n_copies = _apply_train_augmentation(
                        img_out / img.name,
                        lbl_out / lbl.name,
                        img_out,
                        lbl_out,
                        train_augment_copies,
                        contrast_limit,
                        gaussian_noise_std,
                    )
                    if step_log:
                        step_log(f"Aplicando augmentation em {img.name}. Resultado: {n_copies} cópias geradas")
        data_yaml = {
            **base_yaml,
            "path": str(out.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
        }
        (out / "data.yaml").write_text(
            yaml.dump(data_yaml, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        if step_log:
            step_log(f"data.yaml gerado em {out}")

    counts = {
        "n_train": len(train_f),
        "n_val": len(val_f),
        "n_test": len(test_f),
        "n_total": n_total,
        "k_folds": 1,
        "n_groups": n_groups,
    }
    return out / "train", out / "val", out / "test", counts
