"""
Prepara splits train/val/test e estrutura YOLO para treino.
- stratified_per_group: em cada grupo (indivíduo), 80%% treino, 10%% val, 10%% teste por grupo (sem overlap de fotos).
- group_kfold: split por indivíduo (não por foto). Primeiro divide os grupos em train/val/test (ex.: 80%% dos
  indivíduos para train, 10%% val, 10%% test); todas as fotos de um indivíduo ficam no mesmo split (evita vazamento).
  Se k_folds > 1, GroupKFold reparte apenas train+val por fold; test é hold-out fixo de indivíduos.
"""

import random
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import yaml

from src.config import get_full_config
from src.data.augmentation import get_offline_pose_augmentation

# YOLO pose: uma linha por objeto: class cx cy w h kp1_x kp1_y kp1_v ... (8 keypoints)
N_KEYPOINTS = 8
YOLO_POSE_VALS = 5 + N_KEYPOINTS * 3  # 29


def _get_image_files(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in exts]


def _clear_yolo_pose_output(out: Path, step_log: Optional[Callable[[str], None]]) -> None:
    """
    Remove todos os arquivos das pastas train/val/test (e fold_*) em out,
    para que treinamentos e testes futuros usem apenas o split gerado nesta execução.
    """
    for name in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = out / name / sub
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()
                if step_log:
                    step_log(f"Limpeza: arquivos removidos de {d.relative_to(out)}")
    for fold_dir in out.iterdir():
        if fold_dir.is_dir() and fold_dir.name.startswith("fold_"):
            for name in ("train", "val", "test"):
                for sub in ("images", "labels"):
                    d = fold_dir / name / sub
                    if d.exists():
                        for f in d.iterdir():
                            if f.is_file():
                                f.unlink()
            fold_yaml = fold_dir / "data.yaml"
            if fold_yaml.exists():
                fold_yaml.unlink()
            if step_log:
                step_log(f"Limpeza: arquivos removidos de {fold_dir.name}/")
    data_yaml = out / "data.yaml"
    if data_yaml.exists():
        data_yaml.unlink()
        if step_log:
            step_log("Limpeza: data.yaml removido (será recriado).")


def _clear_classification_split_output(out: Path, step_log: Optional[Callable[[str], None]]) -> None:
    """
    Remove todos os arquivos e subpastas de train/val/test em classification_split,
    para que treinamentos e testes futuros do classificador usem apenas o split desta execução.
    """
    for split_name in ("train", "val", "test"):
        split_dir = out / split_name
        if not split_dir.exists():
            continue
        for cow_dir in list(split_dir.iterdir()):
            if cow_dir.is_dir():
                for f in cow_dir.iterdir():
                    if f.is_file():
                        f.unlink()
                cow_dir.rmdir()
        if step_log:
            step_log(f"Limpeza (classificador): arquivos removidos de classification_split/{split_name}/")


def _group_from_stem(stem: str) -> str:
    """Extrai grupo do nome do arquivo (prefixo antes de __) para group k-fold."""
    if "__" in stem:
        return stem.split("__")[0]
    return stem


def _imread_unicode(path: Path):
    """
    Carrega imagem de um Path (suporta nomes com acentos no Windows).
    cv2.imread falha com caminhos Unicode no Windows; usar leitura em bytes + imdecode.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    buf = path.read_bytes()
    if not buf:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _imwrite_unicode(path: Path, img) -> bool:
    """
    Salva imagem em Path (suporta nomes com acentos no Windows).
    cv2.imwrite falha com caminhos Unicode no Windows; usar imencode + write_bytes.
    """
    try:
        import cv2
    except ImportError:
        return False
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        enc = cv2.imencode(".jpg", img)
    elif ext == ".png":
        enc = cv2.imencode(".png", img)
    else:
        enc = cv2.imencode(".jpg", img)
    if enc[0]:
        path.write_bytes(enc[1].tobytes())
        return True
    return False


def _parse_yolo_pose_line(line: str, img_w: int, img_h: int) -> Optional[Tuple[int, List[float], List[Tuple[float, float]], List[int]]]:
    """Retorna (class, bbox_norm [cx,cy,w,h], keypoints_px [(x,y),...], visibilities) ou None."""
    parts = line.strip().split()
    if len(parts) < YOLO_POSE_VALS:
        return None
    vals = [float(x) for x in parts[:YOLO_POSE_VALS]]
    cls = int(vals[0])
    bbox = vals[1:5]  # cx, cy, w, h normalized
    keypoints_px = []
    vis = []
    for j in range(N_KEYPOINTS):
        x_n, y_n, v = vals[5 + j * 3], vals[5 + j * 3 + 1], int(vals[5 + j * 3 + 2])
        keypoints_px.append((x_n * img_w, y_n * img_h))
        vis.append(v)
    return (cls, bbox, keypoints_px, vis)


def _yolo_pose_line_from_bbox_kp(cls: int, bbox: List[float], keypoints_px: List[Tuple[float, float]], vis: List[int], img_w: int, img_h: int) -> str:
    """Monta uma linha YOLO pose a partir de bbox normalizado e keypoints em pixels."""
    parts = [str(cls), f"{bbox[0]:.6f}", f"{bbox[1]:.6f}", f"{bbox[2]:.6f}", f"{bbox[3]:.6f}"]
    for (x, y), v in zip(keypoints_px, vis):
        xn = x / img_w if img_w else 0
        yn = y / img_h if img_h else 0
        xn = max(0, min(1, xn))
        yn = max(0, min(1, yn))
        parts.extend([f"{xn:.6f}", f"{yn:.6f}", str(v)])
    return " ".join(parts)


def _apply_train_augmentation(
    img_path: Path,
    lbl_path: Path,
    out_images: Path,
    out_labels: Path,
    n_copies: int,
    contrast_limit: float,
    gaussian_noise_std: float,
    transform: Optional[Any] = None,
) -> int:
    """
    Gera n_copies imagens augmentadas. Se transform (Albumentations) for passado,
    aplica todas as augmentations planejadas (flip, rotate, blur, HSV, ruído, etc.)
    com atualização de bbox e keypoints; senão usa apenas contraste + ruído (fallback).
    """
    try:
        import numpy as np
    except ImportError:
        return 0
    img = _imread_unicode(img_path)
    if img is None:
        return 0
    label_content = lbl_path.read_text(encoding="utf-8").strip()
    if not label_content:
        return 0
    h, w = img.shape[:2]
    parsed = _parse_yolo_pose_line(label_content, w, h)
    if parsed is None:
        return 0
    cls, bbox, keypoints_px, vis = parsed
    stem = img_path.stem
    ext = img_path.suffix
    count = 0

    for i in range(1, n_copies + 1):
        if transform is not None:
            try:
                transformed = transform(
                    image=img,
                    bboxes=[bbox],
                    class_labels=[cls],
                    keypoints=keypoints_px,
                )
                out_img = transformed["image"]
                t_bboxes = transformed["bboxes"]
                t_kp = transformed["keypoints"]
                if not t_bboxes or not t_kp:
                    continue
                new_bbox = t_bboxes[0]
                out_h, out_w = out_img.shape[:2]
                new_line = _yolo_pose_line_from_bbox_kp(cls, new_bbox, t_kp, vis, out_w, out_h)
            except Exception:
                continue
        else:
            out_img = img.astype(np.float64)
            alpha = 1.0 + random.uniform(-contrast_limit, contrast_limit)
            beta = random.uniform(-20, 20)
            out_img = np.clip(alpha * out_img + beta, 0, 255).astype(np.uint8)
            noise = np.random.normal(0, gaussian_noise_std, out_img.shape).astype(np.float64)
            out_img = np.clip(out_img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
            new_line = label_content

        out_name = f"{stem}_aug{i}{ext}"
        if _imwrite_unicode(out_images / out_name, out_img):
            (out_labels / f"{stem}_aug{i}.txt").write_text(new_line, encoding="utf-8")
            count += 1
    return count


def _create_mosaic_pose(
    list_of_img_label: List[Tuple[Any, str]],
    out_images: Path,
    out_labels: Path,
    mosaic_idx: int,
    size: int = 640,
) -> bool:
    """
    Cria uma imagem mosaic 2x2 a partir de 4 (imagem, linha_yolo).
    Cada imagem é redimensionada para size/2 x size/2 e colocada em um quadrante.
    Retorna True se salvou com sucesso.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False
    if len(list_of_img_label) < 4:
        return False
    half = size // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    all_lines: List[str] = []

    for quad, (img, line) in enumerate(list_of_img_label):
        if img is None or img.size == 0:
            return False
        img_small = cv2.resize(img, (half, half), interpolation=cv2.INTER_LINEAR)
        if quad == 0:
            canvas[0:half, 0:half] = img_small
            dx_norm, dy_norm = 0.0, 0.0
        elif quad == 1:
            canvas[0:half, half:size] = img_small
            dx_norm, dy_norm = 0.5, 0.0
        elif quad == 2:
            canvas[half:size, 0:half] = img_small
            dx_norm, dy_norm = 0.0, 0.5
        else:
            canvas[half:size, half:size] = img_small
            dx_norm, dy_norm = 0.5, 0.5

        parsed = _parse_yolo_pose_line(line, half, half)
        if parsed is None:
            continue
        cls, bbox, kp_px, vis = parsed
        cx, cy, bw, bh = bbox
        new_cx = cx * 0.5 + dx_norm
        new_cy = cy * 0.5 + dy_norm
        new_bw = bw * 0.5
        new_bh = bh * 0.5
        # Keypoints: de pixels no patch (half x half) para normalizados no canvas (size x size)
        new_kp_norm = [((x / half) * 0.5 + dx_norm, (y / half) * 0.5 + dy_norm) for x, y in kp_px]
        new_kp_px = [(xn * size, yn * size) for xn, yn in new_kp_norm]
        line_str = _yolo_pose_line_from_bbox_kp(cls, [new_cx, new_cy, new_bw, new_bh], new_kp_px, vis, size, size)
        all_lines.append(line_str)

    if not all_lines:
        return False
    out_name = f"mosaic_{mosaic_idx}.jpg"
    full_label = "\n".join(all_lines)
    if _imwrite_unicode(out_images / out_name, canvas):
        (out_labels / f"mosaic_{mosaic_idx}.txt").write_text(full_label, encoding="utf-8")
        return True
    return False


def prepare_classification_split(
    unified_dir: Optional[Path] = None,
    step_log: Optional[Callable[[str], None]] = None,
) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """
    Cria split 80/10/10 (train/val/test) para classificação **por indivíduo** (uma classe por vaca).
    Cada vaca (pasta) vai inteira para train, val ou test; nenhum indivíduo aparece em mais de um split (evita vazamento).
    Entrada: data/unified/classification/<nome_vaca>/*.jpg
    Saída:   data/unified/classification_split/train/<nome_vaca>/, val/..., test/...
    Usa train_ratio, val_ratio, test_ratio aplicados à **quantidade de vacas**, não de fotos.
    Retorna (classification_split_path, counts) ou None se pasta classification não existir.
    """
    cfg = get_full_config()
    root = Path(__file__).resolve().parents[2]
    unified = unified_dir or root / cfg.get("paths", {}).get("unified_dir", "data/unified")
    data_cfg = cfg.get("data", {})
    class_dir = unified / "classification"
    if not class_dir.exists():
        return None

    seed = data_cfg.get("random_seed", 42)
    train_ratio = data_cfg.get("train_ratio", 0.8)
    val_ratio = data_cfg.get("val_ratio", 0.1)
    test_ratio = data_cfg.get("test_ratio", 0.1)
    total_r = train_ratio + val_ratio + test_ratio
    if total_r <= 0:
        total_r = 1.0
    train_ratio, val_ratio, test_ratio = train_ratio / total_r, val_ratio / total_r, test_ratio / total_r

    out = unified / "classification_split"
    out.mkdir(parents=True, exist_ok=True)
    _clear_classification_split_output(out, step_log)
    (out / "train").mkdir(exist_ok=True)
    (out / "val").mkdir(exist_ok=True)
    (out / "test").mkdir(exist_ok=True)

    cow_folders = [d for d in class_dir.iterdir() if d.is_dir()]
    if not cow_folders:
        if step_log:
            step_log("Nenhuma pasta de vaca em classification. Split omitido.")
        return None

    # Split por indivíduo: embaralhar vacas e dividir por quantidade de vacas (não de fotos)
    random.seed(seed)
    shuffled_cows = list(cow_folders)
    random.shuffle(shuffled_cows)
    n_cows = len(shuffled_cows)
    n_test_c = max(1, int(round(n_cows * test_ratio))) if n_cows >= 2 else 0
    n_val_c = max(0, int(round(n_cows * val_ratio)))
    n_train_c = n_cows - n_test_c - n_val_c
    if n_train_c < 1:
        n_train_c = 1
        n_test_c = max(0, n_cows - n_val_c - 1)
    if n_test_c < 0:
        n_test_c = 0
    test_cows = shuffled_cows[:n_test_c] if n_test_c else []
    val_cows = shuffled_cows[n_test_c : n_test_c + n_val_c] if n_val_c else []
    train_cows = shuffled_cows[n_test_c + n_val_c :]

    if step_log:
        step_log(
            f"Split classificação por indivíduo: {len(train_cows)} vacas train, {len(val_cows)} val, {len(test_cows)} test. Sem overlap de indivíduos."
        )

    n_train, n_val, n_test = 0, 0, 0
    for split_name, cow_list in [("train", train_cows), ("val", val_cows), ("test", test_cows)]:
        for cow_dir in cow_list:
            cow_name = cow_dir.name
            images = _get_image_files(cow_dir)
            if not images:
                continue
            dest = out / split_name / cow_name
            dest.mkdir(parents=True, exist_ok=True)
            for img in images:
                shutil.copy2(img, dest / img.name)
            if split_name == "train":
                n_train += len(images)
            elif split_name == "val":
                n_val += len(images)
            else:
                n_test += len(images)

    if step_log:
        step_log(f"Classificação split: train={n_train}, val={n_val}, test={n_test}. Salvo em {out}.")
    counts = {"n_train": n_train, "n_val": n_val, "n_test": n_test, "n_classes": len(cow_folders)}
    return (out, counts)


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

    try:
        k_folds = int(pose_cfg.get("k_folds", 1))
    except (TypeError, ValueError):
        k_folds = 1
    k_folds = max(1, k_folds)
    strategy = pose_cfg.get("strategy", "kfold_misturado")

    contrast_limit = aug_cfg.get("contrast_limit", 0.25)
    gaussian_noise_std = aug_cfg.get("gaussian_noise_std", 10.0)
    train_augment_copies = aug_cfg.get("train_augment_copies", 0)
    mosaic_enabled = aug_cfg.get("mosaic_enabled", False)
    image_size = data_cfg.get("image_size", 640)

    images_dir = unified / "keypoints" / "images"
    labels_dir = unified / "keypoints" / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Pastas {images_dir} ou {labels_dir} não encontradas.")

    files: List[Tuple[Path, Path]] = []
    for img in _get_image_files(images_dir):
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.exists():
            files.append((img, lbl))

    random.seed(seed)
    n_total = len(files)
    if step_log:
        step_log(f"Listagem concluída: {n_total} pares imagem/label em {images_dir}")

    if n_total == 0:
        raise ValueError("Nenhum par imagem/label encontrado. Verifique pastas keypoints/images e keypoints/labels.")

    if step_log:
        step_log("Agrupando por pasta (prefixo do nome antes de __)...")

    # Agrupar por pasta (prefixo do nome antes de __)
    groups_dict: Dict[str, List[Tuple[Path, Path]]] = {}
    for img, lbl in files:
        g = _group_from_stem(img.stem)
        groups_dict.setdefault(g, []).append((img, lbl))
    unique_groups = list(groups_dict.keys())
    n_groups = len(unique_groups)

    kp_names = cfg.get("keypoints", {}).get("names", [])
    base_yaml = {
        "names": {0: "cow"},
        "kpt_shape": [len(kp_names), 3],
        "flip_idx": list(range(len(kp_names))),
    }

    out = unified / "yolo_pose"
    out.mkdir(parents=True, exist_ok=True)
    if step_log:
        step_log(f"Diretório de saída: {out}")
    _clear_yolo_pose_output(out, step_log)

    train_f: List[Tuple[Path, Path]] = []
    val_f: List[Tuple[Path, Path]] = []
    test_f: List[Tuple[Path, Path]] = []

    PROGRESS_INTERVAL = 100

    # Estratificado por pasta: 80% train, 10% val, 10% test por grupo (sem overlap)
    if strategy == "stratified_per_group":
        if step_log:
            step_log(f"Estratificado por grupo: {n_groups} grupos. Aplicando 80% train / 10% val / 10% test por grupo...")
        # Por grupo: shuffle -> train_ratio (80%), val_ratio (10%), resto test (10%)
        for gname, gfiles in groups_dict.items():
            random.shuffle(gfiles)
            n = len(gfiles)
            n_train = max(0, int(round(n * train_ratio)))
            n_val = max(0, int(round(n * val_ratio)))
            n_test = n - n_train - n_val  # resto para test (garante soma = n)
            if n_test < 0:
                n_test = 0
                n_train = n - n_val
            train_f.extend(gfiles[: n_train])
            val_f.extend(gfiles[n_train : n_train + n_val])
            test_f.extend(gfiles[n_train + n_val :])
        if step_log:
            step_log(
                f"Split definido: Train={len(train_f)}, Val={len(val_f)}, Test={len(test_f)}. Sem overlap."
            )
        # Augmentation e mosaic apenas em train; val e test recebem só cópia (sem transformação).
        for split_list, name in [(train_f, "train"), (val_f, "val"), (test_f, "test")]:
            img_out = out / name / "images"
            lbl_out = out / name / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            total = len(split_list)
            if step_log:
                step_log(f"Copiando {total} arquivos para {name}/...")
            for i, (img, lbl) in enumerate(split_list):
                shutil.copy2(img, img_out / img.name)
                shutil.copy2(lbl, lbl_out / lbl.name)
                if step_log and (i + 1) % PROGRESS_INTERVAL == 0:
                    step_log(f"  {name}: {i + 1}/{total}")
            if step_log:
                step_log(f"  {name}: {total}/{total} concluído.")
            if name == "train" and train_augment_copies > 0:
                offline_transform = get_offline_pose_augmentation()
                if step_log:
                    if offline_transform is not None:
                        step_log(
                            f"Aplicando todas as augmentations planejadas em train (flip, rotate, blur, HSV, ruído; "
                            f"{train_augment_copies} cópia(s) por imagem)..."
                        )
                    else:
                        step_log(
                            f"Aplicando augmentation em train (contraste + ruído; {train_augment_copies} cópia(s) por imagem)..."
                        )
                for i, (img, lbl) in enumerate(split_list):
                    _apply_train_augmentation(
                        img_out / img.name,
                        lbl_out / lbl.name,
                        img_out,
                        lbl_out,
                        train_augment_copies,
                        contrast_limit,
                        gaussian_noise_std,
                        transform=offline_transform,
                    )
                    if step_log and (i + 1) % PROGRESS_INTERVAL == 0:
                        step_log(f"  augmentation: {i + 1}/{total}")
                if step_log:
                    step_log(f"  augmentation: {total}/{total} concluído.")
            if name == "train" and mosaic_enabled and len(split_list) >= 4:
                n_mosaic = len(split_list) // 4  # máximo possível: uma mosaic a cada 4 imagens (todo o conjunto)
                shuffled_train = list(split_list)
                random.shuffle(shuffled_train)
                img_out = out / "train" / "images"
                lbl_out = out / "train" / "labels"
                if step_log:
                    step_log(f"Gerando {n_mosaic} imagens mosaic (2x2) em train/ (todo o conjunto, 4 imagens por mosaic)...")
                for i in range(n_mosaic):
                    group = shuffled_train[i * 4 : (i + 1) * 4]
                    four = []
                    for img_path, lbl_path in group:
                        im = _imread_unicode(img_path)
                        line = lbl_path.read_text(encoding="utf-8").strip()
                        if not line:
                            break
                        four.append((im, line))
                    if len(four) == 4 and _create_mosaic_pose(four, img_out, lbl_out, i + 1, image_size):
                        if step_log and (i + 1) % PROGRESS_INTERVAL == 0:
                            step_log(f"  mosaic: {i + 1}/{n_mosaic}")
                if step_log:
                    step_log(f"  mosaic: {n_mosaic}/{n_mosaic} concluído.")
        if step_log:
            step_log("Gerando data.yaml...")
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
            step_log(f"data.yaml gerado em {out}. Concluído: train/val/test disjuntos (sem vazamento).")
        counts = {
            "n_train": len(train_f),
            "n_val": len(val_f),
            "n_test": len(test_f),
            "n_total": n_total,
            "k_folds": 1,
            "n_groups": n_groups,
            "strategy": "stratified_per_group",
        }
        return out / "train", out / "val", out / "test", counts

    # Split por indivíduo (grupo) para group_kfold: evita vazamento (train/val/test sem overlap de indivíduos)
    # Caso contrário: split global (random) por imagem.
    if strategy == "group_kfold":
        # Shuffle grupos e dividir por quantidade de grupos (não por fotos)
        shuffled_groups = list(unique_groups)
        random.shuffle(shuffled_groups)
        n_test_g = max(1, int(round(n_groups * test_ratio))) if n_groups >= 2 else 0
        n_val_g = max(0, int(round(n_groups * val_ratio)))
        n_train_g = n_groups - n_test_g - n_val_g
        if n_train_g < 1:
            n_train_g = 1
            n_test_g = max(0, n_groups - n_val_g - 1)
        if n_test_g < 0:
            n_test_g = 0
        test_groups = set(shuffled_groups[:n_test_g]) if n_test_g else set()
        val_groups = set(shuffled_groups[n_test_g : n_test_g + n_val_g]) if n_val_g else set()
        train_groups = set(shuffled_groups[n_test_g + n_val_g :])
        train_f = [(img, lbl) for img, lbl in files if _group_from_stem(img.stem) in train_groups]
        val_f = [(img, lbl) for img, lbl in files if _group_from_stem(img.stem) in val_groups]
        test_f = [(img, lbl) for img, lbl in files if _group_from_stem(img.stem) in test_groups]
        train_val_f = train_f + val_f
        if step_log:
            step_log(
                f"Split por indivíduo (group_kfold): {len(train_groups)} grupos train, {len(val_groups)} val, {len(test_groups)} test. "
                f"Imagens: train={len(train_f)}, val={len(val_f)}, test={len(test_f)}. Sem overlap de indivíduos."
            )
    else:
        # Split global por imagem (comportamento antigo)
        n_test = int(n_total * test_ratio)
        shuffled = list(files)
        random.shuffle(shuffled)
        test_f = shuffled[:n_test]
        train_val_f = shuffled[n_test:]
        train_f = []
        val_f = []

    groups = [_group_from_stem(img.stem) for img, _ in train_val_f]

    if k_folds > 1 and strategy == "group_kfold" and n_groups >= k_folds and len(train_val_f) >= k_folds:
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
                # Augmentation e mosaic apenas em train; val e test (fold) recebem só cópia.
                for split_list, name in [(train_fold, "train"), (val_fold, "val")]:
                    img_out = fold_dir / name / "images"
                    lbl_out = fold_dir / name / "labels"
                    img_out.mkdir(parents=True, exist_ok=True)
                    lbl_out.mkdir(parents=True, exist_ok=True)
                    total = len(split_list)
                    if step_log:
                        step_log(f"Fold {fold_idx + 1}: copiando {total} arquivos para {name}/...")
                    for i, (img, lbl) in enumerate(split_list):
                        shutil.copy2(img, img_out / img.name)
                        shutil.copy2(lbl, lbl_out / lbl.name)
                        if step_log and (i + 1) % 100 == 0:
                            step_log(f"  fold_{fold_idx + 1} {name}: {i + 1}/{total}")
                    if name == "train" and train_augment_copies > 0:
                        offline_transform = get_offline_pose_augmentation()
                        if step_log:
                            step_log(f"Fold {fold_idx + 1}: augmentation em train (todas as planejadas)...")
                        for i, (img, lbl) in enumerate(split_list):
                            _apply_train_augmentation(
                                img_out / img.name,
                                lbl_out / lbl.name,
                                img_out,
                                lbl_out,
                                train_augment_copies,
                                contrast_limit,
                                gaussian_noise_std,
                                transform=offline_transform,
                            )
                            if step_log and (i + 1) % 100 == 0:
                                step_log(f"  fold_{fold_idx + 1} augmentation: {i + 1}/{total}")
                    if name == "train" and mosaic_enabled and len(split_list) >= 4:
                        n_mosaic = len(split_list) // 4
                        shuffled_train = list(split_list)
                        random.shuffle(shuffled_train)
                        if step_log:
                            step_log(f"Fold {fold_idx + 1}: gerando {n_mosaic} mosaic (todo o conjunto)...")
                        for i in range(n_mosaic):
                            group = shuffled_train[i * 4 : (i + 1) * 4]
                            four = []
                            for img_path, lbl_path in group:
                                im = _imread_unicode(img_path)
                                line = lbl_path.read_text(encoding="utf-8").strip()
                                if not line:
                                    break
                                four.append((im, line))
                            if len(four) == 4:
                                _create_mosaic_pose(four, img_out, lbl_out, i + 1, image_size)
                            if step_log and (i + 1) % 100 == 0:
                                step_log(f"  fold_{fold_idx + 1} mosaic: {i + 1}/{n_mosaic}")
                        if step_log:
                            step_log(f"  fold_{fold_idx + 1} mosaic: {n_mosaic}/{n_mosaic} concluído.")
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

    # group_kfold com k_folds <= 1: um único split já por indivíduo (train_f, val_f, test_f já definidos)
    if strategy == "group_kfold" and (len(train_f) + len(val_f) + len(test_f)) > 0:
        if step_log:
            step_log(f"Split único por indivíduo (group_kfold): copiando train={len(train_f)}, val={len(val_f)}, test={len(test_f)}...")
        # Augmentation e mosaic apenas em train; val e test só cópia.
        for split_list, name in [(train_f, "train"), (val_f, "val"), (test_f, "test")]:
            img_out = out / name / "images"
            lbl_out = out / name / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            total = len(split_list)
            if step_log and total:
                step_log(f"Copiando {total} arquivos para {name}/...")
            for i, (img, lbl) in enumerate(split_list):
                shutil.copy2(img, img_out / img.name)
                shutil.copy2(lbl, lbl_out / lbl.name)
                if step_log and (i + 1) % 100 == 0 and total:
                    step_log(f"  {name}: {i + 1}/{total}")
            if name == "train" and train_augment_copies > 0 and split_list:
                offline_transform = get_offline_pose_augmentation()
                if step_log:
                    step_log("Aplicando augmentations em train...")
                for i, (img, lbl) in enumerate(split_list):
                    _apply_train_augmentation(
                        img_out / img.name,
                        lbl_out / lbl.name,
                        img_out,
                        lbl_out,
                        train_augment_copies,
                        contrast_limit,
                        gaussian_noise_std,
                        transform=offline_transform,
                    )
                if step_log:
                    step_log(f"  augmentation: {total}/{total} concluído.")
            if name == "train" and mosaic_enabled and len(split_list) >= 4:
                n_mosaic = len(split_list) // 4
                shuffled_train = list(split_list)
                random.shuffle(shuffled_train)
                if step_log:
                    step_log(f"Gerando {n_mosaic} mosaic em train/...")
                for i in range(n_mosaic):
                    group = shuffled_train[i * 4 : (i + 1) * 4]
                    four = []
                    for img_path, lbl_path in group:
                        im = _imread_unicode(img_path)
                        line = lbl_path.read_text(encoding="utf-8").strip()
                        if not line:
                            break
                        four.append((im, line))
                    if len(four) == 4:
                        _create_mosaic_pose(four, img_out, lbl_out, i + 1, image_size)
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
            step_log(f"data.yaml gerado em {out}. Split por indivíduo (sem vazamento).")
        counts = {
            "n_train": len(train_f),
            "n_val": len(val_f),
            "n_test": len(test_f),
            "n_total": n_total,
            "k_folds": 1,
            "n_groups": n_groups,
            "strategy": "group_kfold",
        }
        return out / "train", out / "val", out / "test", counts

    # Split único por quantidade de imagens (estratégias que não são group_kfold)
    n_train = int(len(train_val_f) * train_ratio)
    n_val = len(train_val_f) - n_train
    train_f = train_val_f[:n_train]
    val_f = train_val_f[n_train:]
    if k_folds <= 1 or n_groups < k_folds:
        if step_log:
            step_log(f"Split único: Train={len(train_f)}, Val={len(val_f)}, Test={len(test_f)}. Copiando...")
        # Augmentation e mosaic apenas em train; val e test só cópia.
        for split_list, name in [(train_f, "train"), (val_f, "val"), (test_f, "test")]:
            img_out = out / name / "images"
            lbl_out = out / name / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            total = len(split_list)
            if step_log:
                step_log(f"Copiando {total} arquivos para {name}/...")
            for i, (img, lbl) in enumerate(split_list):
                shutil.copy2(img, img_out / img.name)
                shutil.copy2(lbl, lbl_out / lbl.name)
                if step_log and (i + 1) % 100 == 0:
                    step_log(f"  {name}: {i + 1}/{total}")
            if step_log:
                step_log(f"  {name}: {total}/{total} concluído.")
            if name == "train" and train_augment_copies > 0:
                offline_transform = get_offline_pose_augmentation()
                if step_log:
                    step_log("Aplicando todas as augmentations planejadas em train...")
                for i, (img, lbl) in enumerate(split_list):
                    _apply_train_augmentation(
                        img_out / img.name,
                        lbl_out / lbl.name,
                        img_out,
                        lbl_out,
                        train_augment_copies,
                        contrast_limit,
                        gaussian_noise_std,
                        transform=offline_transform,
                    )
                    if step_log and (i + 1) % 100 == 0:
                        step_log(f"  augmentation: {i + 1}/{total}")
                if step_log:
                    step_log(f"  augmentation: {total}/{total} concluído.")
            if name == "train" and mosaic_enabled and len(split_list) >= 4:
                n_mosaic = len(split_list) // 4
                shuffled_train = list(split_list)
                random.shuffle(shuffled_train)
                if step_log:
                    step_log(f"Gerando {n_mosaic} imagens mosaic em train/ (todo o conjunto)...")
                for i in range(n_mosaic):
                    group = shuffled_train[i * 4 : (i + 1) * 4]
                    four = []
                    for img_path, lbl_path in group:
                        im = _imread_unicode(img_path)
                        line = lbl_path.read_text(encoding="utf-8").strip()
                        if not line:
                            break
                        four.append((im, line))
                    if len(four) == 4:
                        _create_mosaic_pose(four, img_out, lbl_out, i + 1, image_size)
                if step_log:
                    step_log(f"  mosaic: {n_mosaic} concluído.")
        if step_log:
            step_log("Gerando data.yaml...")
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
            step_log(f"data.yaml gerado em {out}. Concluído.")

    counts = {
        "n_train": len(train_f),
        "n_val": len(val_f),
        "n_test": len(test_f),
        "n_total": n_total,
        "k_folds": 1,
        "n_groups": n_groups,
    }
    return out / "train", out / "val", out / "test", counts
