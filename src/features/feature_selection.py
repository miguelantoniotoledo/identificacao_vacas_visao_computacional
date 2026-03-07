"""
Seleção de features para keypoints e embeddings.

Oferece métodos: mutual_info, rf_importance (Random Forest), PCA.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.config import get_full_config, get_keypoint_names


_DEFAULT_KEYPOINT_NAMES = [
    "withers",
    "back",
    "hook_up",
    "hook_down",
    "hip",
    "tail_head",
    "pin_up",
    "pin_down",
]


def _load_keypoints_and_images(
    labels_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, List[Optional[Path]]]:
    """
    Carrega coordenadas dos keypoints dos arquivos YOLO e resolve o caminho
    das imagens correspondentes (por stem igual).

    Retorna (X, y, image_paths).
    """
    cfg = get_full_config()
    root = Path(__file__).resolve().parents[2]
    unified = root / cfg.get("paths", {}).get("unified_dir", "data/unified")
    images_dir = unified / "keypoints" / "images"

    rows = []
    image_paths: List[Optional[Path]] = []
    # y é placeholder (0) até termos IDs de vaca associados
    y_vals: List[int] = []

    for lbl in sorted(labels_dir.glob("*.txt")):
        line = lbl.read_text().strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        # class x_center y_center w h kp1_x kp1_y kp1_v ...
        try:
            vals = [float(x) for x in parts[5:]]  # só keypoints
            if len(vals) < 24:  # 8 keypoints * 3
                continue
        except ValueError:
            continue

        rows.append(vals[:24])
        y_vals.append(0)

        img_path: Optional[Path] = None
        if images_dir.exists():
            stem = lbl.stem
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                cand = images_dir / f"{stem}{ext}"
                if cand.exists():
                    img_path = cand
                    break
        image_paths.append(img_path)

    if not rows:
        return np.array([]).reshape(0, 0), np.array([]), []

    X = np.array(rows, dtype=float)
    y = np.array(y_vals, dtype=int)
    return X, y, image_paths


def _load_keypoint_data(labels_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega coordenadas dos keypoints dos arquivos YOLO.
    Retorna (X, y) onde X são as features (keypoints) e y é o label (ex: id da vaca).
    """
    X, y, _ = _load_keypoints_and_images(labels_dir)
    return X, y


def _get_kp_names() -> List[str]:
    """Retorna a lista de nomes de keypoints a partir do config ou fallback."""
    names = get_keypoint_names()
    if names:
        return list(names)
    return list(_DEFAULT_KEYPOINT_NAMES)


def _build_geometric_feature_matrix(
    X: np.ndarray,
    kp_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    A partir das coordenadas (x, y, v) dos keypoints, gera features geométricas:
    - comprimentos normalizados de segmentos
    - razões entre comprimentos
    - ângulos (normalizados) entre segmentos
    """
    if X.size == 0:
        return np.empty((0, 0)), []

    n_samples, n_vals = X.shape
    n_kp = n_vals // 3
    if n_kp == 0:
        return np.empty((0, 0)), []

    kp_names = list(kp_names) or _get_kp_names()
    if len(kp_names) != n_kp:
        kp_names = (kp_names + _DEFAULT_KEYPOINT_NAMES)[:n_kp]

    name_to_idx = {name: i for i, name in enumerate(kp_names)}

    def _has(*names: str) -> bool:
        return all(n in name_to_idx for n in names)

    # Definição de segmentos principais
    segment_defs: List[Tuple[str, str]] = []
    for a, b in [
        ("withers", "back"),
        ("back", "hip"),
        ("hip", "tail_head"),
        ("tail_head", "pin_up"),
        ("pin_up", "pin_down"),
    ]:
        if _has(a, b):
            segment_defs.append((a, b))

    # Segmento de referência para normalização de escala
    base_pair: Optional[Tuple[str, str]] = None
    for cand in [("withers", "hip"), ("back", "hip")]:
        if _has(*cand):
            base_pair = cand
            break

    # Definição de ângulos (centro, p1, p2)
    angle_defs: List[Tuple[str, str, str]] = []
    for c, a, b in [
        ("back", "withers", "hip"),
        ("hip", "back", "tail_head"),
        ("pin_up", "hip", "pin_down"),
    ]:
        if _has(c, a, b):
            angle_defs.append((c, a, b))

    if not segment_defs and not angle_defs:
        return np.empty((0, 0)), []

    eps = 1e-6
    feature_names: List[str] = []
    feats_all: List[List[float]] = []

    for i in range(n_samples):
        row = X[i]
        kps3 = row.reshape(n_kp, 3)
        kps = kps3[:, :2]  # (n_kp, 2)
        vis = kps3[:, 2]

        # Comprimentos de segmentos
        seg_lengths = {}
        for a, b in segment_defs:
            ia, ib = name_to_idx[a], name_to_idx[b]
            va = kps[ia]
            vb = kps[ib]
            seg_lengths[(a, b)] = float(np.linalg.norm(va - vb))

        # Comprimento base
        base_len = 1.0
        if base_pair is not None and base_pair in seg_lengths:
            base_len = max(seg_lengths[base_pair], eps)

        feats: List[float] = []
        names_local: List[str] = []

        # 1) Comprimentos normalizados
        for a, b in segment_defs:
            l = seg_lengths.get((a, b), 0.0)
            feats.append(l / base_len if base_len > eps else 0.0)
            names_local.append(f"len_norm_{a}_{b}")

        # 2) Razões entre segmentos adjacentes
        for (a1, b1), (a2, b2) in zip(segment_defs, segment_defs[1:]):
            l1 = seg_lengths.get((a1, b1), 0.0)
            l2 = seg_lengths.get((a2, b2), 0.0)
            feats.append(l1 / (l2 + eps))
            names_local.append(f"ratio_len_{a1}_{b1}_over_{a2}_{b2}")

        # 3) Ângulos normalizados (0–1, onde 1 ~ 180 graus)
        for c, a, b in angle_defs:
            ic, ia, ib = name_to_idx[c], name_to_idx[a], name_to_idx[b]
            vc = kps[ic]
            v1 = kps[ia] - vc
            v2 = kps[ib] - vc
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < eps or n2 < eps:
                theta_norm = 0.0
            else:
                cos_theta = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                theta = float(np.arccos(cos_theta))
                theta_norm = theta / np.pi
            feats.append(theta_norm)
            names_local.append(f"angle_{c}_{a}_{b}")

        # 4) Distância ao centróide (assinatura de forma global, sem escala)
        visible_mask = vis > 0
        if visible_mask.any():
            centroid = kps[visible_mask].mean(axis=0)
            dists_cent = np.linalg.norm(kps - centroid, axis=1)
            # Normalizar por comprimento base para remover escala
            dists_cent_norm = dists_cent / (base_len + eps)
        else:
            dists_cent_norm = np.zeros(n_kp, dtype=float)

        for idx, name in enumerate(kp_names):
            feats.append(float(dists_cent_norm[idx]))
            names_local.append(f"centroid_dist_norm_{name}")

        # 5) Assinatura geométrica: matriz de distâncias entre todos os pares,
        # normalizada pelo "comprimento do animal"
        body_pair: Optional[Tuple[str, str]] = None
        for cand in [("withers", "tail_head"), ("back", "tail_head"), ("withers", "pin_down")]:
            if _has(*cand):
                body_pair = cand
                break

        body_len = 1.0
        if body_pair is not None:
            ia, ib = name_to_idx[body_pair[0]], name_to_idx[body_pair[1]]
            body_len = float(np.linalg.norm(kps[ia] - kps[ib]))
        body_len = max(body_len, eps)

        # Feature explícita de comprimento (em coordenadas normalizadas da imagem)
        feats.append(body_len)
        names_local.append("sig_body_length")

        # Matriz de distâncias normalizada por body_len (assinatura geométrica)
        for ia in range(n_kp):
            for ib in range(ia + 1, n_kp):
                dij = float(np.linalg.norm(kps[ia] - kps[ib]) / body_len)
                feats.append(dij)
                names_local.append(f"sig_dist_{kp_names[ia]}_{kp_names[ib]}")

        if not feature_names:
            feature_names = names_local
        feats_all.append(feats)

    if not feats_all:
        return np.empty((0, 0)), []

    F = np.array(feats_all, dtype=float)
    return F, feature_names


def _build_texture_color_feature_matrix(
    X: np.ndarray,
    image_paths: List[Optional[Path]],
    kp_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Gera features de textura e cor ao redor de cada keypoint,
    a partir das imagens originais.

    Para cada keypoint:
      - média e desvio padrão da intensidade em tons de cinza
      - variância do Laplaciano (medida de textura/detalhe)
      - média de cor (R, G, B) normalizada
    """
    if X.size == 0 or not image_paths:
        return np.empty((0, 0)), []

    try:
        import cv2  # type: ignore
    except Exception:
        return np.empty((0, 0)), []

    n_samples, n_vals = X.shape
    n_kp = n_vals // 3
    if n_kp == 0:
        return np.empty((0, 0)), []

    kp_names = list(kp_names) or _get_kp_names()
    if len(kp_names) != n_kp:
        kp_names = (kp_names + _DEFAULT_KEYPOINT_NAMES)[:n_kp]

    feats_all: List[List[float]] = []
    feature_names: List[str] = []

    for i in range(n_samples):
        row = X[i]
        img_path = image_paths[i] if i < len(image_paths) else None

        # Layout por padrão: zeros
        feats: List[float] = []
        names_local: List[str] = []

        # Carrega imagem (se existir)
        img = None
        if img_path is not None and img_path.exists():
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if img is None:
            # Sem imagem: todas as features ficam 0
            for name in kp_names:
                names_local.extend(
                    [
                        f"tex_{name}_mean_gray",
                        f"tex_{name}_std_gray",
                        f"tex_{name}_lap_var",
                        f"tex_{name}_mean_r",
                        f"tex_{name}_mean_g",
                        f"tex_{name}_mean_b",
                    ]
                )
                feats.extend([0.0] * 6)
            if not feature_names:
                feature_names = names_local
            feats_all.append(feats)
            continue

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kps = row.reshape(n_kp, 3)  # (x, y, v)

        # Tamanho da janela: ~3% do menor lado, mínimo 4 px
        win_radius = max(4, int(0.03 * min(w, h)))

        for ki, name in enumerate(kp_names):
            x_norm, y_norm, v = kps[ki]
            # Se visibilidade for 0 (ausente), zera as features
            if v <= 0:
                feats.extend([0.0] * 6)
                names_local.extend(
                    [
                        f"tex_{name}_mean_gray",
                        f"tex_{name}_std_gray",
                        f"tex_{name}_lap_var",
                        f"tex_{name}_mean_r",
                        f"tex_{name}_mean_g",
                        f"tex_{name}_mean_b",
                    ]
                )
                continue

            x_pix = int(round(float(x_norm) * w))
            y_pix = int(round(float(y_norm) * h))

            x0 = max(0, x_pix - win_radius)
            x1 = min(w, x_pix + win_radius + 1)
            y0 = max(0, y_pix - win_radius)
            y1 = min(h, y_pix + win_radius + 1)

            if x1 <= x0 or y1 <= y0:
                feats.extend([0.0] * 6)
                names_local.extend(
                    [
                        f"tex_{name}_mean_gray",
                        f"tex_{name}_std_gray",
                        f"tex_{name}_lap_var",
                        f"tex_{name}_mean_r",
                        f"tex_{name}_mean_g",
                        f"tex_{name}_mean_b",
                    ]
                )
                continue

            patch_gray = gray[y0:y1, x0:x1]
            patch_color = img[y0:y1, x0:x1, :]

            if patch_gray.size == 0 or patch_color.size == 0:
                feats.extend([0.0] * 6)
                names_local.extend(
                    [
                        f"tex_{name}_mean_gray",
                        f"tex_{name}_std_gray",
                        f"tex_{name}_lap_var",
                        f"tex_{name}_mean_r",
                        f"tex_{name}_mean_g",
                        f"tex_{name}_mean_b",
                    ]
                )
                continue

            # Normalizar por 255 para manter escala em [0, 1]
            mean_gray = float(patch_gray.mean()) / 255.0
            std_gray = float(patch_gray.std()) / 255.0
            lap_var = float(cv2.Laplacian(patch_gray, cv2.CV_64F).var()) / (255.0**2)

            mean_bgr = patch_color.reshape(-1, 3).mean(axis=0) / 255.0
            mean_b, mean_g, mean_r = [float(x) for x in mean_bgr]

            feats.extend([mean_gray, std_gray, lap_var, mean_r, mean_g, mean_b])
            names_local.extend(
                [
                    f"tex_{name}_mean_gray",
                    f"tex_{name}_std_gray",
                    f"tex_{name}_lap_var",
                    f"tex_{name}_mean_r",
                    f"tex_{name}_mean_g",
                    f"tex_{name}_mean_b",
                ]
            )

        if not feature_names:
            feature_names = names_local
        feats_all.append(feats)

    if not feats_all:
        return np.empty((0, 0)), []

    F = np.array(feats_all, dtype=float)
    return F, feature_names


def compute_keypoint_importance(
    labels_dir: Path,
    method: str = "mutual_info",
    random_state: int = 42,
) -> List[Tuple[str, float]]:
    """
    Calcula importância de cada feature derivada dos keypoints:
    - geométricas (comprimentos, razões, ângulos)
    - textura e cor ao redor dos pontos.

    Retorna lista de (nome_da_feature, score) ordenada por score decrescente.
    """
    X_raw, y, img_paths = _load_keypoints_and_images(labels_dir)
    if X_raw.size == 0:
        return []

    kp_names = _get_kp_names()
    F_geom, names_geom = _build_geometric_feature_matrix(X_raw, kp_names)
    F_tex, names_tex = _build_texture_color_feature_matrix(X_raw, img_paths, kp_names)

    if F_geom.size == 0 and F_tex.size == 0:
        return []
    if F_geom.size == 0:
        F = F_tex
        feat_names = names_tex
    elif F_tex.size == 0:
        F = F_geom
        feat_names = names_geom
    else:
        F = np.concatenate([F_geom, F_tex], axis=1)
        feat_names = names_geom + names_tex

    cfg = get_full_config()
    fs_cfg = cfg.get("feature_selection", {})
    method = fs_cfg.get("method", method)

    # Se não houver múltiplas classes em y, caímos em variância como proxy
    try:
        n_classes = len(np.unique(y))
    except Exception:
        n_classes = 1

    if method == "mutual_info" and n_classes > 1:
        try:
            from sklearn.feature_selection import mutual_info_classif

            mi = mutual_info_classif(F, y, random_state=random_state)
            scores_arr = mi
        except Exception:
            scores_arr = np.var(F, axis=0)
    elif method == "rf_importance" and n_classes > 1:
        try:
            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(n_estimators=50, random_state=random_state)
            rf.fit(F, y)
            scores_arr = rf.feature_importances_
        except Exception:
            scores_arr = np.var(F, axis=0)
    else:
        scores_arr = np.var(F, axis=0)

    scores = [(feat_names[i], float(scores_arr[i])) for i in range(len(feat_names))]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def get_top_k_for_training() -> int:
    """Retorna o número de melhores features a usar no treino (config feature_selection.top_k_for_training)."""
    cfg = get_full_config()
    return int(cfg.get("feature_selection", {}).get("top_k_for_training", 15))


def select_top_keypoints(
    labels_dir: Optional[Path] = None,
    top_k: Optional[int] = None,
    for_training: bool = False,
) -> List[str]:
    """
    Retorna os nomes das top_k features mais importantes (geométricas + textura).
    Se for_training=True, usa config top_k_for_training (ex.: 15).
    """
    cfg = get_full_config()
    root = Path(__file__).resolve().parents[2]
    unified = root / cfg.get("paths", {}).get("unified_dir", "data/unified")
    labels_dir = labels_dir or unified / "keypoints" / "labels"
    if not labels_dir.exists():
        return []

    fs_cfg = cfg.get("feature_selection", {})
    if top_k is None and for_training:
        top_k = fs_cfg.get("top_k_for_training", 15)
    top_k = top_k or fs_cfg.get("top_k", 5)
    scores = compute_keypoint_importance(labels_dir)
    return [name for name, _ in scores[:top_k]]


def build_feature_matrix_for_training(
    labels_dir: Optional[Path] = None,
    top_k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Matriz de features com apenas as top_k melhores (para treino).
    Usa top_k_for_training do config (default 15) quando top_k não é informado.
    Retorna (X, y, feature_names).
    """
    cfg = get_full_config()
    root = Path(__file__).resolve().parents[2]
    unified = root / cfg.get("paths", {}).get("unified_dir", "data/unified")
    labels_dir = labels_dir or unified / "keypoints" / "labels"
    if not labels_dir.exists():
        return np.array([]).reshape(0, 0), np.array([]), []

    k = top_k if top_k is not None else get_top_k_for_training()
    scores = compute_keypoint_importance(labels_dir)
    if not scores:
        return np.array([]).reshape(0, 0), np.array([]), []

    top_names = [name for name, _ in scores[:k]]
    X_raw, y, img_paths = _load_keypoints_and_images(labels_dir)
    if X_raw.size == 0:
        return np.array([]).reshape(0, 0), np.array([]), []

    kp_names = _get_kp_names()
    F_geom, names_geom = _build_geometric_feature_matrix(X_raw, kp_names)
    F_tex, names_tex = _build_texture_color_feature_matrix(X_raw, img_paths, kp_names)
    if F_geom.size == 0 and F_tex.size == 0:
        return np.array([]).reshape(0, 0), np.array([]), []
    if F_geom.size == 0:
        F, feat_names = F_tex, names_tex
    elif F_tex.size == 0:
        F, feat_names = F_geom, names_geom
    else:
        F = np.concatenate([F_geom, F_tex], axis=1)
        feat_names = names_geom + names_tex

    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    indices = []
    for name in top_names:
        if name in name_to_idx:
            indices.append(name_to_idx[name])
    if not indices:
        return np.array([]).reshape(0, 0), np.array([]), []
    F_top = F[:, indices]
    names_top = [feat_names[i] for i in indices]
    return F_top, y, names_top


def compute_feature_correlations(
    labels_dir: Optional[Path] = None,
    min_abs_corr: float = 0.0,
) -> List[Tuple[str, str, float]]:
    """
    Analisa a correlação entre todas as features derivadas dos keypoints
    (geométricas + textura/cor).

    Retorna lista de (feature_i, feature_j, correlação_pearson),
    ordenada por |correlação| decrescente e filtrada por min_abs_corr.
    """
    cfg = get_full_config()
    root = Path(__file__).resolve().parents[2]
    unified = root / cfg.get("paths", {}).get("unified_dir", "data/unified")
    labels_dir = labels_dir or unified / "keypoints" / "labels"

    if not labels_dir.exists():
        return []

    X_raw, _, img_paths = _load_keypoints_and_images(labels_dir)
    if X_raw.size == 0 or X_raw.shape[0] < 2:
        return []

    kp_names = _get_kp_names()
    F_geom, names_geom = _build_geometric_feature_matrix(X_raw, kp_names)
    F_tex, names_tex = _build_texture_color_feature_matrix(X_raw, img_paths, kp_names)

    if F_geom.size == 0 and F_tex.size == 0:
        return []
    if F_geom.size == 0:
        F = F_tex
        feat_names = names_tex
    elif F_tex.size == 0:
        F = F_geom
        feat_names = names_geom
    else:
        F = np.concatenate([F_geom, F_tex], axis=1)
        feat_names = names_geom + names_tex

    if F.shape[0] < 2:
        return []

    C = np.corrcoef(F, rowvar=False)
    n_feat = C.shape[0]

    pairs: List[Tuple[str, str, float]] = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            corr = float(C[i, j])
            if abs(corr) >= min_abs_corr:
                pairs.append((feat_names[i], feat_names[j], corr))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs
