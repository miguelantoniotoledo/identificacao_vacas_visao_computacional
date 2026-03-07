"""
Converte anotações Label Studio (JSON) para formato YOLO pose.

Associação imagem <-> keypoints: sempre pelo NOME DA IMAGEM (task.data.img), nunca pelo
nome do arquivo de anotação. Os arquivos em Key_points/ têm nomes como 1, 56, 310 e
esses números se repetem entre pastas (cada vaca tem seu "1", "56", etc.). Por isso
a imagem é localizada pelo caminho/nome em task.data.img e os arquivos de saída
(imagem + .txt) são nomeados com base no nome da imagem (ex.: grupo__nome_imagem.jpg).

Formato YOLO pose: class x_center y_center w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
Coordenadas normalizadas 0-1.

Visibilidade (sempre usamos 0, 1 ou 2):
  0 = ausente (ponto não anotado / não presente na imagem)
  1 = oculto (anotado mas não visível, ex.: ocluído)
  2 = visível (anotado e visível)

Pontos não presentes na anotação do Label Studio são emitidos como (0, 0, 0).
A visibilidade é lida do JSON quando existir (value.visibility ou choices); senão, assume 2.
"""

import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.config import get_full_config, get_keypoint_names


# Ordem canônica dos keypoints (Label Studio)
KEYPOINT_ORDER = [
    "withers", "back", "hook up", "hook down", "hip", "tail head", "pin up", "pin down"
]


def _parse_labelstudio_json(path: Path) -> Optional[Dict[str, Any]]:
    """Carrega e valida JSON do Label Studio."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict) or "result" not in data or "task" not in data:
        return None
    return data


def _normalize_image_path_to_filename(img_path_str: str) -> str:
    """
    Extrai o nome do arquivo de imagem a partir do valor em task.data.img.
    Aceita: '/data/upload/1/xxx.jpg' -> 'xxx.jpg';
    '?d=Users%5C...%5Cfilename.jpg' -> decodifica e retorna 'filename.jpg'.
    """
    s = img_path_str.strip()
    if not s:
        return s
    # Decodificar URL (ex.: %20 -> espaço), como no repo referência
    try:
        from urllib.parse import unquote
        s = unquote(s)
    except Exception:
        pass
    # Formato query string (?d=path) comum em exportações Label Studio
    if s.startswith("?d=") or "?d=" in s:
        try:
            from urllib.parse import unquote
            if s.startswith("?d="):
                decoded = unquote(s[3:].strip())
            else:
                idx = s.find("?d=")
                decoded = unquote(s[idx + 3 :].strip())
            for sep in ("\\", "/"):
                if sep in decoded:
                    return decoded.split(sep)[-1]
            return decoded
        except Exception:
            pass
    return Path(s).name


def _candidate_image_names(normalized_name: str) -> List[str]:
    """
    Lista de nomes candidatos para buscar a imagem (como no repo deteccao_keypoints_vacas).
    Inclui o nome normalizado e a versão sem prefixo de hash (ex.: 8 hex chars).
    """
    candidates = [normalized_name]
    # Label Studio pode prefixar com hash curto (ex.: d2766c60-20260101_205224_baia20_IPC2.jpg)
    without_hash = re.sub(r"^[0-9a-fA-F]{8}-", "", normalized_name)
    if without_hash != normalized_name:
        candidates.append(without_hash)
    return candidates


def _get_image_path(data: Dict[str, Any]) -> Optional[str]:
    """Extrai caminho da imagem do task.data."""
    task = data.get("task", {})
    task_data = task.get("data", {}) if isinstance(task.get("data"), dict) else {}
    img = task_data.get("img") or task_data.get("image")
    if isinstance(img, str):
        return img.strip()
    return None


def _visibility_to_yolo(visibility: Any) -> int:
    """
    Converte valor de visibilidade do Label Studio para YOLO (0, 1 ou 2).
    YOLO: 0=ausente, 1=oculto, 2=visível.
    """
    if visibility is None:
        return 2
    if isinstance(visibility, (int, float)):
        v = int(visibility)
        return 2 if v >= 2 else (1 if v == 1 else 0)
    s = str(visibility).lower().strip()
    if s in ("2", "visible", "visível", "visivel", "v", "yes", "true", "1.0"):
        return 2
    if s in ("1", "occluded", "oculto", "hidden", "occluded", "o"):
        return 1
    return 0


def _get_keypoint_visibility_from_result(
    value: Dict[str, Any],
    result: List[Dict],
    keypoint_result_id: Optional[Any] = None,
) -> int:
    """
    Obtém visibilidade do keypoint: primeiro em value.visibility / value.visible,
    depois em choices no result (parentID apontando para keypoint_result_id).
    """
    v = value
    # Campo direto no value (comum em exportações customizadas)
    vis = v.get("visibility") or v.get("visible") or v.get("v")
    if vis is not None:
        return _visibility_to_yolo(vis)
    # Choices: item type=choices com parentID igual ao id do keypoint
    if keypoint_result_id is not None:
        for r in result:
            if not isinstance(r, dict) or r.get("type") != "choices":
                continue
            if str(r.get("parentID")) != str(keypoint_result_id):
                continue
            val = r.get("value") or {}
            choices = val.get("choices", []) if isinstance(val, dict) else []
            if choices:
                choice = choices[0] if isinstance(choices[0], str) else str(choices[0])
                if "visible" in choice.lower() or "visível" in choice.lower() or "visivel" in choice.lower():
                    return 2
                if "occlud" in choice.lower() or "hidden" in choice.lower() or "ocult" in choice.lower():
                    return 1
                return 0
    return 2  # padrão: visível


def _extract_keypoints_and_bbox(
    result: List[Dict],
    img_w: int,
    img_h: int,
) -> Tuple[Optional[Tuple[float, float, float, float]], List[Tuple[float, float, int]]]:
    """
    Extrai bbox (x_center, y_center, w, h normalizados) e lista de (x, y, visibility).
    Visibilidade: 0=ausente, 1=oculto, 2=visível. Pontos não anotados saem como (0, 0, 0).
    """
    bbox = None
    kp_map: Dict[str, Tuple[float, float, int]] = {}

    for r in result:
        if not isinstance(r, dict):
            continue
        t = r.get("type")
        v = r.get("value") or {}
        if not isinstance(v, dict):
            continue

        if t == "rectanglelabels" and "rectanglelabels" in v and "cow" in str(v.get("rectanglelabels", [])):
            x = float(v.get("x", 0))
            y = float(v.get("y", 0))
            w = float(v.get("width", 0))
            h = float(v.get("height", 0))
            x_center = (x + w / 2) / 100.0
            y_center = (y + h / 2) / 100.0
            bbox = (x_center, y_center, w / 100.0, h / 100.0)

        if t == "keypointlabels":
            labels = v.get("keypointlabels", [])
            if labels:
                name = labels[0] if isinstance(labels[0], str) else str(labels[0])
                x = float(v.get("x", 0)) / 100.0
                y = float(v.get("y", 0)) / 100.0
                kp_id = r.get("id")  # para associar choices por parentID
                vis = _get_keypoint_visibility_from_result(v, result, kp_id)
                kp_map[name] = (x, y, vis)

    # Ordenar keypoints na ordem canônica; pontos não presentes ficam (0, 0, 0) com vis=0
    order = get_keypoint_names() or KEYPOINT_ORDER
    keypoints: List[Tuple[float, float, int]] = []
    for name in order:
        norm = name.lower().replace(" ", "_")
        found = None
        for k, val in kp_map.items():
            if k.lower().replace(" ", "_") == norm:
                found = val
                break
        if found is None:
            for k, val in kp_map.items():
                if norm in k.lower().replace(" ", "_") or k.lower().replace(" ", "_") in norm:
                    found = val
                    break
        if found is not None:
            keypoints.append(found)
        else:
            # Ponto não anotado / não presente: YOLO usa (0, 0, 0)
            keypoints.append((0.0, 0.0, 0))

    return bbox, keypoints


def _tokens_from_expected_image_name(name: str) -> List[str]:
    """
    Extrai tokens do nome esperado para casamento flexível.
    Padrões: cow_id_YYYY_MM_DD_HH_MM_SS_cam_station.jpg,
    cam_ID_XX_YYYYMMDDHHMMSS_station_cam.jpg, YYYYMMDD_HHMMSS_station_cam.jpg.
    Retorna lista de partes significativas: YYYYMMDD, HHMMSS, station_id, cam_ID, etc.
    """
    stem = Path(name).stem
    parts = re.split(r"[-_.]+", stem)
    tokens = []
    for p in parts:
        if len(p) < 2:
            continue
        if re.match(r"^\d{8}$", p) or re.match(r"^\d{6}$", p):
            tokens.append(p)
        elif re.search(r"[a-zA-Z]", p) or len(p) >= 4:
            tokens.append(p)
    return tokens


def _stem_matches_expected_tokens(stem_f: str, expected_name: str) -> bool:
    """
    True se stem_f contém os tokens do nome esperado, aceitando:
    - YYYYMMDD ou YYYY_MM_DD
    - HHMMSS ou HH_MM_SS
    - demais ids (station, cam, cow_id).
    """
    tokens = _tokens_from_expected_image_name(expected_name)
    if not tokens:
        return False
    date_8 = None
    time_6 = None
    rest = []
    for t in tokens:
        if re.match(r"^\d{8}$", t):
            date_8 = t
        elif re.match(r"^\d{6}$", t):
            time_6 = t
        else:
            rest.append(t)
    if date_8 and date_8 not in stem_f:
        alt = f"{date_8[0:4]}_{date_8[4:6]}_{date_8[6:8]}"
        if alt not in stem_f:
            return False
    elif date_8:
        pass  # já está no stem
    if time_6 and time_6 not in stem_f:
        alt = f"{time_6[0:2]}_{time_6[2:4]}_{time_6[4:6]}"
        if alt not in stem_f:
            return False
    elif time_6:
        pass
    return all(r in stem_f for r in rest)


def _find_image_file(
    img_path_str: str,
    search_dirs: List[Path],
    search_root: Optional[Path] = None,
) -> Optional[Path]:
    """
    Localiza arquivo de imagem (como no repo deteccao_keypoints_vacas).
    - Usa nomes candidatos: nome normalizado + nome sem prefixo hash (8 hex).
    - Busca na pasta da vaca (search_dirs); depois rglob em search_root (catálogo inteiro).
    - Fallback: casamento por tokens (data, hora, station, câmera) nos padrões de nome.
    """
    name = _normalize_image_path_to_filename(img_path_str)
    stem = Path(name).stem
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    names_to_try = _candidate_image_names(name)

    for nome_candidato in names_to_try:
        stem_cand = Path(nome_candidato).stem
        for d in search_dirs:
            for ext in exts:
                p = d / (stem_cand + ext)
                if p.exists():
                    return p
            for f in d.rglob("*"):
                if f.is_file() and f.suffix.lower() in exts and (f.stem == stem_cand or f.name == nome_candidato):
                    return f
        # Busca global no catálogo (repo referência: rglob no root)
        if search_root and search_root.exists():
            found = list(search_root.rglob(nome_candidato))
            if found and found[0].is_file():
                return found[0]
            for ext in exts:
                stem_c = Path(nome_candidato).stem
                found = list(search_root.rglob(stem_c + ext))
                if found and found[0].is_file():
                    return found[0]

    # Fallback: casamento por tokens (data, hora, station_id, cam_ID)
    candidates: List[Path] = []
    for d in search_dirs:
        for f in d.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in exts:
                continue
            if _stem_matches_expected_tokens(f.stem, name):
                candidates.append(f)
    if search_root and search_root.exists():
        for f in search_root.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in exts:
                continue
            if _stem_matches_expected_tokens(f.stem, name):
                candidates.append(f)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        candidates.sort(key=lambda p: abs(len(p.stem) - len(stem)))
        return candidates[0]
    return None


def get_conversion_failure_reason(
    json_path: Path,
    search_dirs: List[Path],
    search_root: Optional[Path] = None,
) -> str:
    """
    Retorna o motivo da falha na conversão (para debug).
    Não modifica arquivos; apenas lê o JSON e tenta identificar o passo que falharia.
    """
    data = _parse_labelstudio_json(json_path)
    if not data:
        return "JSON inválido ou sem result/task"

    img_path_str = _get_image_path(data)
    if not img_path_str:
        return "Sem caminho da imagem em task.data (task.data.img ou task.data.image)"

    img_file = _find_image_file(img_path_str, search_dirs, search_root)
    if not img_file or not img_file.exists():
        nome_esperado = _normalize_image_path_to_filename(img_path_str)
        return f"Imagem não encontrada na pasta da vaca (esperado: {nome_esperado})"

    result = data.get("result", [])
    if not result:
        return "Anotação sem result (vazio)"

    w, h = 640, 640
    for r in result:
        if isinstance(r, dict) and "original_width" in r:
            w = int(r.get("original_width", w))
            h = int(r.get("original_height", h))
            break

    bbox, keypoints = _extract_keypoints_and_bbox(result, w, h)
    if not bbox or not keypoints:
        return "Sem bbox ou keypoints válidos na anotação (ex.: rectanglelabels cow + keypointlabels)"

    return "OK"


def convert_single_annotation(
    json_path: Path,
    images_out: Path,
    labels_out: Path,
    search_dirs: List[Path],
    group_prefix: Optional[str] = None,
    search_root: Optional[Path] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Converte um JSON do Label Studio em arquivo YOLO pose e copia a imagem.

    A associação é sempre pelo nome da imagem (task.data.img): localizamos a imagem
    por esse nome e gravamos saída como grupo__nome_imagem.jpg e grupo__nome_imagem.txt.
    O nome do arquivo de anotação (json_path, ex.: "1" ou "56") não é usado para
    nomear a saída, pois se repete entre pastas de vacas.

    search_root: se informado, busca a imagem também em todo o catálogo (rglob).
    Retorna (True, nome_imagem_saida) se sucesso, (False, None) se falha.
    """
    data = _parse_labelstudio_json(json_path)
    if not data:
        return False, None

    img_path_str = _get_image_path(data)
    if not img_path_str:
        return False, None

    img_file = _find_image_file(img_path_str, search_dirs, search_root)
    if not img_file or not img_file.exists():
        return False, None

    result = data.get("result", [])
    if not result:
        return False, None

    # Dimensões da imagem (do primeiro elemento com original_width/height)
    w, h = 640, 640
    for r in result:
        if isinstance(r, dict) and "original_width" in r:
            w = int(r.get("original_width", w))
            h = int(r.get("original_height", h))
            break

    bbox, keypoints = _extract_keypoints_and_bbox(result, w, h)
    if not bbox or not keypoints:
        return False, None

    # Formato YOLO pose: class x_center y_center w h kp1_x kp1_y kp1_v ...
    parts = ["0", f"{bbox[0]:.6f}", f"{bbox[1]:.6f}", f"{bbox[2]:.6f}", f"{bbox[3]:.6f}"]
    for kx, ky, kv in keypoints:
        parts.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])
    line = " ".join(parts)

    # Nome da saída sempre pelo nome da IMAGEM (nunca pelo nome do arquivo de anotação)
    base_name = f"{group_prefix}__{img_file.name}" if group_prefix else img_file.name
    dest_img = images_out / base_name
    if not dest_img.exists() or dest_img.stat().st_mtime < img_file.stat().st_mtime:
        shutil.copy2(img_file, dest_img)

    label_name = dest_img.stem + ".txt"
    label_path = labels_out / label_name
    label_path.write_text(line, encoding="utf-8")

    return True, base_name


def convert_labelstudio_to_yolo_pose(
    raw_dir: Optional[Path] = None,
    unified_dir: Optional[Path] = None,
    step_log: Optional[Callable[[str], None]] = None,
    collect_failure_reasons: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Converte todos os JSONs de Key_points para YOLO pose em data/unified/keypoints.

    Cada arquivo em Key_points/ (ex.: 1, 56) contém uma anotação; a imagem associada
    vem de task.data.img no JSON. A saída é nomeada pelo nome da imagem (nunca pelo
    número do arquivo de anotação, que se repete entre pastas).

    step_log(mensagem) é chamado para cada anotação.
    Se collect_failure_reasons=N, retorna em ["failure_reasons"] as primeiras N falhas (debug).
    """
    cfg = get_full_config()
    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    root = Path(__file__).resolve().parents[2]
    raw = raw_dir or root / paths_cfg.get("raw_dir", "raw")
    unified = unified_dir or root / paths_cfg.get("unified_dir", "data/unified")
    catalogo = raw / data_cfg.get("catalogo_subdir", "catalogo")
    kp_subdir = data_cfg.get("keypoints_subdir", "Key_points")

    images_out = unified / "keypoints" / "images"
    labels_out = unified / "keypoints" / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    converted = 0
    failed = 0
    failure_reasons: List[Tuple[str, str]] = []
    max_reasons = collect_failure_reasons if isinstance(collect_failure_reasons, int) else 0

    if not catalogo.exists():
        return {"converted": 0, "failed": 0, "failure_reasons": []}

    for cow_folder in catalogo.iterdir():
        if not cow_folder.is_dir():
            continue
        kp_folder = cow_folder / kp_subdir
        if not kp_folder.exists():
            continue
        search_dirs = [cow_folder]
        group_prefix = cow_folder.name
        for ann in kp_folder.iterdir():
            if ann.is_file():
                ok, out_name = convert_single_annotation(
                    ann, images_out, labels_out, search_dirs,
                    group_prefix=group_prefix,
                    search_root=catalogo,
                )
                if ok:
                    converted += 1
                    if step_log:
                        step_log(f"Convertendo imagem {out_name} em YOLO. Resultado: sucesso")
                else:
                    failed += 1
                    reason = get_conversion_failure_reason(ann, search_dirs, catalogo)
                    if step_log:
                        step_log(f"Convertendo anotação {ann.name}. Resultado: falha ({reason})")
                    if len(failure_reasons) < max_reasons:
                        try:
                            rel = str(ann.relative_to(catalogo))
                        except ValueError:
                            rel = f"{group_prefix}/{ann.name}"
                        failure_reasons.append((rel, reason))

    out = {"converted": converted, "failed": failed}
    if max_reasons > 0:
        out["failure_reasons"] = failure_reasons
    return out
