#!/usr/bin/env python
"""
Verifica se a unificação e conversão (unify_and_convert) foram feitas corretamente.

Checa:
- Existência das pastas data/unified/keypoints/images e labels
- Mesmo número de imagens e labels
- Cada imagem tem um .txt com mesmo stem (nome sem extensão)
- Formato do label: classe, bbox (4), 8 keypoints x 3 = 24 valores (total 5+24=29 números)
- Opcional: compara com métricas do último run (converted/failed)
- Opcional: plota N imagens com bbox e keypoints (--plot N)

Uso:
  python scripts/verificar_unify_convert.py
  python scripts/verificar_unify_convert.py --amostras 5   # mostra conteúdo de 5 labels
  python scripts/verificar_unify_convert.py --plot 3       # salva N imagens com bbox e keypoints
  python scripts/verificar_unify_convert.py --image "caminho/para/imagem.jpg"  # desenha anotações originais em UMA imagem (label em data/unified/keypoints/labels pelo nome do arquivo)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_full_config, get_keypoint_names

# Para desenhar keypoints e segmentos (mesmo esquema do visualize_keypoints)
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Nomes e segmentos padrão (alinhado com visualize_keypoints)
DEFAULT_KP_NAMES = [
    "withers", "back", "hook_up", "hook_down", "hip", "tail_head", "pin_up", "pin_down",
]
DEFAULT_SEGMENTS: List[Tuple[str, str]] = [
    ("withers", "back"),
    ("back", "hip"),
    ("hip", "tail_head"),
    ("tail_head", "pin_up"),
    ("pin_up", "pin_down"),
    ("pin_down", "tail_head"),   # fecha traseiro
    ("hook_up", "hook_down"),    # ligação entre hooks
]


def _get_kp_names() -> List[str]:
    names = get_keypoint_names()
    if names:
        return list(names)
    return list(DEFAULT_KP_NAMES)


def _plot_imagem_bbox_keypoints(
    img_path: Path,
    label_path: Path,
    out_path: Path,
    kp_names: Sequence[str],
    segments: Sequence[Tuple[str, str]],
) -> bool:
    """Desenha bbox e keypoints/segmentos na imagem e salva em out_path."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]

    line = label_path.read_text(encoding="utf-8").strip()
    parts = line.split()
    if len(parts) < 5 + 8 * 3:
        return False
    vals = [float(x) for x in parts[: 5 + 8 * 3]]
    # Bbox YOLO: classe, cx, cy, w, h (normalizados 0-1)
    cx, cy, bw, bh = vals[1], vals[2], vals[3], vals[4]
    x_center = cx * w
    y_center = cy * h
    box_w = bw * w
    box_h = bh * h
    x1 = int(round(x_center - box_w / 2))
    y1 = int(round(y_center - box_h / 2))
    x2 = int(round(x_center + box_w / 2))
    y2 = int(round(y_center + box_h / 2))
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 165, 0), 2)  # laranja

    kps = np.array(vals[5 : 5 + 8 * 3], dtype=float).reshape(8, 3)
    name_to_idx = {name: i for i, name in enumerate(kp_names)}

    for name, (x_norm, y_norm, v) in zip(kp_names, kps):
        if v <= 0:
            continue
        x = int(round(float(x_norm) * w))
        y = int(round(float(y_norm) * h))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, name, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    for a, b in segments:
        if a not in name_to_idx or b not in name_to_idx:
            continue
        ia, ib = name_to_idx[a], name_to_idx[b]
        xa, ya, va = kps[ia]
        xb, yb, vb = kps[ib]
        if va <= 0 or vb <= 0:
            continue
        xa_pix = int(round(float(xa) * w))
        ya_pix = int(round(float(ya) * h))
        xb_pix = int(round(float(xb) * w))
        yb_pix = int(round(float(yb) * h))
        cv2.line(img, (xa_pix, ya_pix), (xb_pix, yb_pix), (0, 255, 0), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(out_path), img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verificar resultado da unificação e conversão.")
    parser.add_argument(
        "--amostras",
        type=int,
        default=0,
        metavar="N",
        help="Mostrar conteúdo dos primeiros N arquivos de label (0 = não mostrar).",
    )
    parser.add_argument(
        "--plot",
        type=int,
        default=0,
        metavar="N",
        help="Plota N imagens com bbox e keypoints; salva em outputs/statistics/verificar_unify_convert/ (0 = não plotar).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        metavar="PATH",
        help="Uma única imagem para desenhar as anotações originais (keypoints que você anotou). O label é buscado em data/unified/keypoints/labels pelo nome do arquivo (stem). Pode ser caminho para imagem em raw ou em data/unified/keypoints/images.",
    )
    args = parser.parse_args()

    cfg = get_full_config()
    root = Path(__file__).resolve().parents[1]
    unified = root / cfg.get("paths", {}).get("unified_dir", "data/unified")
    images_dir = unified / "keypoints" / "images"
    labels_dir = unified / "keypoints" / "labels"

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ok = True

    print("=== Verificação: unify_and_convert ===\n")

    # 1. Pastas existem?
    if not images_dir.exists():
        print(f"  [FALHA] Pasta de imagens não existe: {images_dir}")
        ok = False
    else:
        print(f"  [OK] Pasta de imagens: {images_dir}")

    if not labels_dir.exists():
        print(f"  [FALHA] Pasta de labels não existe: {labels_dir}")
        ok = False
    else:
        print(f"  [OK] Pasta de labels: {labels_dir}")

    if not ok:
        print("\n  Rode antes: python scripts/unify_and_convert.py")
        sys.exit(1)

    # 2. Contagem
    images = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in exts]
    labels = [f for f in labels_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
    n_img = len(images)
    n_lbl = len(labels)

    print(f"\n  Total de imagens: {n_img}")
    print(f"  Total de labels:  {n_lbl}")

    if n_img != n_lbl:
        print(f"  [AVISO] Número de imagens e labels difere. Esperado igual.")
        ok = False
    else:
        print("  [OK] Quantidades batem.")

    # 3. Cada imagem tem label com mesmo stem?
    stems_img = {f.stem for f in images}
    stems_lbl = {f.stem for f in labels}
    sem_label = stems_img - stems_lbl
    sem_imagem = stems_lbl - stems_img

    if sem_label:
        print(f"  [AVISO] {len(sem_label)} imagens sem label correspondente (ex.: {list(sem_label)[:3]})")
        ok = False
    if sem_imagem:
        print(f"  [AVISO] {len(sem_imagem)} labels sem imagem correspondente (ex.: {list(sem_imagem)[:3]})")
        ok = False
    if not sem_label and not sem_imagem:
        print("  [OK] Cada imagem tem um label com o mesmo nome (stem).")

    # 4. Formato do label: 1 classe + 4 bbox + 8*3 keypoints = 29 números
    esperado_valores = 5 + 8 * 3  # class cx cy w h + 8*(x,y,v)
    erros_formato = 0
    amostras_mostradas = 0

    for lbl in sorted(labels_dir.glob("*.txt"))[: (args.amostras if args.amostras > 0 else n_lbl)]:
        try:
            line = lbl.read_text(encoding="utf-8").strip()
            parts = line.split()
            if len(parts) < esperado_valores:
                erros_formato += 1
                if erros_formato <= 3:
                    print(f"  [FALHA] Label com poucos valores: {lbl.name} ({len(parts)} valores)")
                continue
            vals = [float(x) for x in parts[:esperado_valores]]
            if args.amostras > 0 and amostras_mostradas < args.amostras:
                print(f"\n  Amostra: {lbl.name}")
                print(f"    classe={int(vals[0])} bbox(cx,cy,w,h)={vals[1]:.4f} {vals[2]:.4f} {vals[3]:.4f} {vals[4]:.4f}")
                print(f"    keypoints (x y v)... 8 pontos, 24 valores")
                amostras_mostradas += 1
        except Exception as e:
            erros_formato += 1
            if erros_formato <= 3:
                print(f"  [FALHA] Erro ao ler {lbl.name}: {e}")

    if erros_formato > 0:
        print(f"\n  [AVISO] {erros_formato} label(s) com formato inválido ou incompleto.")
        ok = False
    else:
        print("  [OK] Formato dos labels (classe + bbox + 8 keypoints x,y,v) está correto.")

    # 5. Comparar com último run
    try:
        import json
        latest = root / cfg.get("paths", {}).get("statistics_dir", "outputs/statistics") / "unify_and_convert_latest.json"
        if latest.exists():
            data = json.loads(latest.read_text(encoding="utf-8"))
            m = data.get("metrics", {})
            conv = m.get("converted", 0)
            fail = m.get("failed", 0)
            print(f"\n  Último run: convertidos={conv}, falhas={fail}")
            if conv > 0 and n_img != conv:
                print(f"  [AVISO] Total de imagens atuais ({n_img}) diferente do último 'converted' ({conv}).")
    except Exception:
        pass

    # 6. Uma imagem específica: desenhar anotações originais (--image)
    if args.image and HAS_CV2:
        root = Path(__file__).resolve().parents[1]
        img_path = Path(args.image)
        if not img_path.is_absolute():
            img_path = (root / img_path).resolve()
        if not img_path.exists():
            img_path = images_dir / Path(args.image).name
        if not img_path.exists():
            print(f"  [FALHA] Imagem não encontrada: {args.image}")
        else:
            stem = img_path.stem
            lbl_path = labels_dir / f"{stem}.txt"
            if not lbl_path.exists():
                print(f"  [FALHA] Label não encontrado para esta imagem: {lbl_path} (stem={stem})")
            else:
                out_plot_dir = root / cfg.get("paths", {}).get("statistics_dir", "outputs/statistics") / "verificar_unify_convert"
                out_file = out_plot_dir / f"{stem}_anotacoes_originais{img_path.suffix}"
                kp_names = _get_kp_names()
                if _plot_imagem_bbox_keypoints(img_path, lbl_path, out_file, kp_names, DEFAULT_SEGMENTS):
                    print(f"\n  [OK] Anotações originais desenhadas em: {out_file.relative_to(root)}")
                else:
                    print(f"\n  [FALHA] Não foi possível gerar a imagem.")
    elif args.image and not HAS_CV2:
        print("\n  [AVISO] --image requer opencv-python. Instale: pip install opencv-python")

    # 7. Plotar N imagens com bbox e keypoints (opcional)
    if args.plot > 0 and HAS_CV2:
        out_plot_dir = root / cfg.get("paths", {}).get("statistics_dir", "outputs/statistics") / "verificar_unify_convert"
        kp_names = _get_kp_names()
        sorted_images = sorted([f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in exts])
        n_plot = 0
        for img_path in sorted_images:
            if n_plot >= args.plot:
                break
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            out_file = out_plot_dir / f"amostra_{n_plot + 1}_{img_path.name}"
            if _plot_imagem_bbox_keypoints(img_path, lbl_path, out_file, kp_names, DEFAULT_SEGMENTS):
                n_plot += 1
                print(f"  [PLOT] Salvo: {out_file.relative_to(root)}")
        if n_plot > 0:
            print(f"\n  {n_plot} imagem(ns) com bbox e keypoints em: {out_plot_dir.relative_to(root)}")
        elif args.plot > 0:
            print("\n  [AVISO] Nenhuma imagem foi plotada (verifique labels).")
    elif args.plot > 0 and not HAS_CV2:
        print("\n  [AVISO] --plot requer opencv-python. Instale: pip install opencv-python")

    print()
    if ok:
        print("  Conclusão: unificação e conversão parecem corretas.")
    else:
        print("  Conclusão: há avisos ou falhas. Revise os itens acima.")
    print()


if __name__ == "__main__":
    main()
