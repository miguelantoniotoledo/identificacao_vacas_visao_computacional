#!/usr/bin/env python
"""
Avalia o modelo de keypoints em um conjunto (test ou val).
- Métricas oficiais do Ultralytics (mAP, etc.) via model.val().
- Losses de avaliação: IoU, MSE, L1, Cross Entropy, Focal, Heatmap.

Uso:
  python scripts/evaluate_keypoints.py              # padrão: split test
  python scripts/evaluate_keypoints.py --split val # validação
  python scripts/evaluate_keypoints.py --weights outputs/keypoints/train/weights/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import get_full_config
from src.evaluation import compute_all_pose_losses, load_yolo_pose_label
from src.utils.metrics_logger import get_statistics_dir

N_KEYPOINTS = 8


def _match_pred_to_gt(pred_boxes_xyxy, pred_kp, pred_conf, gt_list, iou_box_xyxy):
    """Escolhe a predição com maior IoU em relação ao primeiro GT (ou único)."""
    if not pred_boxes_xyxy.shape[0] or not gt_list:
        return None
    gt_box = gt_list[0][0]
    best_iou = -1
    best_idx = 0
    for i in range(pred_boxes_xyxy.shape[0]):
        iou = iou_box_xyxy(pred_boxes_xyxy[i], gt_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    return (
        pred_boxes_xyxy[best_idx : best_idx + 1],
        pred_kp[best_idx : best_idx + 1],
        pred_conf[best_idx : best_idx + 1],
        gt_list[0],
    )


def main() -> None:
    try:
        from ultralytics import YOLO
        from src.evaluation.pose_losses import iou_box_xyxy
    except ImportError as e:
        print("Instale dependências: pip install ultralytics numpy. Erro:", e)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Avaliar modelo de keypoints em test ou val (métricas + losses)."
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("test", "val"),
        default="test",
        help="Conjunto a avaliar: test (hold-out) ou val (validação). Padrão: test.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Caminho para o modelo (ex.: outputs/keypoints/train/weights/best.pt).",
    )
    parser.add_argument(
        "--no-losses",
        action="store_true",
        help="Só rodar model.val(); não calcular losses por amostra.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Avaliar apenas esta imagem (caminho); exibe PCK e distância média. O label deve estar em <split>/labels com o mesmo stem.",
    )
    args = parser.parse_args()
    split = args.split

    cfg = get_full_config()
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    root = Path(__file__).resolve().parents[1]
    unified = root / paths.get("unified_dir", "data/unified")
    yolo_pose_dir = unified / "yolo_pose"
    data_yaml = yolo_pose_dir / "data.yaml"
    # Com k-fold, prepare_dataset gera só fold_1/, fold_2/, ... (sem data.yaml na raiz)
    if not data_yaml.exists():
        best_fold = None
        stats_dir = root / paths.get("statistics_dir", "outputs/statistics")
        json_path = stats_dir / "train_keypoints_latest.json"
        if json_path.exists():
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
                best_fold = (data.get("metrics") or {}).get("best_fold")
            except (json.JSONDecodeError, TypeError):
                pass
        if best_fold is not None:
            fold_yaml = yolo_pose_dir / f"fold_{best_fold}" / "data.yaml"
            if fold_yaml.exists():
                data_yaml = fold_yaml
                print(f"Usando conjunto de teste do melhor fold (fold {best_fold}, segundo train_keypoints_latest.json).")
        if not data_yaml.exists():
            for i in range(1, 20):
                fold_yaml = yolo_pose_dir / f"fold_{i}" / "data.yaml"
                if fold_yaml.exists():
                    data_yaml = fold_yaml
                    break
        if not data_yaml.exists():
            print("data.yaml não encontrado em yolo_pose/ nem em yolo_pose/fold_N/. Rode antes: python scripts/prepare_dataset.py")
            sys.exit(1)

    keypoints_out = root / paths.get("outputs_dir", "outputs") / "keypoints"
    default_weights = keypoints_out / "train" / "weights" / "best.pt"

    weights = args.weights
    if not weights:
        weights = default_weights
    else:
        weights = Path(weights)
        if not weights.is_absolute():
            weights = root / weights

    if not weights.exists():
        # Fallback: se train/weights/best.pt não existir (ex.: treino k-fold interrompido), usar o primeiro fold com best.pt
        if weights == default_weights and keypoints_out.exists():
            for i in range(1, 20):
                candidate = keypoints_out / f"fold_{i}" / "weights" / "best.pt"
                if candidate.exists():
                    weights = candidate
                    print(f"Usando modelo do fold {i} (train/weights/best.pt não encontrado): {weights}")
                    break
            else:
                weights = default_weights  # manter original para mensagem de erro
        if not weights.exists():
            print(f"Modelo não encontrado: {weights}")
            print("Rode antes: python scripts/train_keypoints.py (e aguarde o fim de todos os folds para gerar train/weights/best.pt).")
            print("Se o treino foi interrompido, existe best.pt em algum outputs/keypoints/fold_N/weights/?")
            sys.exit(1)

    # Base do dataset: raiz yolo_pose ou fold_N (quando usa k-fold)
    data_base = data_yaml.resolve().parent
    split_images_dir = data_base / split / "images"
    split_labels_dir = data_base / split / "labels"
    # Se --image foi passado, avaliar só essa imagem (procurar em <split> ou pelo path)
    single_image_path = None
    if args.image:
        p = Path(args.image)
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.exists():
            p = split_images_dir / Path(args.image).name
        if p.exists():
            single_image_path = p
            test_image_paths_for_eval = [single_image_path]
        else:
            print(f"Imagem não encontrada: {args.image}")
            sys.exit(1)
    else:
        test_image_paths_for_eval = None

    if not split_images_dir.exists() or not list(split_images_dir.glob("*.*")):
        if not single_image_path:
            print(f"Pasta do split '{split}' vazia ou inexistente: {split_images_dir}")
            sys.exit(1)

    imgsz = data_cfg.get("image_size", 640)
    print("Avaliando no conjunto de TESTE (hold-out)." if split == "test" else "Avaliando no conjunto de VALIDAÇÃO.")
    print(f"  Modelo: {weights}")
    print(f"  Data:   {data_yaml}")
    print(f"  Split:  {split} | imgsz: {imgsz}")
    print()

    model = YOLO(str(weights))

    # 1) Métricas oficiais (mAP, etc.) — baseadas em OKS, não em distância em pixels
    results_val = model.val(data=str(data_yaml), split=split)
    if results_val is not None:
        if hasattr(results_val, "results_dict") and results_val.results_dict:
            d = results_val.results_dict
            print(f"Métricas no {split.upper()} (Ultralytics, baseadas em OKS):")
            for k, v in sorted(d.items()):
                if v is not None and isinstance(v, (int, float)):
                    print(
                        f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
                    )
            print(
                "  [NOTA] mAP50/Precision/Recall usam OKS (Object Keypoint Similarity): "
                "são sensíveis à escala do objeto e podem ser altos mesmo com pontos "
                "visualmente distantes. Para 'proximidade em pixels', use PCK e distância média abaixo."
            )
        elif hasattr(results_val, "metrics"):
            print(f"Métricas no {split.upper()}:", results_val.metrics)
    print()

    # 2) Losses por amostra (IoU, MSE, L1, CE, Focal, Heatmap) e métricas em pixels (PCK, distância)
    if args.no_losses:
        print("Losses de avaliação omitidas (--no-losses).")
        return

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if test_image_paths_for_eval is not None:
        test_image_paths = test_image_paths_for_eval
    else:
        test_image_paths = [
            f
            for f in split_images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_exts
        ]
    if not test_image_paths:
        print(f"Nenhuma imagem no split '{split}' encontrada.")
        return

    predictions = model.predict(
        source=[str(p) for p in test_image_paths],
        imgsz=imgsz,
        verbose=False,
    )

    all_iou = []
    all_mse = []
    all_l1 = []
    all_ce = []
    all_focal = []
    all_heatmap = []
    all_mean_dist_px = []
    all_pck_20 = []
    all_pck_30 = []
    all_diagonal = []  # diagonal da imagem por amostra (para normalizar distância)
    n_skipped = 0
    _printed_first = False

    for img_path, res in zip(test_image_paths, predictions):
        stem = img_path.stem
        labels_dir = split_labels_dir
        try:
            parts = img_path.resolve().parts
            if "fold_" in parts and split in parts:
                idx = parts.index(split)
                if idx > 0:
                    base = Path(*parts[: idx + 1])
                    fold_labels = base / "labels"
                    if (fold_labels / f"{stem}.txt").exists():
                        labels_dir = fold_labels
        except Exception:
            pass
        lbl_path = labels_dir / f"{stem}.txt"
        if not lbl_path.exists():
            for i in range(1, 20):
                fold_lbl = yolo_pose_dir / f"fold_{i}" / split / "labels" / f"{stem}.txt"
                if fold_lbl.exists():
                    lbl_path = fold_lbl
                    break
        # Tamanho real da imagem para GT (labels normalizados) alinhar com pred (pixels).
        # Ultralytics devolve orig_shape como (height, width); sem isso, lemos do disco para evitar escala errada.
        if hasattr(res, "orig_shape") and res.orig_shape is not None:
            try:
                img_h, img_w = res.orig_shape  # (height, width)
            except Exception:
                img_h = img_w = None
        else:
            img_h = img_w = None
        if img_h is None or img_w is None:
            try:
                import cv2
                img = cv2.imread(str(img_path))
                if img is not None and img.size > 0:
                    img_h, img_w = img.shape[:2]
                else:
                    img_w, img_h = imgsz, imgsz
            except Exception:
                img_w, img_h = imgsz, imgsz

        gt_list = load_yolo_pose_label(
            str(lbl_path), img_w=img_w, img_h=img_h, n_keypoints=N_KEYPOINTS
        )
        if not gt_list:
            n_skipped += 1
            continue

        if not hasattr(res, "boxes") or res.boxes is None or len(res.boxes) == 0:
            n_skipped += 1
            continue
        if not hasattr(res, "keypoints") or res.keypoints is None:
            n_skipped += 1
            continue

        pred_xyxy = np.array(res.boxes.xyxy.cpu().numpy(), dtype=np.float64)
        pred_kp = np.array(res.keypoints.xy.cpu().numpy(), dtype=np.float64)
        conf = getattr(res.keypoints, "conf", None)
        if conf is not None:
            pred_conf = np.array(conf.cpu().numpy(), dtype=np.float64)
        else:
            pred_conf = np.ones((pred_kp.shape[0], pred_kp.shape[1]), dtype=np.float64) * 0.5
        if pred_conf.size == 0:
            pred_conf = np.ones_like(pred_kp[..., 0], dtype=np.float64) * 0.5

        matched = _match_pred_to_gt(
            pred_xyxy, pred_kp, pred_conf, gt_list, iou_box_xyxy
        )
        if matched is None:
            n_skipped += 1
            continue

        p_boxes, p_kp, p_conf, (gt_box, gt_kp, gt_vis) = matched
        gt_boxes = gt_box[None, :]
        gt_kp_exp = gt_kp[None, :, :]
        gt_vis_exp = gt_vis[None, :]

        if not _printed_first:
            print("--- Primeira imagem com match (GT vs Pred) ---")
            print("Imagem:", img_path.name)
            print("GT:", gt_kp)
            print("Pred:", p_kp[0])
            print("----------------------------------------------")
            _printed_first = True

        losses = compute_all_pose_losses(
            pred_boxes=p_boxes,
            pred_kp=p_kp,
            pred_conf=p_conf,
            gt_boxes=gt_boxes,
            gt_kp=gt_kp_exp,
            gt_vis=gt_vis_exp,
            box_format="xyxy",
            img_shape=(img_h, img_w),
            heatmap_size=(64, 64),
        )
        all_iou.append(losses["iou_loss"])
        all_mse.append(losses["mse_loss"])
        all_l1.append(losses["l1_loss"])
        all_ce.append(losses["cross_entropy"])
        all_focal.append(losses["focal_loss"])
        all_heatmap.append(losses["heatmap_loss"])
        all_mean_dist_px.append(losses["mean_distance_px"])
        all_pck_20.append(losses["pck_20px"])
        all_pck_30.append(losses["pck_30px"])
        all_diagonal.append(float(np.sqrt(img_w**2 + img_h**2)))
        if single_image_path is not None:
            print(f"  [{img_path.name}] dist_média={losses['mean_distance_px']:.1f}px  PCK@20px={100*losses['pck_20px']:.0f}%  PCK@30px={100*losses['pck_30px']:.0f}%")

    n_eval = len(all_iou)
    print(f"Losses de avaliação no {split.upper()} (média sobre amostras com predição e GT):")
    if n_eval == 0:
        print("  Nenhuma amostra avaliada (sem predição ou sem GT).")
    else:
        print(f"  Amostras: {n_eval} (omitidas: {n_skipped})")
        print(f"  IoU loss       (1 - IoU bbox): {np.mean(all_iou):.6f}")
        print(f"  MSE loss       (L2 keypoints): {np.mean(all_mse):.6f}")
        print(f"  L1 loss        (L1 keypoints): {np.mean(all_l1):.6f}")
        print(f"  Cross entropy  (conf vs vis):  {np.mean(all_ce):.6f}")
        print(f"  Focal loss     (conf vs vis):  {np.mean(all_focal):.6f}")
        print(f"  Heatmap loss   (MSE heatmaps):  {np.mean(all_heatmap):.6f}")
        print()
        print("  Acurácia de proximidade (priorize para relatar resultado real):")
        dist_mean = float(np.mean(all_mean_dist_px))
        pck20 = float(np.mean(all_pck_20))
        pck30 = float(np.mean(all_pck_30))
        # Distância média normalizada pela diagonal da imagem (0 = perfeito, comparável entre resoluções)
        norm_dists = [all_mean_dist_px[i] / all_diagonal[i] for i in range(n_eval) if all_diagonal[i] > 0]
        dist_mean_norm = float(np.mean(norm_dists)) if norm_dists else 0.0
        print(f"  PCK@30px (acuracia de proximidade): {100 * pck30:.1f}%")
        print(f"  PCK@20px:                          {100 * pck20:.1f}%")
        print(f"  Distância média (px):               {dist_mean:.2f}")
        print(f"  Distância média normalizada (0–1):  {dist_mean_norm:.4f}  (÷ diagonal da imagem)")
        print("  [Use PCK@30px ou distância normalizada para comparar entre resoluções; mAP50/OKS é referência.]")

        # Salvar métricas principais em JSON para consulta
        stats_dir = get_statistics_dir(root)
        metrics = {
            "accuracy_proximity_pck30_pct": round(100 * pck30, 1),
            "pck_20px_pct": round(100 * pck20, 1),
            "pck_30px_pct": round(100 * pck30, 1),
            "distance_mean_px": round(dist_mean, 2),
            "distance_mean_normalized": round(dist_mean_norm, 4),
            "pck_20px": round(pck20, 4),
            "pck_30px": round(pck30, 4),
            "n_samples": n_eval,
        }
        latest_path = stats_dir / f"evaluate_keypoints_{split}_latest.json"
        with latest_path.open("w", encoding="utf-8") as f:
            json.dump({"split": split, "metrics": metrics}, f, ensure_ascii=False, indent=2)
        print()
        print(f"  Métricas salvas em: {latest_path}")
        print("  (Para relatório: use accuracy_proximity_pck30_pct ou distance_mean_normalized.)")

    print()
    print("Concluído. Acurácia de proximidade = PCK@30px; mAP50/OKS = referência para literatura.")


if __name__ == "__main__":
    main()
