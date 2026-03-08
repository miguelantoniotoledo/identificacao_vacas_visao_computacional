#!/usr/bin/env python
"""
Avalia o modelo de keypoints no conjunto de TESTE (data/unified/yolo_pose/test).
- Métricas oficiais do Ultralytics (mAP, etc.) via model.val().
- Losses de avaliação: IoU, MSE, L1, Cross Entropy, Focal, Heatmap.

Uso:
  python scripts/evaluate_keypoints.py
  python scripts/evaluate_keypoints.py --weights outputs/keypoints/train/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import get_full_config
from src.evaluation import compute_all_pose_losses, load_yolo_pose_label

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
        description="Avaliar modelo de keypoints no conjunto de TESTE (métricas + losses)."
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
        help="Avaliar apenas esta imagem (caminho); exibe PCK e distância média para ela. O label deve estar em test/labels com o mesmo stem.",
    )
    args = parser.parse_args()

    cfg = get_full_config()
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    root = Path(__file__).resolve().parents[1]
    unified = root / paths.get("unified_dir", "data/unified")
    yolo_pose_dir = unified / "yolo_pose"
    data_yaml = yolo_pose_dir / "data.yaml"

    if not data_yaml.exists():
        print("data.yaml não encontrado. Rode antes: python scripts/prepare_dataset.py")
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

    test_images_dir = yolo_pose_dir / "test" / "images"
    test_labels_dir = yolo_pose_dir / "test" / "labels"
    # Se --image foi passado, avaliar só essa imagem (procurar em test ou pelo path)
    single_image_path = None
    if args.image:
        p = Path(args.image)
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.exists():
            p = test_images_dir / Path(args.image).name
        if p.exists():
            single_image_path = p
            test_image_paths_for_eval = [single_image_path]
        else:
            print(f"Imagem não encontrada: {args.image}")
            sys.exit(1)
    else:
        test_image_paths_for_eval = None

    if not test_images_dir.exists() or not list(test_images_dir.glob("*.*")):
        if not single_image_path:
            print("Pasta de teste vazia ou inexistente:", test_images_dir)
            sys.exit(1)

    imgsz = data_cfg.get("image_size", 640)
    print("Avaliando no conjunto de TESTE (hold-out).")
    print(f"  Modelo: {weights}")
    print(f"  Data:   {data_yaml}")
    print(f"  Split:  test | imgsz: {imgsz}")
    print()

    model = YOLO(str(weights))

    # 1) Métricas oficiais (mAP, etc.) — baseadas em OKS, não em distância em pixels
    results_val = model.val(data=str(data_yaml), split="test")
    if results_val is not None:
        if hasattr(results_val, "results_dict") and results_val.results_dict:
            d = results_val.results_dict
            print("Métricas no TESTE (Ultralytics, baseadas em OKS):")
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
            print("Métricas no TESTE:", results_val.metrics)
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
            for f in test_images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_exts
        ]
    if not test_image_paths:
        print("Nenhuma imagem de teste encontrada.")
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
    n_skipped = 0

    for img_path, res in zip(test_image_paths, predictions):
        stem = img_path.stem
        labels_dir = test_labels_dir
        try:
            parts = img_path.resolve().parts
            if "fold_" in parts and "test" in parts:
                idx = parts.index("test")
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
                fold_lbl = yolo_pose_dir / f"fold_{i}" / "test" / "labels" / f"{stem}.txt"
                if fold_lbl.exists():
                    lbl_path = fold_lbl
                    break
        # Tamanho real da imagem para GT (labels normalizados) alinhar com pred (pixels)
        if hasattr(res, "orig_shape") and res.orig_shape is not None:
            try:
                img_h, img_w = res.orig_shape
            except Exception:
                img_w, img_h = imgsz, imgsz
        else:
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
        if single_image_path is not None:
            print(f"  [{img_path.name}] dist_média={losses['mean_distance_px']:.1f}px  PCK@20px={100*losses['pck_20px']:.0f}%  PCK@30px={100*losses['pck_30px']:.0f}%")

    n_eval = len(all_iou)
    print("Losses de avaliação no TESTE (média sobre amostras com predição e GT):")
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
        print("  Métricas em pixels (proximidade real pred vs GT):")
        print(f"  Distância média (px):  {np.mean(all_mean_dist_px):.2f}")
        print(f"  PCK@20px (% pontos ≤20px): {100 * np.mean(all_pck_20):.1f}%")
        print(f"  PCK@30px (% pontos ≤30px): {100 * np.mean(all_pck_30):.1f}%")
        print("  [PCK = Percentage of Correct Keypoints; reflete melhor a 'acuracia visual' que mAP50/OKS.]")

    print()
    print("Concluído. Use PCK e distância média para avaliar proximidade visual; mAP50/OKS para comparação com literatura.")


if __name__ == "__main__":
    main()
