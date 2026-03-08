"""
Augmentation para imagens e anotações (keypoints/bbox).

Usa albumentations para garantir transformações consistentes em imagem e labels.
Pipeline offline (prepare_dataset): flip, rotate, brilho/contraste, blur, HSV, ruído.
Mosaic: combina 4 imagens em uma (2x2) para aumentar variedade no treino.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import get_full_config


def get_offline_pose_augmentation() -> Optional[Any]:
    """
    Pipeline de augmentation para pose (imagem + bbox + keypoints), sem resize/normalize.
    Usado no prepare_dataset para gerar cópias augmentadas em disco.
    Retorna None se albumentations não estiver instalado.
    """
    try:
        import albumentations as A
    except ImportError:
        return None

    cfg = get_full_config()
    aug_cfg = cfg.get("augmentation", {})
    if not aug_cfg.get("enabled", True):
        return None

    return A.Compose(
        [
            A.HorizontalFlip(p=aug_cfg.get("horizontal_flip", 0.5)),
            A.VerticalFlip(p=aug_cfg.get("vertical_flip", 0.2)),
            A.Rotate(limit=aug_cfg.get("rotate_limit", 15), p=0.5, border_mode=0),
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg.get("brightness_contrast", 0.2),
                contrast_limit=aug_cfg.get("contrast_limit", aug_cfg.get("brightness_contrast", 0.2)),
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=aug_cfg.get("gaussian_blur", 0.1)),
            A.HueSaturationValue(
                hue_shift_limit=int(aug_cfg.get("hue_saturation", 0.1) * 180),
                sat_shift_limit=int(aug_cfg.get("hue_saturation", 0.1) * 255),
                val_shift_limit=0,
                p=0.3,
            ),
            A.GaussNoise(var_limit=(0, aug_cfg.get("gaussian_noise_std", 10.0) ** 2), p=0.3),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_augmentation_transform(
    image_size: int = 640,
    training: bool = True,
) -> Any:
    """
    Retorna pipeline de augmentation baseado no config.yaml.

    Se training=False, retorna apenas resize (sem augmentation).
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        return None

    cfg = get_full_config()
    aug_cfg = cfg.get("augmentation", {})
    if not training or not aug_cfg.get("enabled", True):
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    return A.Compose([
        A.HorizontalFlip(p=aug_cfg.get("horizontal_flip", 0.5)),
        A.VerticalFlip(p=aug_cfg.get("vertical_flip", 0.2)),
        A.Rotate(limit=aug_cfg.get("rotate_limit", 15), p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=aug_cfg.get("brightness_contrast", 0.2),
            contrast_limit=aug_cfg.get("brightness_contrast", 0.2),
            p=0.5,
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=aug_cfg.get("gaussian_blur", 0.1)),
        A.HueSaturationValue(
            hue_shift_limit=aug_cfg.get("hue_saturation", 0.1) * 180,
            sat_shift_limit=aug_cfg.get("hue_saturation", 0.1) * 255,
            val_shift_limit=0,
            p=0.3,
        ),
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))


def build_yolo_augmentation_hint() -> str:
    """
    Retorna string com parâmetros de augmentation para passar ao YOLO.
    O Ultralytics YOLO tem augmentation embutido; esta função documenta
    os equivalentes no config para referência.
    """
    cfg = get_full_config()
    aug = cfg.get("augmentation", {})
    return (
        f"hsv_h={aug.get('hue_saturation', 0.1):.2f} "
        f"hsv_s={aug.get('hue_saturation', 0.1):.2f} "
        f"hsv_v=0 "
        f"degrees={aug.get('rotate_limit', 15)} "
        f"flipud={aug.get('vertical_flip', 0.2):.2f} "
        f"fliplr={aug.get('horizontal_flip', 0.5):.2f}"
    )
