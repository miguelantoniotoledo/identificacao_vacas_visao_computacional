"""
Módulo de seleção e análise de features geométricas.
"""

from .feature_selection import (
    select_top_keypoints,
    compute_keypoint_importance,
    compute_feature_correlations,
    get_top_k_for_training,
    build_feature_matrix_for_training,
)

__all__ = [
    "select_top_keypoints",
    "compute_keypoint_importance",
    "compute_feature_correlations",
    "get_top_k_for_training",
    "build_feature_matrix_for_training",
]
