#!/usr/bin/env python
"""
Análise exploratória (EDA) das features derivadas dos keypoints.
Gera gráficos (distribuições, correlação, PCA) e relatório em outputs/statistics/eda/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_full_config
from src.utils.metrics_logger import create_step_logger, get_statistics_dir
from src.features.feature_selection import (
    _build_geometric_feature_matrix,
    _get_kp_names,
    _load_keypoints_and_images,
    get_top_k_for_training,
    select_top_keypoints,
)


def main() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"Instale dependências: pip install matplotlib numpy scikit-learn. Erro: {e}")
        sys.exit(1)

    cfg = get_full_config()
    root = Path(__file__).resolve().parents[1]
    logger = create_step_logger("analisar_features", root)

    def log_and_print(msg: str) -> None:
        logger.log(msg)
        print(msg, flush=True)

    log_and_print("=== EDA: Análise exploratória das features ===")
    log_and_print("")

    unified = root / cfg.get("paths", {}).get("unified_dir", "data/unified")
    labels_dir = unified / "keypoints" / "labels"
    if not labels_dir.exists():
        logger.log(f"Pasta não encontrada: {labels_dir}. Resultado: falha")
        logger.finalize({"error": "labels_dir_not_found"})
        print(f"Pasta não encontrada: {labels_dir}. Rode antes: python scripts/unify_and_convert.py")
        sys.exit(1)

    out_dir = get_statistics_dir(root) / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_and_print("[1/6] Diretório de saída: " + str(out_dir.relative_to(root)))
    log_and_print("")

    log_and_print("[2/6] Carregando labels de keypoints...")
    X_raw, y, img_paths = _load_keypoints_and_images(labels_dir)
    if X_raw.size == 0:
        logger.log("Nenhum dado de keypoints encontrado. Resultado: falha")
        logger.finalize({"n_samples": 0, "n_features": 0})
        print("Nenhum dado de keypoints encontrado.")
        sys.exit(1)
    log_and_print(f"      Carregadas {len(X_raw)} amostras.")
    log_and_print("")

    log_and_print("[3/6] Construindo features geométricas e de textura/cor...")
    kp_names = _get_kp_names()
    F, feat_names = _build_geometric_feature_matrix(X_raw, kp_names)
    if F.size == 0 or len(feat_names) == 0:
        logger.log("Nenhuma feature geométrica gerada. Resultado: falha")
        logger.finalize({"n_samples": 0, "n_features": 0})
        print("Nenhuma feature geométrica gerada.")
        sys.exit(1)
    n_samples, n_feat = F.shape
    log_and_print(f"      Geradas {n_samples} amostras x {n_feat} features.")
    top_k_train = get_top_k_for_training()
    top_names = select_top_keypoints(labels_dir, for_training=True)
    log_and_print(f"      Para treino: {top_k_train} melhores features (config: top_k_for_training).")
    log_and_print("")
    report_lines = [
        "# Análise exploratória das features (EDA)",
        "",
        f"- Amostras: {n_samples}",
        f"- Features: {n_feat}",
        f"- **Para treino**: apenas as **{top_k_train}** melhores features (config: feature_selection.top_k_for_training)",
        f"- Lista das {min(top_k_train, len(top_names))} melhores para treino: {', '.join(top_names[:top_k_train])}",
        "",
        "## Estatísticas descritivas",
        "",
    ]

    # Histogramas (amostra de features se muitas)
    n_plot = min(12, n_feat)
    indices = np.linspace(0, n_feat - 1, n_plot, dtype=int)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i >= len(indices):
            ax.set_visible(False)
            continue
        idx = indices[i]
        ax.hist(F[:, idx], bins=30, edgecolor="black", alpha=0.7)
        ax.set_title(feat_names[idx][:25] + ("..." if len(feat_names[idx]) > 25 else ""), fontsize=8)
    plt.suptitle("Distribuição de features (amostra)")
    plt.tight_layout()
    fig.savefig(out_dir / "distribuicoes_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    log_and_print("[4/6] Histogramas de distribuição gerados: distribuicoes_features.png")

    # Matriz de correlação (amostra de colunas se muitas)
    if n_feat > 2:
        n_corr = min(25, n_feat)
        idx_corr = np.linspace(0, n_feat - 1, n_corr, dtype=int)
        F_sub = F[:, idx_corr]
        C = np.corrcoef(F_sub.T)
        names_sub = [feat_names[i][:15] for i in idx_corr]
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(names_sub)))
        ax.set_yticks(range(len(names_sub)))
        ax.set_xticklabels(names_sub, rotation=90, ha="right", fontsize=7)
        ax.set_yticklabels(names_sub, fontsize=7)
        plt.colorbar(im, ax=ax)
        plt.title("Matriz de correlação (amostra de features)")
        plt.tight_layout()
        plt.savefig(out_dir / "correlacao_features.png", dpi=150, bbox_inches="tight")
        plt.close()
        log_and_print("[5/6] Matriz de correlação gerada: correlacao_features.png")
    else:
        log_and_print("[5/6] Matriz de correlação: pulada (poucas features).")

    # PCA 2D (se houver amostras suficientes)
    if n_samples >= 10 and n_feat >= 2:
        log_and_print("      Gerando projeção PCA 2D...")
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            F_scaled = scaler.fit_transform(np.nan_to_num(F, nan=0.0))
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(F_scaled)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
            ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
            ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
            ax.set_title("Projeção PCA (2 componentes)")
            plt.tight_layout()
            plt.savefig(out_dir / "pca_2d.png", dpi=150, bbox_inches="tight")
            plt.close()
            report_lines.append(f"- Variância explicada PC1: {100*pca.explained_variance_ratio_[0]:.2f}%")
            report_lines.append(f"- Variância explicada PC2: {100*pca.explained_variance_ratio_[1]:.2f}%")
            report_lines.append("")
            log_and_print("      Projeção PCA 2D gerada: pca_2d.png")
        except Exception:
            logger.log("Gerando projeção PCA 2D. Resultado: falha (exceção)")
            print("      [AVISO] PCA 2D não gerado (exceção).", flush=True)
    elif n_feat >= 2:
        log_and_print("      PCA 2D: pulado (poucas amostras).")

    # Média e desvio por feature (todas as features, sem simplificação)
    report_lines.append("| Feature | Média | Desvio |")
    report_lines.append("|---------|-------|--------|")
    for i, name in enumerate(feat_names):
        mean = float(np.nanmean(F[:, i]))
        std = float(np.nanstd(F[:, i]))
        report_lines.append(f"| {name} | {mean:.4f} | {std:.4f} |")

    (out_dir / "relatorio_eda.md").write_text("\n".join(report_lines), encoding="utf-8")
    log_and_print("[6/6] Relatório EDA gerado: relatorio_eda.md (estatísticas de todas as features)")
    log_and_print("")
    log_and_print("Concluído. Saídas em: " + str(out_dir.relative_to(root)))
    log_path = logger.finalize({"n_samples": n_samples, "n_features": n_feat})
    print(f"  Log e métricas: {log_path}", flush=True)


if __name__ == "__main__":
    main()
