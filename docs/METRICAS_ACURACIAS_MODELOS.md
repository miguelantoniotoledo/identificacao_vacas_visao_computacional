# Métricas e acurácias dos modelos

Este documento reúne as **acurácias** do modelo de keypoints (pose) e do classificador de vacas, além de **exemplos de como reproduzir** a avaliação no conjunto de teste.

---

## 1. Modelo de keypoints (pose)

### Estratégia de split

O dataset de pose usa **split por indivíduo** (group k-fold): as vacas são divididas em train/val/test de forma que **nenhum indivíduo aparece em mais de um split** (evita vazamento). Todas as fotos de uma vaca ficam no mesmo conjunto.

### Métricas de acurácia (conjunto de teste)

As métricas abaixo refletem a **proximidade real** entre keypoints preditos e ground truth (priorize-as em relação ao mAP50/OKS, que é sensível à escala):

| Métrica | Descrição | Valor (exemplo de run) |
|--------|-----------|------------------------|
| **Acurácia de proximidade (PCK@30px)** | % de keypoints a ≤ 30 px do GT | 16,2% |
| **PCK@20px** | % de keypoints a ≤ 20 px do GT | 8,7% |
| **Distância média (px)** | Média da distância pred↔GT por keypoint | 81,63 px |
| **Distância média normalizada** | Média(distância / diagonal da imagem), 0–1 | 0,042 |
| **Amostras** | Número de imagens avaliadas no teste | 132 |

*Os valores da tabela correspondem a um run em `evaluate_keypoints_latest.json` (split test). Atualize após novos runs.*

### Como reproduzir a avaliação no teste

```powershell
# Na raiz do projeto
python scripts/evaluate_keypoints.py --split test
```

- **Saída:** métricas no terminal e em `outputs/statistics/evaluate_keypoints_test_latest.json` (ou `evaluate_keypoints_latest.json`).
- **Validação:** use `--split val` para avaliar no conjunto de validação.

---

## 2. Modelo de classificação (identificação de vacas)

### Estratégia de split

O classificador usa **split 80-10-10 por foto**, **com embaralhamento** (não por indivíduo):

- Em **cada pasta** (uma classe = uma vaca), as imagens são embaralhadas e divididas em ~80% train, ~10% val, ~10% test.
- A **mesma vaca** aparece em train, val e test (em fotos diferentes).
- Train, val e test têm **as mesmas classes**; o modelo é avaliado em indivíduos que já viu no treino.

*Se val/test tivessem apenas indivíduos não vistos no treino (split por indivíduo), a acurácia reportada seria 0 ou enganosa.*

### Métricas de acurácia (val ou test)

| Métrica | Descrição | Referência |
|--------|-----------|------------|
| **top1_acc (Top-1 accuracy)** | Fração de imagens em que a classe prevista é a correta | Principal acurácia |
| **top5_acc (Top-5 accuracy)** | Fração em que a classe correta está entre as 5 mais prováveis | Auxiliar |
| **fitness** | Métrica interna Ultralytics (relacionada a top-5) | Opcional |

*A avaliação chama `model.val(data=<raiz classification_split>, split=val|test)` e lê `top1_acc` / `top5_acc` do resultado (ou de `results_dict`: `metrics/accuracy_top1`, `metrics/accuracy_top5`). Valores atuais em `outputs/statistics/evaluate_classifier_latest.json`.*

### Como reproduzir a avaliação

```powershell
# Na raiz do projeto
python scripts/evaluate_classifier.py --split val   # validação (10% dos dados)
python scripts/evaluate_classifier.py --split test   # hold-out final (10%)
```

- **Saída:** top-1 e top-5 no terminal e em `outputs/statistics/evaluate_classifier_latest.json`.
- **Pré-requisito:** ter rodado `prepare_dataset.py` para gerar `data/unified/classification_split/` com a estratégia 80-10-10 por foto; train/val/test devem ter as mesmas pastas (classes).

---

## 3. Exemplo de pipeline para verificar acurácia no teste

1. **Preparar dados:**
   ```powershell
   python scripts/prepare_dataset.py
   ```
   - **Pose:** split por indivíduo (group k-fold) em `data/unified/yolo_pose/`.
   - **Classificador:** split 80-10-10 por foto em `data/unified/classification_split/` (train/val/test com as mesmas classes).

2. **Treinar keypoints** (usa melhor fold se k_folds > 1):
   ```powershell
   python scripts/train_keypoints.py
   ```

3. **Avaliar keypoints no teste:**
   ```powershell
   python scripts/evaluate_keypoints.py --split test
   ```
   Consultar: `outputs/statistics/evaluate_keypoints_test_latest.json` (ou `evaluate_keypoints_latest.json`).

4. **Treinar classificador:**
   ```powershell
   python scripts/train_classifier.py
   ```

5. **Avaliar classificador (val ou test):**
   ```powershell
   python scripts/evaluate_classifier.py --split val
   python scripts/evaluate_classifier.py --split test
   ```
   Consultar: `outputs/statistics/evaluate_classifier_latest.json`.

Os números deste documento devem ser atualizados após rodar esses passos no seu ambiente; os arquivos `*_latest.json` em `outputs/statistics/` são a referência atual.
