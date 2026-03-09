# Identificação de Vacas – Visão Computacional

Projeto baseado em **YOLO** para:
1. **Detecção de keypoints** anatômicos em imagens de vacas (pose estimation)
2. **Identificação de vacas** a partir da pasta de classificação (uma classe por vaca)

Uma **comparação com o projeto de referência** [deteccao_keypoints_vacas](https://github.com/thalessalvador/deteccao_keypoints_vacas) (o que temos de melhor e possíveis melhorias) está em [docs/COMPARACAO_PROJETO_REFERENCIA.md](docs/COMPARACAO_PROJETO_REFERENCIA.md).

---

## Para avaliadores (pós-graduação)

- **Objetivo:** pipeline reprodutível para (1) estimativa de pose (8 keypoints anatômicos em vacas) com YOLOv8-pose e (2) classificação de identidade com YOLOv8-cls, a partir de dados anotados no Label Studio.
- **Metodologia:** unificação de dados brutos (catálogo + classificação) → conversão Label Studio → YOLO pose → split estratificado por grupo (80/10/10) com opção de k-fold → treino com early stopping; augmentations offline (flip, rotação, blur, HSV, mosaic) no preparo do dataset.
- **Métricas:** keypoints: mAP50, mAP50-95 (OKS), **PCK@20px/30px** e **distância média em pixels** (avaliação no hold-out de teste); classificador: top-1 e top-5 accuracy em val/test. Detalhes em [docs/COMPARACAO_PROJETO_REFERENCIA.md](docs/COMPARACAO_PROJETO_REFERENCIA.md) (OKS vs proximidade em pixels).
- **Reprodutibilidade:** `config.yaml` (seed, ratios, device); pipeline único (`python scripts/pipeline.py`); métricas e logs em `outputs/statistics/` e `outputs/logs/`.
- **Referência:** [thalessalvador/deteccao_keypoints_vacas](https://github.com/thalessalvador/deteccao_keypoints_vacas).

---

## Primeira vez usando o projeto

Se você **não conhece o repositório**, siga na ordem:

1. **Pré-requisitos:** Python 3.10+; opcional: GPU NVIDIA (ver [docs/SETUP_GPU.md](docs/SETUP_GPU.md)).
2. **Clonar/abrir o projeto** e, na **raiz do repositório**, criar e ativar o ambiente:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. **Instalar dependências:**  
   Se for usar GPU, instale primeiro PyTorch com CUDA (veja comentários no topo de `requirements.txt` e [docs/SETUP_GPU.md](docs/SETUP_GPU.md)). Depois:
   ```powershell
   pip install -r requirements.txt
   ```
4. **Colocar os dados brutos** em `raw/` conforme a [Estrutura de dados esperada](#estrutura-de-dados-esperada) (`raw/catalogo/<nome_vaca>/` com imagens e `Key_points/` com JSONs do Label Studio; `raw/classificacao/<nome_vaca>/` com fotos por vaca).
5. **Rodar o pipeline completo** (recomendado na primeira vez):
   ```powershell
   python scripts/pipeline.py
   ```
   Isso executa em sequência: unificação e conversão → preparação dos splits → EDA → treino de keypoints → treino do classificador → visualizações. **Resultados:** modelos em `outputs/keypoints/train/weights/` e `outputs/classifier/train/weights/`; métricas e gráficos em `outputs/statistics/`; log consolidado em `outputs/logs/pipeline_run_<timestamp>.log`.
6. **Rodar sempre na raiz do projeto** (evita erros de caminho). Exemplo: `python scripts/train_keypoints.py`, e não de dentro de `scripts/`.

Para **avaliar** keypoints no conjunto de teste: `python scripts/evaluate_keypoints.py`. Para **inferência** em uma imagem: `python scripts/predict_keypoints.py --image caminho/para/imagem.jpg` ou `python scripts/predict_cow.py --image ...` (classificador).

---

## Fluxo de dados e informações

Este é o fluxo do pipeline: de onde vêm os dados, o que cada etapa faz e para onde vão os resultados.

```
raw/catalogo, raw/classificacao
        ↓
[1] unify_and_convert  →  data/unified/keypoints/ (+ classification/)
        ↓
[2] prepare_dataset    →  data/unified/yolo_pose/ (train/val/test ou fold_1..N)
        ↓
[3] analisar_features  →  outputs/statistics/eda/
        ↓
[4] train_keypoints    →  outputs/keypoints/train/weights/best.pt
        ↓
[5] train_classifier   →  outputs/classifier/train/weights/best.pt
        ↓
[6] visualize_keypoints →  outputs/vis/keypoints/ + keypoints_val/ + keypoints_test/
```

| Passo | Script | O que faz |
|-------|--------|-----------|
| **1/6** | `unify_and_convert.py` | Unifica pastas em `raw/`; converte anotações Label Studio (JSON) → YOLO pose; copia imagens e grava labels em `data/unified/keypoints/`. Nomes com prefixo da pasta para group k-fold. |
| **2/6** | `prepare_dataset.py` | Split 80/10/10 por pasta; **todas as augmentations** (flip, rotate, blur, HSV, ruído) + **mosaic** (2x2) no treino; gera `data.yaml`. |
| **3/6** | `analisar_features.py` | EDA: carrega labels, monta features geométricas, calcula as 15 melhores para treino; gera histogramas, correlação, PCA 2D e relatório em `outputs/statistics/eda/`. |
| **4/6** | `train_keypoints.py` | Treina YOLOv8-pose (keypoints); **early stopping** (patience); se houver k-fold, treina por fold e escolhe o melhor por mAP50-95; salva `best.pt` em `outputs/keypoints/train/weights/`. |
| **5/6** | `train_classifier.py` | Treina YOLOv8-cls (uma classe por pasta em `data/unified/classification/`); **early stopping**; salva modelo em `outputs/classifier/train/`. |
| **6/6** | `visualize_keypoints.py` | Desenha keypoints e segmentos nas imagens; gera **três pastas**: `outputs/vis/keypoints/` (conjunto completo), `outputs/vis/keypoints_val/` (validação) e `outputs/vis/keypoints_test/` (teste), para verificação visual do modelo. |

Ao final, o pipeline grava um **log consolidado** em `outputs/logs/pipeline_run_<timestamp>.log` com data/hora de cada evento e estatísticas dos scripts.

### Resumo: instruções, resultado esperado e localização

| Script | Comando principal | Resultado esperado | Localização do resultado |
|--------|-------------------|-------------------|--------------------------|
| **unify_and_convert** | `python scripts/unify_and_convert.py` | Terminal: contagem de anotações convertidas e falhas. | `data/unified/keypoints/images/`, `data/unified/keypoints/labels/`, `data/unified/classification/<vaca>/`; métricas: `outputs/statistics/unify_and_convert_latest.json` |
| **prepare_dataset** | `python scripts/prepare_dataset.py` | Terminal: contagem train/val/test e confirmação de data.yaml. | Pose: `data/unified/yolo_pose/train/`, `val/`, `test/` (ou `fold_1/`…`fold_N/`); classificação: `data/unified/classification_split/train/`, `val/`, `test/` |
| **analisar_features** | `python scripts/analisar_features.py` | Terminal: top features e confirmação de arquivos gerados. | `outputs/statistics/eda/` (histogramas, correlação, PCA, `relatorio_eda.md`) |
| **train_keypoints** | `python scripts/train_keypoints.py` | Terminal: progresso por época; ao final, melhor fold e mAP50-95. | Modelo: `outputs/keypoints/train/weights/best.pt`; por fold: `outputs/keypoints/fold_N/`; estatísticas: `outputs/statistics/train_keypoints_folds.md`, `train_keypoints_latest.json` |
| **train_classifier** | `python scripts/train_classifier.py` | Terminal: progresso por época e métricas de validação. | Modelo: `outputs/classifier/train/weights/best.pt`; curvas: `outputs/statistics/`; logs: `outputs/logs/` |
| **visualize_keypoints** | `python scripts/visualize_keypoints.py` | Terminal: quantidade por split (completo, val, test). | `outputs/vis/keypoints/` (completo), `outputs/vis/keypoints_val/` (validação), `outputs/vis/keypoints_test/` (teste) — ground truth desenhado para verificação do modelo. |
| **evaluate_keypoints** | `python scripts/evaluate_keypoints.py` (+ `--image path` para uma imagem) | Terminal: mAP (OKS), **PCK e distância média (px)** e losses. mAP50 é baseado em OKS; para “proximidade visual” use PCK. | Logs e métricas em `outputs/logs/`, `outputs/statistics/` |
| **evaluate_classifier** | `python scripts/evaluate_classifier.py --split val\|test` | Terminal: top-1 e top-5 accuracy. | `outputs/statistics/evaluate_classifier_latest.json`; logs: `outputs/logs/` |
| **verificar_unify_convert** | `python scripts/verificar_unify_convert.py` (+ `--plot N` ou `--image path`) | Terminal: OK/FALHA por verificação; com `--plot`/`--image`: confirmação de imagens salvas. | Com `--plot N`: `outputs/statistics/verificar_unify_convert/amostra_*_<nome>.jpg`; com `--image path`: `outputs/statistics/verificar_unify_convert/<stem>_anotacoes_originais.jpg` |
| **predict_cow** | `python scripts/predict_cow.py --image path` ou `--input-dir path` | Terminal: por imagem, classe(es) e confiança; com top-K, lista das K classes. | Imagens anotadas: `outputs/inference/classifier/pred/` |
| **predict_keypoints** | `python scripts/predict_keypoints.py --image path` ou `--input-dir path` | Terminal: coordenadas dos keypoints por imagem (com nomes). Imagem com box "cow" e keypoints com nomes (withers, back, etc.). | Imagens: `outputs/inference/keypoints/pred/` (mesmo nome do arquivo de entrada quando uma única imagem) |

---

## Estrutura de dados esperada

```
raw/
├── catalogo/
│   └── <nome_vaca>/
│       ├── *.jpg, *.png          # Imagens
│       └── Key_points/
│           └── <id>              # JSON do Label Studio (sem extensão)
└── classificacao/
    └── <nome_vaca>/
        └── *.jpg, *.png          # Fotos para treino do classificador
```

## Pipeline

### Passo a passo resumido (ordem recomendada)

| Ordem | Etapa | Pré-requisito | Comando principal |
|-------|--------|----------------|-------------------|
| 0 | Dados em `raw/` | Estrutura `raw/catalogo/`, `raw/classificacao/` | — |
| 1 | Ambiente | Python 3.10+, venv ativado | `pip install -r requirements.txt` |
| 2 | Unificar e converter | Dados em `raw/` | `python scripts/unify_and_convert.py` |
| 3 | Verificar conversão (opcional) | Etapa 2 concluída | `python scripts/verificar_unify_convert.py` ou `--plot 3` |
| 4 | Preparar splits | `data/unified/keypoints/` preenchido | `python scripts/prepare_dataset.py` |
| 5 | EDA (opcional) | Etapa 4 | `python scripts/analisar_features.py` |
| 6 | Treinar keypoints | `data/unified/yolo_pose/` (ou `fold_1/`…) | `python scripts/train_keypoints.py` |
| 7 | Avaliar keypoints no teste | Modelo em `outputs/keypoints/train/weights/best.pt` | `python scripts/evaluate_keypoints.py` |
| 8 | Treinar classificador | `data/unified/classification/` ou `classification_split/` | `python scripts/train_classifier.py` |
| 9 | Avaliar classificador | Modelo treinado | `python scripts/evaluate_classifier.py --split test` |
| 10 | Visualizar keypoints (opcional) | Labels ou predições | `python scripts/visualize_keypoints.py` |

Ou execute tudo em sequência: `python scripts/pipeline.py` (etapas 2, 4, 5, 6, 8 e 10).

### 1. Ambiente

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**GPU NVIDIA (ex.: RTX 5060):** a RTX 50xx usa arquitetura Blackwell e precisa de PyTorch com **CUDA 12.8**. Veja [docs/SETUP_GPU.md](docs/SETUP_GPU.md) para instalação e verificação.

### 2. Rodar o pipeline completo (recomendado)

```powershell
python scripts/pipeline.py
```

Isso executa em sequência:

1. Unificação e conversão dos dados brutos (`unify_and_convert.py`)
2. Preparação dos splits train/val/test (`prepare_dataset.py`)
3. Análise exploratória (EDA) das features (`analisar_features.py`)
4. Treino do modelo de keypoints (YOLO pose) (`train_keypoints.py`)
5. Treino do classificador de vacas (YOLO cls) (`train_classifier.py`)
6. Geração de imagens de visualização com keypoints e retas (`visualize_keypoints.py`)

Cada script grava **métricas e logs** (ver seção *Logs e estatísticas*). O pipeline gera ainda um **log consolidado** (`outputs/logs/pipeline_run_<timestamp>.log`) com o passo a passo e as estatísticas finais de cada processo (F1, acurácia, mAP, etc.).

Flags úteis:

- `--skip-eda`: pula a análise exploratória (EDA)
- `--skip-train-keypoints`: pula o treino de keypoints
- `--skip-train-classifier`: pula o treino do classificador
- `--skip-visualize`: pula as visualizações

Exemplo:

```powershell
python scripts/pipeline.py --skip-train-classifier --skip-visualize
```

### 3. Rodar etapas individualmente (opcional)

#### 3.1. Unificar e converter dados

```powershell
python scripts/unify_and_convert.py
```

- **O que faz:** unifica pastas de `raw`; converte anotações Label Studio (JSON) → formato YOLO pose; copia imagens para `data/unified/`.
- **Resultado esperado:** no terminal, contagem de anotações keypoints convertidas e de imagens de classificação; sem erros, `failed` deve ser 0.
- **Localização do resultado:**
  - Keypoints: `data/unified/keypoints/images/` (imagens) e `data/unified/keypoints/labels/` (um `.txt` por imagem, formato YOLO pose).
  - Classificação: `data/unified/classification/<nome_vaca>/` (imagens por vaca).
  - Métricas do run: `outputs/statistics/unify_and_convert_latest.json` (`converted`, `failed`, etc.).

**Como verificar se deu certo:**

1. **Script automático (recomendado):**
   ```powershell
   python scripts/verificar_unify_convert.py
   ```
   Com amostras de conteúdo dos labels:
   ```powershell
   python scripts/verificar_unify_convert.py --amostras 5
   ```
   Para plotar N imagens com **bbox e keypoints** desenhados (anotações originais):
   ```powershell
   python scripts/verificar_unify_convert.py --plot 3
   ```
   Para **uma imagem específica** (ex.: imagem em raw ou em yolo_pose), desenhar nela as anotações originais; o label é buscado em `data/unified/keypoints/labels/` pelo nome do arquivo (stem):
   ```powershell
   python scripts/verificar_unify_convert.py --image "caminho/para/imagem.jpg"
   ```
   - **Resultado esperado:** no terminal, `[OK]` ou `[FALHA]` por verificação; com `--plot N` ou `--image`, confirmação do caminho do arquivo salvo.
   - **Localização do resultado:** com `--plot N`: `outputs/statistics/verificar_unify_convert/amostra_1_<nome>.jpg`, …; com `--image`: `outputs/statistics/verificar_unify_convert/<stem>_anotacoes_originais.jpg`.
   O script confere: pastas existem, mesmo número de imagens e labels, cada imagem tem um `.txt` com o mesmo nome (stem), formato do label (classe + bbox + 8 keypoints × 3) e compara com o último run em `outputs/statistics/unify_and_convert_latest.json`.

2. **Verificação manual:** conferir que em `data/unified/keypoints/` existem `images/` e `labels/` com a mesma quantidade de arquivos; para uma imagem qualquer (ex.: `foto.jpg`) deve existir `foto.txt` em `labels/` com uma linha no formato YOLO pose (classe, 4 bbox, 24 valores de keypoints). Ver também o JSON do último run: `converted` deve bater com o total e `failed` deve ser 0.

#### 3.2. Preparar splits (train/val/test)

```powershell
python scripts/prepare_dataset.py
```

- **O que faz:**  
  - **Pose (keypoints):** stratified_per_group em cada grupo: **80% train**, **10% val**, **10% test** em `data/unified/yolo_pose/` (train/images, val/images, test/images). Se `pose.k_folds` > 1, gera `fold_1/` … `fold_N/` com train/val/test por fold. Aplica augmentations e mosaic no treino; gera `data.yaml` em cada pasta.
  - **Classificação:** se existir `data/unified/classification/`, gera **split 80-10-10** em `data/unified/classification_split/`: **train/**, **val/**, **test/** (uma pasta por vaca em cada).
- **Resultado esperado:** no terminal, contagem de amostras em train/val/test e confirmação de geração do(s) `data.yaml`.
- **Localização do resultado:**  
  - Pose: `data/unified/yolo_pose/train/`, `val/`, `test/` (ou `data/unified/yolo_pose/fold_N/train/`, `val/`, `test/`).  
  - Classificação: `data/unified/classification_split/train/<vaca>/`, `val/<vaca>/`, `test/<vaca>/`.

Use as mesmas proporções do config: `data.train_ratio: 0.8`, `data.val_ratio: 0.1`, `data.test_ratio: 0.1`.

#### 3.3. Treinar modelo de keypoints

```powershell
python scripts/train_keypoints.py
```

- **O que faz:** treina YOLOv8-pose; usa GPU (configurável em `config.yaml` → `training.device`); augmentation automático pelo Ultralytics; early stopping (patience). Se existir **k-fold** em `data/unified/yolo_pose/fold_1/` … `fold_N/`, treina um modelo por fold e escolhe o melhor por mAP50-95; o `best.pt` do melhor fold é copiado para `outputs/keypoints/train/weights/best.pt`.
- **Resultado esperado:** no terminal, progresso por época (loss, mAP); ao final, mensagem de conclusão e, em k-fold, qual fold foi o melhor e seu mAP50-95.
- **Localização do resultado:**  
  - Modelo principal: `outputs/keypoints/train/weights/best.pt` (e `last.pt`).  
  - Por fold: `outputs/keypoints/fold_N/weights/best.pt`, `results.csv`, gráficos.  
  - Estatísticas: `outputs/statistics/train_keypoints_folds.md`, `outputs/statistics/train_keypoints_latest.json`; logs: `outputs/logs/`.

**Estatísticas comparativas entre folds**

Quando o treino roda em modo k-fold (ex.: 5 folds), o script gera:

- **No log e no JSON de métricas** (`outputs/statistics/train_keypoints_latest.json`):
  - `fold_metrics`: lista `[{ "fold": 1, "mAP50_95": 0.78 }, ...]` com o mAP50-95 de cada fold.
  - `fold_comparison`: resumo com `mAP50_95_mean`, `mAP50_95_std`, `mAP50_95_min`, `mAP50_95_max` e `fold_summary` (tabela por fold).
- **Relatório em Markdown** em `outputs/statistics/train_keypoints_folds.md`:
  - Qual foi o melhor fold e seu mAP50-95.
  - Média, desvio, mínimo e máximo do mAP50-95 entre folds.
  - Tabela | Fold | mAP50-95 | para comparação rápida.

Use esse comparativo para ver a variância entre folds e se algum fold ficou muito atrás (possível problema de split ou dados).

#### 3.3.1. Avaliar o modelo de keypoints no conjunto de **teste**

O split 80/10/10 coloca 10% de cada pasta em **teste** (`data/unified/yolo_pose/test/`). Essas imagens **não são usadas no treino nem na validação**. Para obter as métricas finais e as losses de avaliação no hold-out de teste:

**Como executar**

```powershell
# Avaliação completa (métricas Ultralytics + losses por amostra)
python scripts/evaluate_keypoints.py
```

- **Modelo:** por padrão usa `outputs/keypoints/train/weights/best.pt`. Para outro arquivo:
  ```powershell
  python scripts/evaluate_keypoints.py --weights caminho/para/best.pt
  ```
- **Apenas métricas oficiais** (mAP, precisão, recall), sem calcular as losses:
  ```powershell
  python scripts/evaluate_keypoints.py --no-losses
  ```

**Resultados esperados**

1. **Métricas no TESTE (Ultralytics)** — saída no terminal e via `model.val(split="test")`:
   - `metrics/pose_P`, `metrics/pose_R`: precisão e recall (pose).
   - `metrics/pose_mAP50`, `metrics/pose_mAP50-95`: mAP em IoU 0,5 e média em 0,5:0,95.
   - Outras métricas de detecção/pose conforme a versão do Ultralytics.

2. **Losses de avaliação** (média sobre amostras com predição e GT):
   - **IoU loss** (1 − IoU das caixas): quanto menor, melhor alinhamento das bbox.
   - **MSE loss** (L2 nos keypoints): erro quadrático médio nas coordenadas.
   - **L1 loss** (L1 nos keypoints): erro absoluto médio nas coordenadas.
   - **Cross entropy** (confiança vs. visibilidade): qualidade da confiança por keypoint.
   - **Focal loss** (confiança vs. visibilidade): mesma ideia, com foco em exemplos difíceis.
   - **Heatmap loss** (MSE entre heatmaps): comparação em espaço de mapas de calor.

Exemplo de saída no terminal:

```
Avaliando no conjunto de TESTE (hold-out).
  Modelo: outputs/keypoints/train/weights/best.pt
  Data:   data/unified/yolo_pose/data.yaml
  Split:  test | imgsz: 640

Métricas no TESTE (Ultralytics):
  metrics/pose_P: 0.9234
  metrics/pose_R: 0.8901
  metrics/pose_mAP50: 0.9123
  metrics/pose_mAP50-95: 0.7845
  ...

Losses de avaliação no TESTE (média sobre amostras com predição e GT):
  Amostras: 150 (omitidas: 2)
  IoU loss       (1 - IoU bbox): 0.052341
  MSE loss       (L2 keypoints): 12.345678
  L1 loss        (L1 keypoints): 2.123456
  Cross entropy  (conf vs vis):  0.234567
  Focal loss     (conf vs vis):  0.123456
  Heatmap loss   (MSE heatmaps):  0.012345

Concluído. Use essas métricas e losses como desempenho final no teste.
```

As losses são calculadas por imagem (predição com maior IoU em relação ao GT) e depois é feita a média. Amostras sem predição ou sem label são omitidas da média.

**Localização do resultado:** logs em `outputs/logs/`; métricas em `outputs/statistics/` (conforme gravado pelo script de avaliação).

**mAP50 (OKS) vs “acuracia visual”:** as métricas mAP50 e Precision/Recall do Ultralytics usam **OKS** (Object Keypoint Similarity), que depende da escala do animal: um mesmo erro em pixels pode ser “correto” em OKS em imagens grandes. Por isso um mAP50 alto (ex.: 93%) pode não corresponder a pontos muito próximos do GT. O script agora calcula também **PCK** (Percentage of Correct Keypoints: % de pontos com distância ≤ 20px ou ≤ 30px) e **distância média em pixels**; use essas métricas para avaliar a proximidade real pred vs anotação. Para uma única imagem: `python scripts/evaluate_keypoints.py --image "caminho/para/imagem.jpg"`.

**O que significa cada métrica (keypoints/pose):**

| Métrica | Significado (em uma frase) |
|--------|-----------------------------|
| **mAP50 (P)** | mAP com limiar OKS = 0,5: em quantos % dos casos a predição é considerada "acerto" segundo OKS. Costuma ser alto (~99%); não reflete bem proximidade em pixels. |
| **mAP50-95 (P)** | mAP médio com limiares OKS de 0,5 a 0,95 (mais exigente). Métrica padrão para comparar modelos de pose na literatura. |
| **Precision (P)** | Das predições que o modelo deu como "positivas", quantas % passaram no limiar OKS (ex.: 0,5). |
| **Recall (P)** | Dos ground truths, quantas % foram recuperadas com OKS acima do limiar. |
| **OKS** | Object Keypoint Similarity: nota de 0 a 1 por detecção; considera distância pred↔GT e escala do animal (erro em px "permitido" é maior em animais maiores). |
| **PCK@20px** | % de keypoints com distância ao GT ≤ 20 pixels. Reflete "proximidade real"; 60% = 60% dos pontos dentro de 20 px. |
| **PCK@30px** | Idem com limiar de 30 px. |
| **Distância média (px)** | Média da distância Euclidiana (em pixels) entre cada keypoint predito e o GT. Quanto menor, melhor. |
| **IoU loss** | 1 − IoU das caixas (bbox). Quanto menor, melhor o alinhamento da caixa predita com a GT. |
| **MSE loss** | Erro quadrático médio nas coordenadas (x,y) dos keypoints. Quanto menor, mais perto em L2. |
| **L1 loss** | Erro absoluto médio nas coordenadas dos keypoints. Quanto menor, mais perto em L1. |
| **Cross entropy / Focal** | Qualidade da confiança que o modelo atribui a cada keypoint (comparada à visibilidade no GT). |
| **Heatmap loss** | Diferença entre os mapas de calor da predição e do GT (em espaço redimensionado). |

#### 3.4. Treinar classificador de vacas

```powershell
python scripts/train_classifier.py
```

- **O que faz:** treina YOLOv8-cls (uma classe por vaca). Se existir `data/unified/classification_split/` (criado por `prepare_dataset`), usa **train/** e **val/**; caso contrário, usa `data/unified/classification/` com split interno. Early stopping e métricas de validação.
- **Resultado esperado:** no terminal, progresso por época (loss, accuracy); ao final, confirmação de salvamento do modelo.
- **Localização do resultado:** modelo em `outputs/classifier/train/weights/best.pt`; curvas e métricas em `outputs/statistics/`; logs em `outputs/logs/`.

#### 3.4.1. Validar o classificador (acurácia)

O **validador** (`evaluate_classifier.py`) compara a predição do modelo com o rótulo verdadeiro e calcula top-1 e top-5 accuracy. **Não é o mesmo que** `predict_cow.py` (ver abaixo).

**Qual pasta usar para validar**

- Se você rodou `prepare_dataset` e existe `data/unified/classification_split/`:
  - **Validação (durante/após o treino):** use a pasta **val** → `python scripts/evaluate_classifier.py --split val`
  - **Teste hold-out (desempenho final):** use a pasta **test** → `python scripts/evaluate_classifier.py --split test`
- Se não existir `classification_split/`, o script usa `data/unified/classification/` (comportamento antigo).

```powershell
python scripts/evaluate_classifier.py              # padrão: --split val
python scripts/evaluate_classifier.py --split val   # métricas na validação (10% dos dados)
python scripts/evaluate_classifier.py --split test # métricas no teste hold-out (10% dos dados)
python scripts/evaluate_classifier.py --weights caminho/para/best.pt
```

- **Resultado esperado:** no terminal, top-1 e top-5 accuracy no split escolhido (val ou test).
- **Localização do resultado:** métricas em `outputs/statistics/evaluate_classifier_latest.json`; logs em `outputs/logs/` (arquivo `evaluate_classifier_<timestamp>.log`).

#### 3.5. Visualizar keypoints e retas entre pontos

```powershell
python scripts/visualize_keypoints.py
```

- **O que faz:** lê imagens e labels YOLO pose e desenha pontos anatômicos e segmentos (linha dorsal, garupa, etc.) em cada imagem. Gera visualizações em **três pastas** para permitir verificação do sucesso do modelo:
  - **Conjunto completo:** `data/unified/keypoints/` → `outputs/vis/keypoints/`
  - **Validação:** `data/unified/yolo_pose/val/` (ou `fold_1/val/`) → `outputs/vis/keypoints_val/`
  - **Teste:** `data/unified/yolo_pose/test/` (ou `fold_1/test/`) → `outputs/vis/keypoints_test/`
- **Resultado esperado:** no terminal, contagem de imagens geradas em cada pasta (completo, validação, teste).
- **Localização do resultado:** imagens com keypoints e segmentos em `outputs/vis/keypoints/`, `outputs/vis/keypoints_val/` e `outputs/vis/keypoints_test/`. Use as pastas val e test para inspecionar visualmente as anotações (ground truth) dos conjuntos de validação e teste.
- **Nomes com acento:** o script suporta nomes de arquivo com acento (ex.: pastas "Fábio"); leitura e gravação usam bytes para evitar falhas do OpenCV no Windows.

#### 3.6. Análise exploratória (EDA) das features

```powershell
python scripts/analisar_features.py
```

- **O que faz:** carrega os labels de keypoints em `data/unified/keypoints/labels`, monta features geométricas (e opcionalmente de textura/cor), calcula importância (mutual_info, rf_importance ou PCA) e gera visualizações.
- **Resultado esperado:** no terminal, listagem das top features e confirmação dos arquivos gerados (histogramas, correlação, PCA, relatório).
- **Localização do resultado:** `outputs/statistics/eda/` — por exemplo `distribuicoes_features.png`, `correlacao_features.png`, `pca_2d.png`, `relatorio_eda.md`. O **Relatório Final — Estatísticas e Observações do Modelo** está em [docs/relatorio_final_estatisticas_observacoes_modelo.md](docs/relatorio_final_estatisticas_observacoes_modelo.md).

## Inferência e validação

### predict_cow.py vs evaluate_classifier.py

| Script | Função | Quando usar |
|--------|--------|-------------|
| **predict_cow.py** | **Inferência:** classifica imagens **sem rótulo** (novas fotos). Só prediz a vaca e a confiança; não calcula acurácia. | Quando você tem imagens novas e quer saber “qual vaca é?”. |
| **evaluate_classifier.py** | **Validador:** usa imagens **com rótulo** (pastas val/ ou test/ do split), compara predição com o rótulo e calcula **top-1 e top-5 accuracy**. | Quando você quer medir o desempenho do modelo (validação ou teste hold-out). |

Ou seja: **predict_cow** não é validador; é para uso em produção ou em imagens não anotadas. Para **validar** (acurácia), use **evaluate_classifier** com `--split val` ou `--split test`.

### Classificar qual vaca está na imagem (inferência)

O **predict_cow** carrega o modelo treinado (`outputs/classifier/train/weights/best.pt`), processa as imagens que você passar e, para cada uma, devolve a **classe prevista** (nome da vaca) e a **confiança** (0–1). Não usa rótulos nem calcula acurácia; é só inferência.

**Comportamento:**

1. Carrega o modelo do classificador (YOLO-cls).
2. Lê as imagens (uma via `--image` ou todas em `--input-dir`, incluindo subpastas).
3. Para cada imagem: prediz a classe (e opcionalmente o top-K) e imprime no terminal.
4. Salva cópias das imagens com a classe e a confiança desenhadas em `outputs/inference/classifier/pred/`.
5. Grava log e métricas (número de imagens, confiança média) em `outputs/logs/`.

**Comandos:**

```powershell
# Uma imagem
python scripts/predict_cow.py --image caminho/para/imagem.jpg

# Todas as imagens de uma pasta (e subpastas)
python scripts/predict_cow.py --input-dir caminho/para/pasta

# Top-K classes por imagem (ex.: top-3; o padrão vem de config.yaml → app.top_k)
python scripts/predict_cow.py --input-dir pasta/imagens --top-k 5
```

**Exemplo de uso e saída no terminal:**

```powershell
python scripts/predict_cow.py --image raw/classificacao/1323/20260101_142804_baia23_VIPWX.jpg --top-k 3
```

Saída esperada:

```
Rodando classificação em 1 imagem(ns) (top_k=3)...
raw\classificacao\1323\20260101_142804_baia23_VIPWX.jpg: top-3 → 1323=0.891, OutraVaca=0.065, MaisUma=0.032
Imagens anotadas salvas em: outputs/inference/classifier/pred
```

Com `--top-k 1` (ou omitindo `--top-k` se no config for 1), a linha fica no formato:

```
caminho/para/imagem.jpg: vaca=1323 (confiança=0.891)
```

**Resumo da saída:**

- **Terminal:** para cada imagem, o caminho do arquivo, a classe (nome da vaca) e a confiança; com top-K, as K classes mais prováveis e suas confianças.
- **Arquivos:** imagens anotadas em `outputs/inference/classifier/pred/` (ex.: `image0.jpg`, `image1.jpg`, …).
- **Log:** `outputs/logs/predict_cow_<timestamp>.log` e métricas em `outputs/statistics/`.

**Pré-requisito:** o modelo deve existir em `outputs/classifier/train/weights/best.pt`. Se não existir, o script avisa e pede para rodar antes `python scripts/train_classifier.py`.

### Detectar keypoints em imagem não anotada

**Instruções:**

```powershell
# Uma imagem
python scripts/predict_keypoints.py --image caminho/para/imagem.jpg

# Todas as imagens de uma pasta (recursivo; .jpg, .png, etc.)
python scripts/predict_keypoints.py --input-dir caminho/para/pasta
```

O script usa o modelo em `outputs/keypoints/train/weights/best.pt`. Caminhos relativos (ex.: `data/unified/yolo_pose/...`) são resolvidos a partir da raiz do projeto, mesmo rodando de dentro de `scripts/`.

**Resultado esperado:**

- **Terminal:** para cada imagem, as coordenadas (x, y) de cada keypoint com o **nome** (withers, back, hook_up, hook_down, hip, tail_head, pin_up, pin_down), conforme `config.yaml` → `keypoints.names`.
- **Arquivos:** imagens com a **box** desenhada e o rótulo **"cow"**, e cada **keypoint** com seu nome ao lado; uma imagem por arquivo de entrada. Com uma única imagem, a saída mantém o nome do arquivo; com várias, pode haver sufixo numérico em caso de nome duplicado.

**Localização do resultado:** `outputs/inference/keypoints/pred/` (ex.: `imagem.jpg` ou `image0.jpg`, `image1.jpg`, …).

Requisito: **opencv-python** para desenhar os nomes na imagem; sem ele, as imagens ainda são salvas, mas sem os textos (comportamento padrão do Ultralytics).

## Logs e estatísticas

Cada vez que um script roda, são gravados:

- **Logs** em `outputs/logs/` (ou em `paths.logs_dir` no `config.yaml`):
  - Um arquivo por execução: `<nome_script>_<timestamp>.log` com o passo a passo e as métricas daquele run.
- **Métricas em JSON** em `outputs/statistics/` (ou `paths.statistics_dir`):
  - `<nome_script>_<timestamp>.json` e `<nome_script>_latest.json` (última execução).
- **Gráficos** em `outputs/statistics/`:
  - Nos treinos (keypoints e classificador), curvas de loss/métricas em PNG (ex.: `train_keypoints_curves.png`, `train_classifier_curves.png`).
- **Treino keypoints com k-fold:** além do JSON, é gerado `outputs/statistics/train_keypoints_folds.md` com o **comparativo entre folds** (tabela | Fold | mAP50-95 |, média, desvio, mínimo e máximo). O JSON inclui `fold_metrics` e `fold_comparison`.

O **pipeline** (`python scripts/pipeline.py`) gera ainda:

- `outputs/logs/pipeline_run_<timestamp>.log`: log único com o passo a passo de cada etapa e, ao final, a seção **ESTATÍSTICAS FINAIS POR PROCESSO** (F1, acurácia, mAP, precisão, recall, etc., conforme disponível em cada script).

## Estatísticas do pipeline e conclusão do modelo

Após rodar o pipeline (ou os scripts de treino e avaliação), consulte o **último log de pipeline** (`outputs/logs/pipeline_run_<timestamp>.log`) e os JSON em `outputs/statistics/` para preencher as métricas abaixo. Esta seção serve tanto para **avaliação (pós-graduação)** quanto para **documentar o comportamento do modelo** no seu ambiente.

### Onde encontrar as métricas

| Métrica | Fonte |
|--------|--------|
| **POSE (keypoints)** | `outputs/statistics/train_keypoints_latest.json` (treino/val por fold); `evaluate_keypoints.py` no **teste** → terminal e `outputs/statistics/evaluate_keypoints_*.json` |
| **Classificação** | `outputs/statistics/evaluate_classifier_latest.json` (top-1, top-5 em val ou test); também no log do pipeline |
| **Resumo do pipeline** | `outputs/logs/pipeline_run_<timestamp>.log` (seção final **ESTATÍSTICAS FINAIS POR PROCESSO**) |

### Métricas POSE (keypoints)

Preencha com os valores da **última avaliação no conjunto de teste** (`python scripts/evaluate_keypoints.py`):

| Métrica | Valor (exemplo) | Observação |
|--------|------------------|------------|
| **mAP50 (OKS)** | — | Alto (~99%) é comum; não reflete proximidade em pixels. |
| **mAP50-95 (OKS)** | — | Métrica principal para comparar modelos de pose. |
| **Precision / Recall (pose)** | — | Do Ultralytics no split test. |
| **PCK@20px** | — | % de keypoints a ≤ 20 px do GT; reflete “proximidade real”. |
| **PCK@30px** | — | Idem com limiar 30 px. |
| **Distância média (px)** | — | Média da distância pred↔GT por keypoint; quanto menor, melhor. |
| **Losses (IoU, MSE, L1, etc.)** | — | Opcional; ver saída de `evaluate_keypoints.py`. |

Em **k-fold**, use também `outputs/statistics/train_keypoints_folds.md` para a comparação entre folds (melhor fold, média e desvio de mAP50-95).

### Métricas de classificação

Preencha com os valores de **teste hold-out** (`python scripts/evaluate_classifier.py --split test`):

| Métrica | Valor (exemplo) | Observação |
|--------|------------------|------------|
| **Top-1 accuracy** | — | Fração de imagens em que a classe prevista é a correta. |
| **Top-5 accuracy** | — | Classe correta está entre as 5 mais prováveis. |

O JSON em `outputs/statistics/evaluate_classifier_latest.json` contém esses valores para o último run.

### Conclusão geral do modelo

Use este bloco para descrever, com base no último pipeline e nas métricas acima:

- **Características do modelo:** arquitetura (YOLOv8-pose e YOLOv8-cls), número de keypoints (8), split (80/10/10 ou k-fold), augmentations e early stopping.
- **Qualidade das anotações:** origem (Label Studio), conversão para YOLO pose, verificação com `verificar_unify_convert.py`; possíveis limitações (oclusão, ângulo, consistência entre anotadores).
- **Desempenho resumido:** o que os números de POSE e classificação indicam (ex.: mAP50-95 estável entre folds; PCK@20px em torno de X%; top-1 no teste em Y%).
- **Pontos importantes:** uso de PCK e distância em px além do mAP (OKS); reprodutibilidade via `config.yaml` e pipeline único; referência ao projeto [deteccao_keypoints_vacas](https://github.com/thalessalvador/deteccao_keypoints_vacas) e a [docs/COMPARACAO_PROJETO_REFERENCIA.md](docs/COMPARACAO_PROJETO_REFERENCIA.md).

*Atualize esta seção com os valores reais do seu último pipeline e ajuste a conclusão conforme seus resultados.*

## Configuração (`config.yaml`)

| Seção | Descrição |
|-------|-----------|
| `app` | `name`, `image_size`, `confidence_threshold`, `top_k` (inferência e predict_cow) |
| `paths` | `raw_dir`, `unified_dir`, `outputs_dir`, `logs_dir`, `statistics_dir` |
| `data` | `catalogo_subdir`, `classificacao_subdir`, `keypoints_subdir`, `image_size`, `train_ratio`, `val_ratio`, `test_ratio`, `random_seed` |
| `pose` | `k_folds`, `strategy` (stratified_per_group \| group_kfold \| kfold_misturado) |
| `augmentation` | `enabled`, `horizontal_flip`, `vertical_flip`, `rotate_limit`, `brightness_contrast`, `gaussian_blur`, `hue_saturation`, `train_augment_copies`, `mosaic_enabled` |
| `training` | `device` (GPU ou "cpu"), `epochs`, `batch_size`, `classifier_batch_size`, `patience`, `workers` |
| `keypoints` | `names` (8 pontos anatômicos), `skeleton` (opcional) |
| `feature_selection` | `method` (mutual_info, rf_importance, pca), `top_k`, `top_k_for_training` (EDA e modelos com features) |

## Estrutura do projeto

```
.
├── config.yaml
├── raw/                    # Dados brutos (não versionados)
├── data/
│   └── unified/            # Dados unificados e convertidos
│       ├── keypoints/
│       ├── classification/
│       ├── classification_split/  # train/val/test 80-10-10 (criado por prepare_dataset)
│       └── yolo_pose/      # Splits para treino (pose)
├── models/
├── outputs/
│   ├── logs/              # Logs de cada script e pipeline_run_<timestamp>.log
│   ├── statistics/        # JSON de métricas e PNG de gráficos
│   ├── keypoints/         # Treino pose
│   ├── classifier/        # Treino classificador
│   ├── vis/               # Visualizações (keypoints/, keypoints_val/, keypoints_test/)
│   └── inference/         # Predições
├── src/
│   ├── app/
│   ├── config/
│   ├── data/              # unify, convert, augmentation, prepare
│   ├── features/          # feature selection
│   └── utils/             # metrics_logger (logs, métricas, gráficos)
├── scripts/
│   ├── pipeline.py
│   ├── unify_and_convert.py
│   ├── prepare_dataset.py
│   ├── train_keypoints.py
│   ├── evaluate_keypoints.py   # avaliar keypoints no conjunto de teste
│   ├── train_classifier.py
│   ├── evaluate_classifier.py   # validar classificador (top-1 / top-5 acc)
│   ├── visualize_keypoints.py
│   ├── verificar_unify_convert.py  # verificar unify+convert; --plot N ou --image path
│   ├── predict_cow.py
│   └── predict_keypoints.py
└── tests/
```

## GPU

Para usar GPU, instale PyTorch com suporte CUDA e configure em `config.yaml`:

```yaml
training:
  device: "0"   # ID da GPU (use "cpu" para forçar CPU)
```

**Verificar se o PyTorch está usando CUDA:**

```powershell
python scripts/verificar_cuda.py
```

Se aparecer "CUDA disponível: Não", em geral o PyTorch foi instalado na versão CPU. **Solução:** desinstale e reinstale com o índice CUDA (veja [docs/CUDA_WINDOWS.md](docs/CUDA_WINDOWS.md)):

```powershell
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
```

Depois reinicie o terminal e rode `verificar_cuda.py` de novo. Detalhes e outras versões (cu124, etc.) em [docs/CUDA_WINDOWS.md](docs/CUDA_WINDOWS.md).

## Feature selection

O módulo `src/features` permite analisar quais **features derivadas dos keypoints** são mais discriminativas:

```python
from src.features import (
    select_top_keypoints,
    compute_keypoint_importance,
    compute_feature_correlations,
)
```

Configure o método em `config.yaml` → `feature_selection.method`.

### Tipos de features criadas

- **Geométricas (invariantes a escala relativa):**
  - `len_norm_*`: comprimentos de segmentos normalizados por um segmento de referência (ex.: withers–hip).
  - `ratio_len_*_over_*`: razões entre comprimentos de segmentos adjacentes.
  - `angle_*`: ângulos normalizados (0–1) entre segmentos em torno de pontos-chave (back, hip, pin_up).
  - `centroid_dist_norm_*`: distância normalizada de cada keypoint ao centróide da vaca.
  - `sig_body_length`: comprimento do animal em coordenadas normalizadas (ex.: withers–tail_head).
  - `sig_dist_*_*`: assinatura geométrica baseada na matriz de distâncias normalizadas entre todos os pares de keypoints.

- **Textura e cor ao redor de cada keypoint:**
  - `tex_<kp>_mean_gray`: média de intensidade em tons de cinza na vizinhança do ponto.
  - `tex_<kp>_std_gray`: desvio padrão (contraste local).
  - `tex_<kp>_lap_var`: variância do Laplaciano (medida de textura/detalhe).
  - `tex_<kp>_mean_r`, `tex_<kp>_mean_g`, `tex_<kp>_mean_b`: média de cor (R, G, B) ao redor do ponto.

### Como usar na prática

```python
from pathlib import Path
from src.features import (
    select_top_keypoints,
    compute_keypoint_importance,
    compute_feature_correlations,
)

labels_dir = Path("data/unified/keypoints/labels")

# Top-k features mais importantes (com base no método configurado)
top_feats = select_top_keypoints(labels_dir, top_k=10)

# Importância detalhada de todas as features
importance = compute_keypoint_importance(labels_dir)

# Correlação entre features (por exemplo, |corr| >= 0.7)
corrs = compute_feature_correlations(labels_dir, min_abs_corr=0.7)
```

---

## Observações e troubleshooting

- **Rodar da raiz do projeto:** execute os comandos a partir da pasta raiz do repositório (onde está `config.yaml`). Exemplo: `python scripts/train_keypoints.py`. Assim, caminhos relativos em `config.yaml` e nos scripts funcionam corretamente.
- **CUDA não detectada:** use `python scripts/verificar_cuda.py`. Se aparecer "CUDA disponível: Não", instale PyTorch com o índice CUDA adequado (veja [docs/CUDA_WINDOWS.md](docs/CUDA_WINDOWS.md) e comentários em `requirements.txt`).
- **Erro "No module named 'src'":** certifique-se de estar na raiz do projeto e de ter ativado o ambiente onde instalou as dependências.
- **Verificação das anotações:** após `unify_and_convert.py`, use `verificar_unify_convert.py` para checar formato e contagens; `--amostras N` mostra conteúdo de N labels no terminal; `--plot N` gera N imagens com bbox e keypoints em `outputs/statistics/verificar_unify_convert/`.
- **mAP alto mas pontos longe do GT:** as métricas mAP50/mAP50-95 usam OKS (tolerante à escala). Para "proximidade real" use as métricas **PCK** e **distância média (px)** reportadas por `evaluate_keypoints.py`; ver [docs/COMPARACAO_PROJETO_REFERENCIA.md](docs/COMPARACAO_PROJETO_REFERENCIA.md).

**Documentação adicional:** [docs/COMPARACAO_PROJETO_REFERENCIA.md](docs/COMPARACAO_PROJETO_REFERENCIA.md) (métricas e comparação), [docs/relatorio_final_estatisticas_observacoes_modelo.md](docs/relatorio_final_estatisticas_observacoes_modelo.md) (Relatório Final — Estatísticas e Observações do Modelo), [docs/SETUP_GPU.md](docs/SETUP_GPU.md) e [docs/CUDA_WINDOWS.md](docs/CUDA_WINDOWS.md) (GPU e PyTorch/CUDA).
