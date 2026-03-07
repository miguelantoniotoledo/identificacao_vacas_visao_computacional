# Identificação de Vacas – Visão Computacional

Projeto baseado em **YOLO** para:
1. **Detecção de keypoints** anatômicos em imagens de vacas (pose estimation)
2. **Identificação de vacas** a partir da pasta de classificação (uma classe por vaca)

Uma **comparação com o projeto de referência** [deteccao_keypoints_vacas](https://github.com/thalessalvador/deteccao_keypoints_vacas) (o que temos de melhor e possíveis melhorias) está em [docs/COMPARACAO_PROJETO_REFERENCIA.md](docs/COMPARACAO_PROJETO_REFERENCIA.md).

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
[6] visualize_keypoints →  outputs/vis/keypoints/
```

| Passo | Script | O que faz |
|-------|--------|-----------|
| **1/6** | `unify_and_convert.py` | Unifica pastas em `raw/`; converte anotações Label Studio (JSON) → YOLO pose; copia imagens e grava labels em `data/unified/keypoints/`. Nomes com prefixo da pasta para group k-fold. |
| **2/6** | `prepare_dataset.py` | Split 90/10 (configurável); opcional **group k-fold** por pasta; copia train/val/test; **augmentation** no treino (contraste + ruído gaussiano); gera `data.yaml` para YOLO. |
| **3/6** | `analisar_features.py` | EDA: carrega labels, monta features geométricas, calcula as 15 melhores para treino; gera histogramas, correlação, PCA 2D e relatório em `outputs/statistics/eda/`. |
| **4/6** | `train_keypoints.py` | Treina YOLOv8-pose (keypoints); **early stopping** (patience); se houver k-fold, treina por fold e escolhe o melhor por mAP50-95; salva `best.pt` em `outputs/keypoints/train/weights/`. |
| **5/6** | `train_classifier.py` | Treina YOLOv8-cls (uma classe por pasta em `data/unified/classification/`); **early stopping**; salva modelo em `outputs/classifier/train/`. |
| **6/6** | `visualize_keypoints.py` | Desenha keypoints e segmentos nas imagens de keypoints; salva em `outputs/vis/keypoints/`. |

Ao final, o pipeline grava um **log consolidado** em `outputs/logs/pipeline_run_<timestamp>.log` com data/hora de cada evento e estatísticas dos scripts.

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

### 1. Ambiente

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

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

- Unifica pastas de `raw`
- Converte anotações Label Studio → formato YOLO pose
- Copia imagens para `data/unified/`

#### 3.2. Preparar splits (train/val/test)

```powershell
python scripts/prepare_dataset.py
```

#### 3.3. Treinar modelo de keypoints

```powershell
python scripts/train_keypoints.py
```

- Usa YOLOv8-pose
- Treino em GPU (configurável em `config.yaml` → `training.device`)
- Augmentation automático pelo Ultralytics
- Se existir **k-fold** (group k-fold por pasta, em `data/unified/yolo_pose/fold_1/`, …), treina um modelo por fold e escolhe o melhor por mAP50-95; o `best.pt` do melhor fold é copiado para `outputs/keypoints/train/weights/best.pt` para inferência

#### 3.4. Treinar classificador de vacas

```powershell
python scripts/train_classifier.py
```

- Usa YOLOv8-cls
- Uma classe por pasta em `raw/classificacao/<nome_vaca>/`

#### 3.5. Visualizar keypoints e retas entre pontos

```powershell
python scripts/visualize_keypoints.py
```

- Gera imagens anotadas em `outputs/vis/keypoints/` com:
  - pontos anatômicos
  - retas entre pontos (linha dorsal, garupa, etc.)

#### 3.6. Análise exploratória (EDA) das features

```powershell
python scripts/analisar_features.py
```

- Carrega os labels de keypoints em `data/unified/keypoints/labels`
- Gera estatísticas descritivas, histogramas, matriz de correlação e projeção PCA 2D
- Salva gráficos e relatório em `outputs/statistics/eda/` (`distribuicoes_features.png`, `correlacao_features.png`, `pca_2d.png`, `relatorio_eda.md`)

## Inferência (uso dos modelos treinados)

### Classificar qual vaca está na imagem

```powershell
python scripts/predict_cow.py --image caminho/para/imagem.jpg
```

Ou para uma pasta inteira:

```powershell
python scripts/predict_cow.py --input-dir caminho/para/pasta
```

Para exibir as **top-K** classes por imagem (ex.: top-3 ou top-5), use `--top-k` (o valor padrão vem de `config.yaml` → `app.top_k`):

```powershell
python scripts/predict_cow.py --input-dir pasta/imagens --top-k 5
```

Saída:

- Imprime no terminal, para cada imagem: o nome da vaca (classe YOLO) e a confiança.
- Salva imagens anotadas em `outputs/inference/classifier/pred/`.

### Detectar keypoints em imagem não anotada

```powershell
python scripts/predict_keypoints.py --image caminho/para/imagem.jpg
```

Ou para uma pasta inteira:

```powershell
python scripts/predict_keypoints.py --input-dir caminho/para/pasta
```

Saída:

- Imprime no terminal as coordenadas dos keypoints detectados (em pixels).
- Salva imagens anotadas com keypoints em `outputs/inference/keypoints/pred/`.

## Logs e estatísticas

Cada vez que um script roda, são gravados:

- **Logs** em `outputs/logs/` (ou em `paths.logs_dir` no `config.yaml`):
  - Um arquivo por execução: `<nome_script>_<timestamp>.log` com o passo a passo e as métricas daquele run.
- **Métricas em JSON** em `outputs/statistics/` (ou `paths.statistics_dir`):
  - `<nome_script>_<timestamp>.json` e `<nome_script>_latest.json` (última execução).
- **Gráficos** em `outputs/statistics/`:
  - Nos treinos (keypoints e classificador), curvas de loss/métricas em PNG (ex.: `train_keypoints_curves.png`, `train_classifier_curves.png`).

O **pipeline** (`python scripts/pipeline.py`) gera ainda:

- `outputs/logs/pipeline_run_<timestamp>.log`: log único com o passo a passo de cada etapa e, ao final, a seção **ESTATÍSTICAS FINAIS POR PROCESSO** (F1, acurácia, mAP, precisão, recall, etc., conforme disponível em cada script).

## Configuração (`config.yaml`)

| Seção | Descrição |
|-------|-----------|
| `paths` | `raw_dir`, `unified_dir`, `outputs_dir`, `logs_dir`, `statistics_dir` |
| `data` | `image_size`, `train_ratio`, `val_ratio`, `random_seed` |
| `augmentation` | `horizontal_flip`, `vertical_flip`, `rotate_limit`, etc. |
| `training` | `device` (GPU), `epochs`, `batch_size`, `patience` |
| `keypoints` | Nomes dos 8 pontos anatômicos |
| `feature_selection` | `method` (mutual_info, rf_importance), `top_k` |

## Estrutura do projeto

```
.
├── config.yaml
├── raw/                    # Dados brutos (não versionados)
├── data/
│   └── unified/            # Dados unificados e convertidos
│       ├── keypoints/
│       ├── classification/
│       └── yolo_pose/      # Splits para treino
├── models/
├── outputs/
│   ├── logs/              # Logs de cada script e pipeline_run_<timestamp>.log
│   ├── statistics/        # JSON de métricas e PNG de gráficos
│   ├── keypoints/         # Treino pose
│   ├── classifier/        # Treino classificador
│   ├── vis/               # Visualizações
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
│   ├── train_classifier.py
│   ├── visualize_keypoints.py
│   ├── predict_cow.py
│   └── predict_keypoints.py
└── tests/
```

## GPU

Para usar GPU, instale PyTorch com suporte CUDA e configure em `config.yaml`:

```yaml
training:
  device: "0"   # ID da GPU
```

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
