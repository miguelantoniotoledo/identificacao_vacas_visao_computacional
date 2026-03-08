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
| **2/6** | `prepare_dataset.py` | Split 80/10/10 por pasta; **todas as augmentations** (flip, rotate, blur, HSV, ruído) + **mosaic** (2x2) no treino; gera `data.yaml`. |
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

- Unifica pastas de `raw`
- Converte anotações Label Studio → formato YOLO pose
- Copia imagens para `data/unified/`

**Como verificar se deu certo:**

1. **Script automático (recomendado):**
   ```powershell
   python scripts/verificar_unify_convert.py
   ```
   Com amostras de conteúdo dos labels:
   ```powershell
   python scripts/verificar_unify_convert.py --amostras 5
   ```
   Para plotar N imagens com **bbox e keypoints** desenhados (salvas em `outputs/statistics/verificar_unify_convert/`):
   ```powershell
   python scripts/verificar_unify_convert.py --plot 3
   ```
   O script confere: pastas existem, mesmo número de imagens e labels, cada imagem tem um `.txt` com o mesmo nome (stem), formato do label (classe + bbox + 8 keypoints × 3) e compara com o último run em `outputs/statistics/unify_and_convert_latest.json`.

2. **Verificação manual:** conferir que em `data/unified/keypoints/` existem `images/` e `labels/` com a mesma quantidade de arquivos; para uma imagem qualquer (ex.: `foto.jpg`) deve existir `foto.txt` em `labels/` com uma linha no formato YOLO pose (classe, 4 bbox, 24 valores de keypoints). Ver também o JSON do último run: `converted` deve bater com o total e `failed` deve ser 0.

#### 3.2. Preparar splits (train/val/test)

```powershell
python scripts/prepare_dataset.py
```

Por padrão usa **stratified_per_group**: em cada pasta, **80% treino**, **10% validação** e **10% teste** (conjuntos disjuntos). Config: `data.train_ratio: 0.8`, `data.val_ratio: 0.1`, `data.split_test_ratio: 0.1`.

#### 3.3. Treinar modelo de keypoints

```powershell
python scripts/train_keypoints.py
```

- Usa YOLOv8-pose
- Treino em GPU (configurável em `config.yaml` → `training.device`)
- Augmentation automático pelo Ultralytics
- Se existir **k-fold** (group k-fold por pasta, em `data/unified/yolo_pose/fold_1/`, …), treina um modelo por fold e escolhe o melhor por mAP50-95; o `best.pt` do melhor fold é copiado para `outputs/keypoints/train/weights/best.pt` para inferência

#### 3.3.1. Avaliar o modelo de keypoints no conjunto de **teste**

O split 80/10/10 coloca 10% de cada pasta em **teste** (`data/unified/yolo_pose/test/`). Essas imagens **não são usadas no treino nem na validação**. Para obter as métricas finais no hold-out de teste:

```powershell
python scripts/evaluate_keypoints.py
```

- Usa por padrão o modelo `outputs/keypoints/train/weights/best.pt` e o `data.yaml` em `data/unified/yolo_pose/` (que aponta `test` para `test/images`).
- Para outro modelo: `python scripts/evaluate_keypoints.py --weights caminho/para/best.pt`
- As métricas impressas (mAP, precisão, recall, etc.) são as do **conjunto de teste**, ou seja, a estimativa de desempenho em dados nunca vistos no treino.

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
| `augmentation` | `horizontal_flip`, `vertical_flip`, `rotate_limit`, `contrast_limit`, `gaussian_noise_std`, `train_augment_copies`, `mosaic_enabled` (mosaic usa todo o conjunto: 1 mosaic a cada 4 imagens) |
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
│   ├── evaluate_keypoints.py   # avaliar keypoints no conjunto de teste
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
