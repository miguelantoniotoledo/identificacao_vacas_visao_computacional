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

- **Pose (keypoints):** stratified_per_group em cada grupo: **80% train**, **10% val**, **10% test** em `data/unified/yolo_pose/` (train/images, val/images, test/images).
- **Classificação:** se existir `data/unified/classification/`, o mesmo script gera **split 80-10-10** em `data/unified/classification_split/`:
  - **train/** — treino do classificador (80% das fotos de cada vaca)
  - **val/** — validação durante o treino e para rodar o validador (10%)
  - **test/** — teste hold-out, nunca usado no treino (10%)

Use as mesmas proporções do config: `data.train_ratio: 0.8`, `data.val_ratio: 0.1`, `data.test_ratio: 0.1`.

#### 3.3. Treinar modelo de keypoints

```powershell
python scripts/train_keypoints.py
```

- Usa YOLOv8-pose
- Treino em GPU (configurável em `config.yaml` → `training.device`)
- Augmentation automático pelo Ultralytics
- Se existir **k-fold** (group k-fold por pasta, em `data/unified/yolo_pose/fold_1/`, …), treina um modelo por fold e escolhe o melhor por mAP50-95; o `best.pt` do melhor fold é copiado para `outputs/keypoints/train/weights/best.pt` para inferência.

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

#### 3.4. Treinar classificador de vacas

```powershell
python scripts/train_classifier.py
```

- Usa YOLOv8-cls; uma classe por vaca.
- Se existir `data/unified/classification_split/` (criado por `prepare_dataset`), o treino usa **train/** para treino e **val/** para validação (early stopping e métricas). Caso contrário, usa `data/unified/classification/` e o Ultralytics faz um split interno.

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

- **Métricas:** `top1_acc` (acurácia: a classe com maior confiança é a correta) e `top5_acc` (a classe correta está no top-5).
- Log e métricas em `outputs/logs/` e `outputs/statistics/evaluate_classifier_latest.json`.

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
- **Treino keypoints com k-fold:** além do JSON, é gerado `outputs/statistics/train_keypoints_folds.md` com o **comparativo entre folds** (tabela | Fold | mAP50-95 |, média, desvio, mínimo e máximo). O JSON inclui `fold_metrics` e `fold_comparison`.

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
│       ├── classification_split/  # train/val/test 80-10-10 (criado por prepare_dataset)
│       └── yolo_pose/      # Splits para treino (pose)
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
│   ├── evaluate_classifier.py # validar classificador (top-1 / top-5 acc)
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
