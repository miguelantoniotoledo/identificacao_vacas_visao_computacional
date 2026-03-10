# Identificação de Vacas – Visão Computacional

Projeto em **YOLO** para: (1) **detecção de keypoints** anatômicos em vacas (pose) e (2) **identificação de vacas** por imagem (uma classe por vaca). Dados anotados no Label Studio; pipeline reprodutível via `config.yaml` e `scripts/pipeline.py`.

**Para avaliadores (pós-graduação):** objetivo, metodologia (unificação → conversão → split 80/10/10 → treino YOLOv8-pose e YOLOv8-cls), métricas (keypoints: mAP, **PCK@20px/30px**, distância média px; classificador: top-1/top-5) e reprodutibilidade estão descritos no [Relatório Final](docs/relatorio_final_estatisticas_observacoes_modelo.md).

---

## Como usar

**Pré-requisitos:** Python 3.10+, dados em `raw/` (ver [Estrutura de dados](#estrutura-de-dados)). GPU opcional ([docs/SETUP_GPU.md](docs/SETUP_GPU.md)).

1. Na **raiz do projeto**: `python -m venv .venv` → `.\.venv\Scripts\Activate.ps1` → `pip install -r requirements.txt` (GPU: instale PyTorch com CUDA antes; veja `requirements.txt` e SETUP_GPU).
2. Coloque em `raw/` os dados conforme [Estrutura de dados](#estrutura-de-dados).
3. Rode o pipeline: `python scripts/pipeline.py` (unificação → splits → EDA → treino keypoints → treino classificador → visualizações).
4. Resultados: modelos em `outputs/keypoints/train/weights/best.pt` e `outputs/classifier/train/weights/best.pt`; métricas em `outputs/statistics/`; log em `outputs/logs/pipeline_run_<timestamp>.log`.
5. Sempre execute da raiz (ex.: `python scripts/train_keypoints.py`).

Flags do pipeline: `--skip-eda`, `--skip-train-keypoints`, `--skip-train-classifier`, `--skip-visualize`.

---

## Fluxo e scripts

```
raw/catalogo, raw/classificacao
        ↓
unify_and_convert   →  data/unified/keypoints/ + classification/
        ↓
prepare_dataset     →  data/unified/yolo_pose/ (train/val/test ou folds) + classification_split/
        ↓
analisar_features   →  outputs/statistics/eda/
        ↓
train_keypoints     →  outputs/keypoints/train/weights/best.pt
        ↓
train_classifier    →  outputs/classifier/train/weights/best.pt
        ↓
visualize_keypoints →  outputs/vis/keypoints/ + keypoints_val/ + keypoints_test/
```

| Script | Comando | Saída principal |
|--------|---------|------------------|
| unify_and_convert | `python scripts/unify_and_convert.py` | `data/unified/keypoints/`, `classification/` |
| prepare_dataset | `python scripts/prepare_dataset.py` | `yolo_pose/train|val|test`, `classification_split/` |
| analisar_features | `python scripts/analisar_features.py` | `outputs/statistics/eda/` |
| train_keypoints | `python scripts/train_keypoints.py` | `outputs/keypoints/train/weights/best.pt` |
| train_classifier | `python scripts/train_classifier.py` | `outputs/classifier/train/weights/best.pt` |
| visualize_keypoints | `python scripts/visualize_keypoints.py` | `outputs/vis/keypoints/`, `keypoints_val/`, `keypoints_test/` |
| evaluate_keypoints | `python scripts/evaluate_keypoints.py` | Métricas no teste: mAP (OKS), **PCK@20px/30px**, distância média px → `outputs/statistics/evaluate_keypoints_latest.json` |
| evaluate_classifier | `python scripts/evaluate_classifier.py --split test` | Top-1/top-5 accuracy → `outputs/statistics/evaluate_classifier_latest.json` |
| verificar_unify_convert | `python scripts/verificar_unify_convert.py` [--plot N \| --image path] | Verificação das anotações; imagens em `outputs/statistics/verificar_unify_convert/` |
| predict_cow | `python scripts/predict_cow.py --image path` ou `--input-dir path` | Inferência classificador → `outputs/inference/classifier/pred/` |
| predict_keypoints | `python scripts/predict_keypoints.py --image path` ou `--input-dir path` | Inferência keypoints → `outputs/inference/keypoints/pred/` |

**Keypoints no teste:** priorize **PCK@20px**, **PCK@30px** e **distância média (px)** para “acurácia real”; mAP50 (OKS) costuma ser alto (~99%) e não reflete proximidade em pixels. 

**Classificador:** `evaluate_classifier.py` usa imagens com rótulo (val/test) e calcula acurácia; `predict_cow.py` faz só inferência em imagens novas.

---

## Estrutura de dados

```
raw/
├── catalogo/
│   └── <nome_vaca>/
│       ├── *.jpg, *.png
│       └── Key_points/   # JSONs do Label Studio (sem extensão)
└── classificacao/
    └── <nome_vaca>/
        └── *.jpg, *.png
```

---

## Configuração e estrutura do projeto

**config.yaml:** `paths`, `data` (train_ratio, val_ratio, test_ratio, image_size), `pose` (k_folds, strategy: stratified_per_group | group_kfold | kfold_misturado), `augmentation`, `training` (device, epochs, batch_size, patience), `keypoints`, `feature_selection`.

**Estrutura:** `config.yaml`, `raw/`, `data/unified/` (keypoints, classification, classification_split, yolo_pose), `outputs/` (logs, statistics, keypoints, classifier, vis, inference), `src/`, `scripts/`, `tests/`. Ver árvore em [Estrutura do projeto](#estrutura-do-projeto) abaixo.

---

## Logs e métricas

- Logs: `outputs/logs/<script>_<timestamp>.log`; pipeline: `pipeline_run_<timestamp>.log`.
- Métricas: `outputs/statistics/<script>_latest.json` e gráficos (treinos, EDA).
- Keypoints (teste): PCK e distância em px em `evaluate_keypoints_latest.json` e no terminal.

---

## GPU e troubleshooting

- **GPU:** `config.yaml` → `training.device: "0"`. Verificar: `python scripts/verificar_cuda.py`. Instalação CUDA: [docs/SETUP_GPU.md](docs/SETUP_GPU.md), [docs/CUDA_WINDOWS.md](docs/CUDA_WINDOWS.md).
- **Raiz do projeto:** rodar comandos da pasta onde está `config.yaml`.
- **"No module named 'src'":** ativar o venv e estar na raiz.
- **Anotações:** após unify_and_convert, use `verificar_unify_convert.py` (--plot N ou --image path).

**Documentação:** [Relatório Final](docs/relatorio_final_estatisticas_observacoes_modelo.md), [SETUP_GPU](docs/SETUP_GPU.md), [CUDA_WINDOWS](docs/CUDA_WINDOWS.md). Feature selection: `config.yaml` → `feature_selection`; uso em `src/features`.

---

## Chamadas dos scripts (referência)

Todos os comandos são executados na **raiz do projeto** (`python scripts/<script>.py ...`).

| Script | Argumentos possíveis | Descrição do argumento |
|--------|----------------------|-------------------------|
| **pipeline** | `--skip-train-keypoints` | Pular treino do modelo de keypoints (YOLO pose). |
| pipeline | `--skip-train-classifier` | Pular treino do classificador de vacas (YOLO cls). |
| pipeline | `--skip-visualize` | Pular geração de visualizações de keypoints. |
| pipeline | `--skip-eda` | Pular análise exploratória (EDA) das features. |
| **unify_and_convert** | `--debug` | Imprimir resumo no terminal e as primeiras 15 falhas com motivo (para debugar). |
| **prepare_dataset** | *(nenhum)* | Gera yolo_pose/ e classification_split/ a partir de data/unified. |
| **analisar_features** | *(nenhum)* | Gera EDA em outputs/statistics/eda/. |
| **train_keypoints** | *(nenhum)* | Treina modelo de pose; usa config e data/unified/yolo_pose. |
| **train_classifier** | *(nenhum)* | Treina classificador; usa config e classification_split ou classification. |
| **evaluate_keypoints** | `--split` `test` \| `val` | Conjunto a avaliar (padrão: test). |
| evaluate_keypoints | `--weights` *path* | Caminho do modelo (ex.: outputs/keypoints/train/weights/best.pt). |
| evaluate_keypoints | `--no-losses` | Só rodar model.val(); não calcular losses por amostra. |
| evaluate_keypoints | `--image` *path* | Avaliar apenas esta imagem; label em \<split\>/labels com mesmo stem. |
| **evaluate_classifier** | `--split` `val` \| `test` | Pasta a usar: val (10%) ou test (hold-out). Padrão: val. |
| evaluate_classifier | `--weights` *path* | Caminho do modelo (ex.: outputs/classifier/train/weights/best.pt). |
| **visualize_keypoints** | *(nenhum)* | Gera visualizações em outputs/vis/keypoints/, keypoints_val/, keypoints_test/. |
| **verificar_unify_convert** | `--amostras` *N* | Mostrar conteúdo dos primeiros N arquivos de label (0 = não mostrar). |
| verificar_unify_convert | `--plot` *N* | Plotar N imagens com bbox e keypoints; salva em outputs/statistics/verificar_unify_convert/. |
| verificar_unify_convert | `--image` *path* | Uma única imagem para desenhar anotações originais (keypoints). |
| **predict_cow** | `--image` *path* | Caminho para uma imagem única. |
| predict_cow | `--input-dir` *path* | Diretório contendo imagens para classificação. |
| predict_cow | `--top-k` *K* | Mostrar top-K classes por imagem (default: config app.top_k). |
| **predict_keypoints** | `--image` *path* | Caminho para uma imagem única. |
| predict_keypoints | `--input-dir` *path* | Diretório contendo imagens para detecção de keypoints. |
| **verificar_cuda** | *(nenhum)* | Verifica se PyTorch enxerga CUDA e GPU. |

*Scripts de inferência:* use `--image` **ou** `--input-dir` (ou ambos); pelo menos um é obrigatório em `predict_cow` e `predict_keypoints`.

---

## Estrutura do projeto

```
.
├── config.yaml
├── raw/
├── data/unified/          # keypoints, classification, classification_split, yolo_pose
├── outputs/               # logs, statistics, keypoints, classifier, vis, inference
├── src/                   # config, data, features, utils
├── scripts/               # pipeline, unify_and_convert, prepare_dataset, train_*, evaluate_*, predict_*, visualize_*, verificar_*
└── tests/
```

## Considerações e análise preliminar

Métricas e acurácias dos modelos (keypoints e classificação) e exemplos de como reproduzir a avaliação no teste: [Métricas e acurácias dos modelos](docs/METRICAS_ACURACIAS_MODELOS.md).

- o dataset possui uma especificidade técnica na anotação acima do que pós graduandos possuem, sendo necessária a atribuição de um técnico (veterinário) para geração de anotações mais precisas.
- a qualidade das imagens, método de exposição, distância e tamanho do brete podem ser geradores de menor acurácia no modelo. Além disso a presença de equipamentos ou pessoas que sobrepoem os animais também podem influenciar.
- foram verificados de forma aleatória a presença de vacas fora de suas pastas reais, possuindo outros nomes. Dessa forma a acurácia do modelo também é prejudicada devido à não garantia de não aleatoridade (que previne o leaking no treinamento)
- ao usar split por aleatório tanto para keypoints quanto para o classificador houve um processo de leaking generalizado. Dessa forma optou-se por fazer agrupamentos por vaca (individuo).