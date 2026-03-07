# Comparação com o projeto de referência

Referência: [thalessalvador/deteccao_keypoints_vacas](https://github.com/thalessalvador/deteccao_keypoints_vacas)  
Pipeline em 3 fases: YOLO Pose → geração de features em CSV → identificação com classificadores tabulares (XGBoost, CatBoost, RF, SVM, KNN, MLP, Siamese, etc.), com CLI, k-fold, matriz de confusão e Optuna.

---

## O que temos de melhor (nosso projeto)

| Aspecto | Nosso projeto | Projeto de referência |
|--------|----------------|------------------------|
| **Classificador de identidade** | YOLO-cls direto nas **imagens** (uma pasta por vaca). Mais simples de operar e não exige etapa intermediária de extração de features por imagem. | Classificação **tabular** sobre features geométricas (CSV). Exige fase “gerar-features” rodando o modelo de pose em todas as imagens de classificação. |
| **Features de aparência** | **Textura e cor** ao redor de cada keypoint (`tex_*_mean_gray`, `std_gray`, `lap_var`, `mean_r/g/b`), além de geométricas. | Foco em features **apenas geométricas** (distâncias, razões, ângulos, shape context). Não usa textura/cor local. |
| **Assinatura geométrica** | **Matriz de distâncias** entre todos os pares de keypoints normalizada pelo comprimento do animal (`sig_dist_*`, `sig_body_length`) + distância ao **centróide** por ponto. | Não usa explicitamente “assinatura” como matriz de distâncias normalizada nem centróide como feature global. |
| **Invariância de escala** | Projeto desenhado desde o início para **não usar distâncias absolutas**: só razões, ângulos e comprimentos normalizados. | Também prioriza features relativas; documentação explícita sobre evitar distâncias absolutas. |
| **Logs e métricas** | Cada script grava **log + JSON de métricas** em `outputs/logs` e `outputs/statistics`; **gráficos** (curvas de treino) em PNG; **log consolidado do pipeline** com passo a passo e estatísticas finais. | Logs em `saidas/logs/app.log`; relatórios em `saidas/relatorios/` (JSON, matriz de confusão, gráficos). |
| **Inferência pontual** | Scripts dedicados: **uma foto → qual vaca** (`predict_cow.py`) e **uma foto → keypoints** (`predict_keypoints.py`), com saída em terminal e imagens salvas. | Comando `classificar-imagem` e `inferir-pose` via CLI; fluxo completo documentado. |
| **Configuração** | Tudo em **um único `config.yaml`** na raiz (paths, data, augmentation, training, keypoints, feature_selection). | `config/config.yaml` com muitas seções (pose, classificacao, paths, etc.). |
| **Estrutura de execução** | **Um comando** para pipeline completo (`python scripts/pipeline.py`) com flags para pular etapas. Scripts independentes para cada etapa. | CLI única (`python -m src.cli`) com subcomandos; `pipeline-completo` executa sequência fixa. |

---

## Possíveis melhorias (inspiradas no projeto de referência)

### 1. Validação da fase de pose (k-fold / group k-fold)

- **Referência:** Validação com **k-fold** ou **group k-fold** por sessão/anotador para reduzir vazamento entre frames semelhantes e medir melhor a generalização.
- **Melhoria aqui:** Oferecer opção de treino da pose com k-fold (ex.: 5 folds) e/ou group k-fold por pasta/anotador; registrar métricas por fold e escolher o melhor modelo (ex.: maior mAP50-95).

### 2. Fase “gerar features” a partir das imagens de classificação

- **Referência:** Fase 2 roda o modelo de pose em **todas as imagens** de `dataset_classificacao`, extrai keypoints, calcula features geométricas e gera um **CSV** (uma linha por imagem/instância) para treino do classificador tabular.
- **Melhoria aqui:** Adicionar script `gerar_features_classificacao.py` que: (1) roda o modelo de pose nas imagens de `data/unified/classification/<vaca>/`; (2) calcula nossas features (geométricas + textura/cor + assinatura); (3) salva CSV em `data/unified/classificacao_features/features.csv`. Isso permitiria depois usar **classificadores tabulares** (XGBoost, RF, etc.) como alternativa ao YOLO-cls.

### 3. Classificadores tabulares como alternativa

- **Referência:** Múltiplos modelos (XGBoost, CatBoost, RF, SVM, KNN, MLP, MLP-Torch, Siamese-Torch) treinados sobre o CSV de features; config `classificacao.modelo_padrao`.
- **Melhoria aqui:** Manter YOLO-cls como opção principal e adicionar opção de treino/eval com **XGBoost ou Random Forest** sobre o CSV gerado na fase “gerar features”, com seleção de modelo via config.

### 4. Split de teste fixo por vaca e avaliação formal

- **Referência:** Split **90% treino / 10% teste por vaca**; avaliação no 10% com **matriz de confusão**, accuracy, F1-macro, top-k; cenário com **rejeição** (ex.: `NAO_IDENTIFICADO` quando confiança &lt; limiar).
- **Melhoria aqui:** Garantir que o split de classificação seja **por vaca** (não só shuffle global) e que exista um conjunto de teste reservado; adicionar script `avaliar_classificador.py` que gera matriz de confusão (CSV + PNG), métricas (F1, acurácia, top-k) e, se aplicável, métricas com rejeição.

### 5. Seleção da instância-alvo quando há mais de uma vaca

- **Referência:** Quando há **múltiplas vacas** na imagem, escolhe a instância-alvo por confiança, área do bbox e proximidade ao centro; filtro de qualidade por confiança média dos keypoints.
- **Melhoria aqui:** Em `predict_cow` e na eventual fase “gerar features”, se o modelo de pose retornar várias vacas, implementar critério (ex.: maior confiança, ou bbox mais central) para escolher uma única instância por imagem.

### 6. Augmentação de keypoints (ruído gaussiano) para classificação

- **Referência:** Geração de **cópias sintéticas** com ruído gaussiano nas coordenadas dos keypoints (apenas no treino), aumentando o volume de dados para o classificador.
- **Melhoria aqui:** Opção no config para, ao gerar o CSV de features, criar linhas “augmentadas” com keypoints levemente perturbados (e recalcular features), só para instâncias de treino.

### 7. Otimização de hiperparâmetros (Optuna)

- **Referência:** **Optuna** (ou random search) na fase de classificação para tune de hiperparâmetros dos modelos tabulares.
- **Melhoria aqui:** Se introduzirmos classificadores tabulares, adicionar etapa opcional de otimização (Optuna) com `n_trials` e timeout configuráveis.

### 8. Catálogo ampliado de features geométricas

- **Referência:** Conjunto grande de features (ex.: **shape context** por keypoint, áreas de polígonos normalizadas, índices de conformação, curvatura da coluna, PCA excentricidade, etc.) com lista configurável das que entram no treino.
- **Melhoria aqui:** Estender nosso módulo de features com: shape context (SC) relativo ao withers–tail_head; áreas de polígonos (ex.: pélvico, torácico) normalizadas; desvio da coluna (back/hip em relação à linha dorsal); razões adicionais e ângulos já usados no referência. Manter **seleção de features** (mutual_info, RF) e **correlação** para escolher subconjunto.

### 9. CLI unificada (opcional)

- **Referência:** Um único ponto de entrada: `python -m src.cli <comando>` (preprocessar-pose, treinar-pose, gerar-features, treinar-classificador, avaliar-classificador, classificar-imagem, pipeline-completo).
- **Melhoria aqui:** Opcionalmente criar `src/cli.py` ou `scripts/cli.py` que delegue para os scripts atuais (ex.: `python -m src.cli pipeline` → `scripts/pipeline.py`), para quem preferir um único comando.

### 10. Documentação e EDA de features

- **Referência:** Comando **analisar-features** (EDA) com gráficos e relatório em `docs/analise_features.md`; documentação detalhada das features no README.
- **Melhoria aqui:** Script `analisar_features.py` que, a partir do CSV de features (ou dos labels de keypoints), gera estatísticas descritivas, distribuições e correlações e salva em `outputs/statistics/eda/`; atualizar README com a lista completa de features (geométricas, textura/cor, assinatura) e quando usar cada uma.

### 11. Modelo de pose mais recente

- **Referência:** Uso de **YOLOv26** (yolo26n-pose.pt) quando disponível.
- **Melhoria aqui:** Avaliar troca de `yolov8n-pose.pt` para **yolo26n-pose** (ou versão mais nova do Ultralytics) quando estável, mantendo o mesmo formato de 8 keypoints.

### 12. Rejeição de predição e top-k

- **Referência:** Regra de **rejeição** (ex.: retornar “não identificado” se confiança top-1 &lt; limiar ou se margem top1–top2 for pequena); relatório de **top-k** (top-1, top-3, top-5).
- **Melhoria aqui:** Em `predict_cow`, retornar **top-k** classes com probabilidades; opção de **rejeitar** (ex.: saída “desconhecido”) quando confiança &lt; `confidence_threshold` ou margem &lt; valor configurável; incluir essas métricas no script de avaliação.

---

## Resumo

- **Pontos fortes do nosso projeto:** classificação por imagem (YOLO-cls), features de textura/cor e assinatura geométrica, invariância de escala, logs/métricas/gráficos e pipeline consolidado, inferência simples (predict_cow / predict_keypoints), config único.
- **Melhorias prioritárias sugeridas:** (1) k-fold ou group k-fold na pose; (2) fase “gerar features” + CSV para classificação; (3) split por vaca e avaliação com matriz de confusão e F1/top-k; (4) seleção de instância quando há várias vacas; (5) ampliar catálogo de features geométricas (shape context, áreas, desvio de coluna); (6) opção de classificadores tabulares + Optuna; (7) rejeição e top-k na inferência; (8) EDA de features e documentação das features.

Referência do repositório: [deteccao_keypoints_vacas](https://github.com/thalessalvador/deteccao_keypoints_vacas).
