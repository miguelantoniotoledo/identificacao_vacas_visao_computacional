# Relatório Final — Estatísticas e Observações do Modelo

## Projeto

**Identificação de Vacas por Visão Computacional — Keypoints e Classificação**

A análise é gerada pelo script `analisar_features.py` e os artefatos (gráficos, relatório resumido) ficam em `outputs/statistics/eda/`. Para reproduzir: `python scripts/analisar_features.py`.

---

# 1. Introdução

Este relatório descreve a **Análise Exploratória de Dados (EDA)** das **features derivadas dos keypoints anatômicos** utilizadas neste sistema de identificação de vacas.

Os objetivos da EDA são:

* compreender a estrutura do dataset de keypoints e das features derivadas
* avaliar a qualidade e a discriminabilidade das features
* identificar redundâncias e possíveis problemas nos dados
* orientar a seleção de features para treino (config: `feature_selection.top_k_for_training`)
* analisar a separabilidade no espaço de features (ex.: PCA)

O pipeline utiliza **8 keypoints anatômicos** (withers, back, hook_up, hook_down, hip, tail_head, pin_up, pin_down) anotados no Label Studio e convertidos para formato YOLO pose. A partir das coordenadas e da visibilidade desses pontos, o módulo `src/features` constrói **features geométricas** (e opcionalmente de textura/cor) que representam a morfologia corporal do animal. Essas features podem ser usadas em modelos de classificação supervisionada ou em análises de perfil morfológico.

---

# 2. Visão Geral do Dataset

## 2.1 Estrutura geral

O dataset da EDA é construído a partir dos **labels de keypoints** em `data/unified/keypoints/labels/` (um arquivo `.txt` por imagem, formato YOLO pose: classe, bbox, 8 keypoints × 3 valores). Não há CSV pré-gerado: as features são calculadas sob demanda pelo script.

| Propriedade        | Descrição |
| ------------------ | --------- |
| **Amostras**       | Uma por imagem com label válido em `data/unified/keypoints/labels/` |
| **Número de features** | Variável: geométricas (comprimentos normalizados, razões, ângulos, distância ao centróide, assinatura de distâncias) + opcionalmente textura/cor por keypoint |
| **Classes / grupos** | Identificação por prefixo do nome do arquivo (ex.: pasta/vaca) para análise estratificada no split |

Após rodar `python scripts/analisar_features.py`, o relatório resumido em `outputs/statistics/eda/relatorio_eda.md` contém o número exato de amostras e de features da última execução.

## 2.2 Composição das features

As features são agrupadas em:

* **Comprimentos normalizados** (`len_norm_*`)
* **Razões entre comprimentos** (`ratio_len_*_over_*`)
* **Ângulos anatômicos** (`angle_*`)
* **Distância ao centróide** (`centroid_dist_norm_*`)
* **Comprimento corporal e assinatura geométrica** (`sig_body_length`, `sig_dist_*_*`)
* **Textura e cor** (opcional: `tex_<kp>_mean_gray`, `tex_<kp>_std_gray`, etc.)

---

# 3. Origem das Amostras

| Origem                          | Descrição |
| ------------------------------- | --------- |
| **Dados reais (keypoints)**    | Imagens anotadas no Label Studio, convertidas por `unify_and_convert.py`; cada imagem gera uma amostra na EDA |
| **Augmentation no treino**      | O split de **treino** em `data/unified/yolo_pose/` pode conter imagens aumentadas (flip, rotação, blur, HSV, mosaic) geradas por `prepare_dataset.py`; a EDA é feita sobre os **labels originais** em `data/unified/keypoints/`, não sobre o conjunto já aumentado |

Ou seja: a EDA reflete a distribuição das **anotações originais**. O aumento de dados afeta apenas o treino do modelo de pose e do classificador, não a construção das features nesta análise.

---

# 4. Distribuição de Classes / Grupos

As amostras são agrupadas pelo **prefixo do nome do arquivo** (parte antes de `__`), que corresponde à pasta de origem (ex.: vaca ou sessão). O split 80/10/10 é estratificado por esse grupo para evitar vazamento entre train/val/test.

* Se houver **balanceamento** entre grupos, as métricas de classificação tendem a ser mais estáveis.
* Desbalanceamento pode indicar maior representação de algumas vacas ou sessões; convém verificar em `outputs/statistics/eda/` e nos logs do `prepare_dataset.py`.

---

# 5. Análise de Features

**Nota:** A descrição completa de cada tipo de feature está no README do repositório (seção *Feature selection*) e pode ser consultada em `config.yaml` (keypoints e feature_selection).

## 5.1 Tipos de Features

### Comprimentos normalizados

Exemplo:

```
len_norm_withers_back
len_norm_back_hip
len_norm_hip_tail_head
```

Segmentos entre keypoints adjacentes, normalizados por um segmento de referência (ex.: withers–hip). **Invariantes à escala**, reduzindo o impacto da distância da câmera e do tamanho aparente do animal.

---

### Razões entre comprimentos

Exemplo:

```
ratio_len_withers_back_over_back_hip
ratio_len_back_hip_over_hip_tail_head
```

Razões entre segmentos consecutivos; também invariantes à escala e úteis para proporções corporais.

---

### Ângulos anatômicos

Exemplo:

```
angle_back_withers_hip
angle_hip_back_tail_head
angle_pin_up_hip_pin_down
```

Ângulo no keypoint central (em radianos normalizados 0–1). Capturam **postura e inclinação** da linha dorsal e da região pélvica.

---

### Distância ao centróide

Exemplo:

```
centroid_dist_norm_withers
centroid_dist_norm_back
centroid_dist_norm_hip
```

Distância de cada keypoint ao centróide dos pontos visíveis, normalizada pelo comprimento base. Fornece uma **assinatura de forma global** sem dependência de escala absoluta.

---

### Comprimento corporal e assinatura geométrica

Exemplo:

```
sig_body_length
sig_dist_withers_back
sig_dist_withers_hip
...
```

* `sig_body_length`: comprimento do animal (ex.: withers–tail_head) em coordenadas da imagem.
* `sig_dist_*_*`: matriz de distâncias entre pares de keypoints, normalizada pelo comprimento corporal. Caracteriza a **forma global** da silhueta.

---

### Textura e cor (opcional)

Exemplo:

```
tex_withers_mean_gray
tex_back_std_gray
tex_hip_lap_var
tex_back_mean_r, tex_back_mean_g, tex_back_mean_b
```

Média e desvio em tons de cinza, variância do Laplaciano e médias RGB em uma vizinhança de cada keypoint. Dependem das imagens estarem disponíveis no carregamento (labels + imagens).

---

# 6. Features Mais Informativas

A importância das features é estimada conforme `config.yaml` → `feature_selection.method`:

* **mutual_info**: informação mútua com o rótulo (grupo/vaca)
* **rf_importance**: importância do Random Forest
* **pca**: uso em componentes principais

O número de features utilizadas para treino (quando aplicável) é definido por `feature_selection.top_k_for_training` (ex.: 15). O script `analisar_features.py` lista as **top-k melhores** e grava em `outputs/statistics/eda/relatorio_eda.md`.

### Interpretação

As variáveis mais importantes tendem a capturar:

* proporções corporais (razões e comprimentos normalizados)
* postura (ângulos)
* forma global (distância ao centróide, assinatura de distâncias)

Isso sugere que **diferenças morfológicas entre vacas são capturáveis por relações geométricas entre keypoints**.

---

# 7. Redundância de Features

A matriz de correlação (amostra de features) é gerada em `outputs/statistics/eda/correlacao_features.png`. Alta correlação entre duas features indica **informação redundante**.

* Redundância não é necessariamente um problema para todos os modelos, mas pode orientar:
  * redução de dimensionalidade
  * simplificação do conjunto de features para treino

---

# 8. Features Constantes ou Quase Constantes

Se alguma feature tiver variância muito baixa (ou constante), ela pouco contribui para discriminação. O relatório `relatorio_eda.md` inclui **média e desvio** por feature; desvio próximo de zero indica variabilidade quase nula. Tais features podem ser ignoradas na seleção.

---

# 9. Análise de Variância (PCA)

O script gera uma projeção em 2D usando os dois primeiros componentes principais em `outputs/statistics/eda/pca_2d.png`. O `relatorio_eda.md` registra a **variância explicada** por PC1 e PC2.

| Componente | Uso |
| ---------- | ----- |
| **PC1**    | Maior variância explicada (ex.: 30–40%) |
| **PC2**    | Segunda maior (ex.: 15–20%) |

Interpretação típica:

* **Variância total dos 2 primeiros PCs** em torno de 40–50% indica que existe **estrutura no espaço de features**, mas as classes podem **sobrepor-se** (separabilidade não linear).
* Isso é coerente com a ideia de que algumas vacas têm morfologia semelhante e outras são mais distintas.

---

# 10. Observações Sobre o Dataset

## 10.1 Qualidade das anotações

* Anotações originadas no **Label Studio**, convertidas para YOLO pose por `unify_and_convert.py`.
* Recomenda-se rodar `verificar_unify_convert.py` (e opcionalmente `--plot N` ou `--image path`) para checar consistência entre imagens e labels e formato dos keypoints.

## 10.2 Possível ruído de rótulos

Casos de imagens com rótulo incorreto podem impactar tanto a EDA quanto o desempenho do classificador. Recomenda-se inspeção manual de amostras duvidosas quando a acurácia for menor que o esperado.

## 10.3 Contexto de captura (câmera, baia)

Certas **câmeras** e **baias** podem apresentar taxas de erro maiores, por **diferenças de domínio visual**. Neste projeto, o prefixo do nome do arquivo pode codificar câmera/baia; análises futuras podem estratificar erros por esse contexto.

## 10.4 Nomes de arquivo com acento

Pastas ou arquivos com acento (ex.: "Fábio") são suportados nos scripts que usam leitura/escrita por bytes (ex.: `visualize_keypoints.py`), evitando falhas do OpenCV no Windows.

---

# 11. Conclusões da EDA

Principais conclusões:

* O dataset de keypoints permite construir um conjunto rico de **features geométricas** (e opcionalmente de textura) a partir dos 8 pontos anatômicos.
* As features são em grande parte **invariantes à escala**, adequadas para comparação entre animais e condições de captura.
* A **redundância** entre algumas variáveis (matriz de correlação) e a **variância explicada pelo PCA** indicam que é possível reduzir dimensionalidade ou selecionar um subconjunto (top_k_for_training) sem perda relevante de informação.
* A **separabilidade entre classes** tende a ser moderada: algumas vacas compartilham morfologia semelhante.

A EDA sustenta que **a identificação de vacas por morfologia corporal usando keypoints é viável**, desde que a qualidade das anotações e o balanceamento dos grupos sejam considerados.

---

# 12. Próximos Passos

Sugestões a partir dos resultados da EDA:

* Aumentar o número de **imagens reais anotadas** por vaca para melhorar a generalização.
* Revisar **features constantes ou quase constantes** e excluí-las da seleção.
* Analisar a **matriz de confusão** do classificador (quando disponível) para identificar pares de classes mais confundidas.
* Estudar a influência de **câmera e baia** (ou outro contexto codificado no nome) no desempenho.
* Comparar **métodos de importância** (mutual_info, rf_importance) e o impacto de incluir **textura/cor** nas features.

---

# 13. Considerações Finais

Este relatório descreve a **análise exploratória das features derivadas dos keypoints anatômicos** no projeto de identificação de vacas por visão computacional. As conclusões devem ser atualizadas com os **valores reais** produzidos por `python scripts/analisar_features.py` (amostras, número de features, variância PCA, top-k e tabelas em `outputs/statistics/eda/relatorio_eda.md`).
