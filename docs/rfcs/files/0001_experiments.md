---
title: RFC 0001 - Expansão dos Experimentos
summary: Este RFC propõe a expansão dos experimentos do projeto de mestrado com a introdução de modelos Transformer pré-treinados (como BioBERT, SciBERT e PubMedBERT), embeddings contextualizados (como SBERT), estratégias de aprendizado ativo (via ASReview) e calibração de thresholds para garantir recall ≥ 95%. A proposta visa alinhar a metodologia ao estado da arte em triagem automatizada de revisões sistemáticas da literatura.
date: 2025-06-30
authors:
    - Matheus Silva Martins Mota
---

# Expansão dos Experimentos com Transformers, Embeddings Contextuais e Aprendizado Ativo

## Resumo

Este RFC propõe a expansão metodológica do projeto de mestrado por meio da introdução de modelos Transformer pré-treinados (como BioBERT e SciBERT), vetorização com embeddings contextualizados (ex: SBERT), e avaliação com técnicas de aprendizado ativo. O objetivo é atualizar o pipeline experimental para refletir o estado da arte em triagem automatizada de Revisões Sistemáticas da Literatura (RSL), especialmente em tarefas de classificação binária com alto desbalanceamento de classes e restrição de recall.

## Motivação

A literatura recente (2021–2025) mostra que modelos baseados em Transformers superam arquiteturas como LSTM e Bi-LSTM em tarefas de classificação de textos científicos. Além disso, o uso de embeddings contextualizados e estratégias de aprendizado ativo tem demonstrado ganhos substanciais em recall, WSS@95 e economia de esforço humano.

Essa expansão se justifica pelos seguintes pontos:

- Resultados inconsistentes ao tentar replicar os experimentos do artigo base com LSTM.
- Existência de ferramentas públicas (ASReview, HuggingFace) que oferecem suporte direto a Transformers e Active Learning.
- A necessidade de investigar a aplicabilidade de métodos mais recentes e robustos.

## Proposta

### 1. Modelos a serem testados

BioBERT: modelo BERT pré-treinado com literatura biomédica (PubMed).

- `SciBERT`: modelo BERT treinado com textos científicos (Semantic Scholar).
- `PubMedBERT`: modelo treinado do zero com corpus PubMed.
- `Sentence-BERT`: versão de BERT para embeddings de sentenças.

Cada modelo será `fine-tuned` para a tarefa binária de inclusão/exclusão de resumos.

### 2. Estratégias de vetorização adicionais

- `SBERT embeddings + SVM/XGBoost`: avaliação do uso de embeddings de sentença como entrada para modelos tradicionais.
- Comparação com GloVe, TF-IDF e FastText.

### 3. Aprendizado Ativo

Avaliação de desempenho com simulações de triagem ativa (CAL).

Estratégias:

- Uncertainty sampling (baseado em entropia/softmax)
    - Random sampling (baseline)
    - Core-set selection (caso viável)
- Ferramenta sugerida: ASReview LAB (https://asreview.nl/)

### 4. Calibração de Thresholds

- Ajuste de thresholds com base em:
    - Curvas Precision-Recall
    - Platt Scaling
    - Isotonic Regression
- Foco em calibrar para Recall ≥ 95%.

### 5. Novas métricas a coletar

- `WSS@95` (Work Saved over Sampling at 95% Recall)
- `AUROC`, `PR-AUC`, tempo de inferência
- Simulação de triagem (número de resumos lidos até atingir 95% de recall)

## Alternativas Consideradas

- Continuar com LSTM apenas: descartado por limitação empírica e estado da arte defasado.
- Apenas usar Transformers sem aprendizado ativo: descartado por não capturar o comportamento real da triagem humana assistida por IA.

## Impacto / Compatibilidade

- Impacto positivo: maior robustez dos resultados e alinhamento com a prática científica atual.
- Compatibilidade: plenamente compatível com o pipeline atual. A vetorização e modelos novos podem ser adicionados modularmente.

## Implementação

- Usar `HuggingFace Transformers` para fine-tuning de modelos BERT.
- Integrar `sentence-transformers` para SBERT.
- Empacotar o pipeline com scripts reprodutíveis para cada experimento.
- Utilizar ASReview LAB para experimentos com aprendizado ativo.

## Conclusão

Esta RFC formaliza a proposta de modernização do pipeline experimental do projeto de mestrado, com foco em Transformer models, embeddings contextuais e aprendizado ativo. A implementação dessas melhorias está alinhada com o estado da arte e deve aumentar a validade e a relevância dos resultados obtidos.

## Referências

- Beltagy, I. et al. (2019). BioBERT.
- Lee, J. et al. (2020). PubMedBERT.
- Devlin, J. et al. (2018). BERT.
- Reimers, N., Gurevych, I. (2019). Sentence-BERT.
- Van de Schoot, R. et al. (2021). ASReview: Open-source tool for screening in SRs.
- https://asreview.nl/
- https://huggingface.co/models



