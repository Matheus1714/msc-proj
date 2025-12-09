---
title: Início
---

# Projeto de Mestrado

Bem-vindo à documentação do meu projeto de mestrado. Este espaço organiza, acompanha e documenta todas as etapas do desenvolvimento do trabalho, de forma clara, versionada e acessível.

## 🎯 Objetivo

Este projeto de mestrado tem como foco principal:

> Este trabalho tem como objetivo avaliar e comparar o desempenho de diferentes modelos de Machine Learning na tarefa de classificação de estudos em Revisões Sistemáticas da Literatura (RSL), com vistas a identificar os modelos mais eficazes e contribuir para o desenvolvimento de uma base metodológica sólida que subsidie análises automatizadas futuras nesse domínio.

Além disso, busca-se manter um processo organizado e transparente durante o desenvolvimento, com registros frequentes de progresso e decisões.

## 📂 Estrutura da Documentação

- 📌 **[RFCs](/msc-proj/rfcs/)** — Propostas formais de decisões técnicas, metodológicas ou estruturais.
- 📈 **[Relatórios](/msc-proj/reports/)** — Acompanhamento de progresso, experimentos e análises.
- 🧠 **[Outros](/msc-proj/others/)** — Ideias soltas, cronograma, brainstorms e notas gerais.

## 📅 Andamento

O andamento do projeto pode ser acompanhado pelo [cronograma](/msc-proj/others/schedule/) atualizado e pelos relatórios semanais ou mensais disponíveis na seção de **Relatórios**.

## 🛠️ Tecnologias e Ferramentas

- **Temporal**: Orquestração de workflows distribuídos
- **Python 3.10**: Linguagem principal do projeto
- **TensorFlow/Keras**: Modelos de deep learning (LSTM, BiLSTM)
- **scikit-learn**: Modelos clássicos de ML (SVM)
- **GloVe Embeddings**: Vetores de palavras pré-treinados
- **MkDocs**: Documentação do projeto
- **Docker**: Containerização dos serviços
- **Jupyter Notebooks**: Análises e visualizações

## 🚀 Início Rápido

Para começar a usar o projeto:

1. **Instale as dependências**:
   ```bash
   pipenv install
   pipenv shell
   ```

2. **Configure o ambiente**:
   - Crie um arquivo `.env` com `TEMPORAL_CONNECT=localhost:7233`

3. **Inicie os serviços Docker**:
   ```bash
   docker-compose up -d
   ```

4. **Inicie o worker Temporal** (em terminal separado):
   ```bash
   python scripts/start_ml_worker.py
   ```

5. **Execute os experimentos**:
   ```bash
   python scripts/run_all_experiments.py
   ```

Para mais detalhes, consulte o [README do projeto](https://github.com/matheus1714/msc-proj/blob/master/README.md).

## 📚 Guias Disponíveis

- **[Guia de Estrutura de Workflows](experiment_workflow_structure.md)**: Entenda como os workflows estão organizados
- **[Guia de Fluxo de Experimentos](experiment_flow_diagram.md)**: Visualize o fluxo completo de execução
- **[Guia de Diretórios de Experimentos](experiment_directories_guide.md)**: Saiba onde os resultados são salvos
- **[Guia de Download do GloVe](glove_download_guide.md)**: Como baixar e usar vetores GloVe
- **[Guia de Métricas de Sistema](system_metrics_guide.md)**: Entenda as métricas coletadas

## Links Rápidos

- [Documento Overleaf](https://www.overleaf.com/project/6482050c1c6ea5c00b3344b4)
- [Google Drive Arquivos](https://drive.google.com/drive/folders/12XmtEgzXKUfD6ylQFEi4PRTmDYkzd6h7?usp=sharing)
- [Google Sites Mestrado](https://sites.google.com/view/msc-matheus-mota/doc-geral-artefatos/01-vis%C3%A3o-do-projeto)

---

*Este site é atualizado continuamente conforme o progresso do projeto.*
