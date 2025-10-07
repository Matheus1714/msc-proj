# Estrutura do Fluxo de Experimentos

## Visão Geral

A estrutura de experimentos foi projetada para simular um modelo por vez, seguindo o fluxo:
1. **Preparação de dados** (70% treino, 20% validação, 10% produção)
2. **Tokenização** (reutilizável por estratégia)
3. **Simulação do modelo** (treino → validação → produção)

## Workflows

### ExperimentWorkflow
**Entrada:** `ExperimentWorkflowIn`
- `dataset_id`: ID do dataset
- `model_config`: Configuração do modelo (tipo, hiperparâmetros)
- `tokenizer_strategy`: Estratégia de tokenização
- `model_path`: (opcional) Caminho do modelo pré-treinado

**Saída:** `ExperimentWorkflowOut`
- `model_path`: Caminho do modelo treinado
- `validation_metrics_path`: Caminho das métricas de validação
- `production_metrics_path`: Caminho das métricas de produção
- `final_report_path`: Caminho do relatório final

### TokenizeSharedWorkflow
**Entrada:** `TokenizeSharedWorkflowIn`
- `file_path`: Caminho dos dados
- `strategy`: Estratégia de tokenização

**Saída:** `TokenizeSharedWorkflowOut`
- `tokenized_data_path`: Caminho dos dados tokenizados

### SimulateModelWorkflow
**Entrada:** `SimulateModelWorkflowIn`
- `file_path`: Caminho dos dados tokenizados
- `strategy`: Tipo do modelo (svm, random_forest, etc.)

**Saída:** `SimulateModelWorkflowOut`
- `result`: Caminho do modelo treinado

## Activities

### Preparação de Dados
- `prepare_data_for_experiment_activity`: Divide dados em treino/validação/produção

### Treinamento de Modelos
- `train_model_activity`: Treinamento genérico
- `train_svm_activity`: Treinamento específico para SVM
- `train_random_forest_activity`: Treinamento específico para Random Forest

### Validação e Inferência
- `validate_model_activity`: Validação com 20% dos dados
- `run_production_inference_activity`: Inferência com 10% dos dados
- `aggregate_results_activity`: Agregação dos resultados finais

## Tipos de Dados

### ModelConfig
```python
@dataclass
class ModelConfig(TypedDict):
  name: str
  type: str  # "svm", "random_forest", etc.
  hyperparameters: Dict[str, any]
```

### Métricas de Avaliação
- Accuracy, Precision, Recall, F1-Score
- Matriz de confusão
- Métricas específicas por fase (treino/validação/produção)

## Fluxo de Execução

```
ExperimentWorkflow
├── prepare_data_for_experiment_activity (70%/20%/10%)
├── TokenizeSharedWorkflow
│   └── Aplicar estratégia de tokenização
└── SimulateModelWorkflow
    ├── train_model_activity (70% dados)
    ├── validate_model_activity (20% dados)
    ├── run_production_inference_activity (10% dados)
    └── aggregate_results_activity
```

## Modelos Suportados

### SVM (Support Vector Machine)
- Configurável via hiperparâmetros
- Boa performance para classificação de texto

### Random Forest
- Ensemble de árvores de decisão
- Robusto e interpretável

## Estratégias de Tokenização

- TF-IDF
- Word2Vec
- BERT embeddings
- Outras estratégias customizadas

## Arquivos de Saída

- **Modelos:** `models/{strategy}_{model_name}.pkl`
- **Dados tokenizados:** `data/tokenized_{strategy}_{dataset}.csv`
- **Métricas:** `results/{experiment_id}_metrics.json`
- **Relatórios:** `reports/{experiment_id}_final_report.json`
