# Guia de Diretórios Únicos para Experimentos

Este documento descreve o novo sistema de diretórios únicos implementado para organizar e isolar os resultados de cada execução de experimentos.

## Visão Geral

O sistema agora cria automaticamente um diretório único para cada execução de experimentos, garantindo que:

- **Não há conflitos** entre diferentes execuções
- **Fácil organização** e comparação de resultados
- **Histórico completo** de todas as execuções
- **Isolamento de dados** entre experimentos
- **Facilita debugging** e análise

## Estrutura de Diretórios

### Padrão de Nomenclatura

Os diretórios seguem o padrão: `data/experiments/exp_YYYYMMDD_HHMMSS/`

Exemplo:
```
data/experiments/exp_20251028_113008/
├── prepared_data.csv
├── tokenized_data.csv
├── word_index.json
├── glove_embeddings.npy
├── x_seq.npy
├── y.npy
├── x_train.npy
├── x_val.npy
├── x_test.npy
├── y_train.npy
├── y_val.npy
├── y_test.npy
├── experiment_results.csv
└── machine_specs.txt
```

### ExperimentConfig

A classe `ExperimentConfig` gerencia todos os caminhos de arquivos para um experimento:

```python
from constants import ExperimentConfig

# Criar configuração com ID automático
config = ExperimentConfig()

# Criar configuração com ID personalizado
config = ExperimentConfig("meu_experimento_001")

# Criar diretórios
config.create_directories()
```

## Arquivos Gerados

### Dados de Processamento
- **`prepared_data.csv`**: Dados preparados para o experimento
- **`tokenized_data.csv`**: Dados tokenizados
- **`word_index.json`**: Índice de palavras do tokenizador
- **`glove_embeddings.npy`**: Matriz de embeddings GloVe

### Dados de Treinamento
- **`x_seq.npy`**: Sequências de entrada completas
- **`y.npy`**: Labels completos
- **`x_train.npy`**: Dados de treinamento
- **`x_val.npy`**: Dados de validação
- **`x_test.npy`**: Dados de teste
- **`y_train.npy`**: Labels de treinamento
- **`y_val.npy`**: Labels de validação
- **`y_test.npy`**: Labels de teste

### Resultados
- **`experiment_results.csv`**: Resultados detalhados de todos os experimentos
- **`machine_specs.txt`**: Especificações da máquina e métricas de performance

## Uso

### Execução Automática

O sistema funciona automaticamente quando você executa:

```bash
pipenv run python scripts/run_all_experiments.py
```

O script irá:
1. Criar um diretório único para esta execução
2. Executar todos os experimentos
3. Salvar todos os arquivos no diretório criado
4. Exibir o caminho do diretório no final

### Exemplo de Saída

```
📁 Diretório do experimento: data/experiments/exp_20251028_113008

🎉 Todos os experimentos foram executados!
✅ Experimentos concluídos: 5
❌ Experimentos falharam: 0
📊 Total de experimentos: 5

📁 Arquivos gerados em: data/experiments/exp_20251028_113008
  - Resultados: data/experiments/exp_20251028_113008/experiment_results.csv
  - Especificações: data/experiments/exp_20251028_113008/machine_specs.txt
  - Dados preparados: data/experiments/exp_20251028_113008/prepared_data.csv
  - Dados tokenizados: data/experiments/exp_20251028_113008/tokenized_data.csv
  - Embeddings GloVe: data/experiments/exp_20251028_113008/glove_embeddings.npy
```

## Modificações Realizadas

### 1. Constants.py
- Adicionada classe `ExperimentConfig` para gerenciar caminhos
- Geração automática de IDs únicos baseados em timestamp

### 2. Activities
Todas as activities foram atualizadas para aceitar parâmetros de diretório:
- `prepare_data_for_experiment_activity.py`
- `tokenizer_activity.py`
- `split_data_activity.py`
- `load_glove_embeddings_activity.py`

### 3. Workflows
Todos os workflows foram atualizados para:
- Aceitar `ExperimentConfig` como parâmetro
- Passar caminhos específicos para as activities
- Criar diretórios automaticamente

### 4. Scripts
- `run_all_experiments.py` atualizado para usar o novo sistema
- Exibição de informações sobre arquivos gerados

## Benefícios

### Organização
- **Histórico completo**: Cada execução fica em seu próprio diretório
- **Fácil comparação**: Compare resultados de diferentes execuções
- **Limpeza simples**: Delete diretórios antigos quando necessário

### Debugging
- **Isolamento**: Problemas em uma execução não afetam outras
- **Rastreabilidade**: Identifique facilmente qual execução gerou cada arquivo
- **Análise detalhada**: Examine todos os arquivos intermediários

### Colaboração
- **Compartilhamento**: Compartilhe diretórios específicos de experimentos
- **Reproducibilidade**: Execute experimentos com configurações idênticas
- **Versionamento**: Use controle de versão para rastrear mudanças

## Exemplo de Uso Programático

```python
from constants import ExperimentConfig
from src.workflows.experiments_workflow import ExperimentsWorkflow, ExperimentsWorkflowIn

# Criar configuração
config = ExperimentConfig("experimento_teste")
config.create_directories()

# Executar workflow
workflow_input = ExperimentsWorkflowIn(
    input_data_path="data/academic_works.csv",
    hyperparameters=hyperparameters,
    experiment_config=config
)

# Todos os arquivos serão salvos em config.base_dir
```

## Migração

### Compatibilidade
- O sistema é **totalmente compatível** com o código existente
- Se não for fornecida uma `ExperimentConfig`, o sistema usa os caminhos padrão
- **Não há breaking changes** para usuários existentes

### Atualização Gradual
- Execute experimentos existentes normalmente
- Novos experimentos usarão automaticamente o novo sistema
- Migre gradualmente conforme necessário

## Troubleshooting

### Problemas Comuns

1. **Permissões de diretório**: Certifique-se de que o usuário tem permissão para criar diretórios
2. **Espaço em disco**: Cada experimento cria uma cópia completa dos dados
3. **Limpeza**: Considere limpar diretórios antigos periodicamente

### Soluções

```bash
# Verificar espaço usado
du -sh data/experiments/*

# Limpar experimentos antigos (cuidado!)
rm -rf data/experiments/exp_20251001_*

# Listar experimentos por data
ls -la data/experiments/ | sort -k6,7
```

## Próximos Passos

- [ ] Adicionar compressão de diretórios antigos
- [ ] Implementar limpeza automática de experimentos antigos
- [ ] Adicionar metadados de configuração em cada diretório
- [ ] Criar interface web para visualizar experimentos
- [ ] Implementar comparação automática entre experimentos
