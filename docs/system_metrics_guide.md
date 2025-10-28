# Guia de Métricas de Sistema

Este documento descreve as novas métricas de sistema implementadas no `ExperimentsWorkflow` para monitoramento detalhado de performance durante a execução dos experimentos de machine learning.

## Visão Geral

O sistema agora coleta automaticamente métricas detalhadas de performance durante a execução de cada experimento, incluindo:

- **Métricas de Memória**: Pico e média de uso de RAM
- **Métricas de CPU**: Pico e média de uso de processador
- **Throughput**: Amostras processadas por segundo
- **Latência**: Tempo médio de processamento por amostra
- **Tempos por Etapa**: Carregamento, treinamento e avaliação
- **Escalabilidade**: Eficiências de memória e CPU
- **Eficiência Energética**: Score baseado no uso de recursos

## Estrutura das Métricas

### SystemMetrics

```python
@dataclass
class SystemMetrics:
    # Memória
    peak_memory_mb: float
    average_memory_mb: float
    memory_usage_samples: List[float]
    
    # CPU
    peak_cpu_percent: float
    average_cpu_percent: float
    cpu_usage_samples: List[float]
    
    # Throughput e Latência
    throughput_samples_per_second: float
    average_latency_ms: float
    latency_samples: List[float]
    
    # Tempo por etapa
    data_loading_time_ms: float
    model_training_time_ms: float
    model_evaluation_time_ms: float
    total_execution_time_ms: float
    
    # Escalabilidade
    memory_efficiency: float
    cpu_efficiency: float
    
    # Eficiência energética
    energy_efficiency_score: float
```

### SystemMetricsCollector

O `SystemMetricsCollector` é responsável por coletar as métricas em tempo real:

```python
# Criar coletor
collector = SystemMetricsCollector(sample_interval=0.1)

# Iniciar coleta
collector.start_collection()

# Registrar tempos de etapas específicas
collector.record_step_time("data_loading", start_time, end_time)
collector.record_step_time("model_training", start_time, end_time)
collector.record_step_time("model_evaluation", start_time, end_time)

# Registrar latência e throughput
collector.record_latency(latency_ms)
collector.record_throughput(samples_per_second)

# Parar coleta e obter métricas
collector.stop_collection()
metrics = collector.get_metrics(data_size)
```

## Integração com ExperimentsWorkflow

### ExperimentResult Atualizado

O `ExperimentResult` agora inclui as métricas do sistema:

```python
@dataclass
class ExperimentResult:
    experiment_name: str
    status: str
    execution_time_minutes: float = None
    metrics: EvaluationData = None
    system_metrics: SystemMetrics = None  # NOVO
    error_message: str = None
```

### Coleta Automática

O workflow agora coleta automaticamente as métricas para cada experimento:

1. **Inicia** o coletor antes de cada experimento
2. **Coleta** métricas em tempo real durante a execução
3. **Para** o coletor após a conclusão
4. **Armazena** as métricas no resultado do experimento

## Arquivos de Saída

### CSV de Resultados

O arquivo CSV agora inclui colunas adicionais para as métricas de sistema:

- `peak_memory_mb`: Pico de uso de memória em MB
- `average_memory_mb`: Uso médio de memória em MB
- `peak_cpu_percent`: Pico de uso de CPU em %
- `average_cpu_percent`: Uso médio de CPU em %
- `throughput_samples_per_second`: Throughput em amostras/seg
- `average_latency_ms`: Latência média em ms
- `data_loading_time_ms`: Tempo de carregamento em ms
- `model_training_time_ms`: Tempo de treinamento em ms
- `model_evaluation_time_ms`: Tempo de avaliação em ms
- `total_execution_time_ms`: Tempo total em ms
- `memory_efficiency`: Eficiência de memória
- `cpu_efficiency`: Eficiência de CPU
- `energy_efficiency_score`: Score de eficiência energética

### Arquivo de Especificações da Máquina

O arquivo de especificações agora inclui uma seção detalhada com as métricas de performance de cada experimento:

```
MÉTRICAS DE PERFORMANCE DOS EXPERIMENTOS:
============================================================

  SVM with GloVe and TF-IDF:
    Status: success
    Tempo de execução: 2.45 minutos
    Memória:
      Pico: 1024.50 MB
      Média: 512.25 MB
    CPU:
      Pico: 85.30%
      Média: 45.20%
    Throughput: 150.25 amostras/seg
    Latência média: 6.65 ms
    Tempos por etapa:
      Carregamento de dados: 250.50 ms
      Treinamento do modelo: 120000.00 ms
      Avaliação do modelo: 5000.00 ms
    Eficiências:
      Memória: 0.9756
      CPU: 0.0542
      Energia: 78.50/100
```

## Interpretação das Métricas

### Memória
- **Pico**: Maior uso de RAM durante a execução
- **Média**: Uso médio de RAM durante a execução
- **Eficiência**: Dados processados por MB de RAM usada

### CPU
- **Pico**: Maior uso de CPU durante a execução
- **Média**: Uso médio de CPU durante a execução
- **Eficiência**: Tempo de execução por % de CPU usado

### Throughput e Latência
- **Throughput**: Quantas amostras são processadas por segundo
- **Latência**: Tempo médio para processar uma amostra

### Eficiência Energética
- **Score**: 0-100, baseado no uso de recursos
- **Cálculo**: 100 - (avg_cpu * 0.4 + peak_memory_gb * 0.6)
- **Interpretação**: Maior score = maior eficiência energética

## Exemplo de Uso

```python
# Executar demonstração
python examples/system_metrics_demo.py
```

## Benefícios

1. **Monitoramento Detalhado**: Visibilidade completa do uso de recursos
2. **Otimização**: Identificação de gargalos de performance
3. **Comparação**: Análise comparativa entre diferentes modelos
4. **Escalabilidade**: Avaliação da eficiência de recursos
5. **Debugging**: Identificação de problemas de performance
6. **Documentação**: Registro automático de especificações da máquina

## Considerações Técnicas

- **Overhead Mínimo**: Coleta em thread separada com impacto mínimo
- **Tolerância a Falhas**: Coleta continua mesmo com erros nos experimentos
- **Flexibilidade**: Intervalo de amostragem configurável
- **Compatibilidade**: Funciona com todos os tipos de experimentos existentes
