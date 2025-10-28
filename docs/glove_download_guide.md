# Guia de Download do GloVe Vectors

Este documento descreve como usar a nova funcionalidade de download automático dos vetores GloVe no sistema de experimentos.

## Visão Geral

A funcionalidade de download do GloVe foi implementada como uma activity e workflow do Temporal, permitindo:

- Download automático dos vetores GloVe 6B
- Criação automática de diretórios necessários
- Verificação se os arquivos já existem (evita downloads desnecessários)
- Política de retry em caso de falhas de rede
- Integração automática no workflow de experimentos

## Arquivos Criados

### Activity
- `src/activities/download_glove_vectors_activity.py` - Activity principal para download

### Workflow
- `src/workflows/download_glove_vectors_workflow.py` - Workflow com retry policy

### Scripts
- `scripts/download_glove_vectors.py` - Script standalone para download
- `scripts/test_download_glove.py` - Script de teste

## Como Usar

### 1. Download Standalone

Para baixar apenas os vetores GloVe sem executar experimentos:

```bash
cd /Users/matheusmota/src/github/msc/msc-proj
python scripts/download_glove_vectors.py
```

### 2. Integração Automática

Os vetores GloVe são baixados automaticamente quando você executa os experimentos:

```bash
python scripts/run_all_experiments.py
```

### 3. Teste da Funcionalidade

Para testar apenas o download (sem usar o Temporal):

```bash
python scripts/test_download_glove.py
```

## Configuração

### Dimensões Suportadas

O sistema suporta as seguintes dimensões de embedding:
- 50d (menor arquivo, ideal para testes)
- 100d
- 200d
- 300d (padrão)

### Diretório de Destino

Por padrão, os arquivos são salvos em:
```
data/word_vectors/glove/
```

### Política de Retry

- **Máximo de tentativas**: 3
- **Intervalo inicial**: 5 segundos
- **Intervalo máximo**: 2 minutos
- **Coeficiente de backoff**: 2.0

## Estrutura de Arquivos

Após o download, a estrutura será:

```
data/word_vectors/glove/
├── glove.6B.zip          # Arquivo zip original
├── glove.6B.50d.txt      # Vetores 50d
├── glove.6B.100d.txt     # Vetores 100d
├── glove.6B.200d.txt     # Vetores 200d
└── glove.6B.300d.txt     # Vetores 300d
```

## Comportamento

### Primeira Execução
1. Cria o diretório `data/word_vectors/glove/` se não existir
2. Baixa o arquivo `glove.6B.zip` (~822MB)
3. Extrai todos os arquivos de texto
4. Retorna o caminho para o arquivo específico solicitado

### Execuções Subsequentes
1. Verifica se o arquivo específico já existe
2. Se existir, retorna imediatamente sem fazer download
3. Se não existir, baixa apenas o zip e extrai

### Tratamento de Erros
- **Erro de rede**: Retry automático com backoff exponencial
- **Arquivo corrompido**: Retry automático
- **Erro de permissão**: Falha imediata
- **Espaço insuficiente**: Falha imediatamente

## Logs

O sistema produz logs informativos:

```
📁 Target directory: data/word_vectors/glove
🔢 Embedding dimension: 300d
Downloading GloVe vectors to data/word_vectors/glove...
Downloading GloVe: 100%|████████| 822M/822M [02:15<00:00, 6.07MB/s]
Download completed.
Extracting files...
Files extracted to: data/word_vectors/glove
🎉 GloVe vectors downloaded successfully!
```

## Integração com Experimentos

A funcionalidade está integrada no `ExperimentsWorkflow` e é executada automaticamente antes de qualquer experimento que precise dos vetores GloVe. O workflow:

1. Verifica a dimensão de embedding configurada nos hiperparâmetros
2. Baixa os vetores correspondentes se necessário
3. Continua com a execução dos experimentos
4. Falha se não conseguir baixar os vetores

## Troubleshooting

### Problema: Download falha repetidamente
**Solução**: Verifique sua conexão com a internet e espaço em disco disponível.

### Problema: Arquivo corrompido
**Solução**: Delete o arquivo `glove.6B.zip` e execute novamente.

### Problema: Permissão negada
**Solução**: Verifique as permissões do diretório `data/word_vectors/glove/`.

### Problema: Espaço insuficiente
**Solução**: Libere espaço em disco (o arquivo zip tem ~822MB e os arquivos extraídos ~1.5GB).
