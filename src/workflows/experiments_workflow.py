from dataclasses import dataclass
from temporalio import workflow
from typing import List, Dict, Any
import pandas as pd
import os
import time
import platform
import psutil
import subprocess
from datetime import datetime

from src.workflows.experiment_svm_with_glove_and_tfidf_workflow import (
  ExperimentSVMWithGloveAndTFIDFWorkflow,
  ExperimentSVMWithGloveAndTFIDFWorkflowIn,
  ExperimentSVMWithGloveAndTFIDFHyperparameters,
)
from src.workflows.experiment_lstm_with_glove_workflow import (
  ExperimentLSTMWithGloveWorkflow,
  ExperimentLSTMWithGloveWorkflowIn,
  ExperimentLSTMWithGloveHyperparameters,
)
from src.workflows.experiment_lstm_with_glove_and_attention_workflow import (
  ExperimentLSTMWithGloveAndAttentionWorkflow,
  ExperimentLSTMWithGloveAndAttentionWorkflowIn,
  ExperimentLSTMWithGloveAndAttentionHyperparameters,
)
from src.workflows.experiment_bi_lstm_with_glove_workflow import (
  ExperimentBiLSTMWithGloveWorkflow,
  ExperimentBiLSTMWithGloveWorkflowIn,
  ExperimentBiLSTMWithGloveHyperparameters,
)
from src.workflows.experiment_bi_lstm_with_glove_and_attention_workflow import (
  ExperimentBiLSTMWithGloveAndAttentionWorkflow,
  ExperimentBiLSTMWithGloveAndAttentionWorkflowIn,
  ExperimentBiLSTMWithGloveAndAttentionHyperparameters,
)
from constants import WorflowTaskQueue
from src.utils.calculate_metrics import EvaluationData

@dataclass
class ExperimentsWorkflowIn:
  input_data_path: str
  hyperparameters: Dict[str, Any]

@dataclass
class ExperimentResult:
  experiment_name: str
  status: str  # "success" or "failed"
  execution_time_minutes: float = None
  metrics: EvaluationData = None
  error_message: str = None

@dataclass
class ExperimentsWorkflowOut:
  completed_experiments: List[str]
  failed_experiments: List[str]
  total_experiments: int
  results_file_path: str
  detailed_results: List[ExperimentResult]

@workflow.defn
class ExperimentsWorkflow:
  @workflow.run
  async def run(self, data: ExperimentsWorkflowIn) -> ExperimentsWorkflowOut:
    try:
      workflow.logger.info(f"Iniciando execução de todos os experimentos com dados de: {data.input_data_path}")

      results = []
      execution_times = []
      experiment_tasks = [
        self._run_svm_experiment,
        self._run_lstm_experiment,
        self._run_lstm_attention_experiment,
        self._run_bi_lstm_experiment,
        self._run_bi_lstm_attention_experiment,
      ]
      for exp_func in experiment_tasks:
        try:
          start_time = time.time()
          result = await exp_func(data)
          end_time = time.time()
          execution_time_minutes = (end_time - start_time) / 60
          execution_times.append(execution_time_minutes)
        except Exception as e:
          end_time = time.time()
          execution_time_minutes = (end_time - start_time) / 60
          execution_times.append(execution_time_minutes)
          result = e
        results.append(result)
      
      completed_experiments = []
      failed_experiments = []
      detailed_results = []
      
      experiment_names = [
        "SVM with GloVe and TF-IDF",
        "LSTM with GloVe",
        "LSTM with GloVe and Attention",
        "BiLSTM with GloVe",
        "BiLSTM with GloVe and Attention"
      ]
      
      for i, result in enumerate(results):
        if isinstance(result, Exception):
          workflow.logger.error(f"Experimento {experiment_names[i]} falhou: {result}")
          failed_experiments.append(experiment_names[i])
          detailed_results.append(ExperimentResult(
            experiment_name=experiment_names[i],
            status="failed",
            execution_time_minutes=execution_times[i],
            error_message=str(result)
          ))
        else:
          workflow.logger.info(f"Experimento {experiment_names[i]} concluído com sucesso em {execution_times[i]:.2f} minutos")
          completed_experiments.append(experiment_names[i])
          detailed_results.append(ExperimentResult(
            experiment_name=experiment_names[i],
            status="success",
            execution_time_minutes=execution_times[i],
            metrics=result.metrics
          ))
      
      workflow.logger.info(f"Execução concluída: {len(completed_experiments)} sucessos, {len(failed_experiments)} falhas")
      
      # Gerar arquivo de especificações da máquina
      machine_specs_file_path = await self._generate_machine_specs_file(data.input_data_path)
      
      # Salvar resultados em arquivo
      results_file_path = await self._save_results_to_file(detailed_results, data.input_data_path)
      
      return ExperimentsWorkflowOut(
        completed_experiments=completed_experiments,
        failed_experiments=failed_experiments,
        total_experiments=len(experiment_tasks),
        results_file_path=results_file_path,
        detailed_results=detailed_results
      )
      
    except Exception as e:
      workflow.logger.error(f"Erro ao executar workflow de experimentos: {e}")
      raise e

  async def _run_svm_experiment(self, data: ExperimentsWorkflowIn):
    """Execute SVM with GloVe and TF-IDF experiment"""
    return await workflow.execute_child_workflow(
      ExperimentSVMWithGloveAndTFIDFWorkflow.run,
      arg=ExperimentSVMWithGloveAndTFIDFWorkflowIn(
        input_data_path=data.input_data_path,
        hyperparameters=ExperimentSVMWithGloveAndTFIDFHyperparameters(
          max_words=data.hyperparameters.get("max_words", 20000),
          max_len=data.hyperparameters.get("max_len", 300),
          embedding_dim=data.hyperparameters.get("embedding_dim", 300),
          random_state=data.hyperparameters.get("random_state", 42),
          ngram_range=data.hyperparameters.get("ngram_range", (1, 3)),
          max_iter=data.hyperparameters.get("max_iter", 1000),
        ),
      ),
      id=f"svm-experiment-{workflow.uuid4()}",
      task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    )

  async def _run_lstm_experiment(self, data: ExperimentsWorkflowIn):
    """Execute LSTM with GloVe experiment"""
    return await workflow.execute_child_workflow(
      ExperimentLSTMWithGloveWorkflow.run,
      arg=ExperimentLSTMWithGloveWorkflowIn(
        input_data_path=data.input_data_path,
        hyperparameters=ExperimentLSTMWithGloveHyperparameters(
          max_words=data.hyperparameters.get("max_words", 20000),
          max_len=data.hyperparameters.get("max_len", 300),
          embedding_dim=data.hyperparameters.get("embedding_dim", 300),
          random_state=data.hyperparameters.get("random_state", 42),
          lstm_units=data.hyperparameters.get("lstm_units", 100),
          lstm_dropout=data.hyperparameters.get("lstm_dropout", 0.2),
          lstm_recurrent_dropout=data.hyperparameters.get("lstm_recurrent_dropout", 0.2),
          pool_dropout=data.hyperparameters.get("pool_dropout", 0.5),
          dense_units=data.hyperparameters.get("dense_units", 1),
          dense_activation=data.hyperparameters.get("dense_activation", "sigmoid"),
          batch_size=data.hyperparameters.get("batch_size", 64),
          epochs=data.hyperparameters.get("epochs", 5),
          learning_rate=data.hyperparameters.get("learning_rate", 3e-4),
          loss=data.hyperparameters.get("loss", "binary_crossentropy"),
          metrics=data.hyperparameters.get("metrics", ["accuracy"]),
          n_splits=data.hyperparameters.get("n_splits", 5),
          verbose=data.hyperparameters.get("verbose", 0),
          class_weight_0=data.hyperparameters.get("class_weight_0", 1),
          class_weight_1=data.hyperparameters.get("class_weight_1", 44),
        ),
      ),
      id=f"lstm-experiment-{workflow.uuid4()}",
      task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    )

  async def _run_lstm_attention_experiment(self, data: ExperimentsWorkflowIn):
    """Execute LSTM with GloVe and Attention experiment"""
    return await workflow.execute_child_workflow(
      ExperimentLSTMWithGloveAndAttentionWorkflow.run,
      arg=ExperimentLSTMWithGloveAndAttentionWorkflowIn(
        input_data_path=data.input_data_path,
        hyperparameters=ExperimentLSTMWithGloveAndAttentionHyperparameters(
          max_words=data.hyperparameters.get("max_words", 20000),
          max_len=data.hyperparameters.get("max_len", 300),
          embedding_dim=data.hyperparameters.get("embedding_dim", 300),
          random_state=data.hyperparameters.get("random_state", 42),
          lstm_units=data.hyperparameters.get("lstm_units", 100),
          lstm_dropout=data.hyperparameters.get("lstm_dropout", 0.2),
          lstm_recurrent_dropout=data.hyperparameters.get("lstm_recurrent_dropout", 0.2),
          pool_dropout=data.hyperparameters.get("pool_dropout", 0.5),
          dense_units=data.hyperparameters.get("dense_units", 1),
          dense_activation=data.hyperparameters.get("dense_activation", "sigmoid"),
          batch_size=data.hyperparameters.get("batch_size", 64),
          epochs=data.hyperparameters.get("epochs", 5),
          learning_rate=data.hyperparameters.get("learning_rate", 3e-4),
          loss=data.hyperparameters.get("loss", "binary_crossentropy"),
          metrics=data.hyperparameters.get("metrics", ["accuracy"]),
          n_splits=data.hyperparameters.get("n_splits", 5),
          verbose=data.hyperparameters.get("verbose", 0),
          class_weight_0=data.hyperparameters.get("class_weight_0", 1),
          class_weight_1=data.hyperparameters.get("class_weight_1", 44),
        ),
      ),
      id=f"lstm-attention-experiment-{workflow.uuid4()}",
      task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    )

  async def _run_bi_lstm_experiment(self, data: ExperimentsWorkflowIn):
    """Execute BiLSTM with GloVe experiment"""
    return await workflow.execute_child_workflow(
      ExperimentBiLSTMWithGloveWorkflow.run,
      arg=ExperimentBiLSTMWithGloveWorkflowIn(
        input_data_path=data.input_data_path,
        hyperparameters=ExperimentBiLSTMWithGloveHyperparameters(
          max_words=data.hyperparameters.get("max_words", 20000),
          max_len=data.hyperparameters.get("max_len", 300),
          embedding_dim=data.hyperparameters.get("embedding_dim", 300),
          random_state=data.hyperparameters.get("random_state", 42),
          lstm_units=data.hyperparameters.get("lstm_units", 100),
          lstm_dropout=data.hyperparameters.get("lstm_dropout", 0.2),
          lstm_recurrent_dropout=data.hyperparameters.get("lstm_recurrent_dropout", 0.2),
          pool_dropout=data.hyperparameters.get("pool_dropout", 0.02),
          dense_units=data.hyperparameters.get("dense_units", 1),
          dense_activation=data.hyperparameters.get("dense_activation", "sigmoid"),
          batch_size=data.hyperparameters.get("batch_size", 64),
          epochs=data.hyperparameters.get("epochs", 5),
          learning_rate=data.hyperparameters.get("learning_rate", 1e-4),
          loss=data.hyperparameters.get("loss", "binary_crossentropy"),
          metrics=data.hyperparameters.get("metrics", ["accuracy"]),
          n_splits=data.hyperparameters.get("n_splits", 5),
          verbose=data.hyperparameters.get("verbose", 0),
          class_weight_0=data.hyperparameters.get("class_weight_0", 1),
          class_weight_1=data.hyperparameters.get("class_weight_1", 44),
        ),
      ),
      id=f"bi-lstm-experiment-{workflow.uuid4()}",
      task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    )

  async def _run_bi_lstm_attention_experiment(self, data: ExperimentsWorkflowIn):
    """Execute BiLSTM with GloVe and Attention experiment"""
    return await workflow.execute_child_workflow(
      ExperimentBiLSTMWithGloveAndAttentionWorkflow.run,
      arg=ExperimentBiLSTMWithGloveAndAttentionWorkflowIn(
        input_data_path=data.input_data_path,
        hyperparameters=ExperimentBiLSTMWithGloveAndAttentionHyperparameters(
          max_words=data.hyperparameters.get("max_words", 20000),
          max_len=data.hyperparameters.get("max_len", 300),
          embedding_dim=data.hyperparameters.get("embedding_dim", 300),
          random_state=data.hyperparameters.get("random_state", 42),
          lstm_units=data.hyperparameters.get("lstm_units", 100),
          lstm_dropout=data.hyperparameters.get("lstm_dropout", 0.2),
          lstm_recurrent_dropout=data.hyperparameters.get("lstm_recurrent_dropout", 0.2),
          pool_dropout=data.hyperparameters.get("pool_dropout", 0.02),
          dense_units=data.hyperparameters.get("dense_units", 1),
          dense_activation=data.hyperparameters.get("dense_activation", "sigmoid"),
          batch_size=data.hyperparameters.get("batch_size", 64),
          epochs=data.hyperparameters.get("epochs", 5),
          learning_rate=data.hyperparameters.get("learning_rate", 1e-4),
          loss=data.hyperparameters.get("loss", "binary_crossentropy"),
          metrics=data.hyperparameters.get("metrics", ["accuracy"]),
          n_splits=data.hyperparameters.get("n_splits", 5),
          verbose=data.hyperparameters.get("verbose", 0),
          class_weight_0=data.hyperparameters.get("class_weight_0", 1),
          class_weight_1=data.hyperparameters.get("class_weight_1", 44),
        ),
      ),
      id=f"bi-lstm-attention-experiment-{workflow.uuid4()}",
      task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    )

  async def _generate_machine_specs_file(self, input_data_path: str) -> str:
    """Generate a file with detailed machine specifications"""
    try:
      # Criar diretório data se não existir
      data_dir = "data"
      os.makedirs(data_dir, exist_ok=True)
      
      # Gerar nome do arquivo com timestamp
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      filename = f"machine_specs_{timestamp}.txt"
      filepath = os.path.join(data_dir, filename)
      
      with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=== ESPECIFICAÇÕES DA MÁQUINA ===\n\n")
        
        # Informações básicas do sistema
        f.write("SISTEMA OPERACIONAL:\n")
        f.write(f"  Sistema: {platform.system()}\n")
        f.write(f"  Versão: {platform.release()}\n")
        f.write(f"  Arquitetura: {platform.architecture()[0]}\n")
        f.write(f"  Processador: {platform.processor()}\n")
        f.write(f"  Máquina: {platform.machine()}\n")
        f.write(f"  Nó: {platform.node()}\n")
        f.write(f"  Plataforma: {platform.platform()}\n\n")
        
        # Informações de CPU
        f.write("PROCESSADOR:\n")
        f.write(f"  Núcleos físicos: {psutil.cpu_count(logical=False)}\n")
        f.write(f"  Núcleos lógicos: {psutil.cpu_count(logical=True)}\n")
        f.write(f"  Frequência máxima: {psutil.cpu_freq().max if psutil.cpu_freq() else 'N/A'} MHz\n")
        f.write(f"  Frequência atual: {psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A'} MHz\n")
        
        # Informações de memória
        memory = psutil.virtual_memory()
        f.write(f"\nMEMÓRIA:\n")
        f.write(f"  Total: {memory.total / (1024**3):.2f} GB\n")
        f.write(f"  Disponível: {memory.available / (1024**3):.2f} GB\n")
        f.write(f"  Usada: {memory.used / (1024**3):.2f} GB\n")
        f.write(f"  Percentual usado: {memory.percent}%\n")
        
        # Informações de disco
        disk = psutil.disk_usage('/')
        f.write(f"\nDISCO:\n")
        f.write(f"  Total: {disk.total / (1024**3):.2f} GB\n")
        f.write(f"  Usado: {disk.used / (1024**3):.2f} GB\n")
        f.write(f"  Livre: {disk.free / (1024**3):.2f} GB\n")
        f.write(f"  Percentual usado: {(disk.used / disk.total) * 100:.2f}%\n")
        
        # Informações de GPU (se disponível)
        f.write(f"\nGPU:\n")
        try:
          # Tentar detectar NVIDIA GPU
          result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'], 
                                capture_output=True, text=True, timeout=10)
          if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')
            for i, gpu in enumerate(gpu_info):
              parts = gpu.split(', ')
              if len(parts) >= 3:
                f.write(f"  GPU {i+1}: {parts[0]}\n")
                f.write(f"    Memória: {parts[1]} MB\n")
                f.write(f"    Driver: {parts[2]}\n")
          else:
            f.write("  Nenhuma GPU NVIDIA detectada\n")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
          f.write("  Informações de GPU não disponíveis\n")
        
        # Informações de Python
        f.write(f"\nPYTHON:\n")
        f.write(f"  Versão: {platform.python_version()}\n")
        f.write(f"  Implementação: {platform.python_implementation()}\n")
        f.write(f"  Compilador: {platform.python_compiler()}\n")
        
        # Informações de bibliotecas ML
        f.write(f"\nBIBLIOTECAS DE MACHINE LEARNING:\n")
        try:
          import tensorflow as tf
          f.write(f"  TensorFlow: {tf.__version__}\n")
        except ImportError:
          f.write("  TensorFlow: Não instalado\n")
        
        try:
          import torch
          f.write(f"  PyTorch: {torch.__version__}\n")
        except ImportError:
          f.write("  PyTorch: Não instalado\n")
        
        try:
          import sklearn
          f.write(f"  Scikit-learn: {sklearn.__version__}\n")
        except ImportError:
          f.write("  Scikit-learn: Não instalado\n")
        
        try:
          import pandas as pd
          f.write(f"  Pandas: {pd.__version__}\n")
        except ImportError:
          f.write("  Pandas: Não instalado\n")
        
        try:
          import numpy as np
          f.write(f"  NumPy: {np.__version__}\n")
        except ImportError:
          f.write("  NumPy: Não instalado\n")
        
        # Data e hora da execução
        f.write(f"\nEXECUÇÃO:\n")
        f.write(f"  Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Dados de entrada: {input_data_path}\n")
      
      workflow.logger.info(f"Especificações da máquina salvas em: {filepath}")
      return filepath
      
    except Exception as e:
      workflow.logger.error(f"Erro ao gerar especificações da máquina: {e}")
      return ""

  async def _save_results_to_file(self, detailed_results: List[ExperimentResult], input_data_path: str) -> str:
    """Save experiment results to CSV file in data directory"""
    try:
      # Criar diretório data se não existir
      data_dir = "data"
      os.makedirs(data_dir, exist_ok=True)
      
      # Gerar nome do arquivo com timestamp
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      filename = f"experiment_results_{timestamp}.csv"
      filepath = os.path.join(data_dir, filename)
      
      # Preparar dados para o DataFrame
      results_data = []
      for result in detailed_results:
        if result.status == "success" and result.metrics:
          results_data.append({
            "experiment_name": result.experiment_name,
            "status": result.status,
            "execution_time_minutes": result.execution_time_minutes,
            "true_positives": result.metrics["true_positives"],
            "false_positives": result.metrics["false_positives"],
            "true_negatives": result.metrics["true_negatives"],
            "false_negatives": result.metrics["false_negatives"],
            "total_positives": result.metrics["total_positives"],
            "total_negatives": result.metrics["total_negatives"],
            "precision": result.metrics["precision"],
            "recall": result.metrics["recall"],
            "f2_score": result.metrics["f2_score"],
            "wss95": result.metrics["wss95"],
            "error_message": None
          })
        else:
          results_data.append({
            "experiment_name": result.experiment_name,
            "status": result.status,
            "execution_time_minutes": result.execution_time_minutes,
            "true_positives": None,
            "false_positives": None,
            "true_negatives": None,
            "false_negatives": None,
            "total_positives": None,
            "total_negatives": None,
            "precision": None,
            "recall": None,
            "f2_score": None,
            "wss95": None,
            "error_message": result.error_message
          })
      
      # Criar DataFrame e salvar
      df = pd.DataFrame(results_data)
      df.to_csv(filepath, index=False)
      
      workflow.logger.info(f"Resultados salvos em: {filepath}")
      return filepath
      
    except Exception as e:
      workflow.logger.error(f"Erro ao salvar resultados: {e}")
      return ""
