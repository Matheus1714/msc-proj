from dataclasses import dataclass
from temporalio import workflow
from typing import List, Dict, Any
import pandas as pd
import os
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
      experiment_tasks = [
        self._run_svm_experiment,
        self._run_lstm_experiment,
        self._run_lstm_attention_experiment,
        self._run_bi_lstm_experiment,
        self._run_bi_lstm_attention_experiment,
      ]
      for exp_func in experiment_tasks:
        try:
          result = await exp_func(data)
        except Exception as e:
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
            error_message=str(result)
          ))
        else:
          workflow.logger.info(f"Experimento {experiment_names[i]} concluído com sucesso")
          completed_experiments.append(experiment_names[i])
          detailed_results.append(ExperimentResult(
            experiment_name=experiment_names[i],
            status="success",
            metrics=result.metrics
          ))
      
      workflow.logger.info(f"Execução concluída: {len(completed_experiments)} sucessos, {len(failed_experiments)} falhas")
      
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
