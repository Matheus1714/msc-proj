from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
import os
import time
from datetime import datetime

from src.activities.prepare_data_for_experiment_activity import (
  prepare_data_for_experiment_activity,
  PrepareDataForExperimentIn,
  PrepareDataForExperimentOut,
)
from src.activities.tokenizer_activity import (
  tokenizer_activity,
  TokenizerIn,
  TokenizerOut,
)
from src.activities.load_glove_embeddings_activity import (
  load_glove_embeddings_activity,
  LoadGloveEmbeddingsIn,
  LoadGloveEmbeddingsOut,
)
from src.activities.run_experiment_svm_with_glove_and_tfidf_activity import (
  run_experiment_svm_with_glove_and_tfidf_activity,
  RunExperimentSVMWithGloveAndTFIDFIn,
  RunExperimentSVMWithGloveAndTFIDFOut,
)
from src.activities.run_experiment_lstm_with_glove_activity import (
  run_experiment_lstm_with_glove_activity,
  RunExperimentLSTMWithGloveIn,
  RunExperimentLSTMWithGloveOut,
)
from src.activities.run_experiment_lstm_with_glove_and_attention_activity import (
  run_experiment_lstm_with_glove_and_attention_activity,
  RunExperimentLSTMWithGloveAndAttentionIn,
  RunExperimentLSTMWithGloveAndAttentionOut,
)
from src.activities.run_experiment_bi_lstm_with_glove_activity import (
  run_experiment_bi_lstm_with_glove_activity,
  RunExperimentBiLSTMWithGloveIn,
  RunExperimentBiLSTMWithGloveOut,
)
from src.activities.run_experiment_bi_lstm_with_glove_and_attention_activity import (
  run_experiment_bi_lstm_with_glove_and_attention_activity,
  RunExperimentBiLSTMWithGloveAndAttentionIn,
  RunExperimentBiLSTMWithGloveAndAttentionOut,
)
from src.activities.download_glove_vectors_activity import (
  download_glove_vectors_activity,
  DownloadGloveVectorsIn,
  DownloadGloveVectorsOut,
)
from src.activities.generate_machine_specs_activity import (
  generate_machine_specs_activity,
  GenerateMachineSpecsIn,
  GenerateMachineSpecsOut,
)
from constants import ExperimentConfig
from src.utils.calculate_metrics import EvaluationData
from src.utils.system_metrics import SystemMetrics, SystemMetricsCollector
from src.utils.convert_to_native import convert_to_native

@dataclass
class ExperimentsWorkflowIn:
  input_data_path: str
  hyperparameters: Dict[str, Any]
  experiment_config: ExperimentConfig

@dataclass
class ExperimentResult:
  experiment_name: str
  status: str
  execution_time_minutes: float = None
  metrics: Dict[str, Any] = None
  system_metrics: SystemMetrics = None
  error_message: str = None

@dataclass
class ExperimentsWorkflowOut:
  completed_experiments: List[str]
  failed_experiments: List[str]
  total_experiments: int
  results_file_path: str
  detailed_results: List[ExperimentResult]

class ExperimentsWorkflowSimple:
  async def run(self, data: ExperimentsWorkflowIn) -> ExperimentsWorkflowOut:
    try:
      print(f"Iniciando execução de todos os experimentos com dados de: {data.input_data_path}")

      embedding_dim = data.hyperparameters.get("embedding_dim", 300)
      glove_target_dir = "data/word_vectors/glove"
      
      print(f"Verificando/downloadando GloVe vectors ({embedding_dim}d)...")
      glove_result: DownloadGloveVectorsOut = await download_glove_vectors_activity(
        DownloadGloveVectorsIn(
          target_dir=glove_target_dir,
          embedding_dim=embedding_dim,
        )
      )
      
      if not glove_result.success:
        raise Exception(f"Failed to download GloVe vectors: {glove_result.message}")
      
      print(f"GloVe vectors ready: {glove_result.glove_file_path}")

      results = []
      execution_times = []
      system_metrics_list = []
      glove_file_path = glove_result.glove_file_path
      
      data_size = await self._get_data_size(data.input_data_path)
      
      experiment_tasks = [
        self._run_svm_experiment,
        self._run_lstm_experiment,
        self._run_lstm_attention_experiment,
        self._run_bi_lstm_experiment,
        self._run_bi_lstm_attention_experiment,
      ]
      
      for exp_func in experiment_tasks:
        metrics_collector = SystemMetricsCollector(sample_interval=0.1)
        metrics_collector.start_collection()
        
        try:
          start_time = time.time()
          result = await exp_func(data, glove_file_path)
          end_time = time.time()
          execution_time_minutes = (end_time - start_time) / 60
          execution_times.append(execution_time_minutes)
          
          metrics_collector.stop_collection()
          system_metrics = metrics_collector.get_metrics(data_size)
          system_metrics_list.append(system_metrics)
          
        except Exception as e:
          end_time = time.time()
          execution_time_minutes = (end_time - start_time) / 60
          execution_times.append(execution_time_minutes)
          result = e
          
          metrics_collector.stop_collection()
          system_metrics = metrics_collector.get_metrics(data_size)
          system_metrics_list.append(system_metrics)
          
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
      
      total_experiments = len(experiment_tasks)
      
      for i, result in enumerate(results):
        system_metrics = system_metrics_list[i] if i < len(system_metrics_list) else None
        
        if isinstance(result, Exception):
          print(f"Experimento {experiment_names[i]} falhou: {result}")
          failed_experiments.append(experiment_names[i])
          detailed_results.append(ExperimentResult(
            experiment_name=experiment_names[i],
            status="failed",
            execution_time_minutes=execution_times[i],
            system_metrics=system_metrics,
            error_message=str(result)
          ))
        else:
          print(f"Experimento {experiment_names[i]} concluído com sucesso em {execution_times[i]:.2f} minutos")
          completed_experiments.append(experiment_names[i])
          
          converted_metrics = convert_to_native(result.metrics) if result.metrics else None
          
          detailed_results.append(ExperimentResult(
            experiment_name=experiment_names[i],
            status="success",
            execution_time_minutes=execution_times[i],
            metrics=converted_metrics,
            system_metrics=system_metrics
          ))
      
      print(f"Execução concluída: {len(completed_experiments)} sucessos, {len(failed_experiments)} falhas")
      
      serializable_results = []
      for result in detailed_results:
        result_dict = {
          "experiment_name": result.experiment_name,
          "status": result.status,
          "execution_time_minutes": result.execution_time_minutes,
          "error_message": result.error_message
        }
        
        if result.system_metrics:
          result_dict["system_metrics"] = {
            "peak_memory_mb": result.system_metrics.peak_memory_mb,
            "average_memory_mb": result.system_metrics.average_memory_mb,
            "peak_cpu_percent": result.system_metrics.peak_cpu_percent,
            "average_cpu_percent": result.system_metrics.average_cpu_percent,
            "throughput_samples_per_second": result.system_metrics.throughput_samples_per_second,
            "average_latency_ms": result.system_metrics.average_latency_ms,
            "data_loading_time_ms": result.system_metrics.data_loading_time_ms,
            "model_training_time_ms": result.system_metrics.model_training_time_ms,
            "model_evaluation_time_ms": result.system_metrics.model_evaluation_time_ms,
            "memory_efficiency": result.system_metrics.memory_efficiency,
            "cpu_efficiency": result.system_metrics.cpu_efficiency,
            "energy_efficiency_score": result.system_metrics.energy_efficiency_score,
          }
        
        serializable_results.append(result_dict)
      
      # Gerar arquivo de especificações da máquina
      machine_specs_result: GenerateMachineSpecsOut = await generate_machine_specs_activity(
        GenerateMachineSpecsIn(
          input_data_path=data.input_data_path,
          machine_specs_file_path=data.experiment_config.machine_specs_file_path,
          detailed_results=serializable_results,
        )
      )
      
      results_file_path = await self._save_results_to_file(detailed_results, data.input_data_path, data.experiment_config)
      
      return ExperimentsWorkflowOut(
        completed_experiments=completed_experiments,
        failed_experiments=failed_experiments,
        total_experiments=total_experiments,
        results_file_path=results_file_path,
        detailed_results=detailed_results
      )
      
    except Exception as e:
      print(f"Erro ao executar workflow de experimentos: {e}")
      raise e

  async def _get_data_size(self, input_data_path: str) -> int:
    """Obtém o tamanho dos dados para cálculo de throughput"""
    try:
      df = pd.read_csv(input_data_path)
      return len(df)
    except Exception as e:
      print(f"Erro ao obter tamanho dos dados: {e}")
      return 1

  async def _run_svm_experiment(self, data: ExperimentsWorkflowIn, glove_file_path: str):
    """Execute SVM with GloVe and TF-IDF experiment"""
    data.experiment_config.create_directories()
    
    prepare_data_for_experiment_result: PrepareDataForExperimentOut = await prepare_data_for_experiment_activity(
      PrepareDataForExperimentIn(
        input_data_path=data.input_data_path,
        output_data_path=data.experiment_config.prepared_data_path,
        random_state=data.hyperparameters.get("random_state", 42),
      )
    )

    tokenizer_result: TokenizerOut = await tokenizer_activity(
      TokenizerIn(
        input_data_path=prepare_data_for_experiment_result.output_data_path,
        tokenized_data_path=data.experiment_config.tokenized_data_path,
        word_index_path=data.experiment_config.word_index_path,
        x_seq_path=data.experiment_config.x_seq_path,
        y_path=data.experiment_config.y_path,
        max_words=data.hyperparameters.get("max_words", 20000),
        max_len=data.hyperparameters.get("max_len", 300),
      )
    )

    await load_glove_embeddings_activity(
      LoadGloveEmbeddingsIn(
        glove_file_path=glove_file_path,
        output_path=data.experiment_config.glove_embeddings_path,
        max_words=data.hyperparameters.get("max_words", 20000),
        word_index_path=tokenizer_result.word_index_path,
      )
    )

    experiment_result: RunExperimentSVMWithGloveAndTFIDFOut = await run_experiment_svm_with_glove_and_tfidf_activity(
      RunExperimentSVMWithGloveAndTFIDFIn(
        input_data_path=prepare_data_for_experiment_result.output_data_path,
        y_path=tokenizer_result.y_path,
        random_state=data.hyperparameters.get("random_state", 42),
        max_iter=data.hyperparameters.get("max_iter", 1000),
        ngram_range=data.hyperparameters.get("ngram_range", (1, 3)),
        class_weight_0=data.hyperparameters.get("class_weight_0", 1),
        class_weight_1=data.hyperparameters.get("class_weight_1", 1),
      )
    )

    return experiment_result

  async def _run_lstm_experiment(self, data: ExperimentsWorkflowIn, glove_file_path: str):
    """Execute LSTM with GloVe experiment"""
    data.experiment_config.create_directories()
    
    prepare_data_for_experiment_result: PrepareDataForExperimentOut = await prepare_data_for_experiment_activity(
      PrepareDataForExperimentIn(
        input_data_path=data.input_data_path,
        output_data_path=data.experiment_config.prepared_data_path,
        random_state=data.hyperparameters.get("random_state", 42),
      )
    )

    tokenizer_result: TokenizerOut = await tokenizer_activity(
      TokenizerIn(
        input_data_path=prepare_data_for_experiment_result.output_data_path,
        tokenized_data_path=data.experiment_config.tokenized_data_path,
        word_index_path=data.experiment_config.word_index_path,
        x_seq_path=data.experiment_config.x_seq_path,
        y_path=data.experiment_config.y_path,
        max_words=data.hyperparameters.get("max_words", 20000),
        max_len=data.hyperparameters.get("max_len", 300),
      )
    )

    embedding_matrix_result: LoadGloveEmbeddingsOut = await load_glove_embeddings_activity(
      LoadGloveEmbeddingsIn(
        glove_file_path=glove_file_path,
        output_path=data.experiment_config.glove_embeddings_path,
        max_words=data.hyperparameters.get("max_words", 20000),
        word_index_path=tokenizer_result.word_index_path,
      )
    )

    experiment_result: RunExperimentLSTMWithGloveOut = await run_experiment_lstm_with_glove_activity(
      RunExperimentLSTMWithGloveIn(
        x_seq_path=tokenizer_result.x_seq_path,
        y_path=tokenizer_result.y_path,
        embedding_matrix_path=data.experiment_config.glove_embeddings_path,
        max_len=data.hyperparameters.get("max_len", 300),
        num_words=embedding_matrix_result.num_words,
        embedding_dim=embedding_matrix_result.embedding_dim,
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
        random_state=data.hyperparameters.get("random_state", 42),
        verbose=data.hyperparameters.get("verbose", 0),
        class_weight_0=data.hyperparameters.get("class_weight_0", 1),
        class_weight_1=data.hyperparameters.get("class_weight_1", 44),
      )
    )

    return experiment_result

  async def _run_lstm_attention_experiment(self, data: ExperimentsWorkflowIn, glove_file_path: str):
    """Execute LSTM with GloVe and Attention experiment"""
    data.experiment_config.create_directories()
    
    prepare_data_for_experiment_result: PrepareDataForExperimentOut = await prepare_data_for_experiment_activity(
      PrepareDataForExperimentIn(
        input_data_path=data.input_data_path,
        output_data_path=data.experiment_config.prepared_data_path,
        random_state=data.hyperparameters.get("random_state", 42),
      )
    )

    tokenizer_result: TokenizerOut = await tokenizer_activity(
      TokenizerIn(
        input_data_path=prepare_data_for_experiment_result.output_data_path,
        tokenized_data_path=data.experiment_config.tokenized_data_path,
        word_index_path=data.experiment_config.word_index_path,
        x_seq_path=data.experiment_config.x_seq_path,
        y_path=data.experiment_config.y_path,
        max_words=data.hyperparameters.get("max_words", 20000),
        max_len=data.hyperparameters.get("max_len", 300),
      )
    )

    embedding_matrix_result: LoadGloveEmbeddingsOut = await load_glove_embeddings_activity(
      LoadGloveEmbeddingsIn(
        glove_file_path=glove_file_path,
        output_path=data.experiment_config.glove_embeddings_path,
        max_words=data.hyperparameters.get("max_words", 20000),
        word_index_path=tokenizer_result.word_index_path,
      )
    )

    experiment_result: RunExperimentLSTMWithGloveAndAttentionOut = await run_experiment_lstm_with_glove_and_attention_activity(
      RunExperimentLSTMWithGloveAndAttentionIn(
        x_seq_path=tokenizer_result.x_seq_path,
        y_path=tokenizer_result.y_path,
        embedding_matrix_path=data.experiment_config.glove_embeddings_path,
        max_len=data.hyperparameters.get("max_len", 300),
        num_words=embedding_matrix_result.num_words,
        embedding_dim=embedding_matrix_result.embedding_dim,
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
        random_state=data.hyperparameters.get("random_state", 42),
        verbose=data.hyperparameters.get("verbose", 0),
        class_weight_0=data.hyperparameters.get("class_weight_0", 1),
        class_weight_1=data.hyperparameters.get("class_weight_1", 44),
      )
    )

    return experiment_result

  async def _run_bi_lstm_experiment(self, data: ExperimentsWorkflowIn, glove_file_path: str):
    """Execute BiLSTM with GloVe experiment"""
    data.experiment_config.create_directories()
    
    prepare_data_for_experiment_result: PrepareDataForExperimentOut = await prepare_data_for_experiment_activity(
      PrepareDataForExperimentIn(
        input_data_path=data.input_data_path,
        output_data_path=data.experiment_config.prepared_data_path,
        random_state=data.hyperparameters.get("random_state", 42),
      )
    )

    tokenizer_result: TokenizerOut = await tokenizer_activity(
      TokenizerIn(
        input_data_path=prepare_data_for_experiment_result.output_data_path,
        tokenized_data_path=data.experiment_config.tokenized_data_path,
        word_index_path=data.experiment_config.word_index_path,
        x_seq_path=data.experiment_config.x_seq_path,
        y_path=data.experiment_config.y_path,
        max_words=data.hyperparameters.get("max_words", 20000),
        max_len=data.hyperparameters.get("max_len", 300),
      )
    )

    embedding_matrix_result: LoadGloveEmbeddingsOut = await load_glove_embeddings_activity(
      LoadGloveEmbeddingsIn(
        glove_file_path=glove_file_path,
        output_path=data.experiment_config.glove_embeddings_path,
        max_words=data.hyperparameters.get("max_words", 20000),
        word_index_path=tokenizer_result.word_index_path,
      )
    )

    experiment_result: RunExperimentBiLSTMWithGloveOut = await run_experiment_bi_lstm_with_glove_activity(
      RunExperimentBiLSTMWithGloveIn(
        x_seq_path=tokenizer_result.x_seq_path,
        y_path=tokenizer_result.y_path,
        embedding_matrix_path=data.experiment_config.glove_embeddings_path,
        max_len=data.hyperparameters.get("max_len", 300),
        num_words=embedding_matrix_result.num_words,
        embedding_dim=embedding_matrix_result.embedding_dim,
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
        random_state=data.hyperparameters.get("random_state", 42),
        verbose=data.hyperparameters.get("verbose", 0),
        class_weight_0=data.hyperparameters.get("class_weight_0", 1),
        class_weight_1=data.hyperparameters.get("class_weight_1", 44),
      )
    )

    return experiment_result

  async def _run_bi_lstm_attention_experiment(self, data: ExperimentsWorkflowIn, glove_file_path: str):
    """Execute BiLSTM with GloVe and Attention experiment"""
    data.experiment_config.create_directories()
    
    prepare_data_for_experiment_result: PrepareDataForExperimentOut = await prepare_data_for_experiment_activity(
      PrepareDataForExperimentIn(
        input_data_path=data.input_data_path,
        output_data_path=data.experiment_config.prepared_data_path,
        random_state=data.hyperparameters.get("random_state", 42),
      )
    )

    tokenizer_result: TokenizerOut = await tokenizer_activity(
      TokenizerIn(
        input_data_path=prepare_data_for_experiment_result.output_data_path,
        tokenized_data_path=data.experiment_config.tokenized_data_path,
        word_index_path=data.experiment_config.word_index_path,
        x_seq_path=data.experiment_config.x_seq_path,
        y_path=data.experiment_config.y_path,
        max_words=data.hyperparameters.get("max_words", 20000),
        max_len=data.hyperparameters.get("max_len", 300),
      )
    )

    embedding_matrix_result: LoadGloveEmbeddingsOut = await load_glove_embeddings_activity(
      LoadGloveEmbeddingsIn(
        glove_file_path=glove_file_path,
        output_path=data.experiment_config.glove_embeddings_path,
        max_words=data.hyperparameters.get("max_words", 20000),
        word_index_path=tokenizer_result.word_index_path,
      )
    )

    experiment_result: RunExperimentBiLSTMWithGloveAndAttentionOut = await run_experiment_bi_lstm_with_glove_and_attention_activity(
      RunExperimentBiLSTMWithGloveAndAttentionIn(
        x_seq_path=tokenizer_result.x_seq_path,
        y_path=tokenizer_result.y_path,
        embedding_matrix_path=data.experiment_config.glove_embeddings_path,
        max_len=data.hyperparameters.get("max_len", 300),
        num_words=embedding_matrix_result.num_words,
        embedding_dim=embedding_matrix_result.embedding_dim,
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
        random_state=data.hyperparameters.get("random_state", 42),
        verbose=data.hyperparameters.get("verbose", 0),
        class_weight_0=data.hyperparameters.get("class_weight_0", 1),
        class_weight_1=data.hyperparameters.get("class_weight_1", 44),
      )
    )

    return experiment_result

  async def _save_results_to_file(self, detailed_results: List[ExperimentResult], input_data_path: str, experiment_config: ExperimentConfig = None) -> str:
    """Save experiment results to CSV file in data directory"""
    try:
      if experiment_config:
        filepath = experiment_config.results_file_path
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
      else:
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_results_{timestamp}.csv"
        filepath = os.path.join(data_dir, filename)
      
      results_data = []
      for result in detailed_results:
        base_data = {
          "experiment_name": result.experiment_name,
          "status": result.status,
          "execution_time_minutes": result.execution_time_minutes,
          "error_message": result.error_message
        }

        if result.status == "success" and result.metrics:
          base_data.update({
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
          })
        else:
          base_data.update({
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
          })
        
        if result.system_metrics:
          base_data.update({
            "peak_memory_mb": result.system_metrics.peak_memory_mb,
            "average_memory_mb": result.system_metrics.average_memory_mb,
            "peak_cpu_percent": result.system_metrics.peak_cpu_percent,
            "average_cpu_percent": result.system_metrics.average_cpu_percent,
            "throughput_samples_per_second": result.system_metrics.throughput_samples_per_second,
            "average_latency_ms": result.system_metrics.average_latency_ms,
            "data_loading_time_ms": result.system_metrics.data_loading_time_ms,
            "model_training_time_ms": result.system_metrics.model_training_time_ms,
            "model_evaluation_time_ms": result.system_metrics.model_evaluation_time_ms,
            "total_execution_time_ms": result.system_metrics.total_execution_time_ms,
            "memory_efficiency": result.system_metrics.memory_efficiency,
            "cpu_efficiency": result.system_metrics.cpu_efficiency,
            "energy_efficiency_score": result.system_metrics.energy_efficiency_score,
          })
        else:
          base_data.update({
            "peak_memory_mb": None,
            "average_memory_mb": None,
            "peak_cpu_percent": None,
            "average_cpu_percent": None,
            "throughput_samples_per_second": None,
            "average_latency_ms": None,
            "data_loading_time_ms": None,
            "model_training_time_ms": None,
            "model_evaluation_time_ms": None,
            "total_execution_time_ms": None,
            "memory_efficiency": None,
            "cpu_efficiency": None,
            "energy_efficiency_score": None,
          })
        
        results_data.append(base_data)

      df = pd.DataFrame(results_data)
      df.to_csv(filepath, index=False)
      
      print(f"Resultados salvos em: {filepath}")
      return filepath
      
    except Exception as e:
      print(f"Erro ao salvar resultados: {e}")
      return ""

