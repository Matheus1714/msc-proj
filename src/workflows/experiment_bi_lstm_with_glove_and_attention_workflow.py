from datetime import timedelta
from temporalio import workflow
from dataclasses import dataclass
from typing import TypedDict

from src.activities.load_glove_embeddings_activity import (
  load_glove_embeddings_activity,
  LoadGloveEmbeddingsIn,
  LoadGloveEmbeddingsOut,
)
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
from src.activities.split_data_activity import (
  split_data_activity,
  SplitDataIn,
  SplitDataOut,
)
from src.activities.run_experiment_bi_lstm_with_glove_and_attention_activity import (
  run_experiment_bi_lstm_with_glove_and_attention_activity,
  RunExperimentBiLSTMWithGloveAndAttentionIn,
  RunExperimentBiLSTMWithGloveAndAttentionOut,
)
from constants import (
  WorflowTaskQueue,
  GLOVE_6B_300D_FILE_PATH,
  ExperimentConfig,
)

from src.utils.calculate_metrics import EvaluationData

class ExperimentBiLSTMWithGloveAndAttentionHyperparameters(TypedDict):
  max_words: int
  max_len: int
  embedding_dim: int
  random_state: int
  lstm_units: int
  lstm_dropout: float
  lstm_recurrent_dropout: float
  pool_dropout: float
  dense_units: int
  dense_activation: str
  batch_size: int
  epochs: int
  learning_rate: float
  loss: str
  metrics: list
  n_splits: int
  verbose: int
  class_weight_0: float
  class_weight_1: float

@dataclass
class ExperimentBiLSTMWithGloveAndAttentionWorkflowIn:
  input_data_path: str
  hyperparameters: ExperimentBiLSTMWithGloveAndAttentionHyperparameters
  experiment_config: ExperimentConfig

@dataclass
class ExperimentBiLSTMWithGloveAndAttentionWorkflowOut:
  metrics: EvaluationData

@workflow.defn
class ExperimentBiLSTMWithGloveAndAttentionWorkflow:
  @workflow.run
  async def run(self, data: ExperimentBiLSTMWithGloveAndAttentionWorkflowIn) -> ExperimentBiLSTMWithGloveAndAttentionWorkflowOut:
    data.experiment_config.create_directories()
    prepare_data_for_experiment_result: PrepareDataForExperimentOut = await workflow.execute_activity(
      prepare_data_for_experiment_activity,
      arg=PrepareDataForExperimentIn(
        input_data_path=data.input_data_path,
        output_data_path=data.experiment_config.prepared_data_path,
        random_state=data.hyperparameters["random_state"],
      ),
      start_to_close_timeout=timedelta(minutes=5),
      task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    )

    tokenizer_result: TokenizerOut = await workflow.execute_activity(
      tokenizer_activity,
      arg=TokenizerIn(
        input_data_path=prepare_data_for_experiment_result.output_data_path,
        tokenized_data_path=data.experiment_config.tokenized_data_path,
        word_index_path=data.experiment_config.word_index_path,
        x_seq_path=data.experiment_config.x_seq_path,
        y_path=data.experiment_config.y_path,
        max_words=data.hyperparameters["max_words"],
        max_len=data.hyperparameters["max_len"],
      ),
      start_to_close_timeout=timedelta(minutes=5),
      task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    )

    embedding_matrix_result: LoadGloveEmbeddingsOut = await workflow.execute_activity(
      load_glove_embeddings_activity,
      arg=LoadGloveEmbeddingsIn(
        glove_file_path=GLOVE_6B_300D_FILE_PATH,
        output_path=data.experiment_config.glove_embeddings_path,
        max_words=data.hyperparameters["max_words"],
        word_index_path=tokenizer_result.word_index_path,
      ),
      start_to_close_timeout=timedelta(minutes=5),
      task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    )

    split_data_result: SplitDataOut = await workflow.execute_activity(
      split_data_activity,
      arg=SplitDataIn(
        x_seq_path=tokenizer_result.x_seq_path,
        y_path=tokenizer_result.y_path,
        x_train_path=data.experiment_config.x_train_path,
        x_val_path=data.experiment_config.x_val_path,
        x_test_path=data.experiment_config.x_test_path,
        y_train_path=data.experiment_config.y_train_path,
        y_val_path=data.experiment_config.y_val_path,
        y_test_path=data.experiment_config.y_test_path,
        random_state=data.hyperparameters["random_state"],
      ),
      start_to_close_timeout=timedelta(minutes=5),
      task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    )

    experiment_result: RunExperimentBiLSTMWithGloveAndAttentionOut = await workflow.execute_activity(
      run_experiment_bi_lstm_with_glove_and_attention_activity,
      arg=RunExperimentBiLSTMWithGloveAndAttentionIn(
        input_data_path=data.input_data_path,
        x_train_path=split_data_result.x_train_path,
        y_train_path=split_data_result.y_train_path,
        embedding_matrix_path=data.experiment_config.glove_embeddings_path,
        max_len=data.hyperparameters["max_len"],
        num_words=embedding_matrix_result.num_words,
        embedding_dim=embedding_matrix_result.embedding_dim,
        lstm_units=data.hyperparameters["lstm_units"],
        lstm_dropout=data.hyperparameters["lstm_dropout"],
        lstm_recurrent_dropout=data.hyperparameters["lstm_recurrent_dropout"],
        pool_dropout=data.hyperparameters["pool_dropout"],
        dense_units=data.hyperparameters["dense_units"],
        dense_activation=data.hyperparameters["dense_activation"],
        batch_size=data.hyperparameters["batch_size"],
        epochs=data.hyperparameters["epochs"],
        learning_rate=data.hyperparameters["learning_rate"],
        loss=data.hyperparameters["loss"],
        metrics=data.hyperparameters["metrics"],
        n_splits=data.hyperparameters["n_splits"],
        random_state=data.hyperparameters["random_state"],
        verbose=data.hyperparameters["verbose"],
        class_weight_0=data.hyperparameters["class_weight_0"],
        class_weight_1=data.hyperparameters["class_weight_1"],
      ),
      start_to_close_timeout=timedelta(days=100),
      task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    )

    return ExperimentBiLSTMWithGloveAndAttentionWorkflowOut(
      metrics=experiment_result.metrics,
    )
