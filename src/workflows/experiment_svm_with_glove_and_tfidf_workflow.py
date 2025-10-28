from datetime import timedelta
from temporalio import workflow
from dataclasses import dataclass
from typing import TypedDict, Tuple

from src.activities.load_glove_embeddings_activity import (
  load_glove_embeddings_activity,
  LoadGloveEmbeddingsIn,
)
from src.activities.prepare_data_for_experiment_activity import (
  prepare_data_for_experiment_activity,
  PrepareDataForExperimentIn,
  PrepareDataForExperimentOut,
)
from src.activities.run_experiment_svm_with_glove_and_tfidf_activity import (
  run_experiment_svm_with_glove_and_tfidf_activity,
  RunExperimentSVMWithGloveAndTFIDFIn,
  RunExperimentSVMWithGloveAndTFIDFOut,
)
from src.activities.tokenizer_activity import (
  tokenizer_activity,
  TokenizerIn,
  TokenizerOut,
)
from src.activities.split_data_activity import (
  split_data_activity,
  SplitDataIn,
)
from constants import (
  WorflowTaskQueue,
  GLOVE_6B_300D_FILE_PATH,
  GLOVE_EMBEDDINGS_PATH,
)

class ExperimentSVMWithGloveAndTFIDFHyperparameters(TypedDict):
  max_words: int
  max_len: int
  embedding_dim: int
  random_state: int
  ngram_range: Tuple[int, int]
  max_iter: int
  max_words: int
  random_state: int

@dataclass
class ExperimentSVMWithGloveAndTFIDFWorkflowIn:
  input_data_path: str
  hyperparameters: ExperimentSVMWithGloveAndTFIDFHyperparameters

@dataclass
class ExperimentSVMWithGloveAndTFIDFWorkflowOut:
  ...

@workflow.defn
class ExperimentSVMWithGloveAndTFIDFWorkflow:
  @workflow.run
  async def run(self, data: ExperimentSVMWithGloveAndTFIDFWorkflowIn) -> ExperimentSVMWithGloveAndTFIDFWorkflowOut:
    prepare_data_for_experiment_result: PrepareDataForExperimentOut = await workflow.execute_activity(
      prepare_data_for_experiment_activity,
      arg=PrepareDataForExperimentIn(
        input_data_path=data.input_data_path,
        random_state=data.hyperparameters["random_state"],
      ),
      start_to_close_timeout=timedelta(minutes=5),
      task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    )

    tokenizer_result: TokenizerOut = await workflow.execute_activity(
      tokenizer_activity,
      arg=TokenizerIn(
        input_data_path=prepare_data_for_experiment_result.output_data_path,
        max_words=data.hyperparameters["max_words"],
        max_len=data.hyperparameters["max_len"],
      ),
      start_to_close_timeout=timedelta(minutes=5),
      task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    )

    await workflow.execute_activity(
      load_glove_embeddings_activity,
      arg=LoadGloveEmbeddingsIn(
        glove_file_path=GLOVE_6B_300D_FILE_PATH,
        output_path=GLOVE_EMBEDDINGS_PATH,
        max_words=data.hyperparameters["max_words"],
        word_index_path=tokenizer_result.word_index_path,
      ),
      start_to_close_timeout=timedelta(minutes=5),
      task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    )

    await workflow.execute_activity(
      split_data_activity,
      arg=SplitDataIn(
        x_seq_path=tokenizer_result.x_seq_path,
        y_path=tokenizer_result.y_path,
        random_state=data.hyperparameters["random_state"],
      ),
      start_to_close_timeout=timedelta(minutes=5),
      task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    )

    experiment: RunExperimentSVMWithGloveAndTFIDFOut = await workflow.execute_activity(
      run_experiment_svm_with_glove_and_tfidf_activity,
      arg=RunExperimentSVMWithGloveAndTFIDFIn(
        input_data_path=prepare_data_for_experiment_result.output_data_path,
        y_path=tokenizer_result.y_path,
        random_state=data.hyperparameters["random_state"],
        max_iter=data.hyperparameters["max_iter"],
        ngram_range=data.hyperparameters["ngram_range"],
      ),
      start_to_close_timeout=timedelta(minutes=5),
      task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    )

    return ExperimentSVMWithGloveAndTFIDFWorkflowOut()
