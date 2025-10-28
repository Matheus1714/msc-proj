import os
import asyncio
from dotenv import load_dotenv
from uuid import uuid4

from utils import setup_project_path
setup_project_path()

from temporalio.client import Client
# from src.workflows.experiment_svm_with_glove_and_tfidf_workflow import (
#   ExperimentSVMWithGloveAndTFIDFWorkflow,
#   ExperimentSVMWithGloveAndTFIDFWorkflowIn,
#   ExperimentSVMWithGloveAndTFIDFHyperparameters,
# )
# from src.workflows.experiment_lstm_with_glove_workflow import (
#   ExperimentLSTMWithGloveWorkflow,
#   ExperimentLSTMWithGloveWorkflowIn,
#   ExperimentLSTMWithGloveHyperparameters,
# )
from src.workflows.experiment_lstm_with_glove_and_attention_workflow import (
  ExperimentLSTMWithGloveAndAttentionWorkflow,
  ExperimentLSTMWithGloveAndAttentionWorkflowIn,
  ExperimentLSTMWithGloveAndAttentionHyperparameters,
)
from constants import WorflowTaskQueue

async def main():    
  load_dotenv()
  
  try:
    client = await Client.connect(os.environ.get("TEMPORAL_CONNECT"))
    
    # await client.start_workflow(
    #   ExperimentSVMWithGloveAndTFIDFWorkflow.run,
    #   arg=ExperimentSVMWithGloveAndTFIDFWorkflowIn(
    #     input_data_path=f"data/academic_works.csv",
    #     hyperparameters=ExperimentSVMWithGloveAndTFIDFHyperparameters(
    #       max_words=20000,
    #       max_len=300,
    #       embedding_dim=300,
    #       random_state=42,
    #       ngram_range=(1, 3),
    #       max_iter=1000,
    #     ),
    #   ),
    #   id=f"experiment-svm-with-glove-and-tfidf-workflow-{uuid4()}",
    #   task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    # )

    # await client.start_workflow(
    #   ExperimentLSTMWithGloveWorkflow.run,
    #   arg=ExperimentLSTMWithGloveWorkflowIn(
    #     input_data_path=f"data/academic_works.csv",
    #     hyperparameters=ExperimentLSTMWithGloveHyperparameters(
    #       max_words=20000,
    #       max_len=300,
    #       embedding_dim=300,
    #       random_state=42,
    #       lstm_units=100,
    #       lstm_dropout=0.2,
    #       lstm_recurrent_dropout=0.2,
    #       pool_dropout=0.5,
    #       dense_units=1,
    #       dense_activation="sigmoid",
    #       batch_size=64,
    #       epochs=5,
    #       learning_rate=3e-4,
    #       loss="binary_crossentropy",
    #       metrics=["accuracy"],
    #       n_splits=5,
    #       verbose=0,
    #       class_weight_0=1,
    #       class_weight_1=44,
    #     ),
    #   ),
    #   id=f"experiment-lstm-with-glove-workflow-{uuid4()}",
    #   task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    # )

    await client.start_workflow(
      ExperimentLSTMWithGloveAndAttentionWorkflow.run,
      arg=ExperimentLSTMWithGloveAndAttentionWorkflowIn(
        input_data_path=f"data/academic_works.csv",
        hyperparameters=ExperimentLSTMWithGloveAndAttentionHyperparameters(
          max_words=20000,
          max_len=300,
          embedding_dim=300,
          random_state=42,
          lstm_units=100,
          lstm_dropout=0.2,
          lstm_recurrent_dropout=0.2,
          pool_dropout=0.5,
          dense_units=1,
          dense_activation="sigmoid",
          batch_size=64,
          epochs=5,
          learning_rate=3e-4,
          loss="binary_crossentropy",
          metrics=["accuracy"],
          n_splits=5,
          verbose=0,
          class_weight_0=1,
          class_weight_1=44,
        ),
      ),
      id=f"experiment-lstm-with-glove-and-attention-workflow-{uuid4()}",
      task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    )

  except Exception as e:
      print(f"❌ Erro ao executar workflow: {e}")

if __name__ == "__main__":
  asyncio.run(main())
