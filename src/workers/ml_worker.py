from enum import Enum
from temporalio.worker import UnsandboxedWorkflowRunner

from src.workflows.ml_simulation_workflow import MLSimulationWorkflow
from src.workflows.data_preprocessing_workflow import DataPreprocessingWorkflow

from src.activities.process_files_activity import process_files_activity
from src.activities.merge_processed_files_activity import merge_processed_files_activity

from constants import WorflowTaskQueue

ml_worker = {
    "workflows": [
        MLSimulationWorkflow,
        DataPreprocessingWorkflow,
    ],
    "activities": [
        process_files_activity,
        merge_processed_files_activity,
    ],
    "workflow_runner": UnsandboxedWorkflowRunner(),
    "task_queue": WorflowTaskQueue.ML_TASK_QUEUE.value,
}
