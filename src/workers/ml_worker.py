from enum import Enum
from temporalio.worker import UnsandboxedWorkflowRunner

from src.workflows.data_preprocessing_workflow import DataPreprocessingWorkflow
from src.workflows.experiment_workflow import ExperimentWorkflow
from src.workflows.tokenizer_shared_workflow import TokenizeSharedWorkflow
from src.workflows.simulate_model_workflow import SimulateModelWorkflow

from src.activities.process_files_activity import process_files_activity
from src.activities.merge_processed_files_activity import merge_processed_files_activity
from src.activities.prepare_data_for_experiment_activity import prepare_data_for_experiment_activity
from src.activities.train_model_activity import (
    train_model_activity,
    train_svm_activity,
    train_random_forest_activity
)
from src.activities.validate_model_activity import validate_model_activity
from src.activities.run_production_inference_activity import run_production_inference_activity
from src.activities.aggregate_results_activity import aggregate_results_activity

from constants import WorflowTaskQueue

ml_worker = {
    "workflows": [
        DataPreprocessingWorkflow,
        ExperimentWorkflow,
        TokenizeSharedWorkflow,
        SimulateModelWorkflow,
    ],
    "activities": [
        process_files_activity,
        merge_processed_files_activity,
        prepare_data_for_experiment_activity,
        train_model_activity,
        train_svm_activity,
        train_random_forest_activity,
        validate_model_activity,
        run_production_inference_activity,
        aggregate_results_activity,
    ],
    "workflow_runner": UnsandboxedWorkflowRunner(),
    "task_queue": WorflowTaskQueue.ML_TASK_QUEUE.value,
}
