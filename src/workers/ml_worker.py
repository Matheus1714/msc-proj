from temporalio.worker import UnsandboxedWorkflowRunner

from src.workflows.data_preprocessing_workflow import DataPreprocessingWorkflow
from src.workflows.experiment_svm_with_glove_and_tfidf_workflow import ExperimentSVMWithGloveAndTFIDFWorkflow

from src.activities.process_files_activity import process_files_activity
from src.activities.merge_processed_files_activity import merge_processed_files_activity
from src.activities.load_glove_embeddings_activity import load_glove_embeddings_activity
from src.activities.run_experiment_svm_with_glove_and_tfidf_activity import run_experiment_svm_with_glove_and_tfidf_activity
from src.activities.prepare_data_for_experiment_activity import prepare_data_for_experiment_activity
from src.activities.tokenizer_activity import tokenizer_activity
from src.activities.split_data_activity import split_data_activity


from constants import WorflowTaskQueue

ml_worker = {
    "workflows": [
        DataPreprocessingWorkflow,
        ExperimentSVMWithGloveAndTFIDFWorkflow,
    ],
    "activities": [
        process_files_activity,
        merge_processed_files_activity,
        load_glove_embeddings_activity,
        prepare_data_for_experiment_activity,
        run_experiment_svm_with_glove_and_tfidf_activity,
        tokenizer_activity,
        split_data_activity,
    ],
    "workflow_runner": UnsandboxedWorkflowRunner(),
    "task_queue": WorflowTaskQueue.ML_TASK_QUEUE.value,
}
