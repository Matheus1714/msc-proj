from temporalio.worker import UnsandboxedWorkflowRunner

# Workflows

from src.workflows.data_preprocessing_workflow import DataPreprocessingWorkflow
from src.workflows.experiments_workflow import ExperimentsWorkflow

# Activities

from src.activities.process_files_activity import process_files_activity
from src.activities.merge_processed_files_activity import merge_processed_files_activity
from src.activities.load_glove_embeddings_activity import load_glove_embeddings_activity
from src.activities.prepare_data_for_experiment_activity import prepare_data_for_experiment_activity
from src.activities.tokenizer_activity import tokenizer_activity
from src.activities.split_data_activity import split_data_activity
from src.activities.generate_machine_specs_activity import generate_machine_specs_activity

# Experiments Workflows

from src.workflows.experiment_svm_with_glove_and_tfidf_workflow import ExperimentSVMWithGloveAndTFIDFWorkflow
from src.workflows.experiment_lstm_with_glove_workflow import ExperimentLSTMWithGloveWorkflow
from src.workflows.experiment_lstm_with_glove_and_attention_workflow import ExperimentLSTMWithGloveAndAttentionWorkflow
from src.workflows.experiment_bi_lstm_with_glove_workflow import ExperimentBiLSTMWithGloveWorkflow
from src.workflows.experiment_bi_lstm_with_glove_and_attention_workflow import ExperimentBiLSTMWithGloveAndAttentionWorkflow

# Experiments Activities

from src.activities.run_experiment_svm_with_glove_and_tfidf_activity import run_experiment_svm_with_glove_and_tfidf_activity
from src.activities.run_experiment_lstm_with_glove_activity import run_experiment_lstm_with_glove_activity
from src.activities.run_experiment_lstm_with_glove_and_attention_activity import run_experiment_lstm_with_glove_and_attention_activity
from src.activities.run_experiment_bi_lstm_with_glove_activity import run_experiment_bi_lstm_with_glove_activity
from src.activities.run_experiment_bi_lstm_with_glove_and_attention_activity import run_experiment_bi_lstm_with_glove_and_attention_activity

from constants import WorflowTaskQueue

ml_worker = {
    "workflows": [
        DataPreprocessingWorkflow,
        ExperimentSVMWithGloveAndTFIDFWorkflow,
        ExperimentLSTMWithGloveWorkflow,
        ExperimentLSTMWithGloveAndAttentionWorkflow,
        ExperimentBiLSTMWithGloveWorkflow,
        ExperimentBiLSTMWithGloveAndAttentionWorkflow,
        ExperimentsWorkflow,
    ],
    "activities": [
        run_experiment_svm_with_glove_and_tfidf_activity,
        run_experiment_lstm_with_glove_activity,
        run_experiment_lstm_with_glove_and_attention_activity,
        run_experiment_bi_lstm_with_glove_activity,
        run_experiment_bi_lstm_with_glove_and_attention_activity,
        process_files_activity,
        merge_processed_files_activity,
        load_glove_embeddings_activity,
        prepare_data_for_experiment_activity,
        tokenizer_activity,
        split_data_activity,
        generate_machine_specs_activity,
    ],
    "workflow_runner": UnsandboxedWorkflowRunner(),
    "task_queue": WorflowTaskQueue.ML_TASK_QUEUE.value,
}
