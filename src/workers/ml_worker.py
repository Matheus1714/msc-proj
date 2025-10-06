from enum import Enum
from temporalio.worker import UnsandboxedWorkflowRunner

from src.workflows.ml_simulation_workflow import MLSimulationWorkflow

class WorflowTaskQueue(Enum):
    ML_TASK_QUEUE = "ml-task-queue"

ml_worker = {
    "workflows": [MLSimulationWorkflow],
    "activities": [],
    "workflow_runner": UnsandboxedWorkflowRunner(),
    "task_queue": WorflowTaskQueue.ML_TASK_QUEUE.value,
}
