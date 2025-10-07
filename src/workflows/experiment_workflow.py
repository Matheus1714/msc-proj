from temporalio import workflow
from datetime import timedelta

from src.default_types import (
  ExperimentWorkflowIn, 
  ExperimentWorkflowOut,
  PrepareDataForExperimentIn,
  PrepareDataForExperimentOut,
  TokenizeSharedWorkflowIn,
  TokenizeSharedWorkflowOut,
  SimulateModelWorkflowIn,
  SimulateModelWorkflowOut,
  TrainModelIn,
  TrainModelOut,
  ValidateModelIn,
  ValidateModelOut,
  RunProductionInferenceIn,
  RunProductionInferenceOut,
  AggregateResultsIn,
  AggregateResultsOut
)
from src.activities.prepare_data_for_experiment_activity import prepare_data_for_experiment_activity
from src.workflows.tokenizer_shared_workflow import TokenizeSharedWorkflow
from src.workflows.simulate_model_workflow import SimulateModelWorkflow

@workflow.defn
class ExperimentWorkflow:
  @workflow.run
  async def run(self, data: ExperimentWorkflowIn) -> ExperimentWorkflowOut:
    # 1. Preparar dados (70% treino, 20% validação, 10% produção)
    prepared_data = await workflow.execute_activity(
      prepare_data_for_experiment_activity,
      PrepareDataForExperimentIn(
        id=data.dataset_id,
        file_path=f"data/{data.dataset_id}.csv"
      ),
      start_to_close_timeout=timedelta(minutes=5),
    )

    # 2. Tokenizar dados
    tokenized_data = await workflow.execute_child_workflow(
      TokenizeSharedWorkflow.run,
      TokenizeSharedWorkflowIn(
        file_path=prepared_data.input_data_path,
        strategy=data.tokenizer_strategy
      ),
      id=f"tokenize-{data.dataset_id}-{data.tokenizer_strategy}",
    )

    # 3. Executar simulação do modelo
    simulation_result = await workflow.execute_child_workflow(
      SimulateModelWorkflow.run,
      SimulateModelWorkflowIn(
        file_path=tokenized_data.tokenized_data_path,
        strategy=data.model_config.get("type", "")
      ),
      id=f"simulate-{data.dataset_id}-{data.model_config['name']}",
    )

    return ExperimentWorkflowOut(
      model_path=simulation_result.result,
      validation_metrics_path="",
      production_metrics_path="",
      final_report_path="",
    )
