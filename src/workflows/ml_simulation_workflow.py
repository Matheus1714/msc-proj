from temporalio import workflow

from src.default_types import (
  MLSimulationWorkflowIn,
  MLSimulationWorkflowOut,
)

@workflow.defn
class MLSimulationWorkflow:
    @workflow.run
    async def run(self, data: MLSimulationWorkflowIn) -> MLSimulationWorkflowOut:
        ...
