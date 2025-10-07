from temporalio import workflow
from datetime import timedelta

from src.default_types import (
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

@workflow.defn
class SimulateModelWorkflow:
  @workflow.run
  async def run(self, data: SimulateModelWorkflowIn) -> SimulateModelWorkflowOut:
    # TODO: Implementar lógica de simulação do modelo
    # 1. Treinar modelo com 70% dos dados (se model_path não fornecido)
    # 2. Validar com 20% dos dados
    # 3. Aplicar em 10% dos dados de produção
    # 4. Agregar resultados
    
    # Por enquanto, retorna um caminho fictício
    model_path = f"models/{data.strategy}_model.pkl"
    
    return SimulateModelWorkflowOut(
      result=model_path,
    )