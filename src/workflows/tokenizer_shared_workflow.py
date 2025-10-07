from temporalio import workflow
from datetime import timedelta

from src.default_types import TokenizeSharedWorkflowIn, TokenizeSharedWorkflowOut

@workflow.defn
class TokenizeSharedWorkflow:
  @workflow.run
  async def run(self, data: TokenizeSharedWorkflowIn) -> TokenizeSharedWorkflowOut:
    # TODO: Implementar lógica de tokenização
    # - Carregar dados do arquivo
    # - Aplicar estratégia de tokenização (TF-IDF, Word2Vec, etc.)
    # - Salvar dados tokenizados
    
    return TokenizeSharedWorkflowOut(
      tokenized_data_path=f"data/tokenized_{data.strategy}_{data.file_path.split('/')[-1]}",
    )
