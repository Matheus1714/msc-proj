import asyncio
from temporalio import workflow
from typing import List
from datetime import timedelta
from temporalio.common import RetryPolicy

from src.activities.process_google_drive_file import process_google_drive_file
from src.activities.merge_processed_data import merge_processed_data
from constants import WorflowTaskQueue

from src.default_types import (
  DataPreprocessingWorkflowIn,
  DataPreprocessingWorkflowOut,
  ProcessGoogleDriveFileOut,
  ProcessGoogleDriveFileIn,
  MergeProcessedDataIn,
)

@workflow.defn
class DataPreprocessingWorkflow:
  @workflow.run
  async def run(self, data: DataPreprocessingWorkflowIn) -> DataPreprocessingWorkflowOut:
    try:
      workflow.logger.info(f"Iniciando pre-processamento de {len(data.source_files)} arquivos")
      
      processing_tasks = [
        workflow.execute_activity(
          process_google_drive_file,
          arg=ProcessGoogleDriveFileIn(file_name=file_name, file_id=file_id),
          start_to_close_timeout=timedelta(minutes=10),
          retry_policy=RetryPolicy(
            initial_interval=timedelta(seconds=1),
            maximum_interval=timedelta(minutes=1),
            maximum_attempts=3,
          ),
          task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
        )
        for file_name, file_id in data.source_files
      ]
      
      workflow.logger.info("Aguardando processamento paralelo de todos os arquivos...")
      all_processed_files: List[ProcessGoogleDriveFileOut] = await asyncio.gather(*processing_tasks)
      
      valid_data = [data for data in all_processed_files if data]
      workflow.logger.info(f"Processados com sucesso {len(valid_data)} de {len(data.source_files)} arquivos")
      
      total_works = await workflow.execute_activity(
        merge_processed_data,
        arg=MergeProcessedDataIn(all_processed_files=valid_data, output_path=data.output_path),
        start_to_close_timeout=timedelta(minutes=5),
        task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
      )
      
      workflow.logger.info(f"Pre-processamento concluído: {total_works} trabalhos salvos")
      return total_works

    except Exception as e:
      workflow.logger.error(f"Erro ao executar workflow: {e}")
      raise e