import asyncio
from temporalio import workflow
from typing import List
from datetime import timedelta
from temporalio.common import RetryPolicy

from src.activities.process_files_activity import process_files_activity
from src.activities.merge_processed_files_activity import merge_processed_files_activity
from constants import WorflowTaskQueue

from src.default_types import (
  DataPreprocessingWorkflowIn,
  DataPreprocessingWorkflowOut,
  ProcessGoogleDriveFileOut,
  ProcessGoogleDriveFileIn,
  MergeProcessedDataIn,
)

from constants import SOURCE_INPUT_FILES

@workflow.defn
class DataPreprocessingWorkflow:
  @workflow.run
  async def run(self, data: DataPreprocessingWorkflowIn) -> DataPreprocessingWorkflowOut:
    try:
      workflow.logger.info(f"Iniciando pre-processamento de {len(SOURCE_INPUT_FILES)} arquivos")
      
      concurrency = 5
      all_processed_files: List[ProcessGoogleDriveFileOut | None] = []
      
      for i in range(0, len(SOURCE_INPUT_FILES), concurrency):
        current_batch = SOURCE_INPUT_FILES[i:i+concurrency]
        workflow.logger.info(f"Processando lote {i//concurrency + 1} com {len(current_batch)} arquivos...")
        batch_tasks = [
          workflow.execute_activity(
            process_files_activity,
            arg=ProcessGoogleDriveFileIn(
              file_name=file_name,
              file_path=file_path,
            ),
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(
              initial_interval=timedelta(seconds=5),
              maximum_interval=timedelta(seconds=10),
              maximum_attempts=3,
            ),
            task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
          )
          for file_name, file_path in current_batch
        ]
        batch_results: List[ProcessGoogleDriveFileOut | None] = await asyncio.gather(*batch_tasks)
        all_processed_files.extend(batch_results)
      
      valid_paths = [d.file_path for d in all_processed_files if d is not None]
      workflow.logger.info(f"Processados com sucesso {len(valid_paths)} de {len(SOURCE_INPUT_FILES)} arquivos")
      
      total_processed = await workflow.execute_activity(
        merge_processed_files_activity,
        arg=MergeProcessedDataIn(all_processed_files=valid_paths, output_path=data.output_path),
        start_to_close_timeout=timedelta(minutes=5),
        task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
      )
      
      workflow.logger.info(f"Pre-processamento concluído: {total_processed} trabalhos salvos")
      return DataPreprocessingWorkflowOut(total_processed_works=total_processed)

    except Exception as e:
      workflow.logger.error(f"Erro ao executar workflow: {e}")
      raise e