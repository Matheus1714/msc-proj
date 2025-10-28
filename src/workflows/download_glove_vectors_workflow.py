from dataclasses import dataclass
from temporalio import workflow
from temporalio.common import RetryPolicy
from typing import Optional
from datetime import timedelta

from src.activities.download_glove_vectors_activity import (
    download_glove_vectors_activity,
    DownloadGloveVectorsIn,
    DownloadGloveVectorsOut,
)

@dataclass
class DownloadGloveVectorsWorkflowIn:
    target_dir: str
    embedding_dim: int = 300

@dataclass
class DownloadGloveVectorsWorkflowOut:
    glove_file_path: str
    target_dir: str
    success: bool
    message: str
    attempts: int

@workflow.defn
class DownloadGloveVectorsWorkflow:
    @workflow.run
    async def run(self, data: DownloadGloveVectorsWorkflowIn) -> DownloadGloveVectorsWorkflowOut:
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=5),
            maximum_interval=timedelta(minutes=2),
            maximum_attempts=3,
            backoff_coefficient=2.0,
        )
        
        attempts = 0
        last_error = None
        
        for attempt in range(3):
            attempts += 1
            try:
                result = await workflow.execute_activity(
                    download_glove_vectors_activity,
                    DownloadGloveVectorsIn(
                        target_dir=data.target_dir,
                        embedding_dim=data.embedding_dim,
                    ),
                    start_to_close_timeout=timedelta(minutes=30),
                    retry_policy=retry_policy,
                )
                
                if result.success:
                    return DownloadGloveVectorsWorkflowOut(
                        glove_file_path=result.glove_file_path,
                        target_dir=result.target_dir,
                        success=True,
                        message=result.message,
                        attempts=attempts,
                    )
                else:
                    last_error = result.message
                    print(f"❌ Attempt {attempts} failed: {result.message}")
                    
            except Exception as e:
                last_error = str(e)
                print(f"❌ Attempt {attempts} failed with exception: {str(e)}")
                
            if attempt < 2:
                await workflow.sleep(timedelta(seconds=10))
        
        return DownloadGloveVectorsWorkflowOut(
            glove_file_path="",
            target_dir=data.target_dir,
            success=False,
            message=f"Failed after {attempts} attempts. Last error: {last_error}",
            attempts=attempts,
        )
