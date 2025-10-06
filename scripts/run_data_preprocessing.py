import os
import asyncio
from dotenv import load_dotenv
from uuid import uuid4

from utils import setup_project_path
setup_project_path()

from temporalio.client import Client
from src.workflows.data_preprocessing_workflow import DataPreprocessingWorkflow
from src.default_types import DataPreprocessingWorkflowIn
from constants import WorflowTaskQueue

async def main():    
    load_dotenv()
    
    try:
        client = await Client.connect(os.environ.get("TEMPORAL_CONNECT"))
        
        await client.start_workflow(
            DataPreprocessingWorkflow.run,
            arg=DataPreprocessingWorkflowIn(
                output_path="data/academic_works.csv",
            ),
            id=f"data-preprocessing-workflow-{uuid4()}",
            task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
        )
        
    except Exception as e:
        print(f"❌ Erro ao executar workflow: {e}")

if __name__ == "__main__":
    asyncio.run(main())
