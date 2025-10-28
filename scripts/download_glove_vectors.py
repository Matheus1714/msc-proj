import os
import asyncio
from dotenv import load_dotenv
from uuid import uuid4

from utils import setup_project_path
setup_project_path()

from temporalio.client import Client
from src.workflows.download_glove_vectors_workflow import (
    DownloadGloveVectorsWorkflow,
    DownloadGloveVectorsWorkflowIn,
)
from constants import WorflowTaskQueue

async def main():    
    load_dotenv()
    
    try:
        client = await Client.connect(os.environ.get("TEMPORAL_CONNECT"))
        
        # Default target directory for GloVe vectors
        target_dir = "./data/word_vectors/glove"
        embedding_dim = 300  # Can be changed to 50, 100, 200, or 300
        
        print(f"📁 Target directory: {target_dir}")
        print(f"🔢 Embedding dimension: {embedding_dim}d")
        
        result = await client.execute_workflow(
            DownloadGloveVectorsWorkflow.run,
            arg=DownloadGloveVectorsWorkflowIn(
                target_dir=target_dir,
                embedding_dim=embedding_dim,
            ),
            id=f"download-glove-vectors-{uuid4()}",
            task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
        )
        
        if result.success:
            print("🎉 GloVe vectors downloaded successfully!")
            print(f"✅ File path: {result.glove_file_path}")
            print(f"📁 Target directory: {result.target_dir}")
            print(f"📊 Attempts made: {result.attempts}")
            print(f"💬 Message: {result.message}")
        else:
            print("❌ Failed to download GloVe vectors!")
            print(f"📊 Attempts made: {result.attempts}")
            print(f"💬 Error: {result.message}")
            return 1

    except Exception as e:
        print(f"❌ Error executing workflow: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
