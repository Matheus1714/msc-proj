import os
import asyncio
from enum import Enum
from dotenv import load_dotenv

from temporalio.worker import Worker, UnsandboxedWorkflowRunner
from temporalio.client import Client

load_dotenv()

class WorflowTaskQueue(Enum):
    ML_TASK_QUEUE = "ml-task-queue"

async def main():
    try:
        client = await Client.connect(os.environ.get("TEMPORAL_CONNECT"))

        worker = Worker(
            client,
            task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
            workflows=[],
            activities=[],
            workflow_runner=UnsandboxedWorkflowRunner(),
        )

        print("Workers runnig")

        await worker.run()
    except Exception as e:
        print(str(e))
    finally:
        print("Finish workers")

if __name__ == "__main__":
    asyncio.run(main())
