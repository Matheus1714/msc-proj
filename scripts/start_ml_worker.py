import os
import asyncio
from dotenv import load_dotenv

from temporalio.worker import Worker
from temporalio.client import Client

from utils import setup_project_path

setup_project_path()

from src.workers.ml_worker import ml_worker

load_dotenv()

async def main():
  try:
    client = await Client.connect(os.environ.get("TEMPORAL_CONNECT"))

    worker = Worker(
      client,
      task_queue=ml_worker["task_queue"],
      workflows=ml_worker["workflows"],
      activities=ml_worker["activities"],
      workflow_runner=ml_worker["workflow_runner"],
    )

    print("Workers runnig")

    await worker.run()
  except Exception as e:
    print(str(e))
  finally:
    print("Finish workers")

if __name__ == "__main__":
  asyncio.run(main())
