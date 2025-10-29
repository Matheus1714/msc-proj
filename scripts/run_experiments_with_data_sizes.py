from logging import exception
import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
from uuid import uuid4

from utils import setup_project_path
setup_project_path()

from temporalio.client import Client
from src.workflows.experiments_workflow import (
  ExperimentsWorkflow,
  ExperimentsWorkflowIn,
)
from constants import WorflowTaskQueue, ExperimentConfig

def create_data_subsets(input_file_path: str, output_dir: str, max_rows: int = 2000):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(input_file_path)
    limit = len(df)

    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True).head(max_rows)
    total_rows = min(limit, max_rows)
    percentages = [50, 75, 100]  # 10%, 25%, 50%, 75%, 100%
    
    data_subsets = []
    
    for pct in percentages:
        target_rows = int((pct / 100) * total_rows)

        subset_df = shuffled_df.head(target_rows)

        filename = f"academic_works_{pct}pct_{len(subset_df)}rows.csv"
        filepath = os.path.join(output_dir, filename)
        
        subset_df.to_csv(filepath, index=False)
        
        data_subsets.append((pct, filepath, len(subset_df)))
    
    return data_subsets

async def start_experiments_for_data_size(client, data_size_info, hyperparameters):
    pct, filepath, _ = data_size_info
    
    experiment_config = ExperimentConfig.create()
    experiment_config.create_directories()

    try:
        await client.execute_workflow(
            ExperimentsWorkflow.run,
            arg=ExperimentsWorkflowIn(
                input_data_path=filepath,
                hyperparameters=hyperparameters,
                experiment_config=experiment_config,
            ),
            id=f"experiments-{pct}pct-{uuid4()}",
            task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
        )
    except Exception as e:
        print(e)


async def main():
    load_dotenv()
    
    try:
        client = await Client.connect(os.environ.get("TEMPORAL_CONNECT"))
        
        hyperparameters = {
            "max_words": 20000,
            "max_len": 300,
            "embedding_dim": 300,
            "random_state": 42,
            "lstm_units": 100,
            "lstm_dropout": 0.2,
            "lstm_recurrent_dropout": 0.2,
            "pool_dropout": 0.5,
            "dense_units": 1,
            "dense_activation": "sigmoid",
            "batch_size": 64,
            "epochs": 5,
            "learning_rate": 3e-4,
            "loss": "binary_crossentropy",
            "metrics": ["accuracy"],
            "n_splits": 5,
            "verbose": 0,
            "class_weight_0": 1,
            "class_weight_1": 44,
            "ngram_range": (1, 3),
            "max_iter": 1000,
        }
        
        input_file = "data/academic_works.csv"
        data_subsets_dir = "data/subsets"
        data_subsets = create_data_subsets(input_file, data_subsets_dir, max_rows=400)

        for data_size_info in data_subsets:
            await start_experiments_for_data_size(
                client, 
                data_size_info, 
                hyperparameters
            )

    except Exception as e:
        print(f"❌ Erro ao executar workflow: {e}")

if __name__ == "__main__":
    asyncio.run(main())
