import os
import asyncio
from dotenv import load_dotenv
from uuid import uuid4
from datetime import datetime

from utils import setup_project_path
setup_project_path()

from temporalio.client import Client
from src.workflows.experiments_workflow import (
  ExperimentsWorkflow,
  ExperimentsWorkflowIn,
)
from constants import WorflowTaskQueue, ExperimentConfig
from src.utils.create_subsets_from_csv import create_subsets_from_csv
from src.utils.calculate_class_weights import calculate_class_weights_from_csv

async def start_experiments_for_data_size(client, data_size_info, hyperparameters):
    pct, filepath, _ = data_size_info
    
    # Calcular class weights para este subset específico
    class_weight_0, class_weight_1 = calculate_class_weights_from_csv(filepath)
    print(f"⚖️  [{pct}%] Class weights calculados: class_weight_0={class_weight_0}, class_weight_1={class_weight_1}")
    
    # Atualizar class weights nos hyperparameters para este subset
    hyperparameters_copy = hyperparameters.copy()
    hyperparameters_copy["class_weight_0"] = class_weight_0
    hyperparameters_copy["class_weight_1"] = class_weight_1
    
    experiment_config = ExperimentConfig.create()
    experiment_config.create_directories()

    try:
        await client.execute_workflow(
            ExperimentsWorkflow.run,
            arg=ExperimentsWorkflowIn(
                input_data_path=filepath,
                hyperparameters=hyperparameters_copy,
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
            "class_weight_0": 1,  # Será calculado automaticamente para cada subset
            "class_weight_1": 44,  # Será calculado automaticamente para cada subset
            "ngram_range": (1, 3),
            "max_iter": 1000,
        }
        
        input_file = "data/academic_works.csv"
        data_subsets_dir = "data/subsets"
        data_subsets = create_subsets_from_csv(
            input_file,
            data_subsets_dir,
            percentages=[50, 75, 100],
            max_rows=400
        )

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
