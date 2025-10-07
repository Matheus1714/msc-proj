import os
import asyncio
from dotenv import load_dotenv
from uuid import uuid4

from utils import setup_project_path
setup_project_path()

from temporalio.client import Client
from src.workflows.experiment_workflow import ExperimentWorkflow
from src.default_types import ExperimentWorkflowIn, ModelConfig
from constants import WorflowTaskQueue

async def main():    
    load_dotenv()
    
    try:
        client = await Client.connect(os.environ.get("TEMPORAL_CONNECT"))
        
        # Configuração do experimento Random Forest
        model_config = ModelConfig(
            name="random_forest_experiment_1",
            type="random_forest",
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            }
        )
        
        experiment_input = ExperimentWorkflowIn(
            dataset_id="academic_works",
            model_config=model_config,
            tokenizer_strategy="word2vec",
            model_path=None  # None para treinar novo modelo
        )
        
        await client.start_workflow(
            ExperimentWorkflow.run,
            arg=experiment_input,
            id=f"random-forest-experiment-{uuid4()}",
            task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
        )
        
        print("✅ Experimento Random Forest iniciado com sucesso!")
        print(f"📊 Dataset: {experiment_input.dataset_id}")
        print(f"🌲 Modelo: {model_config['name']} ({model_config['type']})")
        print(f"🔤 Tokenização: {experiment_input.tokenizer_strategy}")
        
    except Exception as e:
        print(f"❌ Erro ao executar experimento: {e}")

if __name__ == "__main__":
    asyncio.run(main())
