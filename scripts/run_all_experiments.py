import os
import asyncio
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

async def main():    
  load_dotenv()
  
  try:
    client = await Client.connect(os.environ.get("TEMPORAL_CONNECT"))
    
    experiment_config = ExperimentConfig.create()
    experiment_config.create_directories()
    
    print(f"📁 Diretório do experimento: {experiment_config.base_dir}")
    
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
    
    result = await client.execute_workflow(
      ExperimentsWorkflow.run,
      arg=ExperimentsWorkflowIn(
        input_data_path="data/academic_works.csv",
        hyperparameters=hyperparameters,
        experiment_config=experiment_config,
      ),
      id=f"all-experiments-workflow-{uuid4()}",
      task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
    )
    
    print("🎉 Todos os experimentos foram executados!")
    print(f"✅ Experimentos concluídos: {len(result.completed_experiments)}")
    print(f"❌ Experimentos falharam: {len(result.failed_experiments)}")
    print(f"📊 Total de experimentos: {result.total_experiments}")
    
    if result.completed_experiments:
      print("\n✅ Experimentos concluídos com sucesso:")
      for experiment in result.completed_experiments:
        print(f"  - {experiment}")
    
    if result.failed_experiments:
      print("\n❌ Experimentos que falharam:")
      for experiment in result.failed_experiments:
        print(f"  - {experiment}")
    
    print(f"\n📁 Arquivos gerados em: {experiment_config.base_dir}")
    print(f"  - Resultados: {experiment_config.results_file_path}")
    print(f"  - Especificações: {experiment_config.machine_specs_file_path}")
    print(f"  - Dados preparados: {experiment_config.prepared_data_path}")
    print(f"  - Dados tokenizados: {experiment_config.tokenized_data_path}")
    print(f"  - Embeddings GloVe: {experiment_config.glove_embeddings_path}")

  except Exception as e:
      print(f"❌ Erro ao executar workflow: {e}")

if __name__ == "__main__":
  asyncio.run(main())
