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
from src.utils.calculate_class_weights import calculate_class_weights_from_csv

# Lista de bases de dados para executar experimentos
DATA_FILES = [
    "data/appenzeller_herzog_2020.csv",
    "data/bannach_brown_2019.csv",
    "data/atypicalantipsychotics.csv",
    "data/aceinhibitors.csv",
    "data/oralhypoglycemics.csv",
]

async def run_experiments_for_dataset(client, input_file, hyperparameters):
    """Executa experimentos para um dataset específico"""
    print(f"\n{'='*80}")
    print(f"📊 Executando experimentos para: {input_file}")
    print(f"{'='*80}\n")
    
    try:
        # Calcular class weights automaticamente para este dataset
        class_weight_0, class_weight_1 = calculate_class_weights_from_csv(input_file)
        print(f"⚖️  Class weights calculados: class_weight_0={class_weight_0}, class_weight_1={class_weight_1}")
        
        # Criar configuração de experimento para este dataset
        experiment_config = ExperimentConfig.create()
        experiment_config.create_directories()
        
        print(f"📁 Diretório do experimento: {experiment_config.base_dir}")
        
        # Atualizar class weights nos hyperparameters
        hyperparameters_copy = hyperparameters.copy()
        hyperparameters_copy["class_weight_0"] = class_weight_0
        hyperparameters_copy["class_weight_1"] = class_weight_1
        
        # Obter nome do arquivo sem extensão para usar no ID do workflow
        dataset_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Executar workflow via Temporal
        await client.execute_workflow(
            ExperimentsWorkflow.run,
            arg=ExperimentsWorkflowIn(
                input_data_path=input_file,
                hyperparameters=hyperparameters_copy,
                experiment_config=experiment_config,
            ),
            id=f"experiments-{dataset_name}-{uuid4()}",
            task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
        )
        
        print(f"\n✅ Experimentos concluídos para {input_file}!")
        print(f"  📁 Arquivos gerados em: {experiment_config.base_dir}")
        print(f"    - Resultados: {experiment_config.results_file_path}")
        print(f"    - Especificações: {experiment_config.machine_specs_file_path}")
        print(f"    - Dados preparados: {experiment_config.prepared_data_path}")
        print(f"    - Dados tokenizados: {experiment_config.tokenized_data_path}")
        print(f"    - Embeddings GloVe: {experiment_config.glove_embeddings_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao executar experimentos para {input_file}: {e}")
        import traceback
        traceback.print_exc()
        return False

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
            "class_weight_0": 1,  # Será calculado automaticamente para cada dataset
            "class_weight_1": 44,  # Será calculado automaticamente para cada dataset
            "ngram_range": (1, 3),
            "max_iter": 1000,
        }
        
        print(f"🚀 Iniciando execução de experimentos para {len(DATA_FILES)} bases de dados")
        print(f"📋 Bases de dados: {', '.join([os.path.basename(f) for f in DATA_FILES])}\n")
        
        results = []
        for input_file in DATA_FILES:
            # Verificar se o arquivo existe
            if not os.path.exists(input_file):
                print(f"⚠️  Arquivo não encontrado: {input_file}. Pulando...")
                continue
            
            success = await run_experiments_for_dataset(client, input_file, hyperparameters)
            results.append((input_file, success))
        
        # Resumo final
        print(f"\n{'='*80}")
        print(f"🎉 Execução concluída para todas as bases de dados!")
        print(f"{'='*80}\n")
        
        successful = sum(1 for _, success in results if success)
        failed = len(results) - successful
        
        print(f"✅ Bases de dados processadas com sucesso: {successful}/{len(DATA_FILES)}")
        print(f"❌ Bases de dados com falhas: {failed}/{len(DATA_FILES)}")
        
    except Exception as e:
        print(f"❌ Erro ao executar script: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

