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

def create_data_subsets(input_file_path: str, output_dir: str, max_rows: int = 200):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(input_file_path)
    total_rows = len(df)
    
    print(f"📊 Arquivo original: {total_rows} linhas")
    print(f"🔒 Limitando a {max_rows} linhas para teste")
    
    percentages = [10, 25, 50, 75, 100]  # 10%, 25%, 50%, 75%, 100%
    
    data_subsets = []
    
    for pct in percentages:
        target_rows = int((pct / 100) * total_rows)
        
        actual_rows = min(target_rows, max_rows)
        
        if actual_rows < 10:
            print(f"⚠️  Pulando {pct}% - muito poucas linhas ({actual_rows})")
            continue
        
        subset_df = df.sample(n=actual_rows, random_state=42)
        
        filename = f"academic_works_{pct}pct_{actual_rows}rows.csv"
        filepath = os.path.join(output_dir, filename)
        
        subset_df.to_csv(filepath, index=False)
        
        data_subsets.append((pct, filepath, actual_rows))
        print(f"✅ {pct}%: {actual_rows} linhas -> {filename}")
    
    return data_subsets

async def start_experiments_for_data_size(client, data_path, data_size_info, hyperparameters, base_experiment_config):
    pct, filepath, num_rows = data_size_info
    
    print(f"🚀 Iniciando experimentos para {pct}% dos dados ({num_rows} linhas)")
    print(f"📁 Arquivo: {filepath}")
    
    experiment_config = ExperimentConfig.create()
    experiment_config.base_dir = f"{base_experiment_config.base_dir}_size_{pct}pct"
    experiment_config.create_directories()
    
    try:
        # Iniciar workflow sem aguardar conclusão
        workflow_handle = await client.start_workflow(
            ExperimentsWorkflow.run,
            arg=ExperimentsWorkflowIn(
                input_data_path=filepath,
                hyperparameters=hyperparameters,
                experiment_config=experiment_config,
            ),
            id=f"experiments-{pct}pct-{uuid4()}",
            task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
        )
        
        print(f"✅ Workflow iniciado para {pct}% - ID: {workflow_handle.id}")
        
        return {
            'data_size_info': data_size_info,
            'workflow_handle': workflow_handle,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Erro ao iniciar workflow para {pct}%: {e}")
        return {
            'data_size_info': data_size_info,
            'workflow_handle': None,
            'success': False,
            'error': str(e)
        }

async def main():
    load_dotenv()
    
    try:
        client = await Client.connect(os.environ.get("TEMPORAL_CONNECT"))
        
        base_experiment_config = ExperimentConfig.create()
        base_experiment_config.create_directories()
        
        print(f"📁 Diretório base dos experimentos: {base_experiment_config.base_dir}")
        
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
        
        print("📊 Criando subconjuntos de dados...")
        data_subsets = create_data_subsets(input_file, data_subsets_dir, max_rows=200)
        
        print(f"\n🎯 Iniciando experimentos para {len(data_subsets)} tamanhos diferentes de dados")
        
        all_workflows = []
        
        for data_size_info in data_subsets:
            result = await start_experiments_for_data_size(
                client, 
                data_size_info[1],  # filepath
                data_size_info, 
                hyperparameters, 
                base_experiment_config
            )
            all_workflows.append(result)
        
        print("\n" + "="*60)
        print("📊 RESUMO DOS WORKFLOWS INICIADOS")
        print("="*60)
        
        successful_starts = 0
        failed_starts = 0
        
        for result in all_workflows:
            pct, filepath, num_rows = result['data_size_info']
            
            if result['success']:
                successful_starts += 1
                workflow_handle = result['workflow_handle']
                print(f"\n✅ {pct}% ({num_rows} linhas):")
                print(f"   - Workflow ID: {workflow_handle.id}")
                print(f"   - Status: INICIADO")
                print(f"   - Arquivo: {filepath}")
            else:
                failed_starts += 1
                print(f"\n❌ {pct}% ({num_rows} linhas): FALHOU AO INICIAR")
                print(f"   - Erro: {result['error']}")
        
        print(f"\n🎉 Total de workflows iniciados: {len(all_workflows)}")
        print(f"✅ Iniciados com sucesso: {successful_starts}")
        print(f"❌ Falhas ao iniciar: {failed_starts}")
        
        workflows_info_file = os.path.join(base_experiment_config.base_dir, "started_workflows_info.txt")
        with open(workflows_info_file, 'w') as f:
            f.write("INFORMAÇÕES DOS WORKFLOWS INICIADOS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Timestamp: {pd.Timestamp.now()}\n")
            f.write(f"Total de workflows: {len(all_workflows)}\n")
            f.write(f"Sucessos: {successful_starts}\n")
            f.write(f"Falhas: {failed_starts}\n\n")
            
            for result in all_workflows:
                pct, filepath, num_rows = result['data_size_info']
                f.write(f"Tamanho: {pct}% ({num_rows} linhas)\n")
                f.write(f"Arquivo: {filepath}\n")
                
                if result['success']:
                    workflow_handle = result['workflow_handle']
                    f.write(f"Status: INICIADO\n")
                    f.write(f"Workflow ID: {workflow_handle.id}\n")
                    f.write(f"Task Queue: {WorflowTaskQueue.ML_TASK_QUEUE.value}\n")
                else:
                    f.write(f"Status: FALHOU AO INICIAR\n")
                    f.write(f"Erro: {result['error']}\n")
                
                f.write("-" * 30 + "\n")
        
        print(f"\n📄 Informações dos workflows salvos em: {workflows_info_file}")
        print("\n🚀 Todos os workflows foram iniciados e estão executando em paralelo!")
        print("💡 Use o Temporal Web UI ou CLI para monitorar o progresso dos workflows.")

    except Exception as e:
        print(f"❌ Erro ao executar workflow: {e}")

if __name__ == "__main__":
    asyncio.run(main())
