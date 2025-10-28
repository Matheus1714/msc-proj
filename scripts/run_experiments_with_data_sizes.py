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
    
    class_counts = df['included'].value_counts()
    print(f"📈 Distribuição de classes: {dict(class_counts)}")
    
    percentages = [10, 25, 50, 75, 100]  # 10%, 25%, 50%, 75%, 100%
    
    data_subsets = []
    
    for pct in percentages:
        target_rows = int((pct / 100) * total_rows)
        actual_rows = min(target_rows, max_rows)
        
        if actual_rows < 20:  # Increased minimum from 10 to 20
            print(f"⚠️  Pulando {pct}% - muito poucas linhas ({actual_rows})")
            continue
        
        # Sample with stratification to maintain class balance
        try:
            # Calculate how many samples to take from each class proportionally
            class_0_count = int(actual_rows * class_counts[0] / total_rows)
            class_1_count = int(actual_rows * class_counts[1] / total_rows)
            
            # Ensure we have at least 2 samples per class
            class_0_count = max(class_0_count, 2)
            class_1_count = max(class_1_count, 2)
            
            # Sample from each class
            class_0_samples = df[df['included'] == 0].sample(n=min(class_0_count, class_counts[0]), random_state=42)
            class_1_samples = df[df['included'] == 1].sample(n=min(class_1_count, class_counts[1]), random_state=42)
            
            subset_df = pd.concat([class_0_samples, class_1_samples], ignore_index=True)
            
            # If we still don't have enough samples, add more from the majority class
            if len(subset_df) < actual_rows * 0.8:
                remaining_needed = actual_rows - len(subset_df)
                additional_samples = df[df['included'] == 0].sample(n=min(remaining_needed, class_counts[0] - len(class_0_samples)), random_state=42)
                subset_df = pd.concat([subset_df, additional_samples], ignore_index=True)
            
            # Check if we have at least 2 samples per class
            subset_class_counts = subset_df['included'].value_counts()
            min_class_count = subset_class_counts.min()
            
            if min_class_count < 2:
                print(f"⚠️  Pulando {pct}% - classe minoritária tem apenas {min_class_count} amostra(s)")
                continue
            
            filename = f"academic_works_{pct}pct_{len(subset_df)}rows.csv"
            filepath = os.path.join(output_dir, filename)
            
            subset_df.to_csv(filepath, index=False)
            
            data_subsets.append((pct, filepath, len(subset_df)))
            print(f"✅ {pct}%: {len(subset_df)} linhas -> {filename}")
            print(f"   📊 Classes: {dict(subset_class_counts)}")
            
        except Exception as e:
            print(f"⚠️  Erro ao criar {pct}%: {e}")
            continue
    
    return data_subsets

async def start_experiments_for_data_size(client, data_path, data_size_info, hyperparameters):
    pct, filepath, num_rows = data_size_info
    
    print(f"🚀 Iniciando experimentos para {pct}% dos dados ({num_rows} linhas)")
    print(f"📁 Arquivo: {filepath}")
    
    experiment_config = ExperimentConfig.create()
    experiment_config.create_directories()
    
    print(f"📂 Diretório do experimento: {experiment_config.base_dir}")
    
    try:
        # Executar workflow e aguardar conclusão
        result = await client.execute_workflow(
            ExperimentsWorkflow.run,
            arg=ExperimentsWorkflowIn(
                input_data_path=filepath,
                hyperparameters=hyperparameters,
                experiment_config=experiment_config,
            ),
            id=f"experiments-{pct}pct-{uuid4()}",
            task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
        )
        
        print(f"✅ Workflow concluído para {pct}% - ID: {result}")
        
        return {
            'data_size_info': data_size_info,
            'workflow_result': result,
            'experiment_config': experiment_config,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Erro ao executar workflow para {pct}%: {e}")
        return {
            'data_size_info': data_size_info,
            'workflow_result': None,
            'success': False,
            'error': str(e)
        }

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
        
        print("📊 Criando subconjuntos de dados...")
        data_subsets = create_data_subsets(input_file, data_subsets_dir, max_rows=200)
        
        print(f"\n🎯 Iniciando experimentos para {len(data_subsets)} tamanhos diferentes de dados")
        
        all_workflows = []
        
        for data_size_info in data_subsets:
            result = await start_experiments_for_data_size(
                client, 
                data_size_info[1],  # filepath
                data_size_info, 
                hyperparameters
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
                workflow_result = result['workflow_result']
                experiment_config = result['experiment_config']
                print(f"\n✅ {pct}% ({num_rows} linhas):")
                print(f"   - Status: CONCLUÍDO")
                print(f"   - Arquivo: {filepath}")
                print(f"   - Diretório: {experiment_config.base_dir}")
                print(f"   - Experimentos concluídos: {len(workflow_result.completed_experiments)}")
                print(f"   - Experimentos falharam: {len(workflow_result.failed_experiments)}")
            else:
                failed_starts += 1
                print(f"\n❌ {pct}% ({num_rows} linhas): FALHOU AO EXECUTAR")
                print(f"   - Erro: {result['error']}")
        
        print(f"\n🎉 Total de workflows executados: {len(all_workflows)}")
        print(f"✅ Executados com sucesso: {successful_starts}")
        print(f"❌ Falhas ao executar: {failed_starts}")
        
        # Usar o diretório do primeiro experimento bem-sucedido ou criar um diretório temporário
        info_dir = None
        for result in all_workflows:
            if result['success'] and 'experiment_config' in result:
                info_dir = result['experiment_config'].base_dir
                break
        
        if not info_dir:
            info_dir = "data/experiments"
            os.makedirs(info_dir, exist_ok=True)
        
        workflows_info_file = os.path.join(info_dir, "started_workflows_info.txt")
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
                    workflow_result = result['workflow_result']
                    experiment_config = result['experiment_config']
                    f.write(f"Status: CONCLUÍDO\n")
                    f.write(f"Diretório: {experiment_config.base_dir}\n")
                    f.write(f"Experimentos concluídos: {len(workflow_result.completed_experiments)}\n")
                    f.write(f"Experimentos falharam: {len(workflow_result.failed_experiments)}\n")
                    f.write(f"Task Queue: {WorflowTaskQueue.ML_TASK_QUEUE.value}\n")
                else:
                    f.write(f"Status: FALHOU AO EXECUTAR\n")
                    f.write(f"Erro: {result['error']}\n")
                
                f.write("-" * 30 + "\n")
        
        print(f"\n📄 Informações dos workflows salvos em: {workflows_info_file}")
        print("\n🚀 Todos os workflows foram executados sequencialmente!")
        print("💡 Cada pasta de experimento contém os resultados intermediários e finais.")

    except Exception as e:
        print(f"❌ Erro ao executar workflow: {e}")

if __name__ == "__main__":
    asyncio.run(main())
