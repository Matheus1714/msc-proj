import os
import asyncio
import subprocess
import sys
from dotenv import load_dotenv

from utils import setup_project_path
setup_project_path()

from constants import GOOGLE_DRIVE_FILES_ID
from temporalio.client import Client
from src.workflows.data_preprocessing_workflow import DataPreprocessingWorkflow


async def run_worker():
    print("🔄 Iniciando worker...")
    
    worker_process = subprocess.Popen([
        sys.executable, "scripts/start_ml_worker.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    await asyncio.sleep(3)
    
    return worker_process


async def run_preprocessing_workflow():
    """Executa o workflow de pre-processamento."""
    load_dotenv()
    
    try:
        # Conectar ao Temporal
        client = await Client.connect(os.environ.get("TEMPORAL_CONNECT"))
        
        print(f"📊 Iniciando pre-processamento de {len(GOOGLE_DRIVE_FILES_ID)} arquivos...")
        
        # Iniciar o workflow de pre-processamento
        workflow_handle = await client.start_workflow(
            DataPreprocessingWorkflow.run,
            args=[GOOGLE_DRIVE_FILES_ID, "data/academic_works.csv"],
            id="data-preprocessing-workflow",
            task_queue="ml-task-queue",
        )
        
        print(f"🚀 Workflow iniciado com ID: {workflow_handle.id}")
        print("⏳ Aguardando processamento paralelo...")
        
        # Aguardar conclusão do workflow
        result = await workflow_handle.result()
        
        print(f"✅ Pre-processamento concluído!")
        print(f"📈 Total de trabalhos processados: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ Erro ao executar workflow: {e}")
        raise


async def main():
    """Função principal que coordena worker e workflow."""
    worker_process = None
    
    try:
        # Iniciar worker
        worker_process = await run_worker()
        
        # Aguardar um pouco mais para garantir que o worker está pronto
        await asyncio.sleep(5)
        
        # Executar workflow
        result = await run_preprocessing_workflow()
        
        print(f"\n🎉 Processamento completo! {result} trabalhos salvos em data/academic_works.csv")
        
    except KeyboardInterrupt:
        print("\n⏹️ Interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro: {e}")
    finally:
        # Finalizar worker
        if worker_process:
            print("🔄 Finalizando worker...")
            worker_process.terminate()
            worker_process.wait()
            print("✅ Worker finalizado")


if __name__ == "__main__":
    asyncio.run(main())
