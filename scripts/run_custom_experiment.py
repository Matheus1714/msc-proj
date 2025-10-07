import os
import asyncio
import argparse
from dotenv import load_dotenv
from uuid import uuid4

from utils import setup_project_path
setup_project_path()

from temporalio.client import Client
from src.workflows.experiment_workflow import ExperimentWorkflow
from src.default_types import ExperimentWorkflowIn, ModelConfig
from constants import WorflowTaskQueue

def create_svm_config(name: str, **kwargs):
    """Cria configuração padrão para SVM"""
    default_params = {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale"
    }
    default_params.update(kwargs)
    
    return ModelConfig(
        name=name,
        type="svm",
        hyperparameters=default_params
    )

def create_random_forest_config(name: str, **kwargs):
    """Cria configuração padrão para Random Forest"""
    default_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    }
    default_params.update(kwargs)
    
    return ModelConfig(
        name=name,
        type="random_forest",
        hyperparameters=default_params
    )

async def main():    
    parser = argparse.ArgumentParser(description="Executar experimento de ML")
    parser.add_argument("--model", choices=["svm", "random_forest"], required=True,
                       help="Tipo do modelo a ser usado")
    parser.add_argument("--dataset", default="academic_works",
                       help="ID do dataset (padrão: academic_works)")
    parser.add_argument("--tokenizer", default="tfidf",
                       help="Estratégia de tokenização (padrão: tfidf)")
    parser.add_argument("--name", help="Nome do experimento (opcional)")
    parser.add_argument("--model-path", help="Caminho do modelo pré-treinado (opcional)")
    
    args = parser.parse_args()
    
    load_dotenv()
    
    try:
        client = await Client.connect(os.environ.get("TEMPORAL_CONNECT"))
        
        # Criar configuração do modelo
        experiment_name = args.name or f"{args.model}_experiment_{uuid4().hex[:8]}"
        
        if args.model == "svm":
            model_config = create_svm_config(experiment_name)
        elif args.model == "random_forest":
            model_config = create_random_forest_config(experiment_name)
        
        experiment_input = ExperimentWorkflowIn(
            dataset_id=args.dataset,
            model_config=model_config,
            tokenizer_strategy=args.tokenizer,
            model_path=args.model_path
        )
        
        workflow_id = f"experiment-{args.model}-{uuid4().hex[:8]}"
        
        await client.start_workflow(
            ExperimentWorkflow.run,
            arg=experiment_input,
            id=workflow_id,
            task_queue=WorflowTaskQueue.ML_TASK_QUEUE.value,
        )
        
        print("✅ Experimento iniciado com sucesso!")
        print(f"🆔 Workflow ID: {workflow_id}")
        print(f"📊 Dataset: {experiment_input.dataset_id}")
        print(f"🤖 Modelo: {model_config['name']} ({model_config['type']})")
        print(f"🔤 Tokenização: {experiment_input.tokenizer_strategy}")
        if args.model_path:
            print(f"📁 Modelo pré-treinado: {args.model_path}")
        
    except Exception as e:
        print(f"❌ Erro ao executar experimento: {e}")

if __name__ == "__main__":
    asyncio.run(main())
