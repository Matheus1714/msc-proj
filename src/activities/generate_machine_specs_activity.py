import os
import platform
import psutil
import subprocess
from datetime import datetime
from temporalio import activity
from dataclasses import dataclass
from typing import List


@dataclass
class GenerateMachineSpecsIn:
    input_data_path: str
    machine_specs_file_path: str
    detailed_results: List[dict] = None

@dataclass
class GenerateMachineSpecsOut:
    machine_specs_file_path: str

@activity.defn
async def generate_machine_specs_activity(data: GenerateMachineSpecsIn) -> GenerateMachineSpecsOut:
    try:
        os.makedirs(os.path.dirname(data.machine_specs_file_path), exist_ok=True)
        
        with open(data.machine_specs_file_path, 'w', encoding='utf-8') as f:
            f.write("=== ESPECIFICAÇÕES DA MÁQUINA ===\n\n")
            
            f.write("SISTEMA OPERACIONAL:\n")
            f.write(f"  Sistema: {platform.system()}\n")
            f.write(f"  Versão: {platform.release()}\n")
            f.write(f"  Arquitetura: {platform.architecture()[0]}\n")
            f.write(f"  Processador: {platform.processor()}\n")
            f.write(f"  Máquina: {platform.machine()}\n")
            f.write(f"  Nó: {platform.node()}\n")
            f.write(f"  Plataforma: {platform.platform()}\n\n")
            
            # Informações de CPU
            f.write("PROCESSADOR:\n")
            f.write(f"  Núcleos físicos: {psutil.cpu_count(logical=False)}\n")
            f.write(f"  Núcleos lógicos: {psutil.cpu_count(logical=True)}\n")
            f.write(f"  Frequência máxima: {psutil.cpu_freq().max if psutil.cpu_freq() else 'N/A'} MHz\n")
            f.write(f"  Frequência atual: {psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A'} MHz\n")
            
            # Informações de memória
            memory = psutil.virtual_memory()
            f.write(f"\nMEMÓRIA:\n")
            f.write(f"  Total: {memory.total / (1024**3):.2f} GB\n")
            f.write(f"  Disponível: {memory.available / (1024**3):.2f} GB\n")
            f.write(f"  Usada: {memory.used / (1024**3):.2f} GB\n")
            f.write(f"  Percentual usado: {memory.percent}%\n")
            
            # Informações de disco
            disk = psutil.disk_usage('/')
            f.write(f"\nDISCO:\n")
            f.write(f"  Total: {disk.total / (1024**3):.2f} GB\n")
            f.write(f"  Usado: {disk.used / (1024**3):.2f} GB\n")
            f.write(f"  Livre: {disk.free / (1024**3):.2f} GB\n")
            f.write(f"  Percentual usado: {(disk.used / disk.total) * 100:.2f}%\n")
            
            # Informações de GPU (se disponível)
            f.write(f"\nGPU:\n")
            try:
                # Tentar detectar NVIDIA GPU
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    gpu_info = result.stdout.strip().split('\n')
                    for i, gpu in enumerate(gpu_info):
                        parts = gpu.split(', ')
                        if len(parts) >= 3:
                            f.write(f"  GPU {i+1}: {parts[0]}\n")
                            f.write(f"    Memória: {parts[1]} MB\n")
                            f.write(f"    Driver: {parts[2]}\n")
                else:
                    f.write("  Nenhuma GPU NVIDIA detectada\n")
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                f.write("  Informações de GPU não disponíveis\n")
            
            # Informações de Python
            f.write(f"\nPYTHON:\n")
            f.write(f"  Versão: {platform.python_version()}\n")
            f.write(f"  Implementação: {platform.python_implementation()}\n")
            f.write(f"  Compilador: {platform.python_compiler()}\n")
            
            # Informações de bibliotecas ML
            f.write(f"\nBIBLIOTECAS DE MACHINE LEARNING:\n")
            
            try:
                import tensorflow as tf
                f.write(f"  TensorFlow: {tf.__version__}\n")
            except ImportError:
                f.write("  TensorFlow: Não instalado\n")
            
            try:
                import torch
                f.write(f"  PyTorch: {torch.__version__}\n")
            except ImportError:
                f.write("  PyTorch: Não instalado\n")
            
            try:
                import sklearn
                f.write(f"  Scikit-learn: {sklearn.__version__}\n")
            except ImportError:
                f.write("  Scikit-learn: Não instalado\n")
            
            try:
                import pandas as pd
                f.write(f"  Pandas: {pd.__version__}\n")
            except ImportError:
                f.write("  Pandas: Não instalado\n")
            
            try:
                import numpy as np
                f.write(f"  NumPy: {np.__version__}\n")
            except ImportError:
                f.write("  NumPy: Não instalado\n")
            
            # Data e hora da execução
            f.write(f"\nEXECUÇÃO:\n")
            f.write(f"  Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Dados de entrada: {data.input_data_path}\n")
            
            # Métricas de performance dos experimentos
            if data.detailed_results:
                f.write(f"\nMÉTRICAS DE PERFORMANCE DOS EXPERIMENTOS:\n")
                f.write(f"  {'='*60}\n")
                
                for result in data.detailed_results:
                    f.write(f"\n  {result.get('experiment_name', 'Unknown')}:\n")
                    f.write(f"    Status: {result.get('status', 'Unknown')}\n")
                    f.write(f"    Tempo de execução: {result.get('execution_time_minutes', 0):.2f} minutos\n")
                    
                    system_metrics = result.get('system_metrics')
                    if system_metrics:
                        f.write(f"    Memória:\n")
                        f.write(f"      Pico: {system_metrics.get('peak_memory_mb', 0):.2f} MB\n")
                        f.write(f"      Média: {system_metrics.get('average_memory_mb', 0):.2f} MB\n")
                        
                        f.write(f"    CPU:\n")
                        f.write(f"      Pico: {system_metrics.get('peak_cpu_percent', 0):.2f}%\n")
                        f.write(f"      Média: {system_metrics.get('average_cpu_percent', 0):.2f}%\n")
                        
                        f.write(f"    Throughput: {system_metrics.get('throughput_samples_per_second', 0):.2f} amostras/seg\n")
                        f.write(f"    Latência média: {system_metrics.get('average_latency_ms', 0):.2f} ms\n")
                        
                        f.write(f"    Tempos por etapa:\n")
                        f.write(f"      Carregamento de dados: {system_metrics.get('data_loading_time_ms', 0):.2f} ms\n")
                        f.write(f"      Treinamento do modelo: {system_metrics.get('model_training_time_ms', 0):.2f} ms\n")
                        f.write(f"      Avaliação do modelo: {system_metrics.get('model_evaluation_time_ms', 0):.2f} ms\n")
                        
                        f.write(f"    Eficiências:\n")
                        f.write(f"      Memória: {system_metrics.get('memory_efficiency', 0):.4f}\n")
                        f.write(f"      CPU: {system_metrics.get('cpu_efficiency', 0):.4f}\n")
                        f.write(f"      Energia: {system_metrics.get('energy_efficiency_score', 0):.2f}/100\n")
                    else:
                        f.write(f"    Métricas do sistema: Não disponíveis\n")
                    
                    error_message = result.get('error_message')
                    if error_message:
                        f.write(f"    Erro: {error_message}\n")
        
        activity.logger.info(f"Especificações da máquina salvas em: {data.machine_specs_file_path}")
        return GenerateMachineSpecsOut(machine_specs_file_path=data.machine_specs_file_path)
        
    except Exception as e:
        activity.logger.error(f"Erro ao gerar especificações da máquina: {e}")
        raise e
