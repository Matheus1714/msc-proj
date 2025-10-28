import psutil
import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class SystemMetrics:
    """Métricas de sistema coletadas durante a execução"""
    # Memória
    peak_memory_mb: float
    average_memory_mb: float
    memory_usage_samples: List[float]
    
    # CPU
    peak_cpu_percent: float
    average_cpu_percent: float
    cpu_usage_samples: List[float]
    
    # Throughput e Latência
    throughput_samples_per_second: float
    average_latency_ms: float
    latency_samples: List[float]
    
    # Tempo por etapa
    data_loading_time_ms: float
    model_training_time_ms: float
    model_evaluation_time_ms: float
    total_execution_time_ms: float
    
    # Escalabilidade
    memory_efficiency: float  # Uso de memória vs dados processados
    cpu_efficiency: float     # Uso de CPU vs tempo de execução
    
    # Eficiência energética (aproximada)
    energy_efficiency_score: float  # Score baseado em CPU e memória

class SystemMetricsCollector:
    """Coletor de métricas de sistema em tempo real"""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.memory_samples = []
        self.cpu_samples = []
        self.latency_samples = []
        self.throughput_samples = []
        self.is_collecting = False
        self.collection_thread = None
        self.start_time = None
        self.end_time = None
        self.step_times = {}
        
    def start_collection(self):
        """Inicia a coleta de métricas"""
        self.is_collecting = True
        self.start_time = time.time()
        self.collection_thread = threading.Thread(target=self._collect_metrics)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
    def stop_collection(self):
        """Para a coleta de métricas"""
        self.is_collecting = False
        self.end_time = time.time()
        if self.collection_thread:
            self.collection_thread.join(timeout=1.0)
            
    def _collect_metrics(self):
        """Loop de coleta de métricas"""
        while self.is_collecting:
            try:
                # Coletar métricas de memória
                memory_info = psutil.virtual_memory()
                self.memory_samples.append(memory_info.used / (1024 * 1024))  # MB
                
                # Coletar métricas de CPU
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_samples.append(cpu_percent)
                
                time.sleep(self.sample_interval)
            except Exception:
                # Ignorar erros de coleta para não interromper o experimento
                pass
                
    def record_step_time(self, step_name: str, start_time: float, end_time: float):
        """Registra o tempo de uma etapa específica"""
        duration_ms = (end_time - start_time) * 1000
        self.step_times[step_name] = duration_ms
        
    def record_latency(self, latency_ms: float):
        """Registra uma amostra de latência"""
        self.latency_samples.append(latency_ms)
        
    def record_throughput(self, samples_per_second: float):
        """Registra uma amostra de throughput"""
        self.throughput_samples.append(samples_per_second)
        
    def get_metrics(self, data_size: int = 1) -> SystemMetrics:
        """Retorna as métricas coletadas"""
        if not self.memory_samples or not self.cpu_samples:
            return self._get_empty_metrics()
            
        # Calcular métricas de memória
        peak_memory = max(self.memory_samples)
        avg_memory = np.mean(self.memory_samples)
        
        # Calcular métricas de CPU
        peak_cpu = max(self.cpu_samples)
        avg_cpu = np.mean(self.cpu_samples)
        
        # Calcular throughput médio
        total_time = (self.end_time - self.start_time) if self.end_time and self.start_time else 1.0
        avg_throughput = data_size / total_time if total_time > 0 else 0
        
        # Calcular latência média
        avg_latency = np.mean(self.latency_samples) if self.latency_samples else 0
        
        # Calcular tempos por etapa
        data_loading_time = self.step_times.get('data_loading', 0)
        model_training_time = self.step_times.get('model_training', 0)
        model_evaluation_time = self.step_times.get('model_evaluation', 0)
        total_time_ms = total_time * 1000
        
        # Calcular eficiências
        memory_efficiency = (data_size / peak_memory) if peak_memory > 0 else 0
        cpu_efficiency = (total_time / avg_cpu) if avg_cpu > 0 else 0
        
        # Calcular score de eficiência energética (0-100)
        # Baseado na relação entre performance e uso de recursos
        energy_score = 100 - (avg_cpu * 0.4 + (peak_memory / 1000) * 0.6)
        energy_score = max(0, min(100, energy_score))
        
        return SystemMetrics(
            peak_memory_mb=peak_memory,
            average_memory_mb=avg_memory,
            memory_usage_samples=self.memory_samples.copy(),
            
            peak_cpu_percent=peak_cpu,
            average_cpu_percent=avg_cpu,
            cpu_usage_samples=self.cpu_samples.copy(),
            
            throughput_samples_per_second=avg_throughput,
            average_latency_ms=avg_latency,
            latency_samples=self.latency_samples.copy(),
            
            data_loading_time_ms=data_loading_time,
            model_training_time_ms=model_training_time,
            model_evaluation_time_ms=model_evaluation_time,
            total_execution_time_ms=total_time_ms,
            
            memory_efficiency=memory_efficiency,
            cpu_efficiency=cpu_efficiency,
            energy_efficiency_score=energy_score
        )
        
    def _get_empty_metrics(self) -> SystemMetrics:
        """Retorna métricas vazias quando não há dados"""
        return SystemMetrics(
            peak_memory_mb=0,
            average_memory_mb=0,
            memory_usage_samples=[],
            
            peak_cpu_percent=0,
            average_cpu_percent=0,
            cpu_usage_samples=[],
            
            throughput_samples_per_second=0,
            average_latency_ms=0,
            latency_samples=[],
            
            data_loading_time_ms=0,
            model_training_time_ms=0,
            model_evaluation_time_ms=0,
            total_execution_time_ms=0,
            
            memory_efficiency=0,
            cpu_efficiency=0,
            energy_efficiency_score=0
        )
