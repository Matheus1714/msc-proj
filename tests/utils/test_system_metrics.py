import pytest
import time
from src.utils.system_metrics import SystemMetricsCollector, SystemMetrics


class TestSystemMetricsCollector:
    """Testes para o coletor de métricas de sistema"""
    
    def test_collector_initialization(self):
        """Testa inicialização do coletor"""
        collector = SystemMetricsCollector(sample_interval=0.1)
        
        assert collector.sample_interval == 0.1
        assert not collector.is_collecting
        assert len(collector.memory_samples) == 0
        assert len(collector.cpu_samples) == 0
    
    def test_start_stop_collection(self):
        """Testa início e parada da coleta"""
        collector = SystemMetricsCollector(sample_interval=0.05)
        
        assert not collector.is_collecting
        collector.start_collection()
        assert collector.is_collecting
        assert collector.start_time is not None
        
        time.sleep(0.1)  # Aguardar um pouco para coletar algumas amostras
        
        collector.stop_collection()
        assert not collector.is_collecting
        assert collector.end_time is not None
    
    def test_collects_metrics(self):
        """Testa que métricas são coletadas durante a execução"""
        collector = SystemMetricsCollector(sample_interval=0.05)
        
        collector.start_collection()
        time.sleep(0.2)  # Aguardar para coletar várias amostras
        collector.stop_collection()
        
        # Deve ter coletado algumas amostras
        assert len(collector.memory_samples) > 0
        assert len(collector.cpu_samples) > 0
    
    def test_record_step_time(self):
        """Testa registro de tempo de etapas"""
        collector = SystemMetricsCollector()
        
        start = time.time()
        time.sleep(0.01)
        end = time.time()
        
        collector.record_step_time('test_step', start, end)
        
        assert 'test_step' in collector.step_times
        assert collector.step_times['test_step'] > 0
    
    def test_record_latency(self):
        """Testa registro de latência"""
        collector = SystemMetricsCollector()
        
        collector.record_latency(10.5)
        collector.record_latency(20.3)
        
        assert len(collector.latency_samples) == 2
        assert collector.latency_samples[0] == 10.5
        assert collector.latency_samples[1] == 20.3
    
    def test_record_throughput(self):
        """Testa registro de throughput"""
        collector = SystemMetricsCollector()
        
        collector.record_throughput(100.0)
        collector.record_throughput(150.0)
        
        assert len(collector.throughput_samples) == 2
        assert collector.throughput_samples[0] == 100.0
        assert collector.throughput_samples[1] == 150.0
    
    def test_get_metrics_with_data(self):
        """Testa obtenção de métricas com dados coletados"""
        collector = SystemMetricsCollector(sample_interval=0.05)
        
        collector.start_collection()
        time.sleep(0.1)
        collector.stop_collection()
        
        collector.record_step_time('data_loading', time.time() - 0.05, time.time() - 0.03)
        collector.record_step_time('model_training', time.time() - 0.03, time.time() - 0.01)
        
        metrics = collector.get_metrics(data_size=100)
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.peak_memory_mb >= 0
        assert metrics.average_memory_mb >= 0
        assert metrics.peak_cpu_percent >= 0
        assert metrics.average_cpu_percent >= 0
        assert metrics.total_execution_time_ms > 0
    
    def test_get_metrics_empty(self):
        """Testa obtenção de métricas sem dados coletados"""
        collector = SystemMetricsCollector()
        
        metrics = collector.get_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.peak_memory_mb == 0
        assert metrics.average_memory_mb == 0
        assert metrics.peak_cpu_percent == 0
        assert metrics.average_cpu_percent == 0
        assert metrics.total_execution_time_ms == 0
    
    def test_metrics_structure(self):
        """Testa que todas as métricas esperadas estão presentes"""
        collector = SystemMetricsCollector(sample_interval=0.05)
        
        collector.start_collection()
        time.sleep(0.1)
        collector.stop_collection()
        
        metrics = collector.get_metrics(data_size=50)
        
        # Verificar todas as propriedades esperadas
        assert hasattr(metrics, 'peak_memory_mb')
        assert hasattr(metrics, 'average_memory_mb')
        assert hasattr(metrics, 'memory_usage_samples')
        assert hasattr(metrics, 'peak_cpu_percent')
        assert hasattr(metrics, 'average_cpu_percent')
        assert hasattr(metrics, 'cpu_usage_samples')
        assert hasattr(metrics, 'throughput_samples_per_second')
        assert hasattr(metrics, 'average_latency_ms')
        assert hasattr(metrics, 'latency_samples')
        assert hasattr(metrics, 'data_loading_time_ms')
        assert hasattr(metrics, 'model_training_time_ms')
        assert hasattr(metrics, 'model_evaluation_time_ms')
        assert hasattr(metrics, 'total_execution_time_ms')
        assert hasattr(metrics, 'memory_efficiency')
        assert hasattr(metrics, 'cpu_efficiency')
        assert hasattr(metrics, 'energy_efficiency_score')
    
    def test_throughput_calculation(self):
        """Testa cálculo de throughput"""
        collector = SystemMetricsCollector(sample_interval=0.05)
        
        collector.start_collection()
        time.sleep(0.1)
        collector.stop_collection()
        
        data_size = 100
        metrics = collector.get_metrics(data_size=data_size)
        
        # Throughput deve ser calculado baseado no tamanho dos dados e tempo
        assert metrics.throughput_samples_per_second >= 0
    
    def test_energy_efficiency_score_range(self):
        """Testa que o score de eficiência energética está no range correto"""
        collector = SystemMetricsCollector(sample_interval=0.05)
        
        collector.start_collection()
        time.sleep(0.1)
        collector.stop_collection()
        
        metrics = collector.get_metrics(data_size=100)
        
        # Score deve estar entre 0 e 100
        assert 0 <= metrics.energy_efficiency_score <= 100
    
    def test_multiple_step_times(self):
        """Testa registro de múltiplos tempos de etapas"""
        collector = SystemMetricsCollector()
        
        base_time = time.time()
        collector.record_step_time('step1', base_time, base_time + 0.01)
        collector.record_step_time('step2', base_time + 0.01, base_time + 0.03)
        collector.record_step_time('step3', base_time + 0.03, base_time + 0.05)
        
        metrics = collector.get_metrics()
        
        assert 'step1' in collector.step_times
        assert 'step2' in collector.step_times
        assert 'step3' in collector.step_times

