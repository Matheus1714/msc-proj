import pytest
import numpy as np
import os
import tempfile
import shutil
from src.activities.split_data_activity import split_data_activity, SplitDataIn, SplitDataOut


class TestSplitDataActivity:
    """Testes para a activity de divisão de dados"""
    
    @pytest.fixture
    def temp_dir(self):
        """Cria um diretório temporário para os testes"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def sample_data(self, temp_dir):
        """Cria dados de exemplo para os testes"""
        # Criar dados de exemplo
        x_seq = np.random.randint(0, 1000, size=(100, 50))
        y = np.random.randint(0, 2, size=100)
        
        x_seq_path = os.path.join(temp_dir, "x_seq.npy")
        y_path = os.path.join(temp_dir, "y.npy")
        
        np.save(x_seq_path, x_seq)
        np.save(y_path, y)
        
        return x_seq, y, x_seq_path, y_path
    
    @pytest.mark.asyncio
    async def test_split_data_creates_files(self, temp_dir, sample_data):
        """Testa que a divisão de dados cria os arquivos esperados"""
        x_seq, y, x_seq_path, y_path = sample_data
        
        split_in = SplitDataIn(
            x_seq_path=x_seq_path,
            y_path=y_path,
            x_train_path=os.path.join(temp_dir, "x_train.npy"),
            x_val_path=os.path.join(temp_dir, "x_val.npy"),
            x_test_path=os.path.join(temp_dir, "x_test.npy"),
            y_train_path=os.path.join(temp_dir, "y_train.npy"),
            y_val_path=os.path.join(temp_dir, "y_val.npy"),
            y_test_path=os.path.join(temp_dir, "y_test.npy"),
            random_state=42
        )
        
        result = await split_data_activity(split_in)
        
        # Verificar que os arquivos foram criados
        assert os.path.exists(result.x_train_path)
        assert os.path.exists(result.x_val_path)
        assert os.path.exists(result.x_test_path)
        assert os.path.exists(result.y_train_path)
        assert os.path.exists(result.y_val_path)
        assert os.path.exists(result.y_test_path)
    
    @pytest.mark.asyncio
    async def test_split_data_proportions(self, temp_dir, sample_data):
        """Testa que a divisão mantém as proporções corretas (70/20/10)"""
        x_seq, y, x_seq_path, y_path = sample_data
        
        split_in = SplitDataIn(
            x_seq_path=x_seq_path,
            y_path=y_path,
            x_train_path=os.path.join(temp_dir, "x_train.npy"),
            x_val_path=os.path.join(temp_dir, "x_val.npy"),
            x_test_path=os.path.join(temp_dir, "x_test.npy"),
            y_train_path=os.path.join(temp_dir, "y_train.npy"),
            y_val_path=os.path.join(temp_dir, "y_val.npy"),
            y_test_path=os.path.join(temp_dir, "y_test.npy"),
            random_state=42
        )
        
        result = await split_data_activity(split_in)
        
        # Carregar os dados divididos
        x_train = np.load(result.x_train_path)
        x_val = np.load(result.x_val_path)
        x_test = np.load(result.x_test_path)
        y_train = np.load(result.y_train_path)
        y_val = np.load(result.y_val_path)
        y_test = np.load(result.y_test_path)
        
        total = len(x_seq)
        
        # Verificar proporções aproximadas (70/20/10)
        assert len(x_train) == len(y_train)
        assert len(x_val) == len(y_val)
        assert len(x_test) == len(y_test)
        
        # Verificar que a soma é igual ao total
        assert len(x_train) + len(x_val) + len(x_test) == total
        
        # Verificar proporções (com tolerância de 5%)
        train_ratio = len(x_train) / total
        val_ratio = len(x_val) / total
        test_ratio = len(x_test) / total
        
        assert 0.65 <= train_ratio <= 0.75  # ~70%
        assert 0.15 <= val_ratio <= 0.25   # ~20%
        assert 0.05 <= test_ratio <= 0.15  # ~10%
    
    @pytest.mark.asyncio
    async def test_split_data_reproducibility(self, temp_dir, sample_data):
        """Testa que a divisão é reproduzível com o mesmo random_state"""
        x_seq, y, x_seq_path, y_path = sample_data
        
        # Primeira divisão
        split_in1 = SplitDataIn(
            x_seq_path=x_seq_path,
            y_path=y_path,
            x_train_path=os.path.join(temp_dir, "x_train1.npy"),
            x_val_path=os.path.join(temp_dir, "x_val1.npy"),
            x_test_path=os.path.join(temp_dir, "x_test1.npy"),
            y_train_path=os.path.join(temp_dir, "y_train1.npy"),
            y_val_path=os.path.join(temp_dir, "y_val1.npy"),
            y_test_path=os.path.join(temp_dir, "y_test1.npy"),
            random_state=42
        )
        
        result1 = await split_data_activity(split_in1)
        
        # Segunda divisão com mesmo random_state
        split_in2 = SplitDataIn(
            x_seq_path=x_seq_path,
            y_path=y_path,
            x_train_path=os.path.join(temp_dir, "x_train2.npy"),
            x_val_path=os.path.join(temp_dir, "x_val2.npy"),
            x_test_path=os.path.join(temp_dir, "x_test2.npy"),
            y_train_path=os.path.join(temp_dir, "y_train2.npy"),
            y_val_path=os.path.join(temp_dir, "y_val2.npy"),
            y_test_path=os.path.join(temp_dir, "y_test2.npy"),
            random_state=42
        )
        
        result2 = await split_data_activity(split_in2)
        
        # Verificar que as divisões são idênticas
        x_train1 = np.load(result1.x_train_path)
        x_train2 = np.load(result2.x_train_path)
        
        np.testing.assert_array_equal(x_train1, x_train2)
    
    @pytest.mark.asyncio
    async def test_split_data_creates_directories(self, temp_dir, sample_data):
        """Testa que os diretórios são criados automaticamente"""
        x_seq, y, x_seq_path, y_path = sample_data
        
        # Usar subdiretórios que não existem
        subdir = os.path.join(temp_dir, "subdir")
        split_in = SplitDataIn(
            x_seq_path=x_seq_path,
            y_path=y_path,
            x_train_path=os.path.join(subdir, "x_train.npy"),
            x_val_path=os.path.join(subdir, "x_val.npy"),
            x_test_path=os.path.join(subdir, "x_test.npy"),
            y_train_path=os.path.join(subdir, "y_train.npy"),
            y_val_path=os.path.join(subdir, "y_val.npy"),
            y_test_path=os.path.join(subdir, "y_test.npy"),
            random_state=42
        )
        
        result = await split_data_activity(split_in)
        
        # Verificar que o diretório foi criado
        assert os.path.exists(subdir)
        assert os.path.exists(result.x_train_path)

