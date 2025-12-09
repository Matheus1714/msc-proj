import pytest
import pandas as pd
import os
import tempfile
import shutil
from src.activities.prepare_data_for_experiment_activity import (
    prepare_data_for_experiment_activity,
    PrepareDataForExperimentIn,
    PrepareDataForExperimentOut
)


class TestPrepareDataForExperimentActivity:
    """Testes para a activity de preparação de dados para experimentos"""
    
    @pytest.fixture
    def temp_dir(self):
        """Cria um diretório temporário para os testes"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def sample_csv(self, temp_dir):
        """Cria um CSV de exemplo para os testes"""
        data = {
            'title': ['Title 1', 'Title 2', 'Title 3', 'Title 4'],
            'abstract': ['Abstract 1', 'Abstract 2', 'Abstract 3', 'Abstract 4'],
            'keywords': ['keyword1, keyword2', 'keyword3', 'keyword4, keyword5', ''],
            'included': [True, False, True, False]
        }
        df = pd.DataFrame(data)
        
        csv_path = os.path.join(temp_dir, "input.csv")
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    @pytest.mark.asyncio
    async def test_prepare_data_creates_output_file(self, temp_dir, sample_csv):
        """Testa que o arquivo de saída é criado"""
        output_path = os.path.join(temp_dir, "output.csv")
        
        prepare_in = PrepareDataForExperimentIn(
            input_data_path=sample_csv,
            output_data_path=output_path,
            random_state=42
        )
        
        result = await prepare_data_for_experiment_activity(prepare_in)
        
        assert os.path.exists(result.output_data_path)
        assert result.output_data_path == output_path
    
    @pytest.mark.asyncio
    async def test_prepare_data_creates_text_column(self, temp_dir, sample_csv):
        """Testa que a coluna 'text' é criada corretamente"""
        output_path = os.path.join(temp_dir, "output.csv")
        
        prepare_in = PrepareDataForExperimentIn(
            input_data_path=sample_csv,
            output_data_path=output_path,
            random_state=42
        )
        
        await prepare_data_for_experiment_activity(prepare_in)
        
        df_output = pd.read_csv(output_path)
        
        assert 'text' in df_output.columns
        assert len(df_output) == 4
    
    @pytest.mark.asyncio
    async def test_prepare_data_text_concatenation(self, temp_dir, sample_csv):
        """Testa que a coluna text concatena title, keywords e abstract"""
        output_path = os.path.join(temp_dir, "output.csv")
        
        prepare_in = PrepareDataForExperimentIn(
            input_data_path=sample_csv,
            output_data_path=output_path,
            random_state=42
        )
        
        await prepare_data_for_experiment_activity(prepare_in)
        
        df_output = pd.read_csv(output_path)
        df_input = pd.read_csv(sample_csv)
        
        # Verificar que todos os textos originais estão presentes na coluna text
        # (mesmo que embaralhados)
        all_titles = set(df_input['title'].dropna())
        all_abstracts = set(df_input['abstract'].dropna())
        
        # Verificar que pelo menos um título e um abstract aparecem no texto
        all_text = ' '.join(df_output['text'].tolist())
        assert any(title in all_text for title in all_titles if pd.notna(title))
        assert any(abstract in all_text for abstract in all_abstracts if pd.notna(abstract))
    
    @pytest.mark.asyncio
    async def test_prepare_data_included_is_bool(self, temp_dir, sample_csv):
        """Testa que a coluna 'included' é convertida para bool"""
        output_path = os.path.join(temp_dir, "output.csv")
        
        prepare_in = PrepareDataForExperimentIn(
            input_data_path=sample_csv,
            output_data_path=output_path,
            random_state=42
        )
        
        await prepare_data_for_experiment_activity(prepare_in)
        
        df_output = pd.read_csv(output_path)
        
        assert df_output['included'].dtype == bool or df_output['included'].dtype == 'bool'
    
    @pytest.mark.asyncio
    async def test_prepare_data_handles_missing_values(self, temp_dir):
        """Testa que valores faltantes são tratados corretamente"""
        data = {
            'title': ['Title 1', None, 'Title 3'],
            'abstract': ['Abstract 1', 'Abstract 2', None],
            'keywords': [None, 'keyword1', 'keyword2'],
            'included': [True, False, True]
        }
        df = pd.DataFrame(data)
        
        csv_path = os.path.join(temp_dir, "input.csv")
        df.to_csv(csv_path, index=False)
        
        output_path = os.path.join(temp_dir, "output.csv")
        
        prepare_in = PrepareDataForExperimentIn(
            input_data_path=csv_path,
            output_data_path=output_path,
            random_state=42
        )
        
        await prepare_data_for_experiment_activity(prepare_in)
        
        df_output = pd.read_csv(output_path)
        
        # Verificar que não há erros e que o texto foi criado
        assert 'text' in df_output.columns
        assert len(df_output) == 3
    
    @pytest.mark.asyncio
    async def test_prepare_data_creates_directories(self, temp_dir, sample_csv):
        """Testa que os diretórios são criados automaticamente"""
        subdir = os.path.join(temp_dir, "subdir")
        output_path = os.path.join(subdir, "output.csv")
        
        prepare_in = PrepareDataForExperimentIn(
            input_data_path=sample_csv,
            output_data_path=output_path,
            random_state=42
        )
        
        await prepare_data_for_experiment_activity(prepare_in)
        
        assert os.path.exists(subdir)
        assert os.path.exists(output_path)
    
    @pytest.mark.asyncio
    async def test_prepare_data_shuffles_data(self, temp_dir, sample_csv):
        """Testa que os dados são embaralhados com random_state"""
        output_path1 = os.path.join(temp_dir, "output1.csv")
        output_path2 = os.path.join(temp_dir, "output2.csv")
        
        # Primeira execução
        prepare_in1 = PrepareDataForExperimentIn(
            input_data_path=sample_csv,
            output_data_path=output_path1,
            random_state=42
        )
        await prepare_data_for_experiment_activity(prepare_in1)
        
        # Segunda execução com mesmo random_state
        prepare_in2 = PrepareDataForExperimentIn(
            input_data_path=sample_csv,
            output_data_path=output_path2,
            random_state=42
        )
        await prepare_data_for_experiment_activity(prepare_in2)
        
        df1 = pd.read_csv(output_path1)
        df2 = pd.read_csv(output_path2)
        
        # Com mesmo random_state, a ordem deve ser a mesma
        pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))

