import pytest
import numpy as np
from src.utils.convert_to_native import convert_to_native


class TestConvertToNative:
    """Testes para conversão de tipos NumPy para tipos nativos do Python"""
    
    def test_convert_numpy_int(self):
        """Testa conversão de inteiros NumPy"""
        assert convert_to_native(np.int32(42)) == 42
        assert convert_to_native(np.int64(100)) == 100
        assert isinstance(convert_to_native(np.int32(42)), int)
        assert isinstance(convert_to_native(np.int64(100)), int)
    
    def test_convert_numpy_float(self):
        """Testa conversão de floats NumPy"""
        assert convert_to_native(np.float32(3.14)) == pytest.approx(3.14)
        assert convert_to_native(np.float64(2.71)) == pytest.approx(2.71)
        assert isinstance(convert_to_native(np.float32(3.14)), float)
        assert isinstance(convert_to_native(np.float64(2.71)), float)
    
    def test_convert_numpy_array(self):
        """Testa conversão de arrays NumPy para listas"""
        arr = np.array([1, 2, 3, 4, 5])
        result = convert_to_native(arr)
        
        assert isinstance(result, list)
        assert result == [1, 2, 3, 4, 5]
    
    def test_convert_numpy_2d_array(self):
        """Testa conversão de arrays 2D NumPy"""
        arr = np.array([[1, 2], [3, 4]])
        result = convert_to_native(arr)
        
        assert isinstance(result, list)
        assert result == [[1, 2], [3, 4]]
    
    def test_convert_dict_with_numpy(self):
        """Testa conversão de dicionários contendo valores NumPy"""
        data = {
            'int_val': np.int32(42),
            'float_val': np.float64(3.14),
            'array_val': np.array([1, 2, 3])
        }
        
        result = convert_to_native(data)
        
        assert isinstance(result, dict)
        assert result['int_val'] == 42
        assert isinstance(result['int_val'], int)
        assert result['float_val'] == pytest.approx(3.14)
        assert isinstance(result['float_val'], float)
        assert result['array_val'] == [1, 2, 3]
        assert isinstance(result['array_val'], list)
    
    def test_convert_list_with_numpy(self):
        """Testa conversão de listas contendo valores NumPy"""
        data = [
            np.int32(1),
            np.float64(2.5),
            np.array([3, 4, 5])
        ]
        
        result = convert_to_native(data)
        
        assert isinstance(result, list)
        assert result[0] == 1
        assert isinstance(result[0], int)
        assert result[1] == pytest.approx(2.5)
        assert isinstance(result[1], float)
        assert result[2] == [3, 4, 5]
        assert isinstance(result[2], list)
    
    def test_convert_nested_structure(self):
        """Testa conversão de estruturas aninhadas"""
        data = {
            'level1': {
                'int': np.int32(10),
                'array': np.array([1, 2]),
                'nested_list': [np.float64(1.1), np.float64(2.2)]
            }
        }
        
        result = convert_to_native(data)
        
        assert isinstance(result, dict)
        assert isinstance(result['level1'], dict)
        assert result['level1']['int'] == 10
        assert isinstance(result['level1']['int'], int)
        assert result['level1']['array'] == [1, 2]
        assert isinstance(result['level1']['array'], list)
        assert result['level1']['nested_list'] == [1.1, 2.2]
        assert all(isinstance(x, float) for x in result['level1']['nested_list'])
    
    def test_convert_python_native_types(self):
        """Testa que tipos nativos do Python não são alterados"""
        assert convert_to_native(42) == 42
        assert convert_to_native(3.14) == 3.14
        assert convert_to_native("string") == "string"
        assert convert_to_native([1, 2, 3]) == [1, 2, 3]
        assert convert_to_native({'key': 'value'}) == {'key': 'value'}
    
    def test_convert_empty_structures(self):
        """Testa conversão de estruturas vazias"""
        assert convert_to_native([]) == []
        assert convert_to_native({}) == {}
        assert convert_to_native(np.array([])) == []

