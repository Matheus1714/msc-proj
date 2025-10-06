import os
import sys

def setup_project_path():
    """
    Adiciona o diretório raiz do projeto ao sys.path.
    Deve ser chamada no início de cada script.
    """
    # Pega o diretório do script atual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Pega o diretório pai (raiz do projeto)
    project_root = os.path.dirname(script_dir)
    # Adiciona ao sys.path se não estiver já presente
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
