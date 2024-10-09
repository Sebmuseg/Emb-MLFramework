# tests/test_framework.py
from api.api import FrameworkAPI

def test_load_and_predict():
    api = FrameworkAPI()
    api.load_model('test_model', 'path_to_model', 'tensorflow')
    assert 'test_model' in api.list_loaded_models()
    
    # Weitere Tests für Quantisierung und Pruning