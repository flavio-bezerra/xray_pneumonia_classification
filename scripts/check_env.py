import tensorflow as tf
import streamlit as st
import matplotlib
import seaborn
import sklearn
import numpy

def check_environment():
    print("Verificando ambiente...")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Streamlit Version: {st.__version__}")
    print(f"Matplotlib Version: {matplotlib.__version__}")
    print(f"Seaborn Version: {seaborn.__version__}")
    print(f"Scikit-learn Version: {sklearn.__version__}")
    print(f"NumPy Version: {numpy.__version__}")
    

    
    print("\n--- Diagnóstico Detalhado de GPU ---")
    built_with_cuda = tf.test.is_built_with_cuda()
    print(f"TensorFlow Built with CUDA: {built_with_cuda}")
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs Detectadas: {gpus}")
    
    import platform
    if platform.system() == "Windows":
        tf_minor_version = int(tf.__version__.split('.')[1])
        if tf_minor_version > 10:
            print("\n[AVISO IMPORTANTE SOBRE WINDOWS]")
            print("O TensorFlow 2.11+ não suporta GPU nativamente no Windows.")
            print("Para usar GPU no Windows com esta versão, é necessário usar WSL2.")
            print("Ou fazer downgrade para TF 2.10: pip install \"tensorflow<2.11\"")

    print("\nVerificação de Importação de Módulos do Projeto:")
    try:
        import sys
        import os
        sys.path.append(os.path.abspath('.'))
        from src.data_loader import load_data
        from src.model import build_model
        from src.utils import make_gradcam_heatmap
        print("Módulos src/ importados com sucesso!")
    except ImportError as e:
        print(f"Erro ao importar módulos src/: {e}")
    except Exception as e:
        print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    check_environment()
