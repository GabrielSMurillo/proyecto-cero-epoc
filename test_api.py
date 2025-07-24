import requests
import os
import logging
import kagglehub
from pathlib import Path

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/predict"

def find_sample_audio_path() -> Path:
    """
    Localiza la carpeta del dataset usando kagglehub y devuelve la ruta 
    de un archivo de audio de ejemplo.
    """
    logger.info("Localizando el dataset con Kaggle Hub...")
    try:
        # Este comando no vuelve a descargar, solo obtiene la ruta de la caché
        dataset_path = kagglehub.dataset_download("vbookshelf/respiratory-sound-database")
        audio_folder_path = Path(dataset_path) / "Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files"
        
        # Buscar cualquier archivo .wav en la carpeta
        sample_files = list(audio_folder_path.glob('*.wav'))
        if not sample_files:
            raise FileNotFoundError("No se encontraron archivos .wav en el directorio del dataset.")
        
        # Devolver el primer archivo encontrado
        return sample_files[0]
    except Exception as e:
        logger.error(f"No se pudo localizar el dataset de Kaggle. Asegúrate de tener tus credenciales configuradas. Error: {e}")
        raise

def test_prediction(file_path: Path):
    """
    Envía un archivo de audio a la API y muestra la respuesta.

    Args:
        file_path (Path): La ruta completa al archivo de audio a clasificar.
    """
    if not file_path.exists():
        logger.error(f"El archivo no se encontró en la ruta: {file_path}")
        return

    logger.info(f"Enviando archivo: {file_path.name} a {API_URL}")
    
    # Abrir el archivo en modo binario ("rb") para enviarlo en la petición
    with open(file_path, "rb") as audio_file:
        files = {"audio_file": (file_path.name, audio_file, "audio/wav")}
        
        try:
            # Enviar la petición POST con el archivo
            response = requests.post(API_URL, files=files, timeout=60)
            
            # Comprobar la respuesta del servidor
            if response.status_code == 200:
                print("\n--- ¡Predicción Recibida! ---")
                print(response.json())
            else:
                print(f"\n--- Error del Servidor ---")
                print(f"Código de Estado: {response.status_code}")
                print(f"Respuesta: {response.text}")

        except requests.exceptions.ConnectionError:
            print(f"\n--- Error de Conexión ---")
            print(f"No se pudo conectar a {API_URL}.")
            print("Asegúrate de que el contenedor Docker esté corriendo.")
        except Exception as e:
            print(f"\nOcurrió un error inesperado: {e}")


if __name__ == "__main__":
    try:
        # 1. Encontrar la ruta de un audio automáticamente
        audio_to_test = find_sample_audio_path()
        
        # 2. Ejecutar la prueba con la ruta encontrada
        test_prediction(audio_to_test)
    except FileNotFoundError as e:
        logger.error(f"No se pudo continuar con la prueba: {e}")
    except Exception as e:
        logger.error(f"Ocurrió un error general: {e}")