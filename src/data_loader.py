"""
Módulo para la carga y preprocesamiento inicial de datos de audio.
Actualizado para parsear la estructura del dataset de Kaggle de forma programática.
"""
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, lfilter

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_metadata_df(dataset_path: Path) -> pd.DataFrame:
    """
    Construye un DataFrame de metadatos a partir de la estructura del dataset.

    Recorre los archivos de audio y sus anotaciones .txt, los combina con los
    diagnósticos de los pacientes y devuelve un DataFrame unificado.
    Cada fila representa un único ciclo respiratorio anotado.

    Args:
        dataset_path (Path): La ruta raíz a la base de datos descargada de Kaggle.

    Returns:
        pd.DataFrame: Un DataFrame con la información consolidada de cada ciclo.
    """
    logger.info("Construyendo DataFrame de metadatos desde la estructura de archivos...")
    
    data_dir = dataset_path / "Respiratory_Sound_Database/Respiratory_Sound_Database"
    audio_dir = data_dir / "audio_and_txt_files"
    diagnosis_file = data_dir / "patient_diagnosis.csv"

    patient_diagnoses = pd.read_csv(diagnosis_file, names=['patient_id', 'diagnosis'])
    
    all_records = []
    
    for txt_file in audio_dir.glob("*.txt"):
        audio_filename = txt_file.with_suffix(".wav").name
        audio_path = audio_dir / audio_filename
        patient_id = int(txt_file.stem.split('_')[0])

        annotations = pd.read_csv(txt_file, sep='\t', names=['start_time', 'end_time', 'crackles', 'wheezes'])
        
        annotations['patient_id'] = patient_id
        annotations['audio_filename'] = audio_filename
        annotations['audio_path'] = str(audio_path) # Convertir Path a string para el DF
        
        all_records.append(annotations)
        
    if not all_records:
        logger.error("No se encontraron registros de anotaciones. Verifica la ruta del dataset.")
        return pd.DataFrame()
        
    full_df = pd.concat(all_records, ignore_index=True)
    final_df = pd.merge(full_df, patient_diagnoses, on='patient_id')
    
    logger.info(f"DataFrame construido con {len(final_df)} ciclos respiratorios.")
    return final_df


def load_audio_segment(file_path: str, start_time: float, end_time: float) -> Tuple[np.ndarray, int]:
    """Carga un segmento específico de un archivo de audio.

    Args:
        file_path (str): Ruta al archivo de audio.
        start_time (float): Tiempo de inicio del segmento en segundos.
        end_time (float): Tiempo de fin del segmento en segundos.

    Returns:
        Tuple[np.ndarray, int]: La señal de audio y la frecuencia de muestreo.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        logger.error(f"Archivo no encontrado: {path_obj}")
        raise FileNotFoundError(f"El archivo no se encontró en la ruta: {path_obj}")
    try:
        duration = end_time - start_time
        signal, sr = librosa.load(path_obj, sr=None, offset=start_time, duration=duration)
        return signal, sr
    except Exception as e:
        logger.error(f"No se pudo cargar el segmento del archivo {path_obj}: {e}")
        raise

def bandpass_filter(signal: np.ndarray, sr: int, lowcut: float, highcut: float, order: int = 5) -> np.ndarray:
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    # LÍNEA NUEVA
# Aseguramos que la frecuencia de corte alta sea siempre un poco menor que Nyquist
    if high >= 1:
        high = 0.999
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(signal))
    return signal / max_val if max_val > 0 else signal