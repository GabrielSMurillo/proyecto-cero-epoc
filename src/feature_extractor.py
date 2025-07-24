"""
Módulo para la extracción de características de señales de audio.
"""
import logging
from typing import Dict, List, Callable

import numpy as np
import librosa

# Configuración del logging
logger = logging.getLogger(__name__)

def extract_mfcc(signal: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """Extrae los Coeficientes Cepstrales en la Frecuencia de Mel (MFCC)."""
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def extract_zcr(signal: np.ndarray, sr: int) -> np.ndarray:
    """Extrae la Tasa de Cruce por Cero (ZCR)."""
    zcr = librosa.feature.zero_crossing_rate(y=signal)
    return np.mean(zcr.T, axis=0)

def extract_rms_energy(signal: np.ndarray, sr: int) -> np.ndarray:
    """Extrae la energía RMS (Root Mean Square)."""
    rms = librosa.feature.rms(y=signal)
    return np.mean(rms.T, axis=0)