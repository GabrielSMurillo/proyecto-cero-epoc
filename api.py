# api.py
import uvicorn 
import sys
from pathlib import Path

# Añadir el directorio actual al PATH
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import logging
from src.inference import SpectrogramPredictor

# --- Configuración ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Clasificación de Sonidos Respiratorios")

# --- Cargar el Modelo (se hace una sola vez al iniciar) ---
MODEL_PATH = Path("models/cnn2d_best_model.pth")
ENCODER_PATH = Path("models/label_encoder.joblib") # Necesitarás guardar este desde el notebook 04

# Comprobar si los archivos del modelo existen
if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
    raise FileNotFoundError("Asegúrate de que 'cnn2d_best_model.pth' y 'label_encoder.joblib' estén en la carpeta /models")

predictor = SpectrogramPredictor(model_path=MODEL_PATH, encoder_path=ENCODER_PATH)
logger.info("Modelo cargado y listo para recibir peticiones.")

# --- Definir el Endpoint de la API ---
@app.post("/predict")
async def predict_audio(audio_file: UploadFile = File(...)):
    """
    Recibe un archivo de audio, lo procesa y devuelve la predicción.
    """
    # Crear un directorio temporal para guardar el archivo subido
    temp_dir = Path("temp_audio")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / audio_file.filename

    try:
        # Guardar el archivo de audio temporalmente
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        logger.info(f"Procesando el archivo: {temp_path}")
        # Realizar la predicción
        prediction = predictor.predict(temp_path)
        
    except Exception as e:
        logger.error(f"Error durante la predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")
    finally:
        # Limpiar el archivo temporal
        if temp_path.exists():
            temp_path.unlink()
            
    return prediction

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Clasificación de Sonidos Respiratorios. Usa el endpoint /predict para clasificar un archivo."}

# Para ejecutar localmente: uvicorn api:app --reload