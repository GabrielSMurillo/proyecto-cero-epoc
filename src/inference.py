# src/inference.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import librosa
import numpy as np
from pathlib import Path
import joblib
# En src/inference.py, al principio del archivo o donde tengas tus importaciones.
from PIL import Image

class SpectrogramPredictor:
    """
    Clase que encapsula todo el pipeline de inferencia para el modelo 2D-CNN.
    """
    def __init__(self, model_path: Path, encoder_path: Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.img_size = 224
        
        # Cargar el codificador de etiquetas
        self.label_encoder = joblib.load(encoder_path)
        self.num_classes = len(self.label_encoder.classes_)
        
        # Cargar la arquitectura del modelo
        self.model = self._load_model_architecture()
        # Cargar los pesos entrenados
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Definir las transformaciones de imagen (sin aumentación)
        self.transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model_architecture(self):
        """Carga la arquitectura de EfficientNet-B0 y reemplaza el clasificador."""
        model = efficientnet_b0() # No cargamos pesos pre-entrenados aquí
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )
        return model

    def _audio_to_spectrogram(self, audio_path: Path) -> Image.Image:
        """Convierte un archivo de audio en una imagen de Mel-espectrograma."""
        y, sr = librosa.load(audio_path, sr=44100)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_DB = librosa.power_to_db(S, ref=np.max)
        
        # Convertir a imagen RGB compatible con PIL
        # Normalizar a 0-255 y convertir a 3 canales
        from PIL import Image
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(4, 4), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        librosa.display.specshow(S_DB, sr=sr, fmax=8000, ax=ax)
        
        # Guardar en un buffer de memoria para convertir a PIL
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        
        return Image.open(buf).convert('RGB')

    def predict(self, audio_path: Path) -> dict:
        """
        Realiza una predicción completa para un archivo de audio.
        """
        with torch.no_grad():
            # Preprocesamiento
            spectrogram_img = self._audio_to_spectrogram(audio_path)
            img_tensor = self.transforms(spectrogram_img).unsqueeze(0).to(self.device)
            
            # Inferencia
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            
            # Post-procesamiento
            pred_class_name = self.label_encoder.inverse_transform([pred_idx.item()])[0]
            
            return {
                "predicted_class": pred_class_name,
                "confidence": f"{confidence.item():.4f}"
            }