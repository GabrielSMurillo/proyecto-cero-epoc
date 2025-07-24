"""
Módulo para entrenar y evaluar modelos de clasificación.
"""
import logging
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report

logger = logging.getLogger(__name__)

class RespiratorySoundClassifier:
    """Una clase flexible para entrenar y evaluar un modelo de clasificación."""

    def __init__(self, model: Any):
        """Inicializa el clasificador con un modelo scikit-learn compatible."""
        self.model = model
        logger.info(f"Clasificador inicializado con el modelo: {type(model).__name__}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Entrena el modelo con los datos proporcionados."""
        logger.info("Iniciando entrenamiento del modelo...")
        self.model.fit(X_train, y_train)
        logger.info("Entrenamiento completado.")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evalúa el modelo entrenado en el conjunto de prueba."""
        logger.info("Iniciando evaluación del modelo...")
        predictions = self.model.predict(X_test)
        
        # Comprobar si el modelo soporta predict_proba para AUC
        if hasattr(self.model, "predict_proba"):
            predict_proba = self.model.predict_proba(X_test)
            # Para el caso multiclase, 'ovr' es una estrategia común
            auc_score = roc_auc_score(y_test, predict_proba, multi_class='ovr')
        else:
            auc_score = None
            logger.warning("El modelo no tiene 'predict_proba', AUC no será calculado.")

        metrics = {
            "f1_score_weighted": f1_score(y_test, predictions, average='weighted'),
            "roc_auc_score": auc_score,
            "classification_report": classification_report(y_test, predictions, output_dict=True)
        }
        return metrics