# PROYECTO CERO: Clasificador de Sonidos Respiratorios para EPOC 🗣️🦠

![Estado del Proyecto](https://img.shields.io/badge/Estado-Fase%201%20%7C%20Exploraci%C3%B3n%20y%20Baseline-blue)
![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Licencia](https://img.shields.io/badge/Licencia-MIT-green)

**Autor:** Gabriel Santiago Murillo Barragán

---

## 🏥 1. Problema y Misión

La Enfermedad Pulmonar Obstructiva Crónica (EPOC) es una de las principales causas de morbilidad y mortalidad en el mundo. Las **exacerbaciones agudas** son eventos que deterioran rápidamente la salud del paciente y generan altos costos al sistema de salud.

### 🔊 Misión del Proyecto:

Desarrollar un clasificador basado en **Machine Learning** capaz de analizar sonidos respiratorios y detectar **biomarcadores tempranos** (sibilancias, crepitantes) que indiquen una posible exacerbación de la EPOC.

> Una detección temprana podría evitar hospitalizaciones y mejorar la calidad de vida.

---

## 🔢 2. Métricas de Éxito

### 🎯 Métricas del Modelo:

* **F1-Score Ponderado** (métrica principal)
* **AUC-ROC** para discriminación robusta
* **Matriz de confusión y Recall por clase** (minimizar Falsos Negativos en "Exacerbación")

### 📊 Métricas de Producto:

* Reducción potencial de hospitalizaciones
* Tiempo de detección previa al evento
* Confianza del paciente (bajo FPR)

---

## 🔄 3. Plan del Proyecto

| Fase      | Descripción                                           |
| --------- | ----------------------------------------------------- |
| ☑️ Fase 1 | EDA, preprocesamiento y extracción de características |
| ☐ Fase 2  | Modelos baseline (RandomForest, GBM)                  |
| ☐ Fase 3  | Deep Learning (CNN 1D, espectrogramas)                |
| ☐ Fase 4  | Pipeline + API + Docker para despliegue               |

---

## 🔮 4. Estructura del Repositorio

```
proyecto_cero/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_EDA_and_Feature_Engineering.ipynb
│   └── sandbox/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_extractor.py
│   └── model_trainer.py
├── tests/
│   ├── __init__.py
│   └── test_feature_extractor.py
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## ⚙️ 5. Cómo Empezar

```bash
# 1. Clonar el repositorio
$ git clone https://github.com/tu_usuario/proyecto_cero.git
$ cd proyecto_cero

# 2. Crear y activar un entorno virtual
$ python -m venv venv
$ source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
$ pip install -r requirements.txt
```

---

## 🎓 Referencias

* ICBHI Respiratory Sound Database
* Guías GOLD para el manejo de EPOC
* Librosa, PyAudioAnalysis, librosa.display

---

## 🔖 Licencia

Este proyecto está licenciado bajo la Licencia MIT.

---

> "Escuchar antes de que sea tarde: hacia una medicina respiratoria predictiva."
