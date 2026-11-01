# Clasificación Binaria de Audio y Pronunciación con Transformers — Audio ML Studio

Plataforma local de escritorio para el análisis acústico, la segmentación fonética/espectral no supervisada y el entrenamiento e inferencia de modelos de aprendizaje profundo (**Transformers** y **Wav2Vec2**) aplicados a la **clasificación binaria de grabaciones de voz y audio** (`Clase 1 = Correcto` / `Clase 0 = Incorrecto`).

El proyecto explora y compara distintas representaciones acústicas y arquitecturas:
1. **Transformer Encoder entrenado desde cero** (`PronunciationTransformer`) sobre dos tipos de secuencia:
   - **Por frames (`frames`)**: Secuencia de frames del Mel-spectrograma normalizado (`input_dim = n_mels`).
   - **Por segmentos fonéticos/acústicos (`segments`)**: Secuencia de unidades obtenidas mediante segmentación espectral no supervisada por distancia coseno (`input_dim = 2 * n_mels + 1`).
2. **Fine-tuning de un modelo preentrenado de habla** (`Wav2Vec2Classifier`, basado en `facebook/wav2vec2-base`) sobre forma de onda cruda.

> 📖 **¿Buscas el manual paso a paso de la interfaz?** Consulta la **[Guía de Uso Completa (`GUIA_DE_USO.md`)](GUIA_DE_USO.md)**.

---

## Parámetros Espectrales Base

- **Frecuencia de muestreo por defecto**: `16,000 Hz` (configurable a `8,000`, `16,000`, `22,050` o `44,100 Hz`)
- **Ventana Hanning**: `20 ms` (`320` muestras a `16 kHz`)
- **Overlap**: `75%` (salto de `80` muestras / `5 ms`, `200 frames/s`)
- **`n_fft`**: `512` (siguiente potencia de 2 respecto a `win_length`)
- **Filtros Mel**: `40` bandas (configurable de `10` a `128`)

---

## Instalación y Ejecución

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Ejecutar como Aplicación de Escritorio Local
En Windows, haz doble clic en `Iniciar_Audio_ML_Studio.bat` o ejecuta:
```bash
python desktop_app.py
```

### Ejecutar en Navegador Web (Streamlit)
```bash
streamlit run app.py
```

---

## Estructura del Proyecto

```text
├── app.py                         # Interfaz interactiva principal (8 módulos)
├── desktop_app.py                 # Lanzador de aplicación de escritorio en ventana nativa
├── Iniciar_Audio_ML_Studio.bat    # Acceso directo de doble clic para Windows
├── GUIA_DE_USO.md                 # Manual y guía de uso paso a paso de la aplicación
├── models/                        # Paquetes de modelos entrenados (.pt + .json)
├── results/                       # Registros JSON de corridas k-fold
└── src/
    ├── config.py                  # Hiperparámetros y configuración centralizada
    ├── preprocessing/
    │   ├── mel_extraction.py      # Carga de audio y extracción de Mel-spectrograma
    │   ├── segmentation.py        # Segmentación espectral no supervisada por distancia coseno
    │   └── augmentation.py        # SpecAugment (máscaras de tiempo y frecuencia)
    ├── datasets/
    │   └── yuhmu_dataset.py       # Datasets de doble fuente (frames, segmentos, waveform) + collate_fn
    ├── models/
    │   ├── transformer_scratch.py # Transformer encoder entrenado desde cero
    │   ├── wav2vec_finetune.py    # Clasificador con Wav2Vec2 preentrenado
    │   └── checkpoint.py          # Guardado y carga de modelos entrenados (.pt + .json)
    ├── inference/
    │   └── predictor.py           # Inferencia sobre audios individuales o lotes .wav
    ├── training/
    │   ├── train.py               # Script y funciones de entrenamiento y exportación de modelos
    │   ├── cross_validation.py    # Validación cruzada estratificada (k-fold)
    │   ├── diagnose_duration.py   # Diagnóstico de atajo por duración del audio
    │   ├── diagnose_segments.py   # Diagnóstico de atajo por número de segmentos y duración
    │   └── permutation_test.py    # Test de permutación de etiquetas contra data leakage
    └── evaluation/
        └── metrics.py             # Accuracy, Precision, Recall, F1 y matriz de confusión
```

---

## Arquitectura y Módulos Principales

### 1. Configuración Centralizada (`src/config.py`)
- **`AudioConfig`**: Parámetros espectrales (`sample_rate = 16000`, `n_mels = 40`, `win_length_ms = 20.0`, `overlap = 0.75`) y validación automática (`validate()`).
- **`SegmentationConfig`**: Parámetros en milisegundos independientes del sample rate (`silence_db = 25.0`, `context_ms = 20.0`, `smoothing_ms = 15.0`, `threshold_k = 3.0`, `min_novelty = 0.08`, `min_segment_ms = 40.0`).
- **`ModelConfig`**: Transformer parametrizable (`d_model = 64`, `nhead = 4`, `num_layers = 2`, `dim_feedforward = 128`, `dropout = 0.2`, `max_seq_len = 500`).
- **`TrainConfig`**: `batch_size = 8`, `lr = 1e-3`, `epochs = 50`, `k_folds = 5`, `seed = 42`, e integración opcional de `SpecAugment`.

### 2. Preprocesamiento (`src/preprocessing/`)
- **`mel_extraction.py`**: Carga en mono y remuestreo (`load_waveform()`), cálculo de Mel-spectrograma en dB (`compute_mel_db()`) y normalización por muestra ($\mu=0, \sigma=1$) transpuesta a `(tiempo, n_mels)` (`normalize_mel()`).
- **`segmentation.py`**: Segmentación fonética sin transcripción (`segment()`). Recorta el silencio inicial/final (`voiced_region()`), calcula la novedad espectral por distancia coseno entre ventanas adyacentes (`window_novelty()`) y detecta fronteras con un umbral adaptativo robusto ($\text{mediana} + k \cdot \text{MAD}$ con piso absoluto `min_novelty`). Cada segmento se resume mediante `segment_features()` en un vector de dimensión `2 * n_mels + 1` (media espectral, desviación estándar y log-duración relativa a 100 ms).
- **`augmentation.py`**: Implementa `SpecAugment` sobre el Mel-spectrograma exclusivamente en el subconjunto de entrenamiento.

### 3. Datasets, Modelos e Inferencia (`src/datasets/`, `src/models/`, `src/inference/`)
- **`yuhmu_dataset.py`**:
  - Soporta dos fuentes independientes de audio (`correct_source` e `incorrect_source`) mediante `resolve_audio_sources()`.
  - `YuhmuPronunciationDataset`: Secuencia de frames `(tiempo, n_mels)`.
  - `SegmentedPronunciationDataset`: Secuencia de segmentos `(n_segmentos, 2 * n_mels + 1)`.
  - `WaveformPronunciationDataset`: Forma de onda cruda normalizada `(muestras,)` para `Wav2Vec2Classifier`.
  - `collate_fn()`: Padding dinámico por lote y máscara booleana `src_key_padding_mask` (`True` en posiciones de padding).
- **`transformer_scratch.py`**: `PronunciationTransformer` con proyección lineal a `d_model`, *positional encoding* aprendible, `TransformerEncoder`, *mean pooling* enmascarado y clasificador lineal binario.
- **`wav2vec_finetune.py`**: `Wav2Vec2Classifier` con soporte para congelar el encoder base y *mean pooling* enmascarado sobre los frames válidos.
- **`checkpoint.py` y `predictor.py`**: Empaquetado de pesos y metadatos en archivos `.pt` + `.json`, reconstrucción automática de modelos y evaluación de nuevos archivos `.wav`.

### 4. Uso por Línea de Comandos (Opcional)

Además de la interfaz de escritorio, todos los módulos pueden ejecutarse desde la terminal:

```bash
# Entrenar sobre frames, segmentos o wav2vec2 y guardar el modelo entrenado (.pt)
python -m src.training.train --representation frames --save-model models/modelo_frames.pt
python -m src.training.train --representation segments --save-model models/modelo_segments.pt
python -m src.training.train --representation wav2vec2 --save-model models/modelo_wav2vec2.pt

# Entrenar indicando dos carpetas personalizadas
python -m src.training.train --representation frames --correct-dir ruta/correctos --incorrect-dir ruta/incorrectos

# Diagnósticos de sesgo y test de permutación
python -m src.training.diagnose_duration
python -m src.training.diagnose_segments
python -m src.training.permutation_test --representation frames
```
