# Clasificación de pronunciación en Yuhmu con Transformer

Proyecto universitario: clasificación binaria (**pronunciación correcta = 1** / **incorrecta = 0**) de palabras en **Yuhmu** (variante del otomí de Ixtenco, lengua en peligro de extinción), utilizando el corpus **Zetzoho-Yuhmu-Speech-Dataset** (*Ramos-Aguilar et al.*), compuesto por **200 audios** (`100` en `data/raw/correct/*.wav` y `100` en `data/raw/incorrect/*.wav`).

El proyecto explora y compara distintas representaciones acústicas y arquitecturas:
1. **Transformer Encoder entrenado desde cero** (`PronunciationTransformer`) sobre dos tipos de secuencia:
   - **Por frames (`frames`)**: Secuencia de frames del Mel-spectrograma normalizado (`input_dim = n_mels = 40`).
   - **Por segmentos fonéticos (`segments`)**: Secuencia de unidades tipo fonema obtenidas mediante segmentación espectral no supervisada (`input_dim = 2 * n_mels + 1 = 81`).
2. **Fine-tuning de un modelo preentrenado de habla** (`Wav2Vec2Classifier`, basado en `facebook/wav2vec2-base`) sobre forma de onda cruda (en fase de integración).

---

## Dataset y Parámetros Base

- **Corpus**: *Zetzoho-Yuhmu-Speech-Dataset* (Ramos-Aguilar et al.): 200 audios crudos (100 correctos + 100 incorrectos).
- **Parámetros de extracción Mel**: Tomados de Ramos-Aguilar et al. (2025), *Automatic Phonetic Segmentation of the Yuhmu Language Using Mel Scale Spectral Parameters* (IEEE LATAM Trans.):
  - Frecuencia de muestreo: `16,000 Hz`
  - Ventana Hanning: `20 ms` (`320` muestras)
  - Overlap: `75%` (salto de `80` muestras / `5 ms`, `200 frames/s`)
  - `n_fft`: `512`
  - Filtros Mel: `40` bandas

---

## Instalación y Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Coloca los audios en:
```text
data/raw/correct/*.wav
data/raw/incorrect/*.wav
```

---

## Estructura del Proyecto

```text
├── app.py                         # Interfaz interactiva en Streamlit
├── data/raw/                      # Audios .wav (correct/ e incorrect/)
├── results/                       # Registros JSON de corridas k-fold
└── src/
    ├── config.py                  # Hiperparámetros y configuración centralizada
    ├── preprocessing/
    │   ├── mel_extraction.py      # Carga de audio y extracción de Mel-spectrograma
    │   ├── segmentation.py        # Segmentación espectral no supervisada por distancia coseno
    │   └── augmentation.py        # SpecAugment (máscaras de tiempo y frecuencia)
    ├── datasets/
    │   └── yuhmu_dataset.py       # Datasets (frames y segmentos) + collate_fn con padding
    ├── models/
    │   ├── transformer_scratch.py # Transformer encoder entrenado desde cero
    │   └── wav2vec_finetune.py    # Clasificador con Wav2Vec2 preentrenado
    ├── training/
    │   ├── train.py               # Script principal de entrenamiento y registro de métricas
    │   ├── cross_validation.py    # Validación cruzada estratificada (5-fold)
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
- **`ModelConfig`**: Transformer ligero (`d_model = 64`, `nhead = 4`, `num_layers = 2`, `dim_feedforward = 128`, `dropout = 0.2`, `max_seq_len = 500`).
- **`TrainConfig`**: `batch_size = 8`, `lr = 1e-3`, `epochs = 50`, `k_folds = 5`, `seed = 42`.

### 2. Preprocesamiento (`src/preprocessing/`)
- **`mel_extraction.py`**: Carga en mono y remuestreo (`load_waveform()`), cálculo de Mel-spectrograma en dB (`compute_mel_db()`) y normalización por muestra ($\mu=0, \sigma=1$) transpuesta a `(tiempo, n_mels)` (`normalize_mel()`).
- **`segmentation.py`**: Segmentación fonética sin transcripción (`segment()`). Recorta el silencio inicial/final (`voiced_region()`), calcula la novedad espectral por distancia coseno entre ventanas adyacentes (`window_novelty()`) y detecta fronteras con un umbral adaptativo robusto ($\text{mediana} + k \cdot \text{MAD}$ con piso absoluto `min_novelty`). Cada segmento se resume mediante `segment_features()` en un vector de dimensión `2 * n_mels + 1` (media espectral, desviación estándar y log-duración relativa a 100 ms).
- **`augmentation.py`**: Implementa `SpecAugment` sobre el Mel-spectrograma.

### 3. Datasets y Modelos (`src/datasets/` y `src/models/`)
- **`yuhmu_dataset.py`**:
  - `YuhmuPronunciationDataset`: Secuencia de frames `(tiempo, n_mels)`.
  - `SegmentedPronunciationDataset`: Secuencia de segmentos `(n_segmentos, 2 * n_mels + 1)`.
  - `collate_fn()`: Padding dinámico por lote y máscara booleana `src_key_padding_mask` (`True` en posiciones de padding).
- **`transformer_scratch.py`**: `PronunciationTransformer` con proyección lineal a `d_model`, *positional encoding* aprendible, `TransformerEncoder`, *mean pooling* enmascarado (ignorando el padding) y clasificador lineal binario.

### 4. Entrenamiento, Diagnóstico y Evaluación (`src/training/` y `src/evaluation/`)
- **Entrenamiento (`train.py` y `cross_validation.py`)**: Validación cruzada estratificada (`StratifiedKFold`, 5 folds) con `BCEWithLogitsLoss` y `Adam`:
  ```bash
  # Entrenar sobre frames del Mel-spectrograma
  python -m src.training.train --representation frames

  # Entrenar sobre segmentos fonéticos
  python -m src.training.train --representation segments
  ```
- **Diagnósticos y controles de calidad**:
  ```bash
  # Evaluar clasificador trivial basado solo en duración
  python -m src.training.diagnose_duration

  # Evaluar clasificador trivial basado en número de segmentos y/o duración
  python -m src.training.diagnose_segments

  # Test de permutación (etiquetas aleatorias) para descartar fuga de datos
  python -m src.training.permutation_test --representation frames
  python -m src.training.permutation_test --representation segments
  ```

### 5. Interfaz Interactiva (`app.py`)
Aplicación en **Streamlit** para explorar el preprocesamiento y calibrar parámetros en tiempo real:
```bash
streamlit run app.py
```
Cuenta con 4 vistas:
1. **Vista general del dataset**: Conteo por clase e histograma/estadísticas de duración en frames.
2. **Explorar audio individual**: Reproductor de audio, forma de onda y Mel-spectrograma.
3. **Comparar correcto vs incorrecto**: Comparación visual y auditiva lado a lado.
4. **Segmentación espectral**: Visualización de fronteras detectadas, curva de distancia coseno, umbral adaptativo y análisis de plausibilidad fonética (`30–200 ms`) en los 200 audios.

---

## Resultados Experimentales (5-Fold Cross-Validation)

| Representación | Accuracy | Precision | Recall | F1-Score | Registro |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Frames (`frames`)** | `98.50% ± 1.22%` | `98.10% ± 2.33%` | `99.00% ± 2.00%` | `98.51% ± 1.22%` | `results/run_frames_20261002_232726.json` |
| **Segmentos (`segments`)** | `96.00% ± 2.00%` | `95.19% ± 3.01%` | `97.00% ± 2.45%` | `96.05% ± 1.95%` | `results/run_segments_20261002_233542.json` |

---

## Próximos Pasos / Pendientes

- [x] Validar carga de los 200 audios con `mel_extraction.py`
- [x] Correr `cross_validation.py` con `PronunciationTransformer` (representación por `frames` y por `segments`)
- [ ] Implementar `Dataset` y `collate_fn` de waveform crudo para `Wav2Vec2Classifier` y comparar métricas
- [ ] Integrar `SpecAugment` en el split de entrenamiento dentro de `cross_validation.py`
- [ ] Actualizar `requirements.txt` con las dependencias adicionales usadas en el proyecto (`soundfile`, `scipy`, `streamlit`)
