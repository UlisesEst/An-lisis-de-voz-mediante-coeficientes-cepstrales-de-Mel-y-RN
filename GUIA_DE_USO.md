# Guía de Uso Completa — Audio ML Studio

Este documento es el manual interactivo paso a paso para utilizar y probar todas las funciones de **Audio ML Studio**.

---

## 1. Cómo Iniciar la Aplicación

* **Opción A (Doble clic en Windows):** Haz doble clic sobre [`Iniciar_Audio_ML_Studio.bat`](Iniciar_Audio_ML_Studio.bat).
* **Opción B (Ventana nativa desde terminal):**
  ```powershell
  .\.venv\Scripts\python.exe desktop_app.py
  ```
* **Opción C (En navegador web):**
  ```powershell
  .\.venv\Scripts\streamlit.exe run app.py
  ```

---

## 2. Recorrido Paso a Paso por la Herramienta

Al abrir la aplicación por primera vez, esta inicia **completamente en blanco**, sin datasets ni modelos pre-cargados.

### Paso 1: Cargar Audios (Barra Lateral — `1. Selección de Audios`)

Tienes dos formas de cargar los audios de tus dos clases:

1. **Modo `Seleccionar carpetas locales` (Recomendado para datasets completos):**
   * En **Clase 1 — Audios Correctos**, haz clic en el botón **`📁`** (o pega la ruta) y selecciona la carpeta con tus grabaciones positivas (por ejemplo: `data/raw/correct`).
   * En **Clase 0 — Audios Incorrectos**, haz clic en el botón **`📁`** (o pega la ruta) y selecciona la carpeta con tus grabaciones negativas (por ejemplo: `data/raw/incorrect`).
2. **Modo `Subir archivos .wav` (Para pruebas rápidas con archivos sueltos):**
   * Arrastra o selecciona múltiples archivos `.wav` en cada uno de los dos recuadros.

* **Indicador de carga:** Debajo verás `Cargados: X correctos · Y incorrectos`.
* **Reiniciar selección:** El botón **`Limpiar selección de carpetas`** vacía ambas rutas para empezar de cero.

---

### Paso 2: Configurar Parámetros Acústicos (Barra Lateral — `3. Parámetros de extracción`)

Desde la barra lateral puedes calibrar en tiempo real la extracción del **Mel-spectrograma**:

| Parámetro | Valor por defecto | Qué controla |
| :--- | :---: | :--- |
| **Sample rate (Hz)** | `16000` | Frecuencia de muestreo a la que se convierten los audios en mono. |
| **Bandas Mel** | `40` | Número de filtros en escala Mel (resolución espectral). |
| **Ventana Hanning (ms)** | `20 ms` | Tamaño de ventana STFT (`320` muestras a 16 kHz). |
| **Overlap (%)** | `75%` | Porcentaje de traslape entre ventanas consecutivas (`5 ms` de salto = `200 frames/s`). |

* Si configuras valores incompatibles (ej. más bandas Mel que bins FFT), el panel mostrará una advertencia automática.
* El botón **`Restablecer valores por defecto`** restaura los parámetros iniciales.

---

### Paso 3: Explorar el Dataset (`2. Herramientas` — Vistas 1, 2 y 3)

1. **`Vista general del dataset`:**
   * Muestra métricas del total de grabaciones por clase.
   * Genera el **histograma de duración** (en frames del Mel-spectrograma) comparando `Correcto` vs `Incorrecto`.
   * Muestra la tabla estadística con media, desviación estándar, mínimo y máximo por clase.
2. **`Explorar audio individual`:**
   * Selecciona la clase y el archivo `.wav`.
   * Escucha el audio y visualiza su **Forma de onda** junto a su **Mel-spectrograma (`dB`)**.
3. **`Comparar correcto vs incorrecto`:**
   * Compara lado a lado un audio de la Clase 1 frente a uno de la Clase 0 (audio y espectrograma Mel).

---

### Paso 4: Calibrar la Segmentación Espectral (`Segmentación espectral`)

Detecta fronteras entre unidades acústicas/fonéticas sin transcripción comparando ventanas adyacentes con distancia coseno:

1. **Ajuste de parámetros en el panel desplegable:**
   * **Umbral de silencio (dB bajo el pico)** (`25 dB`): Recorta el silencio inicial y final (mostrado en gris).
   * **Ventana de contexto (ms)** (`20 ms`): Tamaño de ventana a izquierda y derecha para calcular la novedad espectral.
   * **Suavizado (ms)** (`15 ms`): Promedio móvil sobre la curva de distancia coseno.
   * **Duración mínima de segmento (ms)** (`40 ms`): Separación mínima entre fronteras.
   * **Sensibilidad $k$ (MADs sobre la mediana)** (`3.0`): Ajusta el umbral adaptativo ($\text{mediana} + k \cdot \text{MAD}$).
   * **Distancia coseno mínima (piso absoluto)** (`0.08`): Piso mínimo para evitar fronteras en zonas estables.
2. **Análisis global del dataset:**
   * Haz clic en **`Analizar los N audios cargados con estos parámetros`** para obtener el porcentaje de segmentos en el rango fonético plausible (`30–200 ms`) y la distribución de segmentos por clase.

---

### Paso 5: Auditar Sesgos y Fuga de Datos (`Auditoría y diagnóstico`)

Incluye 3 pruebas para validar científicamente el dataset antes de entrenar:

1. **`1. Diagnóstico por Duración`:**
   * Entrena una Regresión Logística usando **solo el número de frames**. Permite comprobar si la duración por sí sola separa las clases.
2. **`2. Diagnóstico por Número de Segmentos`:**
   * Evalúa baselines lineales usando *Solo número de segmentos*, *Solo duración* y *Segmentos + duración*, además de su correlación.
3. **`3. Test de Permutación (Anti Data-Leakage)`:**
   * Mezcla las etiquetas al azar y entrena el modelo seleccionado con barra de progreso en vivo. Si el Accuracy ronda el **`~50%`**, confirma que no hay fuga de datos.

---

### Paso 6: Entrenar y Guardar un Modelo (`Estudio de entrenamiento`)

Permite entrenar clasificadores mediante **Validación Cruzada Estratificada (`K-Fold`)**:

1. **Punto de partida (`1. Selección de Arquitectura o Modelo Preentrenado`):**
   * **Entrenar modelo desde cero (o Fine-Tuning base):**
     * **Transformer desde 0 — Por Frames (`frames`)**: Secuencia de frames Mel (`input_dim = n_mels`).
     * **Transformer desde 0 — Por Segmentos (`segments`)**: Secuencia de segmentos acústicos (`input_dim = 2 * n_mels + 1`).
     * **Fine-Tuning — Wav2Vec2 (`wav2vec2`)**: Basado en `facebook/wav2vec2-base` sobre onda cruda.
   * **Cargar un modelo ya entrenado (`.pt`) para continuar entrenándolo:**
     * Selecciona un modelo de la carpeta `models/` o sube un `.pt` desde tu PC para continuar su entrenamiento con nuevos audios.
2. **Hiperparámetros (`2. Hiperparámetros de Arquitectura y Entrenamiento`):**
   * Configura `d_model`, `nhead`, `num_layers`, `dim_feedforward`, `dropout`, `epochs`, `batch_size`, `learning_rate`, `k_folds` y el uso de **`SpecAugment`**.
3. **Guardado (`3. Guardado del Modelo`):**
   * Define el nombre del archivo (se guarda en `models/<nombre>.pt` + `.json`) y los nombres de las etiquetas (`Clase 0` y `Clase 1`).
   * Presiona **`Iniciar Entrenamiento K-Fold y Guardar Modelo`**.
   * Revisa las métricas promedio, las **curvas de pérdida (Loss)**, la **Matriz de Confusión acumulada** y descarga el archivo `.pt` si deseas compartirlo.

---

### Paso 7: Evaluar Audios Nuevos (`Probar modelo / Inferencia`)

> Este módulo puede usarse directamente al abrir la aplicación sin necesidad de cargar un dataset en el Paso 1.

1. **Seleccionar Modelo Entrenado (`.pt`):**
   * Elige un modelo de `models/` o sube cualquier archivo `.pt` desde tu PC.
2. **Seleccionar o Grabar Audios para Evaluar:**
   * **Subir archivos `.wav` desde mi PC** (uno o varios archivos).
   * **Grabar audio con el micrófono** directamente en la aplicación.
   * **Seleccionar una carpeta local de prueba** con el botón `📁 Examinar...`.
3. **Resultados e Inspector Acústico:**
   * Consulta la tabla de predicciones con el porcentaje de confianza y expórtala a **`.csv`**.
   * Selecciona cualquier audio evaluado en el **Inspector Acústico** para escucharlo y ver su forma de onda, Mel-spectrograma y segmentación espectral calculados con la configuración exacta del modelo.

---

### Paso 8: Comparar Corridas (`Historial de experimentos`)

* Consulta y compara en tabla y gráficas todas las corridas registradas en `results/*.json`, incluyendo sus curvas de entrenamiento y matrices de confusión.
