"""
Estudio de Escritorio para Procesamiento Acústico, Segmentación Espectral,
Diagnóstico y Entrenamiento de Clasificadores Binarios de Audio.

Inicia completamente limpio (sin datasets ni modelos pre-cargados). El usuario
puede seleccionar de forma independiente sus carpetas/archivos de audios
Correctos (Clase 1) e Incorrectos (Clase 0), calibrar el preprocesamiento,
auditar sesgos del dataset y entrenar modelos desde cero o a partir de un
modelo previamente guardado (.pt).

Uso:
    streamlit run app.py
"""

from dataclasses import astuple
import json
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import torch

from src.config import AUDIO, MODEL, SEGMENTATION, TRAIN, AudioConfig, ModelConfig, SegmentationConfig, TrainConfig
from src.datasets.yuhmu_dataset import resolve_audio_sources
from src.inference.predictor import predict_audio, predict_batch
from src.models.checkpoint import list_saved_models, load_model_package
from src.preprocessing.mel_extraction import compute_mel_db, load_waveform
from src.preprocessing.segmentation import (
    PLAUSIBLE_RANGE_MS, quality_indicators, segment, segment_durations_ms,
)
from src.training.diagnose_duration import run_duration_diagnosis
from src.training.diagnose_segments import run_segments_diagnosis
from src.training.permutation_test import execute_permutation_test
from src.training.train import run_training_experiment

DEFAULTS = AudioConfig()

st.set_page_config(
    page_title="Audio ML Studio — Preprocesamiento y Entrenamiento",
    layout="wide",
)


# ---------- utilidades de selección de carpetas y carga ----------

def pick_folder_dialog(title: str = "Seleccionar carpeta de audios") -> str:
    """Abre el selector nativo de carpetas del sistema operativo."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        folder = filedialog.askdirectory(title=title)
        root.destroy()
        return folder or ""
    except Exception:
        return ""


def save_uploaded_wavs(uploaded_files, subdir: str) -> list[Path]:
    """Guarda archivos .wav subidos desde la interfaz en un directorio temporal
    de sesión para que puedan ser leídos por soundfile/torchaudio."""
    if not uploaded_files:
        return []
    base_dir = Path(tempfile.gettempdir()) / "audio_ml_studio_uploads" / subdir
    base_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for uf in uploaded_files:
        dest = base_dir / uf.name
        dest.write_bytes(uf.getvalue())
        paths.append(dest)
    return sorted(paths)


@st.cache_data
def load_audio_and_mel(path_str: str, sample_rate: int, n_mels: int,
                       win_length_ms: float, overlap: float):
    """Waveform, sample rate y Mel-spectrograma en dB SIN normalizar
    (para visualización interpretable). Se cachea por archivo y por
    combinación de parámetros."""
    cfg = AudioConfig(sample_rate=sample_rate, n_mels=n_mels,
                      win_length_ms=win_length_ms, overlap=overlap)
    waveform = load_waveform(path_str, cfg)
    mel_db = compute_mel_db(waveform, cfg).numpy()  # (n_mels, tiempo)
    return waveform.squeeze(0).numpy(), cfg.sample_rate, mel_db


# ---------- funciones de graficación ----------

def plot_waveform(waveform: np.ndarray, sr: int):
    fig, ax = plt.subplots(figsize=(8, 2.2))
    t = np.linspace(0, len(waveform) / sr, num=len(waveform))
    ax.plot(t, waveform, linewidth=0.7, color="#4C72B0")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.set_title("Forma de onda")
    fig.tight_layout()
    return fig


def plot_mel(mel_db: np.ndarray, cfg: AudioConfig, title: str = "Mel-spectrograma"):
    fig, ax = plt.subplots(figsize=(8, 3))
    duration = mel_db.shape[1] * cfg.hop_length / cfg.sample_rate
    im = ax.imshow(
        mel_db, origin="lower", aspect="auto", cmap="magma",
        extent=[0, duration, 0, cfg.n_mels],
    )
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Banda Mel")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def plot_segmentation(mel_db: np.ndarray, result, cfg: AudioConfig):
    frame_s = cfg.hop_length / cfg.sample_rate
    duration = mel_db.shape[1] * frame_s
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.imshow(mel_db, origin="lower", aspect="auto", cmap="magma",
               extent=[0, duration, 0, cfg.n_mels])
    for b in result.boundaries:
        ax1.axvline(b * frame_s, color="white", linestyle="--", linewidth=1.2)
    ax1.axvspan(0, result.voiced_start * frame_s, color="gray", alpha=0.5)
    ax1.axvspan(result.voiced_end * frame_s, duration, color="gray", alpha=0.5)
    ax1.set_ylabel("Banda Mel")
    ax1.set_title("Fronteras detectadas (gris = silencio descartado)")

    t = np.arange(len(result.novelty)) * frame_s
    ax2.plot(t, result.novelty, color="#4C72B0", linewidth=1)
    ax2.axhline(result.threshold, color="#C44E52", linestyle=":", label="Umbral adaptativo")
    ax2.plot(result.boundaries * frame_s, result.novelty[result.boundaries], "o", color="#C44E52")
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("Distancia coseno")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_confusion_matrix_agg(folds: list[dict], class_names: tuple[str, str] = ("Incorrecto", "Correcto")):
    cm_sum = np.sum([np.array(f["confusion_matrix"]) for f in folds], axis=0)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(cm_sum, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Pred: {class_names[0]}", f"Pred: {class_names[1]}"])
    ax.set_yticklabels([f"Real: {class_names[0]}", f"Real: {class_names[1]}"])
    ax.set_title("Matriz de Confusión Acumulada (K-Fold)")
    for i in range(cm_sum.shape[0]):
        for j in range(cm_sum.shape[1]):
            color = "white" if cm_sum[i, j] > cm_sum.max() / 2 else "black"
            ax.text(j, i, str(cm_sum[i, j]), ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_loss_curves(folds: list[dict]):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for idx, f in enumerate(folds):
        history = f.get("loss_history", [])
        if history:
            ax.plot(range(1, len(history) + 1), history, label=f"Fold {idx + 1}", linewidth=1.4)
    ax.set_xlabel("Época")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Curvas de pérdida durante el entrenamiento")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# ---------- estado inicial ----------

def reset_params():
    st.session_state.update(
        sr=DEFAULTS.sample_rate,
        n_mels=DEFAULTS.n_mels,
        win_ms=int(DEFAULTS.win_length_ms),
        overlap_pct=int(DEFAULTS.overlap * 100),
    )


if "sr" not in st.session_state:
    reset_params()
if "correct_dir_path" not in st.session_state:
    st.session_state.correct_dir_path = ""
if "incorrect_dir_path" not in st.session_state:
    st.session_state.incorrect_dir_path = ""


# ---------- Barra Lateral: 1. Selección de Audios (Punto A) ----------

st.sidebar.header("1. Selección de Audios")
modo_carga = st.sidebar.radio(
    "Modo de selección",
    ["Seleccionar carpetas locales", "Subir archivos .wav"],
    horizontal=True,
)

correct_files: list[Path] = []
incorrect_files: list[Path] = []

if modo_carga == "Seleccionar carpetas locales":
    st.sidebar.markdown("**Clase 1 — Audios Correctos**")
    c_col1, c_col2 = st.sidebar.columns([3, 1])
    with c_col2:
        if st.button("📁", key="btn_pick_correct", help="Examinar carpeta de audios correctos"):
            picked = pick_folder_dialog("Seleccionar carpeta de Audios Correctos (Clase 1)")
            if picked:
                st.session_state.correct_dir_path = picked
    with c_col1:
        st.text_input(
            "Ruta carpeta Correctos",
            key="correct_dir_path",
            placeholder="Ej: C:/mis_audios/correct",
            label_visibility="collapsed",
        )

    st.sidebar.markdown("**Clase 0 — Audios Incorrectos**")
    i_col1, i_col2 = st.sidebar.columns([3, 1])
    with i_col2:
        if st.button("📁", key="btn_pick_incorrect", help="Examinar carpeta de audios incorrectos"):
            picked = pick_folder_dialog("Seleccionar carpeta de Audios Incorrectos (Clase 0)")
            if picked:
                st.session_state.incorrect_dir_path = picked
    with i_col1:
        st.text_input(
            "Ruta carpeta Incorrectos",
            key="incorrect_dir_path",
            placeholder="Ej: C:/mis_audios/incorrect",
            label_visibility="collapsed",
        )

    c_src = st.session_state.correct_dir_path.strip() or None
    i_src = st.session_state.incorrect_dir_path.strip() or None
    if c_src:
        correct_files, _ = resolve_audio_sources(root_dir="", correct_source=c_src, incorrect_source=[])
    if i_src:
        _, incorrect_files = resolve_audio_sources(root_dir="", correct_source=[], incorrect_source=i_src)

    if c_src or i_src:
        if st.sidebar.button("Limpiar selección de carpetas"):
            st.session_state.correct_dir_path = ""
            st.session_state.incorrect_dir_path = ""
            st.rerun()
else:
    up_correct = st.sidebar.file_uploader(
        "Subir Audios Correctos (.wav)",
        type=["wav"],
        accept_multiple_files=True,
        key="up_correct_wavs",
    )
    up_incorrect = st.sidebar.file_uploader(
        "Subir Audios Incorrectos (.wav)",
        type=["wav"],
        accept_multiple_files=True,
        key="up_incorrect_wavs",
    )
    correct_files = save_uploaded_wavs(up_correct, "correct")
    incorrect_files = save_uploaded_wavs(up_incorrect, "incorrect")

st.sidebar.caption(
    f"Cargados: **{len(correct_files)}** correctos · **{len(incorrect_files)}** incorrectos"
)

st.sidebar.divider()

# ---------- Barra Lateral: 2. Navegación ----------

st.sidebar.header("2. Herramientas")
vista = st.sidebar.radio(
    "Módulo activo",
    [
        "Vista general del dataset",
        "Explorar audio individual",
        "Comparar correcto vs incorrecto",
        "Segmentación espectral",
        "Auditoría y diagnóstico",
        "Estudio de entrenamiento",
        "Probar modelo / Inferencia",
        "Historial de experimentos",
    ],
)

st.sidebar.divider()

# ---------- Barra Lateral: 3. Parámetros de extracción ----------

st.sidebar.header("3. Parámetros de extracción")
st.sidebar.selectbox("Sample rate (Hz)", [8000, 16000, 22050, 44100], key="sr")
st.sidebar.slider("Bandas Mel", 10, 128, key="n_mels")
st.sidebar.slider("Ventana Hanning (ms)", 10, 50, step=5, key="win_ms")
st.sidebar.slider("Overlap (%)", 0, 90, step=5, key="overlap_pct")
st.sidebar.button("Restablecer valores por defecto", on_click=reset_params)

cfg = AudioConfig(
    sample_rate=st.session_state.sr,
    n_mels=st.session_state.n_mels,
    win_length_ms=float(st.session_state.win_ms),
    overlap=st.session_state.overlap_pct / 100,
)

try:
    for w in cfg.validate():
        st.sidebar.warning(w)
except ValueError as e:
    st.sidebar.error(str(e))
    st.stop()

st.sidebar.caption(
    f"Ventana: {cfg.win_length} muestras · Salto: {cfg.hop_length} muestras "
    f"({1000 * cfg.hop_length / cfg.sample_rate:.1f} ms) · n_fft: {cfg.n_fft} · "
    f"{cfg.frames_per_second:.0f} frames/s"
)


def get_audio(path: Path):
    return load_audio_and_mel(str(path), cfg.sample_rate, cfg.n_mels,
                              cfg.win_length_ms, cfg.overlap)


# ---------- Configuración global de segmentación en session_state ----------

SEG_DEFAULTS = SegmentationConfig()


def reset_seg_params():
    st.session_state.update(
        silence_db=int(SEG_DEFAULTS.silence_db),
        context_ms=int(SEG_DEFAULTS.context_ms),
        smoothing_ms=int(SEG_DEFAULTS.smoothing_ms),
        threshold_k=SEG_DEFAULTS.threshold_k,
        min_novelty=SEG_DEFAULTS.min_novelty,
        min_segment_ms=int(SEG_DEFAULTS.min_segment_ms),
    )


if "silence_db" not in st.session_state:
    reset_seg_params()


def get_current_seg_cfg() -> SegmentationConfig:
    return SegmentationConfig(
        silence_db=float(st.session_state.silence_db),
        context_ms=float(st.session_state.context_ms),
        smoothing_ms=float(st.session_state.smoothing_ms),
        threshold_k=float(st.session_state.threshold_k),
        min_novelty=float(st.session_state.min_novelty),
        min_segment_ms=float(st.session_state.min_segment_ms),
    )


# ---------- Cuerpo Principal ----------

st.title("Audio ML Studio — Procesamiento Acústico y Entrenamiento")

VISTAS_SIN_DATASET = {"Probar modelo / Inferencia", "Historial de experimentos"}

if not correct_files and not incorrect_files and vista not in VISTAS_SIN_DATASET:
    st.info(
        "👋 **La herramienta está lista y sin datos pre-cargados.**\n\n"
        "Puedes elegir cualquiera de estos dos flujos de trabajo:\n"
        "1. **Explorar, Segmentar, Diagnosticar o Entrenar un modelo:** En el panel izquierdo (**1. Selección de Audios**), "
        "selecciona tu carpeta o archivos `.wav` de **Audios Correctos (Clase 1)** y de **Audios Incorrectos (Clase 0)**.\n"
        "2. **Usar un modelo ya entrenado (`.pt`):** Ve directamente al módulo **Probar modelo / Inferencia** en el menú izquierdo."
    )
    st.stop()

if correct_files or incorrect_files:
    sample_file = (correct_files or incorrect_files)[0]
    original_sr = sf.info(str(sample_file)).samplerate
    if cfg.sample_rate > original_sr:
        st.sidebar.warning(
            f"Los archivos están grabados a {original_sr} Hz. Subir el sample rate "
            "no agrega información, solo interpola."
        )


def require_both_classes():
    if not correct_files or not incorrect_files:
        st.warning(
            "Esta sección requiere tener cargados tanto **Audios Correctos (Clase 1)** "
            "como **Audios Incorrectos (Clase 0)** en la barra lateral."
        )
        st.stop()


# ---------- Vista 1: general ----------
if vista == "Vista general del dataset":
    st.header("Vista general del dataset cargado")

    col1, col2, col3 = st.columns(3)
    col1.metric("Audios Correctos (Clase 1)", len(correct_files))
    col2.metric("Audios Incorrectos (Clase 0)", len(incorrect_files))
    col3.metric("Total de grabaciones", len(correct_files) + len(incorrect_files))

    durations = {"Correcto": [], "Incorrecto": []}
    with st.spinner("Calculando Mel-spectrogramas con los parámetros actuales..."):
        for label, files in [("Correcto", correct_files), ("Incorrecto", incorrect_files)]:
            for f in files:
                _, _, mel_db = get_audio(f)
                durations[label].append(mel_db.shape[1])  # frames

    st.subheader("Distribución de duración (en frames del Mel-spectrograma)")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if durations["Correcto"]:
        ax.hist(durations["Correcto"], bins=15, alpha=0.6, label="Correcto", color="#55A868")
    if durations["Incorrecto"]:
        ax.hist(durations["Incorrecto"], bins=15, alpha=0.6, label="Incorrecto", color="#C44E52")
    ax.set_xlabel("Frames temporales")
    ax.set_ylabel("Número de audios")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Estadísticas por clase")
    rows = []
    for label in ["Correcto", "Incorrecto"]:
        if durations[label]:
            arr = np.array(durations[label])
            rows.append({
                "Clase": label,
                "Cantidad": len(arr),
                "Media (frames)": round(float(arr.mean()), 1),
                "Desv. estándar": round(float(arr.std()), 1),
                "Mínimo": int(arr.min()),
                "Máximo": int(arr.max()),
            })
    if rows:
        st.table(rows)


# ---------- Vista 2: individual ----------
elif vista == "Explorar audio individual":
    st.header("Explorar un audio individual")

    clases_disponibles = []
    if correct_files:
        clases_disponibles.append("Correcto (Clase 1)")
    if incorrect_files:
        clases_disponibles.append("Incorrecto (Clase 0)")

    clase = st.radio("Clase", clases_disponibles, horizontal=True)
    files = correct_files if clase.startswith("Correcto") else incorrect_files
    nombres = [f.name for f in files]
    seleccion = st.selectbox("Archivo", nombres)
    path = files[nombres.index(seleccion)]

    waveform, sr, mel_db = get_audio(path)

    st.audio(str(path))

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_waveform(waveform, sr))
    with col2:
        st.pyplot(plot_mel(mel_db, cfg))

    st.caption(
        f"Duración: {len(waveform)/sr:.2f} s · "
        f"Forma del Mel-spectrograma: {mel_db.shape[1]} frames × {mel_db.shape[0]} bandas"
    )


# ---------- Vista 3: comparación ----------
elif vista == "Comparar correcto vs incorrecto":
    st.header("Comparar pronunciación / clase Correcta vs Incorrecta")
    require_both_classes()

    col1, col2 = st.columns(2)
    with col1:
        nombres_c = [f.name for f in correct_files]
        sel_c = st.selectbox("Audio Correcto (Clase 1)", nombres_c, key="sel_correct")
        path_c = correct_files[nombres_c.index(sel_c)]
        wf_c, sr_c, mel_c = get_audio(path_c)
        st.audio(str(path_c))
        st.pyplot(plot_mel(mel_c, cfg, title="Correcto (Clase 1)"))
        st.caption(f"Duración: {len(wf_c)/sr_c:.2f} s · {mel_c.shape[1]} frames")

    with col2:
        nombres_i = [f.name for f in incorrect_files]
        sel_i = st.selectbox("Audio Incorrecto (Clase 0)", nombres_i, key="sel_incorrect")
        path_i = incorrect_files[nombres_i.index(sel_i)]
        wf_i, sr_i, mel_i = get_audio(path_i)
        st.audio(str(path_i))
        st.pyplot(plot_mel(mel_i, cfg, title="Incorrecto (Clase 0)"))
        st.caption(f"Duración: {len(wf_i)/sr_i:.2f} s · {mel_i.shape[1]} frames")


# ---------- Vista 4: segmentación ----------
elif vista == "Segmentación espectral":
    st.header("Segmentación espectral por distancia coseno")
    st.caption(
        "Fronteras entre unidades fonéticas/acústicas detectadas sin transcripción: picos de la "
        "distancia coseno entre ventanas adyacentes que superan un umbral adaptativo relativo a cada audio."
    )

    with st.expander("Parámetros de segmentación", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.slider("Umbral de silencio (dB bajo el pico)", 10, 60, step=5, key="silence_db")
        c2.slider("Ventana de contexto (ms)", 5, 50, step=5, key="context_ms")
        c3.slider("Suavizado (ms)", 5, 50, step=5, key="smoothing_ms")
        c4, c5, c6 = st.columns(3)
        c4.slider("Duración mínima de segmento (ms)", 10, 120, step=10, key="min_segment_ms")
        c5.slider("Sensibilidad k (MADs sobre la mediana, relativo al audio)", 0.0, 6.0, step=0.5, key="threshold_k")
        c6.slider("Distancia coseno mínima (piso absoluto)", 0.0, 0.5, step=0.01, key="min_novelty")
        st.button("Restablecer parámetros de segmentación", on_click=reset_seg_params)

    seg_cfg = get_current_seg_cfg()

    clases_disponibles = []
    if correct_files:
        clases_disponibles.append("Correcto (Clase 1)")
    if incorrect_files:
        clases_disponibles.append("Incorrecto (Clase 0)")

    clase = st.radio("Clase", clases_disponibles, horizontal=True, key="seg_clase")
    files = correct_files if clase.startswith("Correcto") else incorrect_files
    nombres = [f.name for f in files]
    seleccion = st.selectbox("Archivo", nombres, key="seg_archivo")
    path = files[nombres.index(seleccion)]

    _, _, mel_db = get_audio(path)
    result = segment(mel_db, cfg, seg_cfg)
    ind = quality_indicators(result, cfg)

    st.audio(str(path))
    st.pyplot(plot_segmentation(mel_db, result, cfg))

    m1, m2, m3 = st.columns(3)
    m1.metric("Segmentos detectados", ind["n_segmentos"])
    m2.metric("Duración media", f"{ind['duracion_media_ms']:.0f} ms")
    m3.metric("Segmentos con duración plausible", f"{ind['pct_duracion_plausible']:.0f}%")

    st.subheader("Análisis de todo el dataset cargado")
    st.caption(
        f"Sin transcripciones, la calidad se estima con indicadores indirectos: la duración "
        f"de los segmentos debería caer mayormente en el rango típico de un fonema "
        f"({PLAUSIBLE_RANGE_MS[0]:.0f}-{PLAUSIBLE_RANGE_MS[1]:.0f} ms)."
    )

    total_files = len(correct_files) + len(incorrect_files)
    params_key = (
        cfg.sample_rate, cfg.n_mels, cfg.win_length_ms, cfg.overlap,
        astuple(seg_cfg), len(correct_files), len(incorrect_files),
    )

    if st.button(f"Analizar los {total_files} audios cargados con estos parámetros", type="primary"):
        all_durations = []
        counts = {"Correcto": [], "Incorrecto": []}
        with st.spinner("Segmentando el dataset..."):
            for label, lista in [("Correcto", correct_files), ("Incorrecto", incorrect_files)]:
                for f in lista:
                    _, _, m = get_audio(f)
                    r = segment(m, cfg, seg_cfg)
                    counts[label].append(len(r.segments))
                    all_durations.extend(segment_durations_ms(r, cfg))
        st.session_state.dataset_analysis = {
            "key": params_key, "durations": np.array(all_durations), "counts": counts,
        }

    analysis = st.session_state.get("dataset_analysis")
    if analysis is not None and analysis["key"] != params_key:
        st.info("Los parámetros o los audios cambiaron desde el último análisis. "
                "Presiona el botón para actualizar los resultados.")
    elif analysis is not None:
        all_durations = analysis["durations"]
        counts = analysis["counts"]
        low, high = PLAUSIBLE_RANGE_MS
        pct = ((all_durations >= low) & (all_durations <= high)).mean() * 100

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(all_durations, bins=30, color="#4C72B0")
            ax.axvspan(low, high, color="#55A868", alpha=0.2, label="Rango plausible")
            ax.set_xlabel("Duración del segmento (ms)")
            ax.set_ylabel("Número de segmentos")
            ax.set_title(f"{pct:.0f}% de segmentos en rango plausible")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            max_c = max(counts["Correcto"]) if counts["Correcto"] else 1
            max_i = max(counts["Incorrecto"]) if counts["Incorrecto"] else 1
            bins = np.arange(1, max(max_c, max_i) + 2) - 0.5
            if counts["Correcto"]:
                ax.hist(counts["Correcto"], bins=bins, alpha=0.6, label="Correcto", color="#55A868")
            if counts["Incorrecto"]:
                ax.hist(counts["Incorrecto"], bins=bins, alpha=0.6, label="Incorrecto", color="#C44E52")
            ax.set_xlabel("Segmentos por audio")
            ax.set_ylabel("Número de audios")
            ax.set_title("Segmentos por audio, por clase")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)

        rows_seg = []
        for label in ["Correcto", "Incorrecto"]:
            if counts[label]:
                rows_seg.append({
                    "Clase": label,
                    "Segmentos promedio": round(float(np.mean(counts[label])), 2),
                    "Mínimo": int(np.min(counts[label])),
                    "Máximo": int(np.max(counts[label])),
                })
        if rows_seg:
            st.table(rows_seg)


# ---------- Vista 5: Auditoría y Diagnóstico ----------
elif vista == "Auditoría y diagnóstico":
    st.header("Auditoría científica y control de sesgos del dataset")
    require_both_classes()

    seg_cfg = get_current_seg_cfg()

    st.markdown(
        "Antes de entrenar un modelo profundo, estas pruebas verifican si existe algún "
        "**atajo estadístico trivial** (por ejemplo, que una clase dure sistemáticamente más segundos "
        "que la otra) o **fuga de datos (*data leakage*)**."
    )

    tab_dur, tab_seg, tab_perm = st.tabs([
        "1. Diagnóstico por Duración",
        "2. Diagnóstico por Número de Segmentos",
        "3. Test de Permutación (Anti Data-Leakage)",
    ])

    with tab_dur:
        st.subheader("¿La duración del audio predice por sí sola la clase?")
        st.caption(
            "Entrena un clasificador lineal trivial (Regresión Logística con 5-fold CV) usando "
            "ÚNICAMENTE el número de frames de cada audio."
        )
        if st.button("Ejecutar diagnóstico de duración", type="primary", key="btn_diag_dur"):
            with st.spinner("Evaluando clasificador trivial por duración..."):
                res_dur = run_duration_diagnosis(
                    audio_cfg=cfg,
                    correct_source=correct_files,
                    incorrect_source=incorrect_files,
                )
            st.session_state.diag_dur_result = res_dur

        if "diag_dur_result" in st.session_state:
            r = st.session_state.diag_dur_result
            c1, c2 = st.columns(2)
            c1.metric(
                "Accuracy usando SOLO duración",
                f"{r['mean_accuracy'] * 100:.2f}% ± {r['std_accuracy'] * 100:.2f}%",
            )
            if r["mean_accuracy"] < 0.70:
                c2.success("El sesgo por duración es bajo. El modelo necesitará aprender contenido acústico real.")
            else:
                c2.warning("Atención: la duración por sí sola separa gran parte de las clases.")

            st.table([
                {
                    "Clase": nombre.capitalize(),
                    "Media (frames)": round(st_cls["mean"], 1),
                    "Desv. estándar": round(st_cls["std"], 1),
                    "Mínimo": st_cls["min"],
                    "Máximo": st_cls["max"],
                }
                for nombre, st_cls in r["stats_by_class"].items()
            ])

    with tab_seg:
        st.subheader("¿El conteo de segmentos fonéticos predice por sí solo la clase?")
        st.caption(
            "Evalúa si un modelo lineal simple puede clasificar los audios usando únicamente "
            "cuántos segmentos se detectaron, su duración o ambos."
        )
        if st.button("Ejecutar diagnóstico de segmentos", type="primary", key="btn_diag_seg"):
            with st.spinner("Segmentando y evaluando clasificadores triviales..."):
                res_seg = run_segments_diagnosis(
                    audio_cfg=cfg,
                    seg_cfg=seg_cfg,
                    correct_source=correct_files,
                    incorrect_source=incorrect_files,
                )
            st.session_state.diag_seg_result = res_seg

        if "diag_seg_result" in st.session_state:
            rs = st.session_state.diag_seg_result
            st.metric(
                "Correlación (N° de segmentos vs Duración)",
                f"{rs['correlation_segments_duration']:.2f}",
            )
            st.subheader("Baselines triviales (5-Fold Cross-Validation)")
            st.table([
                {
                    "Variables utilizadas": nombre,
                    "Accuracy Promedio": f"{b['mean'] * 100:.2f}%",
                    "Desv. Estándar": f"± {b['std'] * 100:.2f}%",
                }
                for nombre, b in rs["baselines"].items()
            ])

    with tab_perm:
        st.subheader("Test de Permutación de Etiquetas")
        st.caption(
            "Mezcla las etiquetas (`Correcto` / `Incorrecto`) completamente al azar y entrena el modelo. "
            "Si el pipeline no tiene fuga de datos (*data leakage*), el Accuracy debe caer a ~50% (azar)."
        )
        col_p1, col_p2 = st.columns(2)
        perm_rep = col_p1.selectbox(
            "Representación a evaluar",
            ["frames", "segments"],
            format_func=lambda x: "Transformer por Frames" if x == "frames" else "Transformer por Segmentos",
            key="perm_rep_select",
        )
        perm_epochs = col_p2.slider("Épocas por fold (prueba rápida)", 5, 50, value=20, step=5, key="perm_epochs")

        if st.button("Ejecutar Test de Permutación", type="primary", key="btn_run_perm"):
            prog = st.progress(0.0, text="Iniciando test de permutación...")
            total_steps = TRAIN.k_folds * perm_epochs

            def _perm_cb(fold, k_folds, epoch, epochs, loss):
                step = (fold - 1) * epochs + epoch
                prog.progress(
                    min(step / total_steps, 1.0),
                    text=f"Fold {fold}/{k_folds} · Época {epoch}/{epochs} · Loss: {loss:.4f}",
                )

            res_perm = execute_permutation_test(
                representation=perm_rep,
                audio_cfg=cfg,
                seg_cfg=seg_cfg,
                train_cfg=TrainConfig(epochs=perm_epochs),
                correct_source=correct_files,
                incorrect_source=incorrect_files,
                progress_callback=_perm_cb,
            )
            prog.empty()
            st.session_state.perm_result = res_perm

        if "perm_result" in st.session_state:
            rp = st.session_state.perm_result["summary"]
            m_acc = rp["accuracy"]["mean"]
            c1, c2 = st.columns(2)
            c1.metric("Accuracy con etiquetas aleatorias", f"{m_acc * 100:.2f}% ± {rp['accuracy']['std'] * 100:.2f}%")
            c2.metric("F1-Score con etiquetas aleatorias", f"{rp['f1']['mean'] * 100:.2f}% ± {rp['f1']['std'] * 100:.2f}%")
            if 0.35 <= m_acc <= 0.65:
                st.success("Resultado cercano al 50% (azar): Se descarta fuga de datos en el pipeline.")
            else:
                st.warning("El resultado se aleja significativamente del 50%; revisa posibles duplicados en el dataset.")


# ---------- Vista 6: Estudio de Entrenamiento (Punto B) ----------
elif vista == "Estudio de entrenamiento":
    st.header("Estudio de Entrenamiento de Modelos")
    require_both_classes()

    seg_cfg = get_current_seg_cfg()

    st.subheader("1. Selección de Arquitectura o Modelo Preentrenado")
    modo_modelo = st.radio(
        "Punto de partida del entrenamiento",
        [
            "Entrenar modelo desde cero (o Fine-Tuning base)",
            "Cargar un modelo ya entrenado (.pt) para continuar entrenándolo",
        ],
        horizontal=True,
    )

    initial_state_dict = None
    loaded_meta = None

    if modo_modelo == "Entrenar modelo desde cero (o Fine-Tuning base)":
        arquitectura = st.selectbox(
            "Arquitectura del modelo",
            ["frames", "segments", "wav2vec2"],
            format_func=lambda x: {
                "frames": "Transformer desde 0 — Por Frames del Mel-spectrograma",
                "segments": "Transformer desde 0 — Por Segmentos Fonéticos/Acústicos",
                "wav2vec2": "Fine-Tuning — Wav2Vec2 Preentrenado (Forma de onda cruda)",
            }[x],
        )
    else:
        fuente_pt = st.radio(
            "Origen del modelo (.pt)",
            ["Seleccionar de la carpeta local models/", "Subir archivo .pt desde mi PC"],
            horizontal=True,
        )
        checkpoint_path = None
        if fuente_pt == "Seleccionar de la carpeta local models/":
            saved_list = list_saved_models("models")
            if not saved_list:
                st.info("Aún no hay modelos guardados en la carpeta `models/`. Entrena uno primero o sube un archivo `.pt`.")
                st.stop()
            opciones = [f"{item['label']} ({item['representation']}) — {item['name']}" for item in saved_list]
            sel_idx = st.selectbox("Modelo guardado", range(len(opciones)), format_func=lambda i: opciones[i])
            checkpoint_path = saved_list[sel_idx]["path"]
        else:
            up_pt = st.file_uploader("Subir modelo entrenado (.pt)", type=["pt"])
            if up_pt is not None:
                tmp_pt = Path(tempfile.gettempdir()) / up_pt.name
                tmp_pt.write_bytes(up_pt.getvalue())
                checkpoint_path = tmp_pt

        if checkpoint_path is None:
            st.stop()

        try:
            loaded_model, loaded_meta = load_model_package(checkpoint_path, device="cpu")
            arquitectura = loaded_meta["representation"]
            initial_state_dict = loaded_model.state_dict()
            st.success(
                f"Modelo cargado: **{loaded_meta.get('model_label', checkpoint_path.name)}** · "
                f"Representación: `{arquitectura}` · Dimensiones de entrada: `{loaded_meta['input_dim']}`"
            )
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
            st.stop()

    with st.expander("2. Hiperparámetros de Arquitectura y Entrenamiento", expanded=True):
        col_m1, col_m2, col_m3 = st.columns(3)
        if arquitectura in ("frames", "segments"):
            default_mcfg = ModelConfig.from_dict(loaded_meta.get("model_config") or {}) if loaded_meta else MODEL
            d_model = col_m1.selectbox("Dimensión interna (d_model)", [32, 64, 128, 256], index=[32, 64, 128, 256].index(default_mcfg.d_model) if default_mcfg.d_model in [32, 64, 128, 256] else 1)
            nhead = col_m2.selectbox("Cabezas de atención (nhead)", [2, 4, 8], index=[2, 4, 8].index(default_mcfg.nhead) if default_mcfg.nhead in [2, 4, 8] else 1)
            num_layers = col_m3.slider("Capas Transformer (num_layers)", 1, 6, value=int(default_mcfg.num_layers))
            c_ff, c_drop = st.columns(2)
            dim_feedforward = c_ff.selectbox("Dimensión Feedforward", [64, 128, 256, 512], index=[64, 128, 256, 512].index(default_mcfg.dim_feedforward) if default_mcfg.dim_feedforward in [64, 128, 256, 512] else 1)
            dropout = c_drop.slider("Dropout", 0.0, 0.5, value=float(default_mcfg.dropout), step=0.05)
            freeze_wav2vec = True
        else:
            d_model, nhead, num_layers, dim_feedforward, dropout = (
                MODEL.d_model, MODEL.nhead, MODEL.num_layers, MODEL.dim_feedforward, MODEL.dropout,
            )
            freeze_wav2vec = st.checkbox(
                "Congelar encoder base de Wav2Vec2 (entrenar solo el clasificador lineal — mucho más rápido)",
                value=True,
            )

        st.markdown("---")
        t1, t2, t3, t4 = st.columns(4)
        epochs = t1.number_input("Épocas (epochs)", min_value=1, max_value=300, value=int(TRAIN.epochs), step=5)
        batch_size = t2.selectbox("Batch size", [4, 8, 16, 32], index=1)
        lr = t3.select_slider("Learning rate", options=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)
        k_folds = t4.slider("Folds de validación cruzada (k_folds)", 2, 10, value=int(TRAIN.k_folds))

        aug1, aug2, aug3 = st.columns(3)
        use_spec_aug = aug1.checkbox(
            "Activar SpecAugment en entrenamiento",
            value=False,
            disabled=(arquitectura == "wav2vec2"),
        )
        freq_mask = aug2.slider("Máscara de frecuencia (bandas)", 2, 20, value=8, disabled=not use_spec_aug)
        time_mask = aug3.slider("Máscara de tiempo (frames)", 5, 50, value=20, disabled=not use_spec_aug)

    st.subheader("3. Guardado del Modelo")
    g1, g2, g3 = st.columns([2, 1, 1])
    model_name_input = g1.text_input(
        "Nombre para guardar el modelo entrenado (en carpeta models/)",
        value=f"modelo_{arquitectura}",
    )
    class_0_name = g2.text_input("Etiqueta Clase 0", value="Incorrecto")
    class_1_name = g3.text_input("Etiqueta Clase 1", value="Correcto")

    if st.button("Iniciar Entrenamiento K-Fold y Guardar Modelo", type="primary"):
        model_cfg_custom = ModelConfig(
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
        )
        train_cfg_custom = TrainConfig(
            batch_size=int(batch_size),
            lr=float(lr),
            epochs=int(epochs),
            k_folds=int(k_folds),
            use_spec_augment=bool(use_spec_aug),
            freq_mask_param=int(freq_mask),
            time_mask_param=int(time_mask),
        )

        progress_bar = st.progress(0.0, text="Preparando dataset y modelo...")
        status_text = st.empty()
        total_steps = train_cfg_custom.k_folds * train_cfg_custom.epochs

        def _train_cb(fold, total_folds, epoch, total_epochs, avg_loss):
            step = (fold - 1) * total_epochs + epoch
            progress_bar.progress(
                min(step / total_steps, 1.0),
                text=f"Entrenando Fold {fold}/{total_folds} · Época {epoch}/{total_epochs} · Loss: {avg_loss:.4f}",
            )

        save_target = Path("models") / f"{model_name_input.strip() or 'modelo'}.pt"

        with st.spinner("Ejecutando validación cruzada y exportando el mejor modelo..."):
            exp_result = run_training_experiment(
                representation=arquitectura,
                audio_cfg=cfg,
                seg_cfg=seg_cfg,
                model_cfg=model_cfg_custom,
                train_cfg=train_cfg_custom,
                correct_source=correct_files,
                incorrect_source=incorrect_files,
                freeze_wav2vec=freeze_wav2vec,
                initial_state_dict=initial_state_dict,
                save_model_path=str(save_target),
                model_label=model_name_input.strip() or f"modelo_{arquitectura}",
                class_names=(class_0_name.strip() or "Incorrecto", class_1_name.strip() or "Correcto"),
                progress_callback=_train_cb,
            )

        progress_bar.empty()
        st.session_state.last_train_result = exp_result
        st.success(
            f"¡Entrenamiento completado! Modelo guardado en `{exp_result['saved_model_path']}` "
            f"y registro en `{exp_result['result_json_path']}`."
        )

    if "last_train_result" in st.session_state:
        res = st.session_state.last_train_result
        rec = res["record"]
        summ = rec["summary"]

        st.subheader("Resultados del Entrenamiento (Validación Cruzada)")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{summ['accuracy']['mean'] * 100:.2f}% ± {summ['accuracy']['std'] * 100:.2f}%")
        k2.metric("Precision", f"{summ['precision']['mean'] * 100:.2f}% ± {summ['precision']['std'] * 100:.2f}%")
        k3.metric("Recall", f"{summ['recall']['mean'] * 100:.2f}% ± {summ['recall']['std'] * 100:.2f}%")
        k4.metric("F1-Score", f"{summ['f1']['mean'] * 100:.2f}% ± {summ['f1']['std'] * 100:.2f}%")

        ch1, ch2 = st.columns(2)
        with ch1:
            st.pyplot(plot_loss_curves(rec["folds"]))
        with ch2:
            st.pyplot(plot_confusion_matrix_agg(rec["folds"], (class_0_name, class_1_name)))

        st.subheader("Detalle por Fold")
        st.table([
            {
                "Fold": idx + 1,
                "Accuracy": f"{f['accuracy'] * 100:.2f}%",
                "Precision": f"{f['precision'] * 100:.2f}%",
                "Recall": f"{f['recall'] * 100:.2f}%",
                "F1-Score": f"{f['f1'] * 100:.2f}%",
                "Loss Final": f["loss_history"][-1] if f.get("loss_history") else "-",
            }
            for idx, f in enumerate(rec["folds"])
        ])

        saved_pt = res.get("saved_model_path")
        if saved_pt and Path(saved_pt).exists():
            with open(saved_pt, "rb") as fh:
                st.download_button(
                    label=f"⬇️ Descargar modelo entrenado ({Path(saved_pt).name})",
                    data=fh.read(),
                    file_name=Path(saved_pt).name,
                    mime="application/octet-stream",
                )


# ---------- Vista 7: Probar modelo / Inferencia ----------
elif vista == "Probar modelo / Inferencia":
    st.header("Laboratorio de Prueba e Inferencia con Modelos Entrenados")
    st.caption(
        "Carga un modelo entrenado (`.pt`), evalúa nuevos audios `.wav` o grabaciones de micrófono y "
        "examina su clasificación junto con su análisis acústico y segmentación espectral."
    )

    st.subheader("1. Seleccionar Modelo Entrenado (.pt)")
    origen_inf = st.radio(
        "Fuente del modelo",
        ["Seleccionar de la carpeta local models/", "Subir archivo .pt desde mi PC"],
        horizontal=True,
        key="inf_model_source",
    )

    inf_ckpt_path = None
    if origen_inf == "Seleccionar de la carpeta local models/":
        saved_models = list_saved_models("models")
        if not saved_models:
            st.info(
                "No hay modelos guardados en la carpeta `models/` todavía. "
                "Puedes entrenar uno en **Estudio de entrenamiento** o cambiar a **Subir archivo .pt desde mi PC**."
            )
            st.stop()
        opciones_m = [f"{m['label']} ({m['representation']}) — {m['name']}" for m in saved_models]
        idx_m = st.selectbox("Modelo disponible", range(len(opciones_m)), format_func=lambda i: opciones_m[i])
        inf_ckpt_path = saved_models[idx_m]["path"]
    else:
        up_inf_pt = st.file_uploader("Cargar archivo de modelo (.pt)", type=["pt"], key="up_inf_pt")
        if up_inf_pt is not None:
            tmp_inf_pt = Path(tempfile.gettempdir()) / f"inf_{up_inf_pt.name}"
            tmp_inf_pt.write_bytes(up_inf_pt.getvalue())
            inf_ckpt_path = tmp_inf_pt

    if inf_ckpt_path is None:
        st.stop()

    try:
        inf_model, inf_meta = load_model_package(inf_ckpt_path, device="cpu")
    except Exception as e:
        st.error(f"No se pudo cargar el modelo seleccionado: {e}")
        st.stop()

    acfg_meta = inf_meta.get("audio_config") or {}
    summ_meta = inf_meta.get("summary") or {}
    class_names_meta = inf_meta.get("class_names") or {"0": "Incorrecto", "1": "Correcto"}

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Modelo", inf_meta.get("model_label", inf_ckpt_path.stem))
    mc2.metric("Representación", inf_meta.get("representation", "-"))
    mc3.metric("Parámetros Mel", f"{acfg_meta.get('sample_rate', 16000)} Hz · {acfg_meta.get('n_mels', 40)} bandas")
    acc_str = (
        f"{summ_meta['accuracy']['mean'] * 100:.1f}%"
        if "accuracy" in summ_meta and "mean" in summ_meta["accuracy"]
        else "N/A"
    )
    mc4.metric("Accuracy CV del modelo", acc_str)

    st.subheader("2. Seleccionar o Grabar Audios para Evaluar")
    modo_audio_inf = st.radio(
        "Origen de los audios de prueba",
        [
            "Subir archivos .wav desde mi PC",
            "Grabar audio con el micrófono",
            "Seleccionar una carpeta local de prueba",
        ],
        horizontal=True,
        key="modo_audio_inf",
    )

    test_wav_paths: list[Path] = []

    if modo_audio_inf == "Subir archivos .wav desde mi PC":
        up_test_wavs = st.file_uploader(
            "Selecciona uno o varios archivos .wav para clasificar",
            type=["wav"],
            accept_multiple_files=True,
            key="up_test_wavs",
        )
        test_wav_paths = save_uploaded_wavs(up_test_wavs, "inference_test")

    elif modo_audio_inf == "Grabar audio con el micrófono":
        mic_audio = st.audio_input("Presiona el icono del micrófono para grabar un audio de prueba")
        if mic_audio is not None:
            mic_dir = Path(tempfile.gettempdir()) / "audio_ml_studio_uploads" / "mic"
            mic_dir.mkdir(parents=True, exist_ok=True)
            mic_path = mic_dir / "grabacion_microfono.wav"
            mic_path.write_bytes(mic_audio.getvalue())
            test_wav_paths = [mic_path]

    else:
        if "test_folder_path" not in st.session_state:
            st.session_state.test_folder_path = ""
        tf_col1, tf_col2 = st.columns([4, 1])
        with tf_col2:
            if st.button("📁 Examinar...", key="btn_pick_test_dir"):
                picked_tf = pick_folder_dialog("Seleccionar carpeta con audios .wav de prueba")
                if picked_tf:
                    st.session_state.test_folder_path = picked_tf
        with tf_col1:
            st.text_input(
                "Ruta de carpeta con audios de prueba",
                key="test_folder_path",
                placeholder="Ej: C:/mis_audios/pruebas",
                label_visibility="collapsed",
            )
        tf_str = st.session_state.test_folder_path.strip()
        if tf_str and Path(tf_str).is_dir():
            test_wav_paths = sorted(Path(tf_str).glob("*.wav"))

    if not test_wav_paths:
        st.info("Sube o selecciona al menos un archivo `.wav` (o graba con el micrófono) para ejecutar la inferencia.")
        st.stop()

    with st.spinner(f"Evaluando {len(test_wav_paths)} audio(s) con el modelo cargado..."):
        predictions = predict_batch(test_wav_paths, inf_model, inf_meta, device="cpu")

    st.subheader("3. Resultados de Clasificación")
    n_c1 = sum(1 for p in predictions if p["pred_label"] == 1)
    n_c0 = len(predictions) - n_c1
    avg_conf = float(np.mean([p["confidence"] for p in predictions]))

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total evaluados", len(predictions))
    r2.metric(f"Predichos {class_names_meta.get('1', 'Correcto')} (1)", n_c1)
    r3.metric(f"Predichos {class_names_meta.get('0', 'Incorrecto')} (0)", n_c0)
    r4.metric("Confianza promedio", f"{avg_conf * 100:.1f}%")

    df_preds = pd.DataFrame([
        {
            "Archivo": p["file_name"],
            "Predicción": p["pred_class_name"],
            "Etiqueta": p["pred_label"],
            "Confianza (%)": round(p["confidence"] * 100, 2),
            f"Prob. {class_names_meta.get('1', 'Correcto')} (%)": round(p["prob_correct"] * 100, 2),
            "Duración (s)": round(p["duration_s"], 2),
            "Frames": p["n_frames"],
            "Segmentos": p["n_segments"],
        }
        for p in predictions
    ])
    st.dataframe(df_preds, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Exportar tabla de predicciones (.csv)",
        data=df_preds.to_csv(index=False).encode("utf-8"),
        file_name="predicciones_inferencia.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("4. Inspector Acústico del Audio Evaluado")
    nombres_eval = [p["file_name"] for p in predictions]
    sel_eval_idx = st.selectbox(
        "Selecciona un audio evaluado para ver su desglose espectral",
        range(len(nombres_eval)),
        format_func=lambda i: f"{nombres_eval[i]}  →  {predictions[i]['pred_class_name']} ({predictions[i]['confidence'] * 100:.1f}%)",
    )
    p_sel = predictions[sel_eval_idx]

    if p_sel["pred_label"] == 1:
        st.success(
            f"✅ **Clasificación: {p_sel['pred_class_name']} (Clase 1)** · "
            f"Confianza: **{p_sel['confidence'] * 100:.2f}%** "
            f"(Probabilidad Clase 1: {p_sel['prob_correct'] * 100:.2f}%)"
        )
    else:
        st.error(
            f"❌ **Clasificación: {p_sel['pred_class_name']} (Clase 0)** · "
            f"Confianza: **{p_sel['confidence'] * 100:.2f}%** "
            f"(Probabilidad Clase 1: {p_sel['prob_correct'] * 100:.2f}%)"
        )

    if p_sel["truncated"]:
        st.warning(
            "Nota: Este audio superó la longitud máxima (`max_seq_len`) con la que fue entrenado "
            "el modelo y fue truncado automáticamente para la evaluación."
        )

    st.audio(p_sel["file_path"])

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.pyplot(plot_waveform(p_sel["waveform"], p_sel["sample_rate"]))
    with col_v2:
        st.pyplot(plot_mel(p_sel["mel_db"], p_sel["audio_cfg"], title="Mel-spectrograma (configuración del modelo)"))

    st.pyplot(plot_segmentation(p_sel["mel_db"], p_sel["segmentation"], p_sel["audio_cfg"]))
    q_ind = p_sel["quality"]
    qm1, qm2, qm3 = st.columns(3)
    qm1.metric("Segmentos detectados", q_ind["n_segmentos"])
    qm2.metric("Duración media de segmento", f"{q_ind['duracion_media_ms']:.0f} ms")
    qm3.metric("Segmentos en rango plausible (30-200 ms)", f"{q_ind['pct_duracion_plausible']:.0f}%")


# ---------- Vista 8: Historial de experimentos ----------
else:
    st.header("Historial y Comparador de Experimentos")
    st.caption(
        "Compara todas las corridas de validación cruzada registradas en la carpeta `results/`."
    )

    results_dir = Path("results")
    json_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if results_dir.exists() else []

    if not json_files:
        st.info("No se encontraron registros de experimentos en `results/`.")
        st.stop()

    records = []
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            summ = data.get("summary") or {}
            ac = data.get("audio_config") or {}
            tc = data.get("train_config") or {}
            records.append({
                "Archivo": jf.name,
                "Representación": data.get("representation", "frames"),
                "Sample Rate": ac.get("sample_rate", "-"),
                "Bandas Mel": ac.get("n_mels", "-"),
                "Ventana (ms)": ac.get("win_length_ms", "-"),
                "Overlap": ac.get("overlap", "-"),
                "SpecAugment": "Sí" if tc.get("use_spec_augment") else "No",
                "Accuracy (%)": round(summ.get("accuracy", {}).get("mean", 0.0) * 100, 2),
                "Std Acc (%)": round(summ.get("accuracy", {}).get("std", 0.0) * 100, 2),
                "Precision (%)": round(summ.get("precision", {}).get("mean", 0.0) * 100, 2),
                "Recall (%)": round(summ.get("recall", {}).get("mean", 0.0) * 100, 2),
                "F1-Score (%)": round(summ.get("f1", {}).get("mean", 0.0) * 100, 2),
                "_raw": data,
            })
        except Exception:
            continue

    df_hist = pd.DataFrame([{k: v for k, v in r.items() if k != "_raw"} for r in records])
    st.dataframe(df_hist, use_container_width=True, hide_index=True)

    if len(records) >= 1:
        st.subheader("Comparativa visual de desempeño (Accuracy y F1-Score)")
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        labels_x = [f"{r['Representación']}\n({r['Archivo'][:19]})" for r in records[:8]]
        acc_vals = [r["Accuracy (%)"] for r in records[:8]]
        f1_vals = [r["F1-Score (%)"] for r in records[:8]]
        x = np.arange(len(labels_x))
        w = 0.36
        ax.bar(x - w / 2, acc_vals, width=w, label="Accuracy (%)", color="#4C72B0")
        ax.bar(x + w / 2, f1_vals, width=w, label="F1-Score (%)", color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Porcentaje (%)")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("Inspeccionar corrida específica")
        sel_run = st.selectbox(
            "Registro JSON",
            range(len(records)),
            format_func=lambda i: f"{records[i]['Archivo']} — {records[i]['Representación']} (F1: {records[i]['F1-Score (%)']}%)",
        )
        raw_sel = records[sel_run]["_raw"]
        folds_sel = raw_sel.get("folds") or []
        if folds_sel:
            c_h1, c_h2 = st.columns(2)
            with c_h1:
                if any(f.get("loss_history") for f in folds_sel):
                    st.pyplot(plot_loss_curves(folds_sel))
                else:
                    st.info("Esta corrida histórica no almacenó curva de pérdida por época.")
            with c_h2:
                if all("confusion_matrix" in f for f in folds_sel):
                    st.pyplot(plot_confusion_matrix_agg(folds_sel))
