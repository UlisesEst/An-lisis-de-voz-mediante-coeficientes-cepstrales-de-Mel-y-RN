"""
Interfaz gráfica (Streamlit) para visualizar el preprocesamiento de
audio: forma de onda, Mel-spectrograma, estadísticas del dataset y
comparación entre pronunciación correcta e incorrecta. Los parámetros
de extracción son editables desde la barra lateral.

Uso:
    streamlit run app.py
"""

from dataclasses import astuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

from src.config import AudioConfig, SegmentationConfig
from src.preprocessing.mel_extraction import compute_mel_db, load_waveform
from src.preprocessing.segmentation import (
    PLAUSIBLE_RANGE_MS, quality_indicators, segment, segment_durations_ms,
)

DATA_DIR = Path("data/raw")
DEFAULTS = AudioConfig()  # valores validados para Yuhmu

st.set_page_config(page_title="Yuhmu — Preprocesamiento de audio", layout="wide")


# ---------- utilidades de carga y extracción (cacheadas) ----------

@st.cache_data
def list_files():
    correct = sorted((DATA_DIR / "correct").glob("*.wav"))
    incorrect = sorted((DATA_DIR / "incorrect").glob("*.wav"))
    return correct, incorrect


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


# ---------- parámetros editables ----------

def reset_params():
    st.session_state.update(
        sr=DEFAULTS.sample_rate,
        n_mels=DEFAULTS.n_mels,
        win_ms=int(DEFAULTS.win_length_ms),
        overlap_pct=int(DEFAULTS.overlap * 100),
    )


if "sr" not in st.session_state:
    reset_params()

st.sidebar.header("Parámetros de extracción")
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
if cfg != DEFAULTS:
    st.sidebar.info(
        "Parámetros distintos a los validados para Yuhmu "
        "(40 bandas, 20 ms, 75% overlap, 16 kHz)."
    )


def get_audio(path: Path):
    return load_audio_and_mel(str(path), cfg.sample_rate, cfg.n_mels,
                              cfg.win_length_ms, cfg.overlap)


# ---------- app ----------

st.title("Yuhmu — Visualización del preprocesamiento de audio")

correct_files, incorrect_files = list_files()

if not correct_files and not incorrect_files:
    st.error(
        "No se encontraron audios en data/raw/correct ni data/raw/incorrect. "
        "Verifica que estás corriendo `streamlit run app.py` desde la raíz del proyecto."
    )
    st.stop()

original_sr = sf.info(str((correct_files or incorrect_files)[0])).samplerate
if cfg.sample_rate > original_sr:
    st.sidebar.warning(
        f"Los archivos están grabados a {original_sr} Hz. Subir el sample rate "
        "no agrega información, solo interpola."
    )

vista = st.sidebar.radio(
    "Vista",
    ["Vista general del dataset", "Explorar audio individual",
     "Comparar correcto vs incorrecto", "Segmentación espectral"],
)

# ---------- Vista 1: general ----------
if vista == "Vista general del dataset":
    st.header("Vista general del dataset")

    col1, col2 = st.columns(2)
    col1.metric("Audios correctos", len(correct_files))
    col2.metric("Audios incorrectos", len(incorrect_files))

    durations = {"correct": [], "incorrect": []}
    with st.spinner("Calculando Mel-spectrogramas con los parámetros actuales..."):
        for label, files in [("correct", correct_files), ("incorrect", incorrect_files)]:
            for f in files:
                _, _, mel_db = get_audio(f)
                durations[label].append(mel_db.shape[1])  # frames

    st.subheader("Distribución de duración (en frames del Mel-spectrograma)")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(durations["correct"], bins=15, alpha=0.6, label="Correcto", color="#55A868")
    ax.hist(durations["incorrect"], bins=15, alpha=0.6, label="Incorrecto", color="#C44E52")
    ax.set_xlabel("Frames temporales")
    ax.set_ylabel("Número de audios")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Estadísticas por clase")
    rows = []
    for label in ["correct", "incorrect"]:
        arr = np.array(durations[label])
        rows.append({
            "Clase": label,
            "Media (frames)": round(arr.mean(), 1),
            "Desv. estándar": round(arr.std(), 1),
            "Mínimo": int(arr.min()),
            "Máximo": int(arr.max()),
        })
    st.table(rows)

# ---------- Vista 2: individual ----------
elif vista == "Explorar audio individual":
    st.header("Explorar un audio individual")

    clase = st.radio("Clase", ["correct", "incorrect"], horizontal=True)
    files = correct_files if clase == "correct" else incorrect_files
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
    st.header("Comparar pronunciación correcta vs incorrecta")

    col1, col2 = st.columns(2)
    with col1:
        nombres_c = [f.name for f in correct_files]
        sel_c = st.selectbox("Audio correcto", nombres_c, key="sel_correct")
        path_c = correct_files[nombres_c.index(sel_c)]
        wf_c, sr_c, mel_c = get_audio(path_c)
        st.audio(str(path_c))
        st.pyplot(plot_mel(mel_c, cfg, title="Correcto"))
        st.caption(f"Duración: {len(wf_c)/sr_c:.2f} s")

    with col2:
        nombres_i = [f.name for f in incorrect_files]
        sel_i = st.selectbox("Audio incorrecto", nombres_i, key="sel_incorrect")
        path_i = incorrect_files[nombres_i.index(sel_i)]
        wf_i, sr_i, mel_i = get_audio(path_i)
        st.audio(str(path_i))
        st.pyplot(plot_mel(mel_i, cfg, title="Incorrecto"))
        st.caption(f"Duración: {len(wf_i)/sr_i:.2f} s")

# ---------- Vista 4: segmentación ----------
else:
    st.header("Segmentación espectral por distancia coseno")
    st.caption(
        "Fronteras entre unidades fonéticas detectadas sin transcripción: picos de la "
        "distancia coseno entre ventanas antes y después de cada instante que superan un umbral relativo a cada audio."
    )

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

    seg_cfg = SegmentationConfig(
        silence_db=float(st.session_state.silence_db),
        context_ms=float(st.session_state.context_ms),
        smoothing_ms=float(st.session_state.smoothing_ms),
        threshold_k=float(st.session_state.threshold_k),
        min_novelty=float(st.session_state.min_novelty),
        min_segment_ms=float(st.session_state.min_segment_ms),
    )

    clase = st.radio("Clase", ["correct", "incorrect"], horizontal=True, key="seg_clase")
    files = correct_files if clase == "correct" else incorrect_files
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

    st.subheader("Análisis de todo el dataset")
    st.caption(
        f"Sin transcripciones, la calidad se estima con indicadores indirectos: la duración "
        f"de los segmentos debería caer mayormente en el rango típico de un fonema "
        f"({PLAUSIBLE_RANGE_MS[0]:.0f}-{PLAUSIBLE_RANGE_MS[1]:.0f} ms)."
    )
    # Un botón solo vale True en la ejecución en que se presiona, así que los
    # resultados se guardan en session_state junto con los parámetros usados;
    # si los parámetros cambian, no se muestran resultados desactualizados.
    params_key = (cfg.sample_rate, cfg.n_mels, cfg.win_length_ms, cfg.overlap, astuple(seg_cfg))

    if st.button("Analizar los 200 audios con estos parámetros", type="primary"):
        all_durations = []
        counts = {"correct": [], "incorrect": []}
        with st.spinner("Segmentando el dataset..."):
            for label, lista in [("correct", correct_files), ("incorrect", incorrect_files)]:
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
        st.info("Los parámetros cambiaron desde el último análisis. "
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
            bins = np.arange(1, max(max(counts["correct"]), max(counts["incorrect"])) + 2) - 0.5
            ax.hist(counts["correct"], bins=bins, alpha=0.6, label="Correcto", color="#55A868")
            ax.hist(counts["incorrect"], bins=bins, alpha=0.6, label="Incorrecto", color="#C44E52")
            ax.set_xlabel("Segmentos por audio")
            ax.set_ylabel("Número de audios")
            ax.set_title("Segmentos por audio, por clase")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)

        st.table([
            {"Clase": label,
             "Segmentos promedio": round(float(np.mean(counts[label])), 2),
             "Mínimo": int(np.min(counts[label])),
             "Máximo": int(np.max(counts[label]))}
            for label in ["correct", "incorrect"]
        ])