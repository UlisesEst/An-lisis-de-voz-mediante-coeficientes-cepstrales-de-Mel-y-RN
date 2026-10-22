"""
Configuración centralizada del proyecto.

Los valores por defecto de AudioConfig provienen de:
Ramos-Aguilar et al. (2025), "Automatic Phonetic Segmentation of the
Yuhmu Language Using Mel Scale Spectral Parameters".
Son el punto de partida validado para Yuhmu; para otros idiomas pueden
ajustarse desde la interfaz o desde la línea de comandos.
"""

from dataclasses import asdict, dataclass, fields


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_mels: int = 40
    win_length_ms: float = 20.0
    overlap: float = 0.75
    window_fn: str = "hann"

    @property
    def win_length(self) -> int:
        return int(self.sample_rate * self.win_length_ms / 1000)

    @property
    def hop_length(self) -> int:
        return max(1, int(round(self.win_length * (1 - self.overlap))))

    @property
    def n_fft(self) -> int:
        # siguiente potencia de 2 >= win_length
        n = 1
        while n < self.win_length:
            n *= 2
        return n

    @property
    def n_freq_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def frames_per_second(self) -> float:
        return self.sample_rate / self.hop_length

    def validate(self) -> list:
        """Lanza error si la configuración es imposible y devuelve una
        lista de advertencias si es válida pero poco recomendable."""
        if not 0 <= self.overlap < 1:
            raise ValueError("El overlap debe estar en el rango [0, 1).")
        if self.win_length < 2:
            raise ValueError("La ventana es demasiado corta para este sample rate.")

        warnings = []
        if self.n_mels > self.n_freq_bins:
            warnings.append(
                f"{self.n_mels} bandas Mel superan las {self.n_freq_bins} frecuencias "
                f"disponibles (n_fft={self.n_fft}); habrá filtros vacíos. "
                "Reduce las bandas o aumenta la ventana."
            )
        if not 10 <= self.win_length_ms <= 50:
            warnings.append(
                "Ventanas fuera de 10-50 ms son poco habituales en procesamiento de habla."
            )
        return warnings

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AudioConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class SegmentationConfig:
    """Parámetros de la segmentación espectral por distancia coseno.
    Todos los parámetros temporales están en milisegundos para que el
    comportamiento no dependa del sample rate ni del salto configurados.
    Los valores por defecto se calibraron visualmente sobre el corpus de
    Yuhmu partiendo de criterios de fonética general; para otros corpus
    pueden ajustarse desde la interfaz."""
    silence_db: float = 25.0       # frames este número de dB bajo el pico = silencio (calibrado en Yuhmu)
    context_ms: float = 20.0       # ventana antes/después de cada instante que se compara
    smoothing_ms: float = 15.0     # ancho del promedio móvil sobre la curva de novedad
    threshold_k: float = 3.0       # frontera = pico > mediana + k * MAD (regla de valores atípicos)
    min_novelty: float = 0.08      # piso absoluto de distancia coseno (evita fronteras en sonidos estables)
    min_segment_ms: float = 40.0   # separación mínima entre fronteras


@dataclass
class ModelConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.2
    max_seq_len: int = 500  # mínimo; se amplía según la duración real de los audios


@dataclass
class TrainConfig:
    batch_size: int = 8
    lr: float = 1e-3
    epochs: int = 50
    k_folds: int = 5
    seed: int = 42
    device: str = "cuda"  # se ajusta en tiempo de ejecución si no hay GPU


AUDIO = AudioConfig()
SEGMENTATION = SegmentationConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()