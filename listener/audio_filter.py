"""
Abstraction layer for audio filtering.

Responsibility: define the contract that any audio filter must fulfil,
so the rest of the system depends on an abstraction and not on a
concrete library (noisereduce, webrtcvad, etc.).

Design patterns applied:
    Strategy   (GoF Behavioral) — the filter algorithm is interchangeable
                                  at runtime without modifying the consumer.

Principles applied:
    OCP  — new filter implementations can be added without touching
           existing code.
    DIP  — high-level modules (AudioStream, Listener) depend on
           AudioFilter, not on noisereduce directly.

BioRobotics Laboratory, Facultad de Ingenieria, UNAM
"""

from abc import ABC, abstractmethod


# ============================================================
# Abstract Strategy
# ============================================================


class AudioFilter(ABC):
    """
    Contract for all audio filter implementations.

    Receives raw PCM bytes (int16, mono) and returns filtered
    PCM bytes in the same format.
    """

    @abstractmethod
    def filter(self, pcm_bytes: bytes) -> bytes:
        """
        Apply the filter to a raw PCM frame.

        Args:
            pcm_bytes: Raw audio bytes (int16, mono).

        Returns:
            Filtered audio bytes (int16, mono).
        """


# ============================================================
# Concrete Strategy — noisereduce
# ============================================================


class NoiseReduceFilter(AudioFilter):
    """
    Concrete filter that uses the noisereduce library.

    This is the only class that imports noisereduce.
    Swapping the library only requires adding a new subclass here.
    """

    def __init__(self, sample_rate: int = 48000) -> None:
        self._sample_rate = sample_rate

    def filter(self, pcm_bytes: bytes) -> bytes:
        import numpy as np
        import noisereduce as nr

        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32)
        reduced = nr.reduce_noise(y=audio_float, sr=self._sample_rate)
        return reduced.astype(np.int16).tobytes()


# ============================================================
# Null Object — no filtering
# ============================================================


class PassThroughFilter(AudioFilter):
    """
    Filter that returns the audio unchanged.

    Useful for testing or when noise reduction is disabled,
    avoiding None checks in consuming code (Null Object pattern).
    """

    def filter(self, pcm_bytes: bytes) -> bytes:
        return pcm_bytes
