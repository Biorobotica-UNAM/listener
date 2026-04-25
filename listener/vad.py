"""
Voice Activity Detection abstraction and Silero implementation.

Responsibility: detect whether a PCM frame contains speech,
so the recognizer knows when to start and stop recording.

Design patterns applied:
    Strategy       (GoF Behavioral)  — VAD algorithm is swappable.
    Null Object    (GoF Behavioral)  — AlwaysVoiceVAD avoids None checks.
    Factory Method (GoF Creational)  — SileroVAD.from_parameters().

Principles applied:
    SRP  — detection logic isolated from accumulation logic.
    OCP  — new VAD backends added by subclassing.
    DIP  — WhisperRecognizer depends on VoiceActivityDetector, not Silero.

BioRobotics Laboratory, Facultad de Ingenieria, UNAM
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# ============================================================
# Abstract Strategy
# ============================================================


class VoiceActivityDetector(ABC):
    """Contract for all VAD implementations."""

    @abstractmethod
    def is_speech(self, pcm_bytes: bytes) -> bool:
        """
        Return True if the frame contains speech.

        Args:
            pcm_bytes: Raw PCM bytes (int16, mono).

        Returns:
            True if speech detected, False otherwise.
        """


# ============================================================
# Concrete Strategy — Silero VAD
# ============================================================


class SileroVAD(VoiceActivityDetector):
    """
    Neural VAD using Silero VAD v5 API.
    """

    _SILERO_SAMPLE_RATE = 16000

    def __init__(self, model, sample_rate: int) -> None:
        self._model = model
        self._input_rate = sample_rate

    @classmethod
    def from_parameters(cls, sample_rate: int, logger=None) -> "SileroVAD":
        """
        Factory Method: load Silero VAD using the v5 API.
        """
        import torch

        if logger:
            logger.info("Loading Silero VAD model...")

        # v5 API: el modelo se importa directamente, no via utils
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        model.eval()

        if logger:
            logger.info("Silero VAD ready.")

        return cls(model, sample_rate)

    def is_speech(self, pcm_bytes: bytes) -> bool:
        """
        Return True if Silero detects speech in any 512-sample chunk.

        Silero v5 requires exactly 512 samples at 16kHz per inference call.
        We split the resampled frame into fixed windows and return True
        if any window exceeds the confidence threshold.
        """
        import torch

        _SILERO_CHUNK = 512  # samples requeridos por Silero a 16kHz

        int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        float32 = int16.astype(np.float32) / 32768.0

        if self._input_rate != self._SILERO_SAMPLE_RATE:
            float32 = self._resample(
                float32, self._input_rate, self._SILERO_SAMPLE_RATE
            )

        # Dividir en chunks exactos de 512 — descartar el residuo final
        num_chunks = len(float32) // _SILERO_CHUNK
        for i in range(num_chunks):
            chunk = float32[i * _SILERO_CHUNK : (i + 1) * _SILERO_CHUNK]
            tensor = torch.from_numpy(chunk).unsqueeze(0)  # [1, 512]

            with torch.no_grad():
                confidence = self._model(tensor, self._SILERO_SAMPLE_RATE).item()

            if confidence > 0.5:
                return True

        return False

    @staticmethod
    def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        ratio = to_rate / from_rate
        new_len = int(len(audio) * ratio)
        return np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)


# ============================================================
# Null Object — always reports speech (useful for testing)
# ============================================================


class AlwaysVoiceVAD(VoiceActivityDetector):
    """
    VAD that always reports speech.
    Useful for unit tests or environments with
    controlled audio, avoiding None checks.
    """

    def is_speech(self, pcm_bytes: bytes) -> bool:
        return True
