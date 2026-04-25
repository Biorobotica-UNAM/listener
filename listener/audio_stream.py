"""
Facade over PyAudio for managing the audio input stream.

Responsibility: own the full lifecycle of the PyAudio stream
(open, read, restart, close) so no other class needs to know
about PyAudio internals.

Design patterns applied:
    Facade     (GoF Structural) — hides PyAudio behind a clean interface,
                                  mirroring ClipsEngine hiding clips.Environment.

Principles applied:
    SRP  — one class, one reason to change: audio I/O.
    DIP  — consumers depend on AudioStream, not on pyaudio directly.

BioRobotics Laboratory, Facultad de Ingenieria, UNAM
"""

import pyaudio

from listener.audio_filter import AudioFilter, PassThroughFilter


# ============================================================
# Constants
# ============================================================

_FORMAT = pyaudio.paInt16
_CHANNELS = 1
_FRAMES_PER_BUFFER = 4096


# ============================================================
# Facade
# ============================================================


class AudioStream:
    """
    Facade over PyAudio.

    Manages a single mono int16 capture stream with optional
    audio filtering (Strategy pattern via AudioFilter).

    Public API:
        read()   -> bytes   read one frame, apply filter
        restart()           close and reopen the stream
        close()             release all PyAudio resources
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        audio_filter: AudioFilter | None = None,
        device_index: int | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._filter: AudioFilter = audio_filter or PassThroughFilter()
        self._device_index = device_index
        self._audio: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None
        self._open()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def read(self) -> bytes:
        """
        Read one frame from the microphone and apply the audio filter.

        Returns:
            Filtered PCM bytes (int16, mono).
        """
        raw = self._stream.read(_FRAMES_PER_BUFFER, exception_on_overflow=False)
        return self._filter.filter(raw)

    def restart(self) -> None:
        """Close the current stream and open a fresh one."""
        self.close()
        self._open()

    def close(self) -> None:
        """Release all PyAudio resources."""
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._audio is not None:
            self._audio.terminate()
            self._audio = None

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _open(self) -> None:
        """Open a new PyAudio stream."""
        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            format=_FORMAT,
            channels=_CHANNELS,
            rate=self._sample_rate,
            input=True,
            frames_per_buffer=_FRAMES_PER_BUFFER,
            input_device_index=self._device_index,
        )
