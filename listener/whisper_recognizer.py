"""
Whisper-based implementation of SpeechRecognizer.

Replaces the fragile RMS-based VAD with SileroVAD for robust
end-of-utterance detection in noisy environments.

BioRobotics Laboratory, Facultad de Ingenieria, UNAM
"""

from typing import Optional

import numpy as np

from listener.audio_stream import AudioStream
from listener.speech_recognizer import SpeechRecognizer
from listener.vad import VoiceActivityDetector, SileroVAD

_FRAMES_PER_BUFFER = 4096
_SILENCE_DURATION_S = 1.5  # sustained non-speech → end of utterance
_MAX_RECORD_S = 20.0


class WhisperRecognizer(SpeechRecognizer):
    """
    Whisper-based SpeechRecognizer with neural VAD.

    Uses SileroVAD to detect utterance boundaries robustly,
    then transcribes the full buffer with faster-whisper.

    Do not instantiate directly — use WhisperRecognizer.from_parameters().
    """

    def __init__(
        self,
        model,
        vad: VoiceActivityDetector,
        sample_rate: int,
        language: Optional[str],
        grammar_rules: Optional[list],
        logger=None,
    ) -> None:
        self._model = model
        self._vad = vad
        self._sample_rate = sample_rate
        self._language = language
        self._grammar_rules = grammar_rules
        self._logger = logger

        frames_per_second = sample_rate / _FRAMES_PER_BUFFER
        self._silence_frames = int(_SILENCE_DURATION_S * frames_per_second)
        self._max_frames = int(_MAX_RECORD_S * frames_per_second)

    # ──────────────────────────────────────────────────────────────────────────
    # Factory Method
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_parameters(
        cls,
        model,
        sample_rate: int,
        language: Optional[str] = None,
        grammar_rules: Optional[list] = None,
        vad: Optional[VoiceActivityDetector] = None,
        logger=None,
    ) -> "WhisperRecognizer":
        """
        Factory Method: build a WhisperRecognizer.

        Creates a SileroVAD automatically if none is provided,
        following the same convention as NoiseReduceFilter in AudioStream.

        Args:
            model:         Loaded faster_whisper.WhisperModel.
            sample_rate:   Audio capture rate in Hz.
            language:      BCP-47 code ("en", "es"), or None for auto-detect.
            grammar_rules: Optional allowlist to bias transcription.
            vad:           VoiceActivityDetector instance, or None to use Silero.
            logger:        ROS2 logger (or None).
        """
        resolved_vad = vad or SileroVAD.from_parameters(sample_rate, logger)

        if logger:
            lang_label = language or "auto-detect"
            grammar_label = "with grammar hint" if grammar_rules else "free-form"
            logger.info(
                f"WhisperRecognizer ready — language={lang_label}, {grammar_label}."
            )

        return cls(model, resolved_vad, sample_rate, language, grammar_rules, logger)

    # ──────────────────────────────────────────────────────────────────────────
    # SpeechRecognizer contract
    # ──────────────────────────────────────────────────────────────────────────

    def recognize(self, audio_stream: AudioStream) -> str:
        """Block until a complete utterance is detected and transcribed."""
        import rclpy

        if self._logger:
            self._logger.info("Listening (Whisper + Silero VAD)...")

        pcm_buffer = self._collect_audio(audio_stream)
        if pcm_buffer is None:
            return ""

        text = self._transcribe(pcm_buffer)
        text = self._filter_grammar(text)

        if self._logger:
            self._logger.info(f"Recognised: {text!r}")

        return text

    # ──────────────────────────────────────────────────────────────────────────
    # Template Method steps
    # ──────────────────────────────────────────────────────────────────────────

    def _collect_audio(self, audio_stream: AudioStream) -> Optional[np.ndarray]:
        """
        Accumulate frames using SileroVAD to detect utterance boundaries.

        Waits for speech to start, then records until silence is
        sustained for _SILENCE_DURATION_S seconds.
        """
        import rclpy

        frames: list[bytes] = []
        consecutive_silent = 0
        speech_started = False

        while rclpy.ok() and len(frames) < self._max_frames:
            frame = audio_stream.read()
            is_speech = self._vad.is_speech(frame)

            if is_speech:
                speech_started = True
                consecutive_silent = 0
                frames.append(frame)
            elif speech_started:
                consecutive_silent += 1
                frames.append(frame)  # keep tail for natural sentence endings

                if consecutive_silent >= self._silence_frames:
                    break

        if not frames:
            return None

        raw = b"".join(frames)
        int16 = np.frombuffer(raw, dtype=np.int16)
        return int16.astype(np.float32) / 32768.0

    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe with faster-whisper."""
        initial_prompt = ", ".join(self._grammar_rules) if self._grammar_rules else None
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            initial_prompt=initial_prompt,
            vad_filter=True,
            beam_size=5,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def _filter_grammar(self, text: str) -> str:
        """Discard output not matching any grammar rule (if rules are set)."""
        if not self._grammar_rules or not text:
            return text

        lower = text.lower()
        if any(rule.lower() in lower for rule in self._grammar_rules):
            return text

        if self._logger:
            self._logger.info(f"Not in grammar, discarding: {text!r}")
        return ""
