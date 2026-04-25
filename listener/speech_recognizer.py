"""
Abstraction and Vosk implementation for speech recognition.

Responsibility: define the contract for ASR engines and provide
the Vosk-based implementation.

Design patterns applied:
    Strategy   (GoF Behavioral) — the recognition algorithm is swappable
                                  (Vosk today, Whisper tomorrow) without
                                  touching the ROS2 node.
    Factory Method (GoF Creational) — VoskRecognizer.from_parameters()
                                  centralises construction logic so the
                                  caller does not need to know about
                                  model paths or grammar loading.

Principles applied:
    OCP  — new ASR engines are added by subclassing, not by modifying.
    DIP  — the ROS2 node depends on SpeechRecognizer, not on Vosk.
    SRP  — grammar loading, model loading, and recognition are each
           handled in focused, well-named methods.

BioRobotics Laboratory, Facultad de Ingenieria, UNAM
"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from vosk import KaldiRecognizer, Model

from listener.audio_stream import AudioStream


# ============================================================
# Abstract Strategy
# ============================================================


class SpeechRecognizer(ABC):
    """
    Contract for all speech recognition implementations.

    A recognizer receives audio frames one at a time and returns
    the recognised text when it detects a complete utterance.
    """

    @abstractmethod
    def recognize(self, audio_stream: AudioStream) -> str:
        """
        Block until a high-confidence utterance is recognised.

        Args:
            audio_stream: Source of audio frames.

        Returns:
            Recognised text, or empty string if recognition failed.
        """


# ============================================================
# Concrete Strategy — Vosk
# ============================================================


class VoskRecognizer(SpeechRecognizer):
    """
    Vosk-based implementation of SpeechRecognizer.

    Encapsulates model loading, grammar loading, recognizer
    creation, and the confidence-gated recognition loop.

    Do not instantiate directly — use VoskRecognizer.from_parameters().
    """

    _MIN_CONFIDENCE = 0.8

    def __init__(self, recognizer: KaldiRecognizer, logger=None) -> None:
        self._recognizer = recognizer
        self._logger = logger

    # ──────────────────────────────────────────────────────────────────────────
    # Factory Method
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_parameters(
        cls,
        model: Model,
        sample_rate: int,
        grammar_rules: Optional[list] = None,
        logger=None,
    ) -> "VoskRecognizer":
        """
        Factory Method: build a VoskRecognizer from its dependencies.

        Centralises recognizer construction so the ROS2 node stays clean.

        Args:
            model:        Loaded Vosk Model object.
            sample_rate:  Audio capture rate in Hz.
            grammar_rules: Optional list of allowed words/phrases.
            logger:       ROS2 logger (or None).

        Returns:
            A fully configured VoskRecognizer instance.
        """
        if grammar_rules:
            kaldi = KaldiRecognizer(model, sample_rate, json.dumps(grammar_rules))
            if logger:
                logger.info("Using grammar-constrained recognizer.")
        else:
            kaldi = KaldiRecognizer(model, sample_rate)
            if logger:
                logger.info("Using free-form recognizer.")

        kaldi.SetWords(True)
        return cls(kaldi, logger)

    # ──────────────────────────────────────────────────────────────────────────
    # SpeechRecognizer contract
    # ──────────────────────────────────────────────────────────────────────────

    def recognize(self, audio_stream: AudioStream) -> str:
        """
        Block until a high-confidence utterance is detected.

        Frames with minimum word-confidence below the threshold
        are logged and discarded.
        """
        import rclpy

        if self._logger:
            self._logger.info("Listening...")

        while rclpy.ok():
            frame = audio_stream.read()
            if not self._recognizer.AcceptWaveform(frame):
                continue

            result = json.loads(self._recognizer.Result())
            text = result.get("text", "")
            if not text:
                continue

            confidences = [w.get("conf", 0.0) for w in result.get("result", [])]
            if confidences and min(confidences) > self._MIN_CONFIDENCE:
                if self._logger:
                    self._logger.info(f"Recognised (conf={confidences}): {text}")
                return text

            if self._logger:
                self._logger.info(
                    f"Low confidence (conf={confidences}), discarding: {text}"
                )

        return ""


# ============================================================
# Model Loader (pure function — no state needed)
# ============================================================


def load_vosk_model(
    package_share_dir: str, model_path: str, logger=None
) -> Optional[Model]:
    """
    Locate and load a Vosk model from the ROS2 package share directory.

    Extracted from the node to keep loading logic testable in isolation.

    Args:
        package_share_dir: Absolute path to the ROS2 package share directory.
        model_path:        Relative path to the model inside the share dir.
        logger:            ROS2 logger (or None).

    Returns:
        A loaded Vosk Model, or None on failure.
    """
    full_path = Path(package_share_dir) / model_path

    if not full_path.exists():
        if logger:
            logger.error(f"Model not found at: {full_path}")
        return None

    try:
        if logger:
            logger.info(f"Loading Vosk model from: {full_path}")
        return Model(str(full_path))
    except Exception as exc:
        if logger:
            logger.error(f"Failed to load model: {exc}")
        return None


# ============================================================
# Grammar Loader (pure function — no state needed)
# ============================================================


def load_grammar(
    package_share_dir: str, grammar_filename: str, logger=None
) -> Optional[list]:
    """
    Load a JSON grammar file from the package share directory.

    Args:
        package_share_dir: Absolute path to the ROS2 package share directory.
        grammar_filename:  Filename inside <share>/grammar/.
        logger:            ROS2 logger (or None).

    Returns:
        Parsed grammar list, or None if not found / invalid.
    """
    grammar_path = Path(package_share_dir) / "grammar" / grammar_filename

    if not grammar_path.is_file():
        if logger:
            logger.warning(f"Grammar file not found: {grammar_path}")
        return None

    try:
        with open(grammar_path, "r", encoding="utf-8") as f:
            grammar = json.load(f)
        if logger:
            logger.info(f"Grammar loaded: {grammar_path}")
        return grammar
    except Exception as exc:
        if logger:
            logger.warning(f"Failed to load grammar: {exc}")
        return None
