"""
Loader for faster-whisper models.

Extracted as a pure function — no state, fully testable in isolation —
mirroring load_vosk_model() in speech_recognizer.py.

BioRobotics Laboratory, Facultad de Ingenieria, UNAM
"""

from typing import Optional


def load_whisper_model(
    model_size: str = "small",
    device: str = "cuda",
    compute_type: str = "float16",  # "int8" for CPU, "float16" for GPU, "float32" for max precision
    logger=None,
):
    """
    Load a faster-whisper model.

    Args:
        model_size:    "tiny", "base", "small", "medium", "large-v3".
        device:        "cpu" or "cuda".
        compute_type:  "int8" (CPU-friendly), "float16" (GPU), "float32".
        logger:        ROS2 logger (or None).

    Returns:
        A loaded WhisperModel instance, or None on failure.
    """
    try:
        from faster_whisper import WhisperModel
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            if logger:
                logger.warn("CUDA requested but not available, falling back to CPU")
            device = "cpu"
            compute_type = "int8"

        if logger:
            logger.info(
                f"Loading Whisper model: size={model_size}, "
                f"device={device}, compute_type={compute_type}"
            )

        return WhisperModel(model_size, device=device, compute_type=compute_type)

    except Exception as exc:
        if logger:
            logger.error(f"Failed to load Whisper model: {exc}")
        return None
