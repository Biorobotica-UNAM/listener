"""
ROS2 node for offline speech recognition (ASR).

Responsibility: expose a ROS2 service and coordinate the ASR
pipeline. Contains no audio or recognition logic — delegates
everything to AudioStream, AudioFilter, and SpeechRecognizer.

Service exposed:
    <listener_service>   (listener_interfaces/Listen)

Design patterns applied:
    Facade       (GoF Structural)   — the node hides the ASR pipeline
                                      behind a single service call.
    Strategy     (GoF Behavioral)   — AudioFilter and SpeechRecognizer
                                      are swappable without modifying
                                      this node.
    Factory Method (GoF Creational) — VoskRecognizer.from_parameters()
                                      builds the recognizer; the node
                                      only asks for one.
    Null Object  (GoF Behavioral)   — PassThroughFilter avoids None
                                      checks for the no-filter case.

Principles applied:
    SRP  — this class handles ROS2 communication only.
    DIP  — depends on SpeechRecognizer and AudioStream abstractions,
           not on Vosk or PyAudio directly.

BioRobotics Laboratory, Facultad de Ingenieria, UNAM
"""

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from listener_interfaces.srv import Listen

from listener.audio_filter import NoiseReduceFilter
from listener.audio_stream import AudioStream
from listener.speech_recognizer import (
    VoskRecognizer,
    load_grammar,
    load_vosk_model,
)


class ListenerNode(Node):
    """
    ROS2 node that exposes the ASR pipeline as a service.

    Responsibility: handle ROS2 communication and delegate
    every audio and recognition task to dedicated components.

    This class contains no Vosk or PyAudio logic.
    """

    def __init__(self) -> None:
        super().__init__("listener_node")

        self._sample_rate = self.declare_parameter("sample_rate", 48000).value
        self._model_path = self.declare_parameter("model_path", "model").value
        self._service_name = self.declare_parameter("listener_service", "listen").value

        self._package_share_dir = self._resolve_package_share()
        self._model = self._load_model()
        self._audio_stream = self._build_audio_stream()

        self._srv = self.create_service(Listen, self._service_name, self._handle_listen)
        self.get_logger().info(f"ASR service ready: /{self._service_name}")

    # ──────────────────────────────────────────────────────────────────────────
    # Initialization helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_package_share(self) -> str:
        """Resolve the ROS2 package share directory path."""
        try:
            return get_package_share_directory("listener")
        except Exception as exc:
            self.get_logger().error(f"Could not resolve package share: {exc}")
            rclpy.shutdown()
            raise

    def _load_model(self):
        """Load the Vosk model; shut down if not found."""
        model = load_vosk_model(
            self._package_share_dir, self._model_path, self.get_logger()
        )
        if model is None:
            self.get_logger().error("Vosk model could not be loaded. Shutting down.")
            rclpy.shutdown()
            raise RuntimeError("Vosk model not found.")
        return model

    def _build_audio_stream(self) -> AudioStream:
        """Compose AudioStream with the noise-reduction filter."""
        audio_filter = NoiseReduceFilter(self._sample_rate)
        return AudioStream(
            sample_rate=self._sample_rate,
            audio_filter=audio_filter,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Service callback
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_listen(self, request, response):
        """
        Service callback for the Listen service.

        Restarts the audio stream on every call to flush stale
        buffers, then recognises one utterance.
        """
        if not request.enable:
            self.get_logger().info("Listener disabled by request.")
            response.success = True
            response.recognized_speech = ""
            return response

        grammar_rules = self._load_grammar_if_requested(request.grammar)
        self._audio_stream.restart()

        recognizer = VoskRecognizer.from_parameters(
            model=self._model,
            sample_rate=self._sample_rate,
            grammar_rules=grammar_rules,
            logger=self.get_logger(),
        )

        response.recognized_speech = recognizer.recognize(self._audio_stream)
        response.success = True
        return response

    def _load_grammar_if_requested(self, grammar_filename: str):
        """Load grammar rules only when a filename was provided."""
        if not grammar_filename:
            return None
        return load_grammar(
            self._package_share_dir, grammar_filename, self.get_logger()
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Shutdown
    # ──────────────────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        """Release audio resources before destroying the node."""
        self.get_logger().info("Closing audio stream...")
        self._audio_stream.close()
        super().destroy_node()


# ============================================================
# Entry point
# ============================================================


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ListenerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
