#!/usr/bin/env python3

import os
import json
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from vosk import Model, KaldiRecognizer
import pyaudio
import numpy as np
import noisereduce as nr

from listener_interfaces.srv import Listen


class NoiseReducer:
    """Clase para filtrar ruido con noisereduce"""

    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate

    def filter(self, pcm_bytes):
        # Convertir bytes a int16
        audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)
        # noisereduce espera float64 o float32
        audio_float = audio_data.astype(np.float32)
        # Reducir ruido usando el frame actual
        reduced = nr.reduce_noise(y=audio_float, sr=self.sample_rate)
        # Convertir de vuelta a int16
        return reduced.astype(np.int16).tobytes()


class Listener(Node):
    def __init__(self):
        super().__init__("listener_node")

        self.language = self.declare_parameter("language", "en").value
        self.model_path = self.declare_parameter("model_path", "model").value
        self.sample_rate = self.declare_parameter("sample_rate", 48000).value
        self.grammar_path = self.declare_parameter("grammar_path", "").value
        self.service_name = self.declare_parameter("listener_service", "listen").value

        self.model = self._load_model(self.model_path)
        if not self.model:
            self.get_logger().error("No se pudo cargar el modelo Vosk")
            rclpy.shutdown()
            return

        self.audio, self.stream = self._init_audio_stream(self.sample_rate)
        self.noise_reducer = NoiseReducer(self.sample_rate)

        self.srv = self.create_service(
            Listen, self.service_name, self.handle_listen_service
        )
        self.get_logger().info(
            f"Servicio de reconocimiento de voz listo: {self.service_name}"
        )

    # ------------------------- Carga de modelo -------------------------
    def _load_model(self, path: str):
        try:
            package_share_dir = get_package_share_directory("listener")
            model_full_path = os.path.join(package_share_dir, path)
        except Exception as e:
            self.get_logger().error(f"No se pudo resolver el paquete: {e}")
            return None
        if not os.path.exists(model_full_path):
            self.get_logger().error(f"Modelo no encontrado en {model_full_path}")
            return None
        try:
            self.get_logger().info(f"Cargando modelo desde: {model_full_path}")
            return Model(model_full_path)
        except Exception as e:
            self.get_logger().error(f"Error al cargar el modelo: {e}")
            return None

    # ------------------------- Inicialización de audio -------------------------
    def _init_audio_stream(self, rate: int):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            input=True,
            frames_per_buffer=4096,
            input_device_index=None,
        )
        return audio, stream

    def _restatr_audio_stream(self):
        try:
            self.get_logger().info("Reiniciando stream de audio...")

            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()

            if self.audio is not None:
                self.audio.terminate()

        except Exception as e:
            self.get_logger().warn(f"Error cerrando audio: {e}")

        self.audio, self.stream = self._init_audio_stream(self.sample_rate)

    # ------------------------- Gramática opcional -------------------------
    def _load_grammar(self, grammar_file: str):
        package_share_dir = get_package_share_directory("listener")
        full_path = os.path.join(package_share_dir, "grammar", grammar_file)
        if not os.path.isfile(full_path):
            self.get_logger().warn(f"Archivo de gramática no encontrado: {full_path}")
            return None
        try:
            with open(full_path, "r") as f:
                grammar = json.load(f)
            self.get_logger().info(f"Gramática cargada: {full_path}")
            return grammar
        except Exception as e:
            self.get_logger().warn(f"No se pudo cargar la gramática: {e}")
            return None

    # ------------------------- Recognizer -------------------------
    def _create_recognizer(self, grammar_rules=None):
        if grammar_rules:
            recognizer = KaldiRecognizer(
                self.model, self.sample_rate, json.dumps(grammar_rules)
            )
            self.get_logger().info("Usando gramática dinámica")
        else:
            recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.get_logger().info("Usando reconocimiento por defecto")
        recognizer.SetWords(True)
        return recognizer

    # ------------------------- Reconocimiento de voz -------------------------
    def _recognize_speech(self, recognizer: KaldiRecognizer) -> str:
        self.get_logger().info("Escuchando...")
        while rclpy.ok():
            raw_data = self.stream.read(4096, exception_on_overflow=False)
            # Pasar audio por el NoiseReducer
            data = self.noise_reducer.filter(raw_data)

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    confs = [w.get("conf", 0) for w in result.get("result", [])]
                    if confs and min(confs) > 0.8:
                        self.get_logger().info(
                            f"Reconocido con {confs}% de confianza: {text}"
                        )
                        return text
                    else:
                        self.get_logger().info(
                            f" {confs}% de confianza, ignorando: {text}"
                        )
        return ""

    # ------------------------- Servicio ROS -------------------------
    def handle_listen_service(self, request, response):
        if request.enable:
            grammar_rules = (
                self._load_grammar(request.grammar) if request.grammar else None
            )
            self._restatr_audio_stream()
            recognizer = self._create_recognizer(grammar_rules)
            response.recognized_speech = self._recognize_speech(recognizer)
            response.success = True
        else:
            self.get_logger().info("Listener deshabilitado")
            response.success = True
            response.recognized_speech = ""
        return response

    # ------------------------- Shutdown -------------------------
    def shutdown(self):
        self.get_logger().info("Cerrando audio...")
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()


# ------------------------- Main -------------------------
def main(args=None):
    rclpy.init(args=args)
    node = Listener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
