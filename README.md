# listener

This repository provides a ROS 2 node for **offline speech recognition**, supporting both **Vosk** and **faster-whisper** (with **Silero VAD**) as interchangeable ASR engines, with real-time **noise reduction** and a **service-based interface**.

## Architecture

The package follows Clean Code, SOLID principles, and GoF design patterns:

- **Strategy** — `SpeechRecognizer`, `AudioFilter`, and `VoiceActivityDetector` are swappable at runtime without modifying consumers.
- **Factory Method** — `VoskRecognizer.from_parameters()` and `WhisperRecognizer.from_parameters()` centralise construction.
- **Facade** — `AudioStream` hides PyAudio internals behind a clean interface.
- **Null Object** — `PassThroughFilter` and `AlwaysVoiceVAD` eliminate None checks.

```
AudioStream (PyAudio Facade)
    │
    ├── AudioFilter (Strategy)
    │       ├── NoiseReduceFilter   ← noisereduce
    │       └── PassThroughFilter   ← Null Object
    │
    └── SpeechRecognizer (Strategy)
            ├── VoskRecognizer      ← Vosk + KaldiRecognizer
            └── WhisperRecognizer   ← faster-whisper + Silero VAD
                    └── VoiceActivityDetector (Strategy)
                            ├── SileroVAD       ← neural VAD (torch.hub)
                            └── AlwaysVoiceVAD  ← Null Object
```

## Description

The `listener_node` exposes a ROS 2 service that, when enabled, opens the microphone, listens until it detects a valid utterance, and returns the recognized text.

Main features:
- **Offline** speech recognition (Vosk or faster-whisper)
- Audio capture using **PyAudio**
- Per-frame **noise reduction** using `noisereduce`
- **Neural VAD** using Silero VAD (Whisper engine only)
- Optional **dynamic grammar** support (JSON)
- ROS 2 service interface (`listener_interfaces/srv/Listen`)
- **Selectable ASR engine** via ROS parameter

## Dependencies

### System packages
```bash
sudo apt install portaudio19-dev
```

### Python packages

**Common (both engines):**
```bash
pip install pyaudio noisereduce numpy
```

**Vosk engine:**
```bash
pip install vosk
```

**Whisper engine:**
```bash
pip install faster-whisper torch
```

> `faster-whisper` downloads the selected Whisper model automatically from Hugging Face on first use. No FFmpeg installation required.

> Silero VAD is loaded automatically via `torch.hub` on first use and cached at `~/.cache/torch/hub/`.

**GPU acceleration (optional, NVIDIA only):**

For CUDA 12 + cuDNN 9:
```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
export LD_LIBRARY_PATH=$(python3 -c \
  'import os, nvidia.cublas.lib, nvidia.cudnn.lib; \
   print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + \
         os.path.dirname(nvidia.cudnn.lib.__file__))')
```

For CUDA 11 + cuDNN 8, pin ctranslate2:
```bash
pip install --force-reinstall ctranslate2==3.24.0
```

### Summary table

| Package         | Required for     | Notes                                      |
|-----------------|------------------|--------------------------------------------|
| `pyaudio`       | Both             | Requires `portaudio19-dev` system package  |
| `noisereduce`   | Both             | Applied per-frame before recognition       |
| `numpy`         | Both             | Audio buffer manipulation                  |
| `vosk`          | Vosk engine      | Requires model downloaded separately       |
| `faster-whisper`| Whisper engine   | Downloads model from HuggingFace on first run |
| `torch`         | Whisper engine   | Required by Silero VAD via torch.hub       |

## Behaviour

When the service is called:

**If `enable = true`:**
1. (Optional) Loads a grammar from a JSON file.
2. Restarts the audio stream to flush stale buffers.
3. Builds the configured recognizer (Vosk or Whisper).
4. Starts listening from the microphone, applying the noise filter per frame.
5. **Vosk path:** streams frames until `AcceptWaveform` returns True, then returns the result if minimum word confidence > 0.8.
6. **Whisper path:** uses Silero VAD to detect utterance start and end, accumulates the buffer, then transcribes with faster-whisper.
7. Returns the recognized text.

**If `enable = false`:**
- Does not listen and returns an empty response.

The service returns:
- `success = true` (as long as the node is running)
- `recognized_speech` containing the recognized text (or `""` if nothing valid is detected)

## Interfaces

### Service Server

Service:
- `listen` (default) [`listener_interfaces::srv::Listen`]

Request:
- `enable` [`bool`] → enables or disables listening
- `grammar` [`string`] → grammar file name (optional, `.json`)

Response:
- `success` [`bool`]
- `recognized_speech` [`string`]

## ROS Parameters

| Parameter            | Default    | Description                                                                 |
|----------------------|------------|-----------------------------------------------------------------------------|
| `asr_engine`         | `"vosk"`   | ASR backend: `"vosk"` or `"whisper"`                                        |
| `sample_rate`        | `16000`    | Microphone capture rate in Hz. Use 16000 for Whisper (native rate)          |
| `model_path`         | `"model"`  | Vosk only — relative path to model inside the package share directory       |
| `whisper_model_size` | `"small"`  | Whisper only — `"tiny"`, `"base"`, `"small"`, `"medium"`, `"large-v3"`     |
| `whisper_language`   | `"en"`     | Whisper only — BCP-47 code (`"en"`, `"es"`). Empty string = auto-detect     |
| `listener_service`   | `"listen"` | Name of the ROS 2 service                                                   |

### Recommended settings by hardware

| Hardware              | `asr_engine` | `whisper_model_size` | Expected latency |
|-----------------------|--------------|----------------------|------------------|
| CPU only              | `whisper`    | `small`              | ~2–3 s           |
| NVIDIA GPU (≥4GB)     | `whisper`    | `medium`             | ~0.5–1 s         |
| Jetson / edge GPU     | `whisper`    | `small`              | ~2 s             |
| Any (grammar-bounded) | `vosk`       | N/A                  | <0.5 s           |

## Grammar files

If `request.grammar` is provided, the node will attempt to load a JSON file from:
```
<listener_share>/grammar/<request.grammar>
```

- **Vosk:** grammar is passed directly to `KaldiRecognizer` as a hard FST constraint.
- **Whisper:** grammar phrases are passed as `initial_prompt` to bias transcription (soft constraint), and the result is post-filtered against the allowlist.

If the file does not exist or parsing fails, a warning is logged and the node continues in free-form recognition mode.

## Usage examples

### CLI
```bash
# Vosk (default)
ros2 service call /listen listener_interfaces/srv/Listen "{enable: true, grammar: ''}"

# With grammar file
ros2 service call /listen listener_interfaces/srv/Listen "{enable: true, grammar: 'commands.json'}"
```

### Python client
```python
import rclpy
from rclpy.node import Node
from listener_interfaces.srv import Listen

class ListenerClient(Node):
    def __init__(self):
        super().__init__("listener_client")
        self.cli = self.create_client(Listen, "listen")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for listen service...")

    def call(self, grammar: str = "") -> str:
        req = Listen.Request()
        req.enable = True
        req.grammar = grammar
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().recognized_speech

def main():
    rclpy.init()
    node = ListenerClient()
    text = node.call()
    print("Recognised:", text)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
```

## Notes

- `frames_per_buffer = 4096`, `channels = 1` (mono).
- Silero VAD processes the audio in **512-sample chunks** at 16 kHz internally — this is handled automatically regardless of the configured `sample_rate`.
- Setting `sample_rate = 16000` is strongly recommended when using the Whisper engine — it matches Whisper and Silero's native rate, eliminating resampling overhead.
- The `pynvml` deprecation warning on startup is harmless; install `nvidia-ml-py` to suppress it: `pip install nvidia-ml-py`.
- ALSA and JACK warnings at startup are benign — they reflect unused audio backends being probed by PyAudio.

## Shutdown

When the node is closed (Ctrl+C):
- The audio stream is stopped and PyAudio is properly terminated.
- The node is destroyed and `rclpy.shutdown()` is called.
