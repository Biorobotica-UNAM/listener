# listener
This repository provides a ROS 2 node for **offline speech recognition** using **Vosk**, with real-time **noise reduction** and a **service-based interface**.

## Description
The `listener_node` exposes a service that, when enabled, opens the microphone, listens until it detects a valid utterance, and returns the recognized text.

Main features:
- **Offline** speech recognition (Vosk)
- Audio capture using **PyAudio**
- Per-frame **noise reduction** using `noisereduce`
- Optional **dynamic grammar** support (JSON)
- ROS 2 service interface (`listener_interfaces/srv/Listen`)

## Behaviour
When the service is called:

1. If `enable = true`:
   1. (Optional) Loads a grammar from a JSON file.
   2. Creates a `KaldiRecognizer` with or without grammar.
   3. Starts listening from the microphone.
   4. Filters the audio using noise reduction.
   5. Waits until Vosk accepts a complete utterance (`AcceptWaveform`).
   6. If the recognized text has **minimum confidence > 0.8**, it returns it.
   7. If confidence is low, it ignores the result and keeps listening.

2. If `enable = false`:
   - It does not listen and returns an empty response.

The service returns:
- `success = true` (as long as the node is running)
- `recognized_speech` containing the recognized text (or `""` if nothing valid is detected)

## Noise filtering
The node includes an internal `NoiseReducer` class that:
1. Converts PCM audio (`int16`) into `float32`
2. Applies `nr.reduce_noise(...)`
3. Converts the audio back to `int16`

This is applied **on every audio frame** read from the stream.

## Interfaces
### Service Server
Service:
- `listen` (default) [`listener_interfaces::srv::Listen`]

Request:
- `enable` [`bool`] → enables or disables listening
- `grammar` [`string`] → grammar file name (optional)

Response:
- `success` [`bool`]
- `recognized_speech` [`string`]

## ROS Parameters
The node uses the following parameters:

- `language` (default: `"en"`)  
  *Currently not used directly in the node logic, but kept as a configurable parameter.*

- `model_path` (default: `"model"`)  
  Relative path to the Vosk model inside the `listener` package share directory.

- `sample_rate` (default: `48000`)  
  Microphone sample rate.

- `grammar_path` (default: `""`)  
  *Currently not used directly.*  
  Grammar files are searched in:  
  `share/listener/grammar/<file.json>`

- `listener_service` (default: `"listen"`)  
  Name of the ROS service to be created.

## Grammar files
If `request.grammar` is provided, the node will attempt to load a JSON file from:
```php-template
<listener_share>/grammar/<request.grammar>
```

If the file does not exist or parsing fails:
- A warning is logged
- The node continues in normal recognition mode (no grammar)

## Usage example
Calling from CLI (typical example):

```bash
ros2 service call /listen listener_interfaces/srv/Listen "{enable: true, grammar: ''}"
```

## Usage example
```python

import rclpy
from rclpy.node import Node
from listener_interfaces.srv import Listen

class TestClient(Node):
    def __init__(self):
        super().__init__("listener_test_client")
        self.cli = self.create_client(Listen, "listen")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for listen service...")

    def call(self):
        req = Listen.Request()
        req.enable = True
        req.grammar = ""  # or "commands.json"
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

def main():
    rclpy.init()
    node = TestClient()
    res = node.call()
    print("success:", res.success)
    print("recognized_speech:", res.recognized_speech)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
```

## Important notes
* Recognition depends on the microphone being available and correctly configured.
* The node uses `frames_per_buffer=4096` and `channels=1`.
* The acceptance threshold is conservative: **it only returns text if the minimum word confidence is above 0.8**.
* Applying noise reduction frame-by-frame can increase CPU usage, but improves robustness in noisy environments.

## Shutdown
When the node is closed (Ctrl+C):
* The audio stream is stopped
* PyAudio is properly terminated
* The node is destroyed and `rclpy.shutdown()` is called
