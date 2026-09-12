# Sources

- Sipeed MA-USB8 user guide: <https://wiki.sipeed.com/hardware/en/modules/micarray_usbboard_bl616.html>
- Sipeed MicArray overview: <https://wiki.sipeed.com/hardware/en/modules/micarray.html>
- ROS4HRI standard: <https://ros4hri.github.io/standard.html>
- REP-155: <https://www.ros.org/reps/rep-0155.html>
- ODAS: <https://github.com/introlab/odas>
- Hugging Face speech-to-speech: <https://github.com/huggingface/speech-to-speech>

Notes reflected in this repository:

- Sipeed MA-USB8 is treated as a UAC2 device exposing 8ch S16_LE 48 kHz input.
- CH0-CH5 are treated as raw outer-ring channels for DOA/beamforming.
- CH6 is treated as the board beam/average output for debugging, not the primary array input.
- CH7 is treated as a raw/reference candidate and must be verified during channel calibration.
- The serial sound-field hotmap can be used as an auxiliary DOA signal, but this repo keeps raw PCM as the primary portable interface.
