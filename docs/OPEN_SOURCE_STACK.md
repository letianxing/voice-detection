# 声学开源组件分层

这份清单记录声学系统的推荐开源组件，和本仓库当前默认实现的关系。

| 编号 | 模块 | 组件 | 本仓库位置 |
| --- | --- | --- | --- |
| 1 | 音频采集/设备抽象 | `sounddevice`/PortAudio、miniaudio、ALSA/WASAPI/CoreAudio | `voice_detection/io.py`、`voice_detection/cli.py`、`scripts/local_voice_dashboard.py` |
| 2 | 空间前端/鸡尾酒 | ODAS、pyroomacoustics | 默认轻量 DOA/beamforming 在 `doa.py`、`beamforming.py`；ODAS 为生产增强 |
| 3 | AEC/NS/AGC/解混响 | WebRTC APM、RNNoise、DeepFilterNet、noisereduce、nara_wpe | 默认 self-echo 与自适应 NS 在 `noise.py`、`signal_enhancement.cpp` |
| 4 | 说话人分离/声纹 | pyannote.audio、SpeechBrain ECAPA-TDNN、resemblyzer | `speaker_label` 与 ASR 测试日志预留字段 |
| 5 | 神经分离/增强 | asteroid SepFormer/MossFormer2、torchaudio ConvTasNet/DPRNN | 默认关闭，建议 overlap 高时远端 4090 调用 |
| 6 | Speech-to-speech 下游 | HF speech-to-speech、Silero VAD v5、Parakeet-TDT、FunASR/SenseVoice/faster-whisper、Qwen3-TTS/CosyVoice/GPT-SoVITS | 本仓库只提供目标音频/转写记录；S2S 在注意力门控后运行 |
| 7 | 视觉注意力/融合 | MediaPipe、YOLOv8/OpenCV、ROS4HRI | 由 `vision-detection` 和 `robot-attention-perception` 消费本仓库输出 |
| 8 | 调试可视化 | pyodas、ODAS Studio、rqt_human_radar | 本仓库提供本地 dashboard；ODAS 可视化后续接入 |

工程原则：

- raw multichannel -> DOA/beamforming -> single enhanced channel -> AEC/NS/dereverb -> VAD/ASR/S2S。
- 单通道 Mac/Windows 麦自动降级，没有可靠空间抑制，只能做 VAD、轻降噪、AEC、自适应阈值和后续 diarization。
- QuickTime 和系统录音只用于排障；阵列测试必须确认 8ch 原始输入。
