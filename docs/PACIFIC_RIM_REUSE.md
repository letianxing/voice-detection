# Pacific Rim 复用边界

本仓库只复用 `pacific-rim` 中稳定、与业务运行时无关的数据约定和算法部件，不依赖它的生成代码或通信运行时。

## 已接入

| 来源 | 当前实现 | 用途 |
| --- | --- | --- |
| `voice_service/voice/audio_script_plan.py` | `voice_detection/audio_script.py` | 可复现地回放 `AudioMsg` 场景，驱动 attention/ASR 接口测试 |
| `voice_service/msg/AudioMsg` | `voice_detection/pacific_rim_compat.py` | 旧系统和脚本事件与 `AcousticTrack` 互转 |
| `tts_service/msg/AudioFrame` | `voice_detection/audio_frame_codec.py` | TTS PCM、播放参考和 AEC 的统一 JSON/ROS2 边界 |
| `tts/stream/chi_concat.py` | `voice_detection/streaming_audio.py` | 先交叉淡化、再切固定 PCM 包，避免播放边界爆音 |
| TTS 音频发布路径 | `scripts/tts_aec_reference_bridge.py` | 订阅 `/tts_service/tts_audio`，镜像 `/voice/playback_reference` 并生成本地 AEC 参考流 |

## 没有直接复制

- `CommunicationRuntimeThread`、生成 provider/consumer 和 service scaffold：它们属于 `pacific-rim` 部署框架，不应成为感知系统的强依赖。
- `AudioIntent.msg`：意图不属于 ASR 或鸡尾酒前端，由下游意图路由处理。
- LiteTTS-AUV 模型装载器：依赖特定 CUDA 模型和权重，保留在 downstream TTS 服务。当前仓库只复用流式音频边界。
- 中文文本正规化：属于 TTS 模型前处理，不应进入麦克风/声学前端。

## TTS -> AEC 回环

先在 ROS2 环境启动参考桥：

```bash
python scripts/tts_aec_reference_bridge.py \
  --topic /tts_service/tts_audio \
  --output-jsonl /tmp/tts_playback_reference.jsonl
```

本地处理：

```bash
python -m voice_detection.cli run-live \
  --profile mac_builtin \
  --device default \
  --playback-reference-jsonl /tmp/tts_playback_reference.jsonl
```

Mac/Jetson 采集、4090 处理：

```bash
python -m voice_detection.cli run-live-remote \
  --profile sipeed_6_plus_1_usb_array \
  --device "设备名或编号" \
  --server-host 4090机器IP \
  --playback-reference-jsonl /tmp/tts_playback_reference.jsonl
```

JSONL 适配器用于本机最小打通和可回溯测试。音频文件读取在后台线程完成，不阻塞 PortAudio 回调；生产部署应把同一个 `AudioFrame` 契约换成进程内 ring buffer、共享内存或 ROS2 loaned message。

实时读取默认从 JSONL 文件末尾开始，不回放旧会话数据。Pacific Rim TTS 消息若没有填写时间戳，桥会用 ROS2 接收时刻补齐，保证参考帧能与麦克风采集帧对齐。

当前 `LinearEchoCanceller` 是已对齐参考信号上的轻量线性回消，可运行、可测试，但不等价于机器人量产 AEC。阵列、扬声器和机身到位后，需要测量播放延迟，并接入 WebRTC AEC3 处理非线性回声、时钟漂移和长尾混响。

## 脚本回放

```bash
python -m voice_detection.cli replay-audio-script \
  --script config/audio_script.example.json \
  --no-wait
```

该命令输出本仓库标准的 ROS4HRI JSON 映射，可直接送给现有 bridge 或注意力 JSON 输入。`AudioMsg` 没有原始 PCM、VAD 能量、overlap 和 self-echo 信息，所以只用于接口与状态机测试，不能冒充真实声学验收。
