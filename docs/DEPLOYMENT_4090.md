# 4090 + Mac + 阵列麦克风部署

## 拓扑

```text
Sipeed 6+1 USB array / Mac mic
        |
        v
Mac or Jetson upper-computer voice-detection
  capture + device selection + raw 8ch PCM + playback reference
        |
        +-> local quick VAD/monitoring
        +-> raw multichannel stream to 4090 acoustic service
        |
        v
4090 acoustic algorithms
  spatial front-end + AEC/NS/dereverb + neural separation + ASR optional
        |
        v
Mac/Jetson ROS2 topics
  /humans/voices/...       ROS4HRI voice input
  /voice/acoustic_tracks   engineering track for attention
        |
        +-> robot-attention-perception gates one target stream
                |
                v
              speech-to-speech
```

The upstream/downstream loop is intentional: the host near the USB microphone
captures synchronized raw channels and playback reference; the 4090 may process
the heavy acoustic algorithms; the host then republishes the returned tracks and
target stream into ROS2. Do not collapse this into a stereo OS recording path.

## 启动方式

Mac 本地设备选择：

```bash
python scripts/local_voice_dashboard.py --host 127.0.0.1 --port 8090
```

命令行指定设备：

```bash
python -m voice_detection.cli run-live --profile mac_builtin --device default
python -m voice_detection.cli run-live --profile sipeed_6_plus_1_usb_array --device 2
```

4090 侧后续应提供一个常驻服务，接收 Mac 推送的 PCM chunk，返回增强后的 `AcousticTrack`、目标音频和 ASR 结果。当前仓库已经把本地/云端边界放在 `FrontendOutput`，后续接 WebRTC、gRPC 或 ROS2 DDS bridge 都不需要改注意力接口。

当前最小远端协议已经可跑：

```bash
# 4090
python -m voice_detection.cli run-remote-server --host 0.0.0.0 --port 9097

# Mac/Jetson
python -m voice_detection.cli run-live-remote \
  --profile sipeed_6_plus_1_usb_array \
  --device 2 \
  --server-host 192.168.1.20 \
  --server-port 9097
```

## 多通道采集注意事项

- macOS QuickTime、系统“语音备忘录”和多数系统录音入口通常按单声道/立体声打开设备，只适合排查“有没有声音”，不适合阵列 DOA/波束。
- Mac 正式采集用 PortAudio/sounddevice：确认 `query_devices()` 里 Sipeed 设备 `max_input_channels >= 8`，并用 `channels=8`、`samplerate=48000` 打开。
- Jetson/Linux 正式采集先用 ALSA 验证：`arecord -l`，再试 `arecord -D <device> -c 8 -r 48000 -f S16_LE /tmp/sipeed.wav`。
- 录制敲击测试：依次靠近每个麦克风轻敲，确认 CH0-CH5 外环顺序；不要在 DOA 前对每通道做谱降噪，避免破坏相位。
- CH6/CH7 的含义要实测。当前 profile 把 CH0-CH5 当外环 DOA 输入，中心/参考通道只作为 AEC 或监测候选，不作为主波束输入。
- 48 kHz 是阵列工作采样率；完成波束/AEC/降噪后，再为 ASR/S2S 重采样到 16 kHz 或模型需要的采样率。

## 分层降噪顺序

1. 原始多通道 PCM：只做 DC removal 和幅度保护。
2. 空间域：DOA/source tracking/beamforming，优先用 CH0-CH5 的相位信息。
3. 回声域：用机器人 TTS 播放参考或硬件参考通道做 AEC/self-echo 判定。
4. 时频域：对单路增强音频做轻量自适应噪声抑制、可选 dereverb。
5. 模型域：overlap 高或噪声很差时，按需调用 SepFormer/MossFormer2 等神经分离。
6. S2S：只接收注意力门控放行后的目标音频，不直接吃全场混音。
