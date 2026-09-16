# voice-detection

## ROS4HRI/Jazzy 兼容边界

本仓库保留声学算法和测试协议，不绑定具体 ROS 发行版。`ros4hri_voice_bridge.py` 将工程输出转换为
REP-155 的 `/humans/voices/tracked`、`/humans/voices/<voiceID>/is_speaking`、`speech` 和 `features`。
在 Jazzy workspace 中运行时，标准消息由 `hri_msgs` 提供；voice bridge 仍可独立运行，便于离线 JSONL
回放和现有测试。

机器人鸡尾酒声学前端，负责把不同麦克风设备统一成 ROS4HRI 可消费的 `/humans/voices` 输入，并给注意力门控提供声源方向、清晰度、重叠概率和目标音频流。

边界：

- 本仓库做：采集、通道映射、降噪/AEC 接口、VAD、DOA、声源跟踪、beamforming、ASR 测试日志、ROS4HRI voice topic 发布。
- 本仓库不做最终决策：不判断是否该回应、不判断权限、不执行动作。
- `speech-to-speech` 是下游对话：只消费 `robot-attention-perception` 门控放行后的目标分离/波束音频。

### Sipeed 6+1 弱声调优基线

Sipeed 6+1 以原始 `8ch / 48 kHz / PCM16` 打开；稳定基线前端先做去直流、DOA/延时求和、
AEC 参考抵消、保守降噪，再进入 VAD 和 ASR。默认参数针对远距离小声但不削波的首轮验收：

| 项目 | 默认值 | 现场含义 |
| --- | ---: | --- |
| VAD `min_rms_dbfs` | `-52` | 与原始稳定版及 C++ 核心保持一致 |
| VAD `threshold_db` | `8 dB` | 与原始稳定版及 C++ 核心保持一致 |
| VAD noise alpha | `0.96` | 与 C++ 核心保持一致，避免噪声底长期漂移 |
| AGC | 默认关闭（增益 `0 dB`） | 保证送入 ASR 的波形与原始稳定版一致 |
| 预缓存 / 端点静音 | `180 / 480 ms` | 可用环境变量显式调节 |
| ASR 输入 | 单声道、16-bit、16 kHz | Whisper 最终识别；Vosk 仅用于 partial |

若面板显示 `last_vad.active=false`，先看 `rms_dbfs/noise_floor_dbfs/snr_db`：RMS 有值但 SNR
低于 6 dB，问题在 VAD/AGC/降噪；VAD 已 active 但文字错误，才继续排查 ASR 模型域不匹配。

声纹模型为 sherpa-onnx 兼容的 3D-Speaker CampPlus 中文 16 kHz 模型：
`weights/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx`。说一段至少 0.5 秒的清晰语音后，
可登记身份：

```bash
curl -X POST http://127.0.0.1:8090/api/enroll-speaker \
  -H 'content-type: application/json' \
  -d '{"speaker_id":"owner","speaker_role":"owner"}'
```

后续匹配使用余弦相似度，默认阈值 `0.48`；真实部署应让同一人采集 3 段不同句子后再验收。

### 流式 VAD + ASR 观测

启用本地 Whisper 最终模型和 Vosk 流式模型时，8090 `/api/state` 会持续提供 `last_streaming_transcript`：
`text` 是 partial，`is_final=false`；句尾会更新为 final，并附带
`first_result_latency_ms` 与 `final_latency_ms`。8092 测试记录区同步显示这三个字段。
因此可以把“VAD 已触发但 ASR 没 partial”与“partial 很快、只是端点静音等待较长”区分开。

### 目标说话人 TSE（SpeakerBeam）

8090 的 `Cocktail mode` 可选择 `SpeakerBeam TSE` 或 `Auto overlap switch`。当前后端使用 REAL-TSE 的
`spk_emb_causal_100`（192D speaker embedding、causal BSRNN）；它在最终 utterance 的 ASR worker 中提取已注册目标人的语音，
然后交给 Sherpa，失败时自动回退到 Baseline DOA/BF。模型位于 `weights/real-tse/pretrained/spk_emb_causal_100/`，
注册说话人后参考音频保存到 `config/speaker_references/target.wav`。状态接口中的 `tse_backend` 和 `last_tse` 可用于记录实际是否启用。

## 快速验证

```bash
cd ~/Golands/voice-detection
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m voice_detection.cli profiles
python3 -m voice_detection.cli simulate --profile sipeed_6_plus_1_usb_array
python3 -m voice_detection.cli write-asr-template --output /tmp/asr_acceptance.jsonl
```

## 分层开源组件选型

| 层 | 默认实现 | 可插拔增强 |
| --- | --- | --- |
| 音频采集/设备抽象 | Python `sounddevice`/PortAudio；C++ core 为后续 PortAudio/miniaudio 留接口 | macOS CoreAudio、Windows WASAPI、Linux/Jetson ALSA、miniaudio |
| 空间前端/鸡尾酒 | raw multichannel -> DOA -> delay-and-sum beamforming -> source track | ODAS、Sipeed 串口 16x16 声场图、pyroomacoustics 离线仿真 |
| 回声消除/降噪/解混响 | playback reference 相关性 self-echo 判定；自适应轻量噪声抑制 | WebRTC APM(AEC/NS/AGC)、RNNoise、DeepFilterNet、nara_wpe |
| 说话人分离/识别 | `speaker_label`/track ID 占位，ASR 测试不做身份结论 | pyannote.audio、SpeechBrain ECAPA-TDNN、resemblyzer |
| 神经语音分离/增强 | 默认关闭，只在 overlap 高时接入 | asteroid SepFormer/MossFormer2、torchaudio ConvTasNet/DPRNN |
| Speech-to-speech 下游 | 本仓库只输出注意力门控前的目标音频和转写记录 | HF speech-to-speech、Silero VAD、Parakeet-TDT、FunASR/SenseVoice/faster-whisper、Qwen3-TTS/CosyVoice/GPT-SoVITS |
| 视觉/注意力融合 | 输出 ROS4HRI `/humans/voices` 和 `/voice/acoustic_tracks` | `vision-detection` 的 `/vision/people`，`robot-attention-perception` 的 ROS4HRI attention |
| 调试可视化 | `scripts/local_voice_dashboard.py` | pyodas、ODAS Studio、rqt_human_radar |

正式阵列链路不要用 QuickTime 或系统录音做采集入口；这些工具通常只给 1-2 声道，无法保证拿到 Sipeed 的原始 8ch 同步 PCM。Mac 上用 PortAudio/sounddevice 枚举并以 `channels=8` 打开 UAC2 设备；Jetson/Linux 上用 ALSA/PortAudio，先用 `arecord -l`、`arecord -D <device> -c 8 -r 48000 -f S16_LE` 验证。

## 4090 部署启动

安装基础依赖：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m pip install sounddevice
```

列出设备：

```bash
python -m voice_detection.cli list-devices
```

用 Mac/Windows 默认麦克风启动：

```bash
python -m voice_detection.cli run-live --profile mac_builtin --device default
```

用 Sipeed 6+1 USB 环麦启动：

```bash
python -m voice_detection.cli run-live \
  --profile sipeed_6_plus_1_usb_array \
  --device "你的系统设备名或编号"
```

4090 远端算法服务：

```bash
# 4090 机器
python -m voice_detection.cli run-remote-server --host 0.0.0.0 --port 9097

# Mac 或 Jetson 上位机：采集 USB 原始多通道，推给 4090，消费回传声源轨迹
python -m voice_detection.cli run-live-remote \
  --profile sipeed_6_plus_1_usb_array \
  --device "你的系统设备名或编号" \
  --server-host 4090机器IP \
  --server-port 9097
```

这个回路是声学系统内部的正常闭环：上位机离硬件近，负责原始 8ch 采集和播放参考；4090 负责重算法；上位机再把增强音频、`AcousticTrack` 和 ROS4HRI topic 发布给本地机器人系统。

启动本地选择界面：

```bash
python scripts/local_voice_dashboard.py --host 127.0.0.1 --port 8090
```

打开：

```text
http://127.0.0.1:8090
```

简单 TTS 播放状态调试：

```bash
python -m voice_detection.cli speak --text "我听到了"
python -m voice_detection.cli speak --text "我听到了" --dry-run
```

Mac 上默认调用系统 `say`。这个命令用于本地闭环调试和 playback state 记录；真实 AEC 仍应接入机器人实际播放链路的参考音频。

## 离线验收工具

生成 ASR 场景记录模板：

```bash
python -m voice_detection.cli write-asr-template \
  --output ~/Golands/voice-detection/data/asr_acceptance_template.jsonl
```

分析已经录好的多通道 WAV：

```bash
python -m voice_detection.cli analyze-wav \
  --profile sipeed_6_plus_1_usb_array \
  --wav /path/to/8ch_48k_s16le.wav
```

`analyze-wav` 会输出采样率、通道数、每通道 RMS/peak dBFS、静音/削波标记、通道相关矩阵和可用时的 DOA 估计。它用于先检查“是否真的拿到了原始 8ch 同步音频”，不替代真实环境标定。

## ROS4HRI 输出

标准前置输入：

- `/humans/voices/tracked`：当前声源 ID 列表。
- `/humans/voices/<voice_id>/is_speaking`：该 voice 是否正在说话。
- `/humans/voices/<voice_id>/features`：RMS、ZCR、HNR、pitch、MFCC 占位字段。
- `/humans/voices/<voice_id>/speech`：ASR 增量和最终转写。
- `/humans/candidate_matches`：voice/person 候选匹配，通常由 attention/person manager 或后续声纹模块补充。

工程扩展输出：

- `/voice/acoustic_tracks`：JSON，包含 `track_id`、DOA、clarity、overlap、self_echo 等，给 `robot-attention-perception`。
- `/voice/asr_records`：JSONL 风格 ASR 测试记录，覆盖你给的场景记录字段。
- `/voice/playback_reference`：TTS PCM 参考帧镜像；输入源是 `/tts_service/tts_audio`，用于 AEC 和 self-echo 判断。

启动 ROS4HRI bridge：

```bash
python scripts/ros4hri_voice_bridge.py --input-jsonl /tmp/voice_tracks.jsonl
```

## Pacific Rim 兼容

`~/Golands/pacific-rim/module/service/voice_service` 里的 `voice_service/msg/AudioMsg` 可以作为脚本化测试或旧链路兼容输入，字段可映射到本仓库的 `AcousticTrack`/`SpeechTranscript`。它不是原始音频协议，不能替代 VAD、DOA、overlap、self-echo 或目标增强音频。

`~/Golands/pacific-rim/module/service/tts_service` 里的 `tts_service/msg/AudioFrame` 适合作为 TTS 播放音频和 AEC playback reference 的桥接格式。本仓库已提供 `voice_detection/audio_frame_codec.py` 做 JSON payload 编解码，并把参考帧随采集帧发送到 4090。

脚本回放：

```bash
python -m voice_detection.cli replay-audio-script \
  --script config/audio_script.example.json \
  --no-wait
```

把 Pacific Rim TTS 音频接到本地/远端 AEC：

```bash
python scripts/tts_aec_reference_bridge.py \
  --topic /tts_service/tts_audio \
  --output-jsonl /tmp/tts_playback_reference.jsonl

python -m voice_detection.cli run-live-remote \
  --profile sipeed_6_plus_1_usb_array \
  --device "你的系统设备名或编号" \
  --server-host 4090机器IP \
  --playback-reference-jsonl /tmp/tts_playback_reference.jsonl
```

当前轻量线性 AEC 是可运行基线；硬件到位后的正式机器人链路应替换为 WebRTC AEC3。完整复用边界和回环说明见 `docs/PACIFIC_RIM_REUSE.md`。

## 目录

- `voice_detection/frontend.py`：鸡尾酒前端主流程。
- `voice_detection/doa.py`：GCC-PHAT DOA。
- `voice_detection/beamforming.py`：delay-and-sum beamforming。
- `voice_detection/vad.py`：能量 VAD。
- `voice_detection/tracker.py`：声源 track 稳定 ID。
- `voice_detection/ros4hri.py`：ROS4HRI topic 映射。
- `voice_detection/asr.py`：ASR 测试记录和下游 adapter 边界。
- `voice_detection/calibration.py`：离线 WAV 通道、电平和 DOA 可用性分析。
- `voice_detection/test_plan.py`：ASR 场景记录模板生成。
- `voice_detection/pacific_rim_compat.py`：`voice_service/msg/AudioMsg` 和当前声学 track 的兼容映射。
- `voice_detection/audio_frame_codec.py`：`tts_service/msg/AudioFrame` 风格音频 payload 编解码。
- `voice_detection/audio_script.py`：Pacific Rim 风格脚本场景解析和确定性回放。
- `voice_detection/streaming_audio.py`：流式 TTS 交叉淡化与固定 PCM 分包。
- `voice_detection/aec.py`：播放参考时间对齐、后台 JSONL 接入和轻量 AEC。
- `scripts/local_voice_dashboard.py`：设备选择和实时声学可视化界面。
- `scripts/tts_aec_reference_bridge.py`：`/tts_service/tts_audio` 到 AEC 参考流的 ROS2 桥。

## 声学反射、音乐与正式大脑兼容

新增异步音乐/曲风分类、BPM、节拍、独立声学反射与ROS2发布。安装、topic、坐标约定及正式大脑修复见 [AUDIO_CAPABILITIES.md](docs/AUDIO_CAPABILITIES.md)。默认通过注意力门控后才向正式大脑发送ASR/意图；旁人转写仍供原有记忆链路使用。

## 仿生打断增量（2026-09-15）

加入BioAcoustics、VAP轮次预测、实际PCM淡出和语义声音事件。VAP模型安装：`python scripts/setup_vap.py`；离线联调：`python scripts/verify_bio_integration.py`（加载模型，无音频设备采集）。VAP只辅助已放行的对话，不单独触发打断。完整交接见[AGENTS.md](AGENTS.md)。
