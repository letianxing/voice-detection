# 49 场景手工测试手册

本手册只收集语音识别场景。测试时记录原话、环境、说话方式、最终识别文字、差异和时间；不在 ASR 记录里判断意图、是否在对机器人说、身份、权限、回应、转头或移动。

## 启动入口

声学单测：

```bash
cd ~/Golands/robot-attention-perception
OPEN_BROWSER=1 scripts/run_acoustic_test.sh
```

视觉单测：

```bash
cd ~/Golands/robot-attention-perception
OPEN_BROWSER=1 scripts/run_visual_test.sh
```

完整真实 ROS2 测试：

```bash
cd ~/Golands/robot-attention-perception
OPEN_BROWSER=1 scripts/run_full_test.sh
```

停止所有本项目测试服务：

```bash
cd ~/Golands/robot-attention-perception
scripts/stop_mac_first_test.sh
```

## 设备组合

在完整面板中，摄像头和麦克风独立选择：

| 组合 | 用途 |
| --- | --- |
| Mac 摄像头 + Mac 内置麦 | 第一阶段单人、ASR、视觉注意力 |
| Mac 摄像头 + Sipeed 6+1 | 第二阶段 DOA、阵列、多人语音 |
| RealSense D405/D435i + Sipeed 6+1 | RGB-D 距离、遮挡、空间注意力 |
| INDEMIND M1 + Sipeed 6+1 | Linux/Jetson 双目、深度、IMU、运动场景 |

选择 `sipeed_6_plus_1_usb_array` 后，界面只显示至少 8 个输入通道的声卡。不要用 QuickTime 录阵列原始数据。

## 每个场景的统一记录

1. 先填写 `actual_utterance`，原样保留停顿、重复、改口、未说完部分。
2. 填写地点、嘴部到麦克风直线距离、相对方向、背景声、遮挡和机器人状态。
3. 填写语速、音量声级计读数、口音、语言，以及笑声/咳嗽/轻声/喊话。
4. 从声学面板或完整面板记录 `asr_result.text`、`language`、`clarity`、`confidence`、`is_final`。
5. 记录 `speech_started_ms`、`speech_ended_ms`、`result_emitted_ms` 和延迟。
6. 对照原句记录句首/句尾、否定词、数字、专名、背景语音和不同说话人的混入。

生成表格：

```bash
cd ~/Golands/voice-detection
VENV=~/Golands/robot-attention-perception/.venv-mac
$VENV/bin/python -m voice_detection.cli write-asr-template \
  --output data/asr_acceptance_scenarios.jsonl
```

## 49 条执行顺序

| 阶段 | 场景范围 | 推荐设备/环境 |
| --- | --- | --- |
| A | 基础语句、数字、英文、否定、转述、时间、条件、改口 | Mac 麦克风，0.5m，安静房间 |
| B | 语速、音量、口音、笑声、咳嗽、停顿、吃东西 | Mac 麦克风；声级计记录实际值 |
| C | 距离、方向、背向、遮挡、空调、电视、音乐、生活噪声、多人 | Sipeed 8ch，固定阵列，逐项改变一个变量 |
| D | 机器人播放、插话、相近内容、运动噪声 | Sipeed + 实际扬声器/机器人 |
| E | 长时间、网络中断、ASR 重启 | 本地/4090 ASR，保留启动日志 |

### A. 基础语句与关键信息

依次执行 `quiet_natural_sentence`、`short_word_no`、`short_word_stop`、`long_two_sentences`、`names_places_products`、`numbers_dates_units`、`mixed_zh_en`、`negation`、`double_negation`、`reported_speech`、`time_order`、`condition_hypothesis`、`multi_information`、`correction`。

每项先静音，再完整说一次。短词场景需要额外在连续句中重复短词，检查它是否被粘到前后句。数字场景记录中文数字与阿拉伯数字对应关系；中英文场景分别检查英文、中文及边界。

### B. 说话方式

执行 `fast_speech`、`quiet_voice`、`accent_common`、`cough_laugh`、`pause_repeat_fillers`、`eating_while_speaking`。

只改变语速、音量或说话方式，不同时改变距离和背景。笑声、咳嗽、清嗓放在句首、句中、句尾分别记录。

### C. 空间与日常噪声

执行 `distance_0_5m`、`distance_1_5m`、`distance_3m`、`distance_6m_call`、`front_back_left_right`、`side_back_speech`、`doorway_occlusion`、`steady_noise`、`tv_chat_noise`、`transient_life_noise`、`music_noise`、`outdoor_weather_traffic`、`strong_noise_mid_sentence`。

Sipeed 先运行：

```bash
cd ~/Golands/robot-attention-perception
scripts/check_sipeed.sh <设备编号> 10
```

每个方向或距离都记录实际测量值。此阶段不仅看文字，也看声学面板的 `azimuth_deg`、`clarity`、`overlap_probability`，但方向正确性另记为声源定位测试。

### D. 回声与运动

执行 `echo_room`、`robot_playback_only`、`robot_speaking_barge_in`、`similar_robot_user_content`、`robot_motion_noise`。

机器人播放时必须接入 `/tts_service/tts_audio` 播放参考桥，检查 `self_echo_probability` 和是否错误产生 ASR。机器人是否停止播报、转头或移动不在本表判断。

### E. 多人和连续异常

执行 `sequential_speakers`、`two_speakers_overlap`、`two_speakers_full_overlap`、`similar_direction_volume`、`multi_direction_overlap`、`moving_speakers_swap`、`first_after_long_silence`、`continuous_turns`、`long_duration_mixed_conditions`、`long_running_restart`、`network_drop_recover`。

多人场景指定主要测试者；如果系统不能拆分声音，只记录最终混合文字，并把“能否分开”交给多人声音处理测试。网络和重启场景记录中断前后文字是否重复或合并。

## 当前边界

Mac 可先执行 A、B 和部分 C/E。Sipeed 到位后执行阵列相关 C 和多人 E。真实机器人播放、运动、机械噪声和量产 AEC 需要机器人硬件；D405 主要适合近距离 7cm-50cm，不替代远距离深度相机。
