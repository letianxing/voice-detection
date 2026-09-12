# 语音识别测试场景覆盖

本仓库按你给的测试标准记录 ASR 结果，但不在 ASR 层判断意图、权限、是否对机器人说或机器人动作。

注意：这份软件仓库不能单靠单元测试宣称所有场景“通过”。单元测试只验证字段、链路和边界；真正通过需要在指定麦克风、距离、方向、噪声、多人和机器人自播条件下实测，并把每轮记录写入 `/voice/asr_records` 或 `hri-memory-service`。

## 必须记录

- 实际说话原句：保留停顿、重复、改口和未说完的部分。
- 环境条件：场所、距离、方向、背景声、遮挡、机器人静止/移动/播报状态。
- 说话方式：语速、音量、口音、语言、轻声/喊话/笑声/咳嗽/改口。
- ASR 最终文字：包括未出文字、重复出现、很晚出现。
- 差异：句首、句尾、否定词、数字、名称、单位、背景语音混入、多人拼接。
- 时间：用户开始、结束、识别结果出现时间。
- 可选质量：语言、clarity/SNR proxy。

## 与声学前端字段的关系

| 测试观察 | 字段 |
| --- | --- |
| 是否有说话 | `AcousticTrack.voice_activity` |
| 声音是否清楚 | `AcousticTrack.clarity` 和 `AudioFeatures.rms/hnr` |
| 多人重叠风险 | `AcousticTrack.overlap_probability` |
| 机器人自播报污染 | `AcousticTrack.self_echo_probability` |
| 最终识别文字 | `SpeechTranscript.text/is_final/language/confidence` |
| 识别延迟 | `SpeechTranscript.started_ms/ended_ms/emitted_ms` |
| 多人能否分开 | 每个 `track_id` 一条转写；不能分离则只记录混合 `track_id` |

## 下游边界

`speech-to-speech` 只消费注意力门控后的目标音频流。ASR 测试中出现“停”“不用了”“不是现在开始”等文字时，本仓库只负责转写和记录，不给出动作结论。
