# ASR 验收执行矩阵

本矩阵只安排语音识别场景，不定义轮数、评分、通过条件和发布结论。ASR 仅记录原话、最终文字、语言/清晰度、差异和时间；意图、对机器人说、身份、权限、回应和动作均由其他测试负责。

## 条件范围

| 条件 | 范围 | 固定位置 |
| --- | --- | --- |
| 近距离 | 0.3-1 m | 0.5 m |
| 日常距离 | >1-2 m | 1.5 m |
| 远距离 | >2-5 m | 3 m |
| 远距离召唤 | >5-8 m | 6 m |
| 轻声 | 40-<50 dB(A) | 不贴近耳语 |
| 正常 | 50-70 dB(A) | 日常交谈 |
| 提高音量 | >70-80 dB(A) | 非喊叫 |
| 喊话 | >80-90 dB(A) | 短时间远距喊话 |

## 执行阶段

| 阶段 | 能执行的范围 | 不能据此下结论的范围 |
| --- | --- | --- |
| `mac_first_test` | 基础语句、关键信息、语速音量、口音、停顿、待机首句、连续多轮、基础噪声和回声记录 | DOA、远场、多人分离、机器人运动 |
| `spatial_hardware` | Sipeed 原始 8ch 下的方向、距离、多人重叠、音乐/电视、目标分离 | 机器人关节噪声和完整双工 AEC |
| `robot_hardware` | 机器人播放、转头、行走、风扇/关节声 | 意图和运动决策结论 |
| `networked_asr` | 4090 网络正常、变慢、中断、恢复 | 离线 ASR 不适用此项 |

## 生成记录表

```bash
cd ~/Golands/voice-detection
python -m voice_detection.cli write-asr-template \
  --output data/asr_acceptance_scenarios.jsonl
```

生成文件逐行包含 `test_stage` 和 `required_capabilities`，并保留用户文档要求的 environment、speaking_style、asr_result、diff_notes 和 timing 字段。
