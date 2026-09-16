# 流式目标说话人提取（SpeakerBeam/WeSep）

当前系统保留 Sherpa Zipformer 作为中文流式 ASR。TSE 前端采用可选、可回退接口：

```text
Sipeed BF → optional TargetSpeakerExtractor → Sherpa partial/final → ROS4HRI
```

`voice_detection/target_speaker.py` 的 `CausalSpeakerBeamExtractor` 使用 REAL-TSE `spk_emb_causal_100` checkpoint，对已注册目标人的最终 utterance 做目标说话人提取，再交给 Sherpa；partial 仍使用实时 Sherpa 音频流。模型未配置、没有参考音频、超时或健康检查失败时，自动使用 passthrough，不影响 VAD/DOA/Attention。

模型目录：`weights/real-tse/pretrained/spk_emb_causal_100/`。注册接口会把最近一次清晰语音保存为 `config/speaker_references/target.wav`，作为 TSE enrollment。首次加载模型可能需要数秒，后续 utterance 推理通常低于 1 秒；`last_tse` 会报告 backend、latency 和健康状态。

SpeakerBeam/WeSep 的 checkpoint 需要单独验证中文和 Mac M4 的 RTF，未通过真实 Sipeed 双人重叠测试前不默认开启。
