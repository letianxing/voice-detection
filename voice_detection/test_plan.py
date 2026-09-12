from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


ASR_TEMPLATE_VERSION = "2026-09-04"


def default_asr_scenarios() -> list[dict[str, object]]:
    """Template rows for the user's ASR-only acceptance scenarios.

    These rows intentionally avoid intent, addressee, permission, identity,
    robot action, and attention conclusions. They are meant to capture what
    was said, what ASR emitted, and timing/difference notes.
    """

    rows = [
        _row("quiet_natural_sentence", "基础语句", "安静环境中的自然语句", "请帮我记一下明天下午三点去二号会议室找小王。"),
        _row("short_word_no", "基础语句", "单字、短词和短句", "不"),
        _row("short_word_stop", "基础语句", "单字、短词和短句", "停"),
        _row("long_two_sentences", "基础语句", "长句和连续两句", "我先去一楼拿文件，然后到三号会议室。等一下你提醒我给小李打电话。"),
        _row("names_places_products", "基础语句", "人名、地点和产品词汇", "小王在西门旁边的 Reachy Mini 展台等我。"),
        _row("numbers_dates_units", "基础语句", "数字、日期和单位", "十二号下午三点十三分到二号楼十三层。"),
        _row("mixed_zh_en", "基础语句", "中英文混合", "帮我把 Qwen TTS 和 ROS4HRI 的测试结果发给 Alex。"),
        _row("negation", "关键信息", "否定词", "别关灯，我今天不出去。"),
        _row("double_negation", "关键信息", "多重否定", "不是不去，我没说不要。"),
        _row("reported_speech", "关键信息", "转述和引用", "小王说他明天来，他说先别走。"),
        _row("time_order", "关键信息", "时间和先后关系", "现在先别动，待会儿再去二号会议室。"),
        _row("condition_hypothesis", "关键信息", "条件和假设", "如果下雨就不出去，要是他来了再告诉我。"),
        _row("multi_information", "关键信息", "多项信息组合", "明天下午三点去二号会议室找小王，但别叫小李。"),
        _row("correction", "说话方式", "犹豫和改口", "去一号……不，二号会议室。"),
        _row("fast_speech", "说话方式", "快慢语速", "你先过来一下然后把那个杯子放到桌子右边。"),
        _row("quiet_voice", "说话方式", "音量变化", "轻声说话时也要把句首和句尾记完整。"),
        _row("accent_common", "说话方式", "常见口音", "今天晚上八点半到办公室找我。"),
        _row("cough_laugh", "说话方式", "笑声、咳嗽和清嗓", "咳嗽之后继续说，把这句话完整记下来。"),
        _row("pause_repeat_fillers", "说话方式", "停顿、重复和口头语", "嗯，那个，就是明天……明天再去。"),
        _row("eating_while_speaking", "说话方式", "边吃东西边说话", "我吃完以后再去会议室。"),
        _row("distance_0_5m", "距离方向遮挡", "不同距离", "我在近距离说一句完整的话。"),
        _row("distance_1_5m", "距离方向遮挡", "不同距离", "我在日常距离说一句完整的话。"),
        _row("distance_3m", "距离方向遮挡", "不同距离", "我在远距离说一句完整的话。"),
        _row("distance_6m_call", "距离方向遮挡", "不同距离", "远距离召唤时请把这句话记下来。"),
        _row("side_back_speech", "距离方向遮挡", "用户背向或侧向说话", "我现在侧对机器人说话。"),
        _row("front_back_left_right", "距离方向遮挡", "不同相对方向", "我从不同方向重复这句话。"),
        _row("doorway_occlusion", "距离方向遮挡", "门口和局部遮挡", "我在隔断后面说这句话。"),
        _row("steady_noise", "日常噪声", "空调、风扇和设备持续声", "空调开着的时候请识别这句话。"),
        _row("tv_chat_noise", "日常噪声", "餐桌聊天和电视", "电视和旁人聊天时只记录主要测试者这句话。"),
        _row("transient_life_noise", "日常噪声", "偶尔出现的生活声音", "关门声出现时不要截断这句话。"),
        _row("music_noise", "日常噪声", "音乐播放", "播放音乐的时候不要把歌词混进这句话。"),
        _row("outdoor_weather_traffic", "日常噪声", "户外风雨和车辆声", "风声和车辆声里请识别这句话。"),
        _row("strong_noise_mid_sentence", "日常噪声", "说话途中出现强噪声", "强噪声出现前、中、后都要记录差异。"),
        _row("echo_room", "回声和机器人播放", "走廊、玻璃门和空会议室", "回声环境里不要重复句尾。"),
        _row("robot_playback_only", "回声和机器人播放", "只有机器人播放声音", ""),
        _row("robot_speaking_barge_in", "回声和机器人播放", "用户在机器人说话时插入一句话", "不是现在开始，等一下。"),
        _row("similar_robot_user_content", "回声和机器人播放", "用户与机器人说相近内容", "不是现在开始。"),
        _row("robot_motion_noise", "回声和机器人播放", "机器人运动时产生的声音", "机器人转动时也要听清楚这句话。"),
        _row("sequential_speakers", "多人说话", "两至三人依次说话", "第一位测试者先说这句话。"),
        _row("two_speakers_overlap", "多人说话", "两人部分重叠", "主要测试者说的这句话需要单独记录。"),
        _row("two_speakers_full_overlap", "多人说话", "两人完全重叠", "其中一个人包含否定词，不要走。"),
        _row("similar_direction_volume", "多人说话", "相近方向和不同音量", "两个人在相近方向说话。"),
        _row("multi_direction_overlap", "多人说话", "多人从不同方向同时说话", "主要测试者从正前方说这句话。"),
        _row("moving_speakers_swap", "多人说话", "两人边走边说话并互换位置", "我边走边说，请不要截断。"),
        _row("first_after_long_silence", "连续使用和异常", "长时间安静后的第一句话", "安静很久后的第一句话。"),
        _row("continuous_turns", "连续使用和异常", "连续多轮说话", "这是连续多轮里的第一句。"),
        _row("long_duration_mixed_conditions", "连续使用和异常", "长时间连续测试", "长时间测试恢复后的第一句话。"),
        _row("long_running_restart", "连续使用和异常", "重新启动语音识别", "重启之后第一句话不要丢句首。"),
        _row("network_drop_recover", "连续使用和异常", "网络变慢或中断时说话", "网络恢复后不要重复中断前的文字。"),
    ]
    for row in rows:
        stage, capabilities = _scenario_requirements(str(row["scenario_id"]))
        row["test_stage"] = stage
        row["required_capabilities"] = capabilities
    return rows


def write_asr_template(path: str | Path, fmt: str = "jsonl", rows: Iterable[dict[str, object]] | None = None) -> None:
    scenario_rows = list(rows if rows is not None else default_asr_scenarios())
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        output_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in scenario_rows) + "\n",
            encoding="utf-8",
        )
        return
    if fmt == "csv":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scenario_rows[0].keys()))
            writer.writeheader()
            for row in scenario_rows:
                writer.writerow(row)
        return
    raise ValueError(f"unsupported ASR template format: {fmt}")


def _row(scenario_id: str, category: str, scenario_name: str, utterance_expected: str) -> dict[str, object]:
    return {
        "template_version": ASR_TEMPLATE_VERSION,
        "scenario_id": scenario_id,
        "category": category,
        "scenario_name": scenario_name,
        "utterance_expected": utterance_expected,
        "actual_utterance": "",
        "environment": {
            "place": "",
            "distance_m": None,
            "relative_direction_deg": None,
            "background_sound": "",
            "occlusion": "",
            "robot_state": "",
        },
        "speaking_style": {
            "speed": "",
            "volume_dba": None,
            "accent": "",
            "language": "",
            "notes": "",
        },
        "asr_result": {
            "text": "",
            "language": "",
            "clarity": None,
            "confidence": None,
            "is_final": None,
        },
        "diff_notes": {
            "missing_beginning": False,
            "missing_ending": False,
            "missing_keywords": [],
            "extra_words": [],
            "number_or_name_changes": [],
            "background_speech_mixed_in": False,
            "speaker_mix": False,
            "notes": "",
        },
        "timing": {
            "speech_started_ms": None,
            "speech_ended_ms": None,
            "result_emitted_ms": None,
            "latency_ms": None,
        },
        "test_stage": "hardware_required",
        "required_capabilities": [],
    }


def _scenario_requirements(scenario_id: str) -> tuple[str, list[str]]:
    sipeed = {
        "distance_3m", "distance_6m_call", "front_back_left_right", "side_back_speech",
        "tv_chat_noise", "music_noise", "strong_noise_mid_sentence", "robot_speaking_barge_in",
        "similar_robot_user_content", "sequential_speakers", "two_speakers_overlap",
        "two_speakers_full_overlap", "similar_direction_volume", "multi_direction_overlap",
        "moving_speakers_swap",
    }
    robot = {"robot_motion_noise"}
    network = {"network_drop_recover"}
    if scenario_id in robot:
        return "robot_hardware", ["robot_motion", "playback_reference", "microphone"]
    if scenario_id in network:
        return "networked_asr", ["network_conditioner", "asr"]
    if scenario_id in sipeed:
        return "spatial_hardware", ["multichannel_array", "doa", "separation", "asr"]
    return "mac_first_test", ["microphone", "vad", "asr"]
