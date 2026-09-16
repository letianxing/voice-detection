"""Pure protocol adaptation for the production brain's existing ROS topics."""
from collections import deque
import json
import math
import os
from pathlib import Path
import re
from .doa import normalize_angle_deg

ROOT = Path(__file__).resolve().parents[1]


def explicit_intent(text):
    compact = re.sub(r"^(?:reachy|机器人|小艾克斯)[，,。！!\s]*", "", text.strip(), flags=re.I).strip("，,。！!？? ")
    for intent, phrases in {"follow": {"跟着我", "跟我来", "请跟着我"},
                            "stop": {"停止", "停下", "停止跟随", "别跟着我"},
                            "hello": {"你好", "嗨"}}.items():
        if compact in phrases:
            return intent
    return None


class SpeakerIDs:
    def __init__(self, path=None):
        self.path = Path(path or ROOT / "config" / "brain_speaker_ids.json")
        self.mapping = json.loads(self.path.read_text()) if self.path.exists() else {}

    def numeric(self, label):
        if not label or label in {"unknown", "-1"}:
            return "-1"
        if label in self.mapping:
            return str(self.mapping[label])
        used = set(self.mapping.values())
        proposed = int(label) if str(label).isdigit() and 0 < int(label) <= 32767 else None
        if proposed in used or proposed is None:
            proposed = next(i for i in range(1, 32768) if i not in used)
        self.mapping[label] = proposed
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.mapping, ensure_ascii=False, indent=2))
        temporary.replace(self.path)
        return str(proposed)


class BrainTopicMapper:
    def __init__(self, config=None, ids=None):
        self.config = config or json.loads((ROOT / "config" / "brain_topics.json").read_text())
        self.ids = ids or SpeakerIDs()
        self.seen_events = deque(maxlen=512)
        self.seen_turns = deque(maxlen=512)
        self.seen_raw_turns = deque(maxlen=512)
        self.last_partial = None
        self.last_state = 0
        self.last_music = None

    def advertised(self):
        c = self.config
        types = {c["voice_topic"]: c["voice_type"], c["direction_topic"]: "std_msgs/msg/Float32",
                 c["state_topic"]: "std_msgs/msg/String", c["music_bpm_topic"]: "std_msgs/msg/Float32",
                 c["music_genre_topic"]: "std_msgs/msg/String"}
        for key in ("asr_partial_topic", "asr_final_topic"):
            if c.get(key): types[c[key]] = "std_msgs/msg/String"
        if c.get("intent_topic"):
            types[c["intent_topic"]] = "audio_msgs/msg/AudioIntent"
        for key in ("startle_topic", "orient_topic", "music_state_topic", "music_beat_topic"):
            types[c[key]] = "std_msgs/msg/Bool"
        return types

    def direction(self, angle):
        if angle is None or not math.isfinite(float(angle)):
            return None
        return normalize_angle_deg(float(angle) * self.config["direction_sign"] + self.config["direction_offset_deg"])

    def messages(self, state, stamp_ms, attention=None):
        c, output = self.config, []
        scene = state.get("audio_scene") or {}
        acoustic = scene.get("acoustic") or {}
        recent = state.get("running") and 0 <= stamp_ms - int(acoustic.get("stamp_ms") or 0) < 300
        angle = self.direction(acoustic.get("direction_deg")) if recent and acoustic.get("direction_valid") else None
        if angle is not None:
            output.append((c["direction_topic"], {"data": angle}))
        for event in scene.get("events", []):
            if event["id"] in self.seen_events or not 0 <= stamp_ms - event["stamp_ms"] < 1000:
                continue
            self.seen_events.append(event["id"])
            topic = {"startle": "startle_topic", "orient": "orient_topic", "music_beat": "music_beat_topic"}.get(event["kind"])
            if topic:
                event_angle = self.direction(event.get("direction_deg"))
                if event["kind"] == "orient" and event_angle is None:
                    continue
                if event_angle is not None:
                    output.append((c["direction_topic"], {"data": event_angle}))
                output.append((c[topic], {"data": True}))
        music = scene.get("music") or {}
        valid = state.get("running") and music.get("valid") and 0 <= stamp_ms - int(music.get("stamp_ms") or 0) < 8000
        playing = bool(valid and music.get("music_state"))
        music_payload = (playing, float(music.get("bpm") or 0) if playing and music.get("bpm_valid") else 0.,
                         music.get("genre", "unknown") if playing and music.get("genre_valid") else "unknown")
        # Publish reset on stale/invalid input so the production brain cannot retain an old BPM forever.
        if music_payload != self.last_music or stamp_ms - self.last_state >= 1000:
            self.last_music = music_payload
            for key, value in zip(("music_state_topic", "music_bpm_topic", "music_genre_topic"), music_payload):
                output.append((c[key], {"data": value}))
        partial = state.get("last_streaming_transcript") or {}
        partial_key = (partial.get("track_id"), partial.get("started_ms"), partial.get("emitted_ms"), partial.get("text"))
        if c.get("asr_partial_topic") and partial.get("text") and not partial.get("is_final") and partial_key != self.last_partial and 0 <= stamp_ms-int(partial.get("emitted_ms") or 0) < 1500:
            output.append((c["asr_partial_topic"], {"data": json.dumps(partial,ensure_ascii=False)}))
            self.last_partial = partial_key
        gates = {turn["utterance_id"]: turn.get("attention", {}) for turn in (attention or {}).get("utterances", [])}
        for turn in state.get("transcripts", []):
            key = turn.get("utterance_id") or f"{turn.get('track_id')}:{turn.get('emitted_ms')}"
            if key in self.seen_turns or not turn.get("is_final") or not turn.get("text") or not 0 <= stamp_ms - int(turn.get("emitted_ms") or 0) < 5000:
                continue
            if c.get("asr_final_topic") and key not in self.seen_raw_turns:
                output.append((c["asr_final_topic"], {"data": json.dumps(turn,ensure_ascii=False)}))
                self.seen_raw_turns.append(key)
            gate = gates.get(key, {})
            if c.get("require_attention", True) and not (gate.get("listen") and gate.get("addressed_to_robot") and gate.get("confidence", 0) >= .52):
                continue
            speaker = turn.get("speaker") or {}
            direction = turn.get("direction") or {}
            voice_angle = self.direction(direction.get("direction_deg")) if direction.get("direction_valid") else None
            typed = c["voice_type"] != "std_msgs/msg/String"
            if typed and voice_angle is None:
                continue  # AudioMsg has no validity field; never fabricate angle=0.
            payload = {"user_id": self.ids.numeric(str(speaker.get("speaker_id") or "unknown")),
                       "internal_id": turn.get("track_id", ""), "role": speaker.get("speaker_role", "unknown"),
                       "asr_text": turn["text"], "emotion": "unknown",
                       "facing_robot": bool(gate.get("addressed_to_robot")),
                       "speaker_vector": turn.get("speaker_vector", [])}
            if voice_angle is not None:
                payload["angle"] = voice_angle
            if not typed:
                payload.update(vad=True, angle_valid=voice_angle is not None,
                               speaker_label=speaker.get("speaker_id", "unknown"), utterance_id=key,
                               stamp_ms=turn.get("emitted_ms"), addressed_to_robot=bool(gate.get("addressed_to_robot")))
                payload = {"data": json.dumps(payload, ensure_ascii=False, allow_nan=False)}
            output.append((c["voice_topic"], payload))
            intent = explicit_intent(turn["text"])
            if intent and c.get("intent_topic"):
                output.append((c["intent_topic"], {"intent": intent, "prob": 1.0}))
            self.seen_turns.append(key)
        if stamp_ms - self.last_state >= 1000:
            self.last_state = stamp_ms
            canonical = {"stamp_ms": stamp_ms, "running": state.get("running"), "audio_scene": scene,
                         "last_transcript": state.get("last_transcript"), "speaker_ids": self.ids.mapping}
            output.append((c["state_topic"], {"data": json.dumps(canonical, ensure_ascii=False, allow_nan=False)}))
        return output
