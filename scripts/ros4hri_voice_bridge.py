#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    try:
        import rclpy
        from hri_msgs.msg import IdsList, LiveSpeech
        from std_msgs.msg import Bool, String
    except Exception as exc:
        raise SystemExit(
            "ROS4HRI bridge requires rclpy, hri_msgs and std_msgs in a sourced ROS2 environment: "
            f"{exc}"
        )

    rclpy.init()
    node = rclpy.create_node("voice_detection_ros4hri_bridge")
    tracked_pub = node.create_publisher(IdsList, "/humans/voices/tracked", 10)
    engineering_pub = node.create_publisher(String, "/voice/acoustic_tracks", 10)
    speech_pubs = {}
    speaking_pubs = {}
    feature_pubs = {}

    def publisher_for(topic, msg_type, cache):
        if topic not in cache:
            cache[topic] = node.create_publisher(msg_type, topic, 10)
        return cache[topic]

    path = Path(args.input_jsonl)
    sleep_s = 1.0 / max(args.rate_hz, 0.1)
    offset = 0
    try:
        while rclpy.ok():
            if not path.exists():
                time.sleep(sleep_s)
                continue
            text = path.read_text(encoding="utf-8")
            if offset > len(text):
                offset = 0
            chunk = text[offset:]
            offset = len(text)
            for line in chunk.splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                voice_id = payload.get("voice_id") or payload.get("track_id")
                engineering_track = payload.get("engineering_track")
                if not voice_id and isinstance(engineering_track, dict):
                    voice_id = engineering_track.get("track_id")
                if not voice_id:
                    continue
                tracked = IdsList()
                tracked.header.stamp = node.get_clock().now().to_msg()
                tracked.ids = [voice_id]
                tracked_pub.publish(tracked)

                speaking = Bool()
                speaking.data = bool(payload.get("is_speaking", True))
                publisher_for(f"/humans/voices/{voice_id}/is_speaking", Bool, speaking_pubs).publish(speaking)

                speech_payload = payload.get("speech")
                if speech_payload:
                    speech = LiveSpeech()
                    speech.header.stamp = node.get_clock().now().to_msg()
                    speech.incremental = str(speech_payload.get("incremental", ""))
                    speech.final = str(speech_payload.get("final", ""))
                    speech.confidence = float(speech_payload.get("confidence", 0.0))
                    speech.locale = str(speech_payload.get("locale", ""))
                    publisher_for(f"/humans/voices/{voice_id}/speech", LiveSpeech, speech_pubs).publish(speech)

                features_payload = payload.get("features")
                if not features_payload and isinstance(engineering_track, dict):
                    features_payload = engineering_track.get("features")
                if features_payload:
                    features = String()
                    features.data = json.dumps(features_payload, ensure_ascii=False)
                    publisher_for(f"/humans/voices/{voice_id}/features", String, feature_pubs).publish(features)

                engineering = String()
                engineering.data = json.dumps(payload, ensure_ascii=False)
                engineering_pub.publish(engineering)
                rclpy.spin_once(node, timeout_sec=0.0)
            if args.once:
                break
            time.sleep(sleep_s)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
