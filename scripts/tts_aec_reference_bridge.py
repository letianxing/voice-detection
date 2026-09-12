#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge TTS AudioFrame packets into the microphone AEC reference path.")
    parser.add_argument("--topic", default="/tts_service/tts_audio")
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    try:
        import rclpy
        from std_msgs.msg import String
        from tts_service.msg import AudioFrame
    except Exception as exc:
        raise SystemExit(
            "requires sourced ROS2 with rclpy, std_msgs and tts_service/msg/AudioFrame: "
            f"{exc}"
        )

    output = Path(args.output_jsonl).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node("tts_aec_reference_bridge")
    mirror = node.create_publisher(String, "/voice/playback_reference", 10)

    def receive(message: AudioFrame) -> None:
        stamp_sec = message.header.stamp.sec
        stamp_nanosec = message.header.stamp.nanosec
        if stamp_sec == 0 and stamp_nanosec == 0:
            stamp = node.get_clock().now().to_msg()
            stamp_sec = stamp.sec
            stamp_nanosec = stamp.nanosec
        payload = {
            "header": {
                "frame_id": message.header.frame_id,
                "stamp_sec": stamp_sec,
                "stamp_nanosec": stamp_nanosec,
            },
            "sample_rate": message.sample_rate,
            "channels": message.channels,
            "encoding": message.encoding,
            "interleaved": message.interleaved,
            "data": list(message.data),
        }
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with output.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        mirrored = String()
        mirrored.data = line
        mirror.publish(mirrored)

    node.create_subscription(AudioFrame, args.topic, receive, 10)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
