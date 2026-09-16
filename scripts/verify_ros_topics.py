#!/usr/bin/env python3
"""Run inside the isolated ROS2 test container while verify_ros_publish.py publishes."""
import json
import time
import rclpy
from std_msgs.msg import Bool, Float32, String
from audio_msgs.msg import AudioIntent
rclpy.init()
node = rclpy.create_node('voice_contract_probe')
observed = {}
def receive(topic):
    def callback(message): observed[topic] = message.data if hasattr(message, "data") else {"intent": message.intent, "prob": message.prob}
    return callback
subscriptions = []
for topic, message in {'/voice_msg': String, '/sound_direction': Float32,
                       '/startle_trigger': Bool, '/rear_turn_trigger': Bool,
                       '/audio/music_state': Bool, '/music_bpm': Float32, '/audio_intent_result': AudioIntent,
                       '/music_genre': String, '/audio/music_beat': Bool}.items():
    subscriptions.append(node.create_subscription(message, topic, receive(topic), 10))
print('READY', flush=True)
deadline = time.monotonic() + 25
def complete():
    return len(observed) == 9 and observed.get("/music_bpm") == 120.0 and observed.get("/music_genre") == "jazz" and observed.get("/audio/music_state")
while time.monotonic() < deadline and not complete():
    rclpy.spin_once(node, timeout_sec=.1)
try:
    assert len(observed) == 9, observed
    voice = json.loads(observed['/voice_msg'])
    assert voice['speaker_label'] == 'contract-test-person'
    assert voice['speaker_vector'] == [.125, -.375]
    assert observed['/sound_direction'] == 90.0
    assert observed['/music_bpm'] == 120.0 and observed['/music_genre'] == 'jazz'
    assert observed['/audio_intent_result']['intent'] == 'hello'
    assert observed['/startle_trigger'] and observed['/rear_turn_trigger'] and observed['/audio/music_beat']
    print(json.dumps(observed, ensure_ascii=False), flush=True)
finally:
    node.destroy_node()
    rclpy.shutdown()
