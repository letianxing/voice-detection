#!/usr/bin/env python3
"""Synthetic contract fixture sent only to the dedicated local test ROS domain."""
import json
from pathlib import Path
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_detection.brain_topics import ROOT, SpeakerIDs
from voice_detection.ros_publisher import BrainRosPublisher
with tempfile.TemporaryDirectory() as directory:
    config = json.loads((ROOT / 'config/brain_topics.json').read_text())
    config['require_attention'] = False
    started = time.monotonic()
    def state():
        stamp = int(time.time() * 1000)
        if time.monotonic() - started < 1:
            return {'running': False}
        return {'running': True, 'audio_scene': {
            'acoustic': {'stamp_ms': stamp, 'direction_deg': 0, 'direction_valid': True},
            'music': {'valid': True, 'stamp_ms': stamp, 'music_state': True, 'genre': 'jazz', 'genre_valid': True, 'bpm': 120, 'bpm_valid': True},
            'events': [{'id': str(i), 'kind': kind, 'stamp_ms': stamp, 'direction_deg': 0} for i, kind in enumerate(['startle','orient','music_beat'])]},
            'transcripts': [{'utterance_id': 'fixture', 'track_id': 'fixture', 'is_final': True, 'emitted_ms': stamp, 'text': '你好',
                             'speaker': {'speaker_id': 'contract-test-person', 'speaker_role': 'known'}, 'speaker_vector': [.125, -.375],
                             'direction': {'direction_deg': 0, 'direction_valid': True}}]}
    publisher = BrainRosPublisher(state, 'ws://127.0.0.1:19090', config)
    publisher.mapper.ids = SpeakerIDs(Path(directory) / 'ids.json')
    try:
        time.sleep(6)
        print(publisher.status())
        assert publisher.connected and not publisher.error and publisher.published > 0
    finally:
        publisher.close()
