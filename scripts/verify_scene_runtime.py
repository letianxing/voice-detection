#!/usr/bin/env python3
"""Feed recorded audio through the asynchronous live DSP path, without devices."""
from pathlib import Path
import sys
import time
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    import librosa
    import numpy as np
    from voice_detection.profiles import get_profile
    from voice_detection.scene_runtime import SceneRuntime
    from voice_detection.types import AudioFrame
    scene = SceneRuntime(get_profile('mac_builtin'))
    try:
        deadline = time.monotonic() + 30
        while not scene.snapshot()['classifier_ready'] and time.monotonic() < deadline:
            if scene.snapshot()['error']:
                raise RuntimeError(scene.snapshot()['error'])
            time.sleep(.1)
        assert scene.snapshot()['classifier_ready'], scene.snapshot()
        audio, _ = librosa.load(librosa.ex('brahms'), sr=48000, duration=12)
        for offset in range(0, len(audio) - 960, 960):
            chunk = audio[offset:offset + 960]
            scene.submit(AudioFrame(chunk[:, None], 48000, int(time.time() * 1000)), chunk)
            time.sleep(.02)
        state = scene.snapshot()
        print(json.dumps({k: state[k] for k in ('music','acoustic','error','dropped_frames')}, ensure_ascii=False), flush=True)
        assert state['music']['valid'] and state['music']['music_state'], state
        assert state['acoustic']['direction_deg'] is None
        for _ in range(40):
            chunk = np.zeros(960, np.float32)
            scene.submit(AudioFrame(chunk[:, None], 48000, int(time.time() * 1000)), chunk)
            time.sleep(.02)
        assert not scene.snapshot()['music']['music_state'], scene.snapshot()
        print('live DSP music + silence reset OK')
    finally:
        scene.close()
        assert not scene.thread.is_alive()
        assert not scene.process.is_alive()

if __name__ == '__main__':
    main()
