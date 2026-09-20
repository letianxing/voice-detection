import unittest
import numpy as np
from voice_detection.doa import estimate_azimuth_gcc
from voice_detection.beamforming import mixdown

class DeadChannelTests(unittest.TestCase):
    def test_single_live_channel_never_invents_60_degree_bearing(self):
        samples=np.zeros((960,8),np.float32);samples[:,0]=np.random.default_rng(1).normal(0,.02,960)
        result=estimate_azimuth_gcc(samples,48000,[[.03*i,0,0] for i in range(8)],range(6))
        self.assertIsNone(result.azimuth_deg);self.assertEqual(result.confidence,0)
        np.testing.assert_allclose(mixdown(samples),samples[:,0])


class ReopenSilentArrayTests(unittest.TestCase):
    """插上阵列后只有一个通道有信号：重开采集流，而不是整场没有方位。

    Measured in the field: the array enumerated as eight channels, channel 0
    carried -36 dBFS and channels 1-7 were digital silence, so direction,
    beamforming and array echo cancellation were all off and the only symptom
    was one line of warning text. Opening the stream again — same device, same
    parameters — brought all eight channels back immediately.
    """

    def monitor(self, levels, calls, supports=True):
        from voice_detection.dashboard import LiveMonitor
        monitor = LiveMonitor.__new__(LiveMonitor)
        import threading
        monitor.lock = threading.RLock()
        monitor.running = True
        monitor.capture_warning = ""
        monitor.reopen_count = 0
        monitor._dead_channel_strikes = 0
        monitor._last_reopen_ms = 0
        monitor.desired = {"profile": "sipeed_6_plus_1_usb_array", "device": "MicArray",
                           "seconds": 0., "tse": "baseline"}

        class Profile:
            supports_raw_multichannel = supports

        class Scene:
            profile = Profile()

            def snapshot(self):
                return {"acoustic": {"channel_rms": levels}}

        monitor.scene = Scene()

        class Playback:
            error = ""

            def close(self): (calls or []).append("playback_closed")

            def warmup(self, rate=16000): (calls or []).append("playback_warmed")

        monitor.playback = Playback()
        monitor._reload_devices = lambda: calls.append("reload")
        monitor._teardown = lambda: calls.append("teardown")
        monitor._start_once = lambda *args: calls.append(("start",) + args)
        return monitor

    def dead(self):
        return [0.012] + [0.0] * 7

    def test_one_live_channel_twice_reopens_the_stream(self):
        calls = []
        monitor = self.monitor(self.dead(), calls)
        monitor._repair_dead_channels(monitor.desired)
        self.assertEqual(calls, [])          # one observation is not a verdict
        monitor._repair_dead_channels(monitor.desired)
        self.assertEqual(calls[0], "teardown")
        self.assertIn("reload", calls)      # 设备表刷新前先关掉播放流
        start = next(item for item in calls if isinstance(item, tuple) and item[0] == "start")
        self.assertEqual(start[1], "sipeed_6_plus_1_usb_array")
        self.assertEqual(monitor.reopen_count, 1)
        self.assertIn("重开采集流", monitor.capture_warning)

    def test_a_working_array_is_left_alone(self):
        calls = []
        monitor = self.monitor([0.012, 0.011, 0.013, 0.010, 0.012, 0.011, 0.004, 0.009], calls)
        for _ in range(4):
            monitor._repair_dead_channels(monitor.desired)
        self.assertEqual(calls, [])

    def test_a_silent_room_is_not_a_broken_array(self):
        calls = []
        monitor = self.monitor([0.0] * 8, calls)
        for _ in range(4):
            monitor._repair_dead_channels(monitor.desired)
        self.assertEqual(calls, [])

    def test_the_laptop_microphone_is_not_expected_to_have_channels(self):
        calls = []
        monitor = self.monitor(self.dead(), calls, supports=False)
        for _ in range(4):
            monitor._repair_dead_channels(monitor.desired)
        self.assertEqual(calls, [])

    def test_it_does_not_reopen_over_and_over(self):
        calls = []
        monitor = self.monitor(self.dead(), calls)
        for _ in range(8):
            monitor._repair_dead_channels(monitor.desired)
        self.assertEqual(monitor.reopen_count, 1)


class SupervisorAlwaysRunsTests(unittest.TestCase):
    """采集起来了也要有人看着阵列。

    The repair was written and then never ran: the supervisor thread was only
    started when the first attempt to open the device failed, so in the session
    where capture came up normally and the array delivered channel 0 with seven
    silent channels, nothing ever looked. The panel showed the warning for
    fifteen minutes with capture_reopens at zero.
    """

    def test_a_successful_start_still_arms_the_supervisor(self):
        from voice_detection.dashboard import LiveMonitor
        import threading
        monitor = LiveMonitor.__new__(LiveMonitor)
        monitor.lock = threading.RLock()
        monitor.running = True
        monitor.desired = None
        monitor._supervisor = None
        armed = []
        monitor._start_once = lambda *args: None
        monitor.stop = lambda: None
        monitor._ensure_supervisor = lambda: armed.append(True)

        LiveMonitor.start(monitor, "sipeed_6_plus_1_usb_array", "MicArray", 0.)

        self.assertEqual(armed, [True])
        self.assertEqual(monitor.desired["device"], "MicArray")
