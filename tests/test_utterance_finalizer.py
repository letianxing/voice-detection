import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from voice_detection.utterance_finalizer import UtteranceFinalizer
from voice_detection.asr import SherpaProcessStreamingSession
from voice_detection.dashboard import LiveMonitor
from voice_detection.streaming_asr import Utterance
from voice_detection.types import SpeechTranscript


class FinalizerTests(unittest.TestCase):
    def test_submission_does_not_wait_for_slow_finalization(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        names = []
        def process(*args):
            names.append(threading.current_thread().name)
            entered.set()
            release.wait(2)
            finished.set()
        worker = UtteranceFinalizer(process, self.fail)
        try:
            start = time.monotonic()
            self.assertTrue(worker.submit(None, None, None, {}))
            self.assertLess(time.monotonic() - start, .1)
            self.assertTrue(entered.wait(1))
            self.assertFalse(finished.is_set())
            self.assertEqual(names, ['utterance-finalizer'])
        finally:
            release.set()
            worker.close()

    def test_timeout_partial_is_not_mislabeled_as_final(self):
        session = SherpaProcessStreamingSession.__new__(SherpaProcessStreamingSession)
        session.pending_audio = np.zeros(0, dtype=np.float32)
        session.commands = Mock()
        session.results = Mock()
        session.results.get.side_effect = [('partial', '尚未完成'), queue.Empty()]
        session.close = Mock()
        self.assertIsNone(session.finish(1000))
        session.close.assert_called_once()

    def test_speaker_embedding_survives_final_transcript_for_enrollment(self):
        monitor = LiveMonitor()
        monitor._finalizer = SimpleNamespace(stop_event=threading.Event())
        monitor.speaker_embedder = Mock()
        monitor.speaker_embedder.embed.return_value = [.25, -.5]
        monitor.speaker_profiles = Mock()
        monitor.speaker_profiles.match.return_value = ('person1', 'owner', .9)
        monitor.speaker_profiles.rank.return_value = [{'speaker_id':'person1','score':.9,'pool_size':1}]
        utterance = Utterance('v1', np.ones(16000, np.float32), 16000, 1000, 2000)
        monitor._asr_worker = Mock()
        monitor._asr_worker.submit.side_effect = lambda u: monitor._on_transcript(
            SpeechTranscript('v1', '测试', True, started_ms=u.started_ms, ended_ms=u.ended_ms, emitted_ms=2100))
        monitor._finalize_utterance(utterance, None, None, {})
        self.assertEqual(monitor.last_speaker['embedding'], [.25, -.5])
        self.assertEqual(monitor.last_transcript['speaker_vector'], [.25, -.5])
        self.assertNotIn('embedding', monitor.last_transcript['speaker'])

    def test_stopped_capture_cannot_publish_into_new_capture(self):
        monitor = LiveMonitor()
        monitor._capture_generation = 2
        monitor._finalizer = SimpleNamespace(stop_event=threading.Event())
        monitor.speaker_embedder = Mock()
        monitor._asr_worker = Mock()
        utterance = Utterance('v1', np.ones(16000, np.float32), 16000, 1000, 2000)
        monitor._finalize_utterance(utterance, None, None, {'_capture_generation': 1})
        monitor.speaker_embedder.embed.assert_not_called()
        monitor._asr_worker.submit.assert_not_called()
