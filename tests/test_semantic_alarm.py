import unittest,queue,threading,time
from unittest.mock import Mock
from voice_detection.scene_runtime import SceneRuntime,empty_music

class SemanticAlarmTests(unittest.TestCase):
    def runtime(self):
        r=SceneRuntime.__new__(SceneRuntime);r.results=queue.Queue();r.lock=threading.RLock();r.process=None
        r.state_data={'music':empty_music('test'),'error':''};r.music_on=False;r.last_semantic_alarm=-100000;r._event=Mock()
        r.direction_at=Mock(return_value={'direction_valid':True,'direction_deg':30});return r
    def test_music_does_not_veto_confirmed_recent_glass_event(self):
        r=self.runtime();now=int(time.time()*1000)
        result={'ready':True,'stamp_ms':now,'music_probability':.95,'event_source_stamp_ms':now-200,'event_window_ms':2000,'event_scores':{'Shatter':.95}}
        r.results.put(result);r._poll_classifier()
        self.assertTrue(r.state_data['music']['music_state']);r._event.assert_called_once()
        self.assertEqual(r._event.call_args.args[0]['reason'],'semantic_alarm_confirmed')
        r.results.put(result);r._poll_classifier();r._event.assert_called_once()
    def test_stale_or_weak_event_does_not_trigger(self):
        now=int(time.time()*1000)
        for source,score in [(now-5000,.99),(now-100,.4)]:
            r=self.runtime();r.results.put({'ready':True,'stamp_ms':now,'music_probability':.8,'event_source_stamp_ms':source,'event_scores':{'Shatter':score}});r._poll_classifier();r._event.assert_not_called()

    def test_cough_wins_over_weaker_alarm_label_and_is_deduplicated(self):
        r=self.runtime();r.last_cough=-100000;now=int(time.time()*1000)
        item={'ready':True,'stamp_ms':now,'music_probability':0,'event_source_stamp_ms':now-200,'event_scores':{'Cough':.96,'Screaming':.9}}
        r.results.put(item);r._poll_classifier()
        r._event.assert_called_once()
        self.assertEqual(r._event.call_args.args[0]['kind'],'cough')
        r.results.put(item);r._poll_classifier();r._event.assert_called_once()
