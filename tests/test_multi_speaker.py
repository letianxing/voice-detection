import unittest,tempfile
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
from types import SimpleNamespace
from voice_detection.multi_speaker import MultiSpeakerTranscriber

class MultiSpeakerTests(unittest.TestCase):
    def setup_processor(self,root):
        for name in ['a','b']:(root/(name+'.wav')).write_bytes(b'test')
        store=Mock(profiles={p:{'role':'known','reference_path':str(root/(p+'.wav'))} for p in ['a','b']})
        embedder=Mock();asr=Mock();asr.transcribe.return_value=SimpleNamespace(text='测试内容')
        return MultiSpeakerTranscriber(store,embedder,asr,root),store,asr
    def test_independent_verification_rejects_wrong_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            processor,store,asr=self.setup_processor(Path(tmp))
            store.rank.side_effect=[[{'speaker_id':'a','score':.9}],[{'speaker_id':'a','score':.8}]]
            extractor=Mock();extractor.process_utterance.return_value=SimpleNamespace(audio=np.ones(16000),enabled=True,healthy=True)
            with patch('voice_detection.multi_speaker.CausalSpeakerBeamExtractor',return_value=extractor):result=processor.transcribe(np.ones(16000),16000,0,1000)
            self.assertEqual([t['person_id'] for t in result['turns']],['a'])
            self.assertFalse(result['streaming']);self.assertEqual(result['rejected'][0]['speaker_id'],'b')
    def test_separation_failure_never_fabricates_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            processor,store,asr=self.setup_processor(Path(tmp))
            extractor=Mock();extractor.process_utterance.return_value=SimpleNamespace(enabled=False,healthy=False)
            with patch('voice_detection.multi_speaker.CausalSpeakerBeamExtractor',return_value=extractor):result=processor.transcribe(np.zeros(16000),16000,0,1000)
            self.assertEqual(result['turns'],[]);asr.transcribe.assert_not_called()
