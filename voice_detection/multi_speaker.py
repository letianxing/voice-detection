"""Per-enrolled-speaker extraction with independent identity checks.

This operates on completed bounded utterances, not sub-word live separation.
Unverified outputs are not attributed to a person or used to train profiles.
"""
from pathlib import Path
import time
import numpy as np
from .target_speaker import CausalSpeakerBeamExtractor

class MultiSpeakerTranscriber:
    def __init__(self,store,embedder,asr,model_dir,max_speakers=4):
        self.store=store;self.embedder=embedder;self.asr=asr;self.model_dir=Path(model_dir)
        self.limit=max_speakers;self.extractor=None
    def transcribe(self,audio,rate,started_ms,ended_ms,cancel=None):
        samples=np.asarray(audio,dtype=np.float32).reshape(-1)
        if len(samples)>rate*12:raise ValueError('overlap segment exceeds 12 second budget')
        profiles=[(key,p) for key,p in self.store.profiles.items() if p.get('role') in {'owner','known'} and p.get('reference_path') and Path(p['reference_path']).is_file()]
        if len(profiles)>self.limit:raise ValueError('too many enrolled sources for local extraction budget')
        if len(profiles)<2:return {'ready':False,'reason':'at_least_two_clean_enrollment_references_required','turns':[]}
        outputs=[];rejected=[];started=time.perf_counter()
        for identity,profile in profiles:
            if cancel and cancel.is_set():return {'ready':False,'reason':'cancelled','turns':[]}
            if self.extractor is None:self.extractor=CausalSpeakerBeamExtractor(self.model_dir,Path(profile['reference_path']))
            self.extractor.reference_path=Path(profile['reference_path'])
            result=self.extractor.process_utterance(samples,rate)
            if not result.enabled or not result.healthy:
                rejected.append({'speaker_id':identity,'reason':'extraction_unavailable'});continue
            vector=self.embedder.embed(result.audio,rate)
            ranked=self.store.rank(vector) if vector is not None else []
            if not ranked or ranked[0]['speaker_id']!=identity or ranked[0]['score']<.65 or (len(ranked)>1 and ranked[0]['score']-ranked[1]['score']<.12):
                rejected.append({'speaker_id':identity,'reason':'separated_voice_identity_not_verified'});continue
            duplicate=any(len(result.audio)==len(old) and abs(float(np.dot(result.audio,old)/(np.linalg.norm(result.audio)*np.linalg.norm(old)+1e-9)))>.95 for old in [o[1] for o in outputs])
            if duplicate:
                rejected.append({'speaker_id':identity,'reason':'duplicate_separated_signal'});continue
            outputs.append((identity,result.audio,ranked[0]['score']))
        turns=[]
        for identity,signal,score in outputs:
            if cancel and cancel.is_set():return {'ready':False,'reason':'cancelled','turns':[]}
            track='separated:'+identity
            transcript=self.asr.transcribe(track,signal,rate)
            if transcript and transcript.text.strip():
                turns.append({'person_id':identity,'text':transcript.text,'started_ms':started_ms,'ended_ms':ended_ms,
                              'identity_score':score,'source':'target_speaker_extraction','timing_scope':'whole_segment_not_word_aligned'})
        return {'ready':True,'turns':turns,'rejected':rejected,'latency_ms':round((time.perf_counter()-started)*1000,2),'streaming':False}
