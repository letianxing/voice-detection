"""Bounded optional VAP worker: mic residual + actual robot playback reference."""
import multiprocessing as mp
import queue
import time
from pathlib import Path
import numpy as np


def worker(requests,results,root):
    try:
        import sys,torch,os
        sys.path.insert(0,str(Path(root)/'vendor/voice_activity_projection'))
        from vap.model import VapGPT,VapConfig
        torch.set_num_threads(2)
        model=VapGPT(VapConfig(load_pretrained=0)).eval()
        model.load_state_dict(torch.load(Path(root)/'weights/vap/model.pt',map_location='cpu',weights_only=True),strict=True)
        device=os.environ.get('VOICE_VAP_DEVICE') or ('mps' if torch.backends.mps.is_available() else 'cpu')
        model=model.to(device)
        with torch.inference_mode():
            warm=model(torch.zeros(1,2,64000,device=device))['logits'].softmax(-1)
            model.objective.probs_next_speaker_aggregate(warm,from_bin=0,to_bin=1).cpu()
            model.objective.probs_next_speaker_aggregate(warm,from_bin=2,to_bin=3).cpu()
            (-(warm[:,-1]*warm[:,-1].clamp_min(1e-9).log2()).sum(-1)).cpu()
        results.put({'ready':True,'valid':False,'backend':'vap_stereo_50hz','device':device})
        while True:
            item=requests.get()
            if item is None:return
            stamp,audio,observed_ms=item;started=time.perf_counter()
            with torch.inference_mode():
                out=model(torch.from_numpy(audio).float().unsqueeze(0).to(device));probs=out['logits'].softmax(-1)
                now=model.objective.probs_next_speaker_aggregate(probs,from_bin=0,to_bin=1)[0,-1]
                future=model.objective.probs_next_speaker_aggregate(probs,from_bin=2,to_bin=3)[0,-1]
                entropy=float((-(probs[:,-1]*probs[:,-1].clamp_min(1e-9).log2()).sum(-1)).item())
            payload={'ready':True,'valid':True,'stamp_ms':stamp,'backend':'vap_stereo_50hz','p_user_now':float(now[0]),'p_robot_now':float(now[1]),
                     'p_user_future':float(future[0]),'p_robot_future':float(future[1]),'entropy_bits':entropy,'inference_ms':round((time.perf_counter()-started)*1000,2),
                     'device':device,'context_ms':audio.shape[-1]*1000/16000,'observed_context_ms':observed_ms,'input':'aec_mic_plus_playback_reference','calibrated_for_mandarin':False}
            try:results.put_nowait(payload)
            except queue.Full:pass
    except Exception as exc:
        try:results.put({'ready':False,'valid':False,'error':str(exc)},timeout=.2)
        except queue.Full:pass


class TurnPredictor:
    def __init__(self):
        ctx=mp.get_context('spawn');self.requests=ctx.Queue(maxsize=1);self.results=ctx.Queue(maxsize=4)
        self.process=ctx.Process(target=worker,args=(self.requests,self.results,str(Path(__file__).resolve().parents[1])),daemon=True)
        self.process.start();self.state={'ready':False,'valid':False,'backend':'vap_stereo_50hz'};self.last=0
    def submit(self,audio,stamp):
        if stamp-self.last<200:return
        try:
            observed_ms=audio.shape[-1]*1000/16000
            audio=np.pad(audio[:,-64000:],((0,0),(max(0,64000-audio.shape[-1]),0)))
            self.requests.put_nowait((stamp,np.ascontiguousarray(audio),observed_ms));self.last=stamp
        except queue.Full:pass
    def snapshot(self):
        while True:
            try:self.state.update(self.results.get_nowait())
            except queue.Empty:break
        if not self.process.is_alive():self.state.update(valid=False,ready=False,error=self.state.get('error') or 'VAP worker exited')
        result=dict(self.state)
        if time.time()*1000-result.get('stamp_ms',0)>1000:result['valid']=False
        return result
    def close(self):
        try:self.requests.put_nowait(None)
        except queue.Full:pass
        self.process.join(.5)
        if self.process.is_alive():self.process.terminate();self.process.join(1)
        for q in [self.requests,self.results]:q.cancel_join_thread();q.close()
