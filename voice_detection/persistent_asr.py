"""Keep the Sherpa model resident; route independent streaming sessions by ID."""
import multiprocessing as mp
import queue
import threading
import time
import uuid
from pathlib import Path
import numpy as np
from .types import SpeechTranscript


def _worker(model_dir, commands, results, ready, load_ms):
    import sherpa_onnx
    start = time.perf_counter()
    root = Path(model_dir)
    model = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(root/'encoder.int8.onnx'), decoder=str(root/'decoder.onnx'),
        joiner=str(root/'joiner.int8.onnx'), tokens=str(root/'tokens.txt'),
        num_threads=2, sample_rate=16000, feature_dim=80, decoding_method='greedy_search', provider='cpu')
    load_ms.value = (time.perf_counter()-start)*1000
    ready.set()
    streams = {}
    while True:
        command, key, audio = commands.get()
        if command == 'stop': return
        if command == 'cancel':
            streams.pop(key, None)
            continue
        stream = streams.setdefault(key, model.create_stream()) if key not in streams else streams[key]
        start = time.perf_counter()
        if command == 'accept':
            stream.accept_waveform(16000, audio)
        elif command == 'finish':
            stream.accept_waveform(16000, np.zeros(4800, np.float32))
            stream.input_finished()
        while model.is_ready(stream): model.decode_stream(stream)
        results.put((key, 'final' if command == 'finish' else 'partial', model.get_result(stream) or '', (time.perf_counter()-start)*1000))
        if command == 'finish': streams.pop(key, None)


class SherpaEngine:
    def __init__(self, model_dir):
        ctx = mp.get_context('spawn')
        self.commands, self.results = ctx.Queue(maxsize=64), ctx.Queue(maxsize=64)
        self.ready, self.load_ms = ctx.Event(), ctx.Value('d', 0.)
        self.closed = threading.Event()
        self.lock = threading.RLock()
        self.sessions = {}
        self.last_decode_ms = None
        self.process = ctx.Process(target=_worker, args=(model_dir, self.commands, self.results, self.ready, self.load_ms), daemon=True)
        self.process.start()
        self.thread = threading.Thread(target=self._route, daemon=True, name='sherpa-results')
        self.thread.start()

    def _route(self):
        while not self.closed.is_set():
            try: key, kind, text, elapsed = self.results.get(timeout=.1)
            except queue.Empty: continue
            except (OSError, EOFError, ValueError): return
            with self.lock:
                self.last_decode_ms = elapsed
                target = self.sessions.get(key)
                if target is not None: target.put((kind, text, elapsed))

    def session(self, track_id, language):
        if self.closed.is_set() or (self.process.exitcode is not None):
            raise RuntimeError('Sherpa model worker is not running')
        return StreamingSession(self, track_id, language)

    def close(self):
        if self.closed.is_set(): return
        self.closed.set()
        with self.lock:
            for target in self.sessions.values(): target.put(('error','Sherpa stopped',0))
        try: self.commands.put_nowait(('stop','',None))
        except queue.Full: pass
        self.process.join(timeout=.5)
        if self.process.is_alive():
            self.process.terminate(); self.process.join(timeout=1)
        self.thread.join(timeout=.3)
        for channel in (self.commands, self.results):
            channel.cancel_join_thread(); channel.close()


class StreamingSession:
    def __init__(self, engine, track_id, language):
        self.engine, self.track_id, self.language = engine, track_id, language
        self.key = uuid.uuid4().hex
        self.results = queue.Queue()
        with engine.lock: engine.sessions[self.key] = self.results
        self.pending = np.zeros(0,np.float32)
        self.started_ms = None
        self.last_partial = ''
        self.decode_ms = None
        self.closed = False
        self.finished = False

    def accept(self, audio, sample_rate_hz, stamp_ms):
        from .asr import resample_mono
        if self.closed: return self.last_partial
        if self.started_ms is None: self.started_ms = stamp_ms
        self.pending = np.concatenate((self.pending, resample_mono(audio, sample_rate_hz,16000)))
        if self.pending.size >= 1600:  # 100 ms batches; model is already loaded.
            try:
                self.engine.commands.put_nowait(('accept',self.key,self.pending.copy()))
                self.pending = np.zeros(0,np.float32)
            except queue.Full:
                if self.pending.size > 160000: raise RuntimeError('Sherpa input backlog exceeds 10 seconds')
        while True:
            try: kind, text, elapsed = self.results.get_nowait()
            except queue.Empty: break
            if kind == 'error': raise RuntimeError(text)
            if kind == 'partial': self.last_partial, self.decode_ms = str(text), elapsed
        return self.last_partial

    def finish(self, ended_ms):
        try:
            if self.pending.size:
                self.engine.commands.put(('accept',self.key,self.pending.copy()),timeout=1)
            self.engine.commands.put(('finish',self.key,None),timeout=1)
            deadline = time.monotonic()+10
            while time.monotonic()<deadline:
                kind, text, elapsed = self.results.get(timeout=max(.1,deadline-time.monotonic()))
                if kind == 'error': raise RuntimeError(text)
                if kind == 'final':
                    self.finished = True
                    self.decode_ms = elapsed
                    text = str(text).replace(' ','').strip()
                    return SpeechTranscript(self.track_id,text,True,language=self.language,started_ms=self.started_ms,ended_ms=ended_ms) if text else None
            return None
        finally:
            self.close()

    def close(self):
        if self.closed: return
        self.closed = True
        with self.engine.lock: self.engine.sessions.pop(self.key,None)
        if not self.finished and not self.engine.closed.is_set():
            try: self.engine.commands.put_nowait(('cancel',self.key,None))
            except queue.Full: pass
