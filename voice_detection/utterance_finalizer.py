"""Keep streaming-ASR finish and speaker embedding out of PortAudio callbacks."""
import queue
import threading


class UtteranceFinalizer:
    def __init__(self, process, on_error):
        self.process, self.on_error = process, on_error
        self.items = queue.Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.active_stream = None
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name="utterance-finalizer")
        self.thread.start()

    @staticmethod
    def close_stream(stream):
        if stream is not None and hasattr(stream, "close"):
            stream.close()

    def submit(self, utterance, stream, track, context):
        try:
            self.items.put_nowait((utterance, stream, track, context))
            return True
        except queue.Full:
            self.dropped += 1
            self.on_error("utterance_finalizer_queue_full")
            threading.Thread(target=self.close_stream, args=(stream,), daemon=True).start()
            return False

    def _run(self):
        while not self.stop_event.is_set():
            try:
                item = self.items.get(timeout=.1)
            except queue.Empty:
                continue
            self.active_stream = item[1]
            try:
                self.process(*item)
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.on_error(f"utterance_finalizer: {exc}")
            finally:
                self.close_stream(self.active_stream)
                self.active_stream = None

    def close(self):
        self.stop_event.set()
        self.close_stream(self.active_stream)
        while True:
            try:
                self.close_stream(self.items.get_nowait()[1])
            except queue.Empty:
                break
        self.thread.join(timeout=2)
