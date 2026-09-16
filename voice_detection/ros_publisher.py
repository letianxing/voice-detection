"""Reconnecting rosbridge publisher; network I/O never runs in the audio callback."""
import json
import threading
import time
from urllib.request import build_opener, ProxyHandler
from .brain_topics import BrainTopicMapper


class BrainRosPublisher:
    def __init__(self, state_provider, url, config=None):
        self.state_provider, self.url = state_provider, url
        self.mapper = BrainTopicMapper(config)
        self.stop_event = threading.Event()
        self.connected, self.error, self.published, self.failed = False, "", 0, 0
        self.attention = {}
        self.last_attention = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name="voice-brain-ros")
        self.thread.start()

    def status(self):
        return {"enabled": True, "connected": self.connected, "error": self.error,
                "published": self.published, "failed": self.failed, "topics": self.mapper.advertised()}

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3)

    def _get_attention(self):
        if not self.mapper.config.get("require_attention", True):
            return {}
        if time.monotonic() - self.last_attention < .2:
            return self.attention
        self.last_attention = time.monotonic()
        try:
            with build_opener(ProxyHandler({})).open(self.mapper.config["attention_url"], timeout=.15) as response:
                self.attention = json.load(response)
        except Exception:
            self.attention = {}
        return self.attention

    def _run(self):
        try:
            import websocket
        except ImportError as exc:
            self.error = str(exc)
            return
        while not self.stop_event.is_set():
            connection = None
            try:
                connection = websocket.create_connection(self.url, timeout=.3, http_no_proxy=["127.0.0.1", "localhost"])
                for topic, type_name in self.mapper.advertised().items():
                    connection.send(json.dumps({"op": "advertise", "topic": topic, "type": type_name}))
                self.connected, self.error = True, ""
                while not self.stop_event.is_set():
                    attention = self._get_attention()
                    messages = self.mapper.messages(self.state_provider(), int(time.time() * 1000), attention)
                    for topic, message in messages:
                        connection.send(json.dumps({"op": "publish", "topic": topic, "msg": message}, allow_nan=False))
                        self.published += 1
                    # Consume rosbridge status responses and surface schema errors.
                    connection.settimeout(.001)
                    try:
                        raw = connection.recv()
                        if not raw:
                            raise ConnectionError("rosbridge closed")
                        status = json.loads(raw)
                        if status.get("op") == "status" and status.get("level") == "error":
                            self.error = str(status.get("msg"))
                    except websocket.WebSocketTimeoutException:
                        pass
                    finally:
                        connection.settimeout(.3)
                    self.stop_event.wait(.04)
            except Exception as exc:
                self.error = str(exc)
                self.failed += 1
            finally:
                self.connected = False
                if connection:
                    connection.close()
            self.stop_event.wait(.5)
