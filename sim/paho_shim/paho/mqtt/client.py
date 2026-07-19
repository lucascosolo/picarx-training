#!/usr/bin/env python3
"""
paho.mqtt.client shim for training simulation.

The robot's broker_client.Bus does `import paho.mqtt.client as mqtt`
and uses a small slice of the paho API. During simulated training the
launcher prepends sim/paho_shim to PYTHONPATH so the UNMODIFIED robot
modules import this file instead, and their traffic rides the
simulator's minibus (see sim/minibus.py) rather than a real MQTT
broker - which the training machine doesn't have.

Only the surface Bus actually touches is implemented:

    client = mqtt.Client()
    client.on_connect = fn            # fired once, on connect
    client.connect(host, port, keepalive)
    client.loop_start()
    client.subscribe(topic)
    client.message_callback_add(topic_filter, callback)
    client.publish(topic, payload_json_str)

Callbacks receive (client, userdata, msg) where msg has .topic (str)
and .payload (bytes) - exactly what Bus's _on_message expects.

To run against a real broker instead, simply don't put this shim on
PYTHONPATH: the real paho package takes over and nothing else changes.
"""
import json
import os
import socket
import threading

# Import minibus for its topic matching + wire protocol. The shim dir is
# self-contained on PYTHONPATH, so locate sim/ relative to this file.
import sys
_SIM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
from sim.minibus import topic_matches  # noqa: E402


class MQTTMessage:
    __slots__ = ("topic", "payload")

    def __init__(self, topic, payload):
        self.topic = topic
        self.payload = payload


class Client:
    def __init__(self, *args, **kwargs):
        self.on_connect = None
        self.on_message = None
        self._sock = None
        self._send_lock = threading.Lock()
        self._callbacks = []          # (topic_filter, fn)
        self._cb_lock = threading.Lock()
        self._subscribed = []
        self._connected = False

    # ---- paho API surface ----

    def connect(self, host, port=1883, keepalive=60):
        # SIM_BUS_PORT lets the launcher redirect all modules to a
        # different port without touching them (default: same as asked).
        port = int(os.environ.get("SIM_BUS_PORT", port))
        host = os.environ.get("SIM_BUS_HOST", host)
        self._sock = socket.create_connection((host, port), timeout=10.0)
        self._sock.settimeout(None)
        self._connected = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True,
                                        name="paho-shim-read")
        self._reader.start()
        if self.on_connect:
            try:
                self.on_connect(self, None, {}, 0)
            except Exception as e:
                print(f"paho shim: on_connect raised: {e!r}")
        return 0

    def loop_start(self):
        # The reader thread already runs; nothing more to do.
        return None

    def loop_stop(self):
        return None

    def disconnect(self):
        self._connected = False
        try:
            self._sock.close()
        except OSError:
            pass

    def subscribe(self, topic, qos=0):
        if topic not in self._subscribed:
            self._subscribed.append(topic)
        self._send({"op": "sub", "topic": topic})
        return (0, 0)

    def message_callback_add(self, topic_filter, callback):
        with self._cb_lock:
            self._callbacks.append((topic_filter, callback))

    def publish(self, topic, payload=None, qos=0, retain=False):
        if isinstance(payload, bytes):
            payload = payload.decode()
        self._send({"op": "pub", "topic": topic, "payload": payload or ""})
        return _PublishResult()

    # ---- internals ----

    def _send(self, msg):
        if not self._connected:
            return
        data = (json.dumps(msg) + "\n").encode()
        try:
            with self._send_lock:
                self._sock.sendall(data)
        except OSError as e:
            print(f"paho shim: send failed: {e}")

    def _read_loop(self):
        buf = b""
        while self._connected:
            try:
                chunk = self._sock.recv(65536)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    self._dispatch(line)

    def _dispatch(self, line):
        try:
            msg = json.loads(line.decode())
        except (ValueError, UnicodeDecodeError):
            return
        if msg.get("op") != "pub":
            return
        topic = msg.get("topic", "")
        mqtt_msg = MQTTMessage(topic, msg.get("payload", "").encode())
        with self._cb_lock:
            matched = [fn for pat, fn in self._callbacks if topic_matches(pat, topic)]
        for fn in matched:
            try:
                fn(self, None, mqtt_msg)
            except Exception as e:
                # paho swallows callback exceptions on its network thread;
                # broker_client also guards - belt and braces.
                print(f"paho shim: callback for {topic} raised: {e!r}")
        if not matched and self.on_message:
            try:
                self.on_message(self, None, mqtt_msg)
            except Exception as e:
                print(f"paho shim: on_message raised: {e!r}")


class _PublishResult:
    rc = 0

    def wait_for_publish(self, timeout=None):
        return None
