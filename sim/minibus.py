#!/usr/bin/env python3
"""
Minimal in-process pub/sub bus - the training simulator's stand-in for
an MQTT broker.

The robot's modules all talk through broker_client.Bus, whose entire
API surface is publish(topic, dict) / subscribe(topic, callback) with
exact-topic JSON messages over paho-mqtt. The development machine this
simulator targets has no MQTT broker and no pip, so this module
provides the same semantics over a ~200-line stdlib TCP server:

  wire protocol: one JSON object per line ("\n"-delimited), either
    {"op": "sub", "topic": "picarx/state/world"}
    {"op": "pub", "topic": "picarx/state/world", "payload": "<json string>"}

  server: fans every "pub" out to every connection whose subscription
  filters match the topic. MQTT-style "+" and "#" wildcards are
  supported even though the current modules only use exact topics.

Fidelity note: this is a transport substitute, not a protocol
substitute - topic names and payload bytes are identical to what the
real robot puts on MQTT, which is the abstraction the training design
relies on. Swapping in a real broker later only means removing the
paho shim from PYTHONPATH (see sim/paho_shim/) and running mosquitto.
"""
import json
import socket
import threading


DEFAULT_PORT = 1883  # same port the modules' Bus() dials, so they need no config


def topic_matches(pattern, topic):
    """MQTT topic filter matching ('+' one level, '#' rest)."""
    if pattern == topic:
        return True
    p_parts = pattern.split("/")
    t_parts = topic.split("/")
    for i, p in enumerate(p_parts):
        if p == "#":
            return True
        if i >= len(t_parts):
            return False
        if p != "+" and p != t_parts[i]:
            return False
    return len(p_parts) == len(t_parts)


class BusServer:
    """Threaded TCP fanout server. start() returns once listening."""

    def __init__(self, host="127.0.0.1", port=DEFAULT_PORT):
        self.host = host
        self.port = port
        self._server = None
        self._clients = []          # list of _ClientConn
        self._clients_lock = threading.Lock()
        self._running = False
        self._accept_thread = None

    def start(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        # With port=0 the OS assigns a free ephemeral port atomically at
        # bind; read it back so parallel runs never collide or race on a
        # fixed port list.
        self.port = self._server.getsockname()[1]
        self._server.listen(32)
        self._running = True
        self._accept_thread = threading.Thread(target=self._accept_loop,
                                               daemon=True, name="bus-accept")
        self._accept_thread.start()

    def stop(self):
        self._running = False
        try:
            self._server.close()
        except OSError:
            pass
        with self._clients_lock:
            for c in self._clients:
                c.close()
            self._clients.clear()

    # ---------- internals ----------

    def _accept_loop(self):
        while self._running:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            client = _ClientConn(conn, self)
            with self._clients_lock:
                self._clients.append(client)
            threading.Thread(target=client.read_loop, daemon=True,
                             name="bus-client").start()

    def _drop(self, client):
        with self._clients_lock:
            if client in self._clients:
                self._clients.remove(client)

    def _fanout(self, topic, payload_str):
        line = (json.dumps({"op": "pub", "topic": topic,
                            "payload": payload_str}) + "\n").encode()
        with self._clients_lock:
            targets = [c for c in self._clients if c.wants(topic)]
        for c in targets:
            c.send_bytes(line)


class _ClientConn:
    def __init__(self, sock, server):
        self.sock = sock
        self.server = server
        self.subs = []
        self._subs_lock = threading.Lock()
        self._send_lock = threading.Lock()

    def wants(self, topic):
        with self._subs_lock:
            return any(topic_matches(p, topic) for p in self.subs)

    def send_bytes(self, data):
        try:
            with self._send_lock:
                self.sock.sendall(data)
        except OSError:
            self.close()
            self.server._drop(self)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def read_loop(self):
        buf = b""
        while True:
            try:
                chunk = self.sock.recv(65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    self._handle(line)
        self.close()
        self.server._drop(self)

    def _handle(self, line):
        try:
            msg = json.loads(line.decode())
        except (ValueError, UnicodeDecodeError):
            return
        op = msg.get("op")
        if op == "sub":
            topic = msg.get("topic")
            if topic:
                with self._subs_lock:
                    if topic not in self.subs:
                        self.subs.append(topic)
        elif op == "pub":
            topic = msg.get("topic")
            if topic:
                self.server._fanout(topic, msg.get("payload", ""))


class BusClient:
    """Client for BusServer with the same shape as broker_client.Bus:
    publish(topic, dict) / subscribe(topic, callback). Used directly by
    the simulator process; robot modules reach the same server through
    the paho shim instead."""

    def __init__(self, host="127.0.0.1", port=DEFAULT_PORT):
        self.sock = socket.create_connection((host, port), timeout=5.0)
        self.sock.settimeout(None)
        self._send_lock = threading.Lock()
        self._callbacks = []        # (pattern, fn(payload_dict))
        self._cb_lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_loop, daemon=True,
                                        name="bus-client-read")
        self._reader.start()

    def publish(self, topic, payload: dict):
        self._send({"op": "pub", "topic": topic, "payload": json.dumps(payload)})

    def subscribe(self, topic, callback):
        with self._cb_lock:
            self._callbacks.append((topic, callback))
        self._send({"op": "sub", "topic": topic})

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def _send(self, msg):
        data = (json.dumps(msg) + "\n").encode()
        with self._send_lock:
            self.sock.sendall(data)

    def _read_loop(self):
        buf = b""
        while True:
            try:
                chunk = self.sock.recv(65536)
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
        try:
            payload = json.loads(msg.get("payload", ""))
        except ValueError:
            return
        with self._cb_lock:
            cbs = [fn for pat, fn in self._callbacks if topic_matches(pat, topic)]
        for fn in cbs:
            try:
                fn(payload)
            except Exception as e:  # mirror broker_client's guarded delivery
                print(f"minibus: callback for {topic} raised: {e!r}")
