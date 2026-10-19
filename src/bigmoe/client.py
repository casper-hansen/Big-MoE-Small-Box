import json
import urllib.error
import urllib.request

from .errors import BoxError


def _sse(resp):
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                return
            yield json.loads(body)


def _messages(messages):
    return [{"role": "user", "content": messages}] if isinstance(messages, str) else list(messages)


class Client:
    def __init__(self, url: str = "http://127.0.0.1:30000", model: str = None, timeout: float = 14400):
        self.url = url.rstrip("/")
        if self.url.endswith("/v1"):
            self.url = self.url[:-3]
        self.model = model
        self.timeout = timeout

    def __repr__(self):
        return f"Client({self.url!r})"

    def request(self, path: str, body=None, *, stream=False, timeout=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            raise BoxError(f"{path}: HTTP {e.code}: {e.read()[:2000].decode(errors='replace')}") from None
        except urllib.error.URLError as e:
            raise BoxError(f"{self.url}{path}: {e.reason} (is the server running?)") from None
        if stream:
            return resp
        with resp:
            return json.loads(resp.read())

    def health(self) -> bool:
        try:
            return bool(self.request("/health", timeout=5).get("ok"))
        except BoxError:
            return False

    def models(self) -> list:
        return self.request("/v1/models")["data"]

    def server_info(self) -> dict:
        return self.request("/server_info", timeout=30)

    def stats(self) -> dict:
        return self.request("/stats", timeout=30)

    def tokenize(self, text: str) -> list:
        return self.request("/tokenize", {"prompt": text})["tokens"]

    def _chat_body(self, messages, *, thinking, max_tokens, temperature, extra):
        body = {"model": self.model or "default", "messages": _messages(messages), "temperature": temperature}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        kw = dict(extra.pop("chat_template_kwargs", None) or {})
        kw.setdefault("enable_thinking", bool(thinking))
        body["chat_template_kwargs"] = kw
        body.update(extra)
        return body

    def chat_completion(self, messages, *, thinking=False, max_tokens=None, temperature=0.0, **extra) -> dict:
        return self.request("/v1/chat/completions", self._chat_body(messages, thinking=thinking, max_tokens=max_tokens,
                                                                    temperature=temperature, extra=extra))

    def chat(self, messages, *, stream=False, thinking=False, max_tokens=None, temperature=0.0, **extra):
        """stream=True yields only the answer's text; the reasoning is in chat_completion / chat_stream_events."""
        if not stream:
            return self.chat_completion(messages, thinking=thinking, max_tokens=max_tokens, temperature=temperature,
                                        **extra)["choices"][0]["message"]["content"]
        return self._chat_text_stream(messages, thinking, max_tokens, temperature, extra)

    def _chat_text_stream(self, messages, thinking, max_tokens, temperature, extra):
        for ev in self.chat_stream_events(messages, thinking=thinking, max_tokens=max_tokens, temperature=temperature,
                                          **extra):
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    yield piece

    def chat_stream_events(self, messages, *, thinking=False, max_tokens=None, temperature=0.0, **extra):
        body = self._chat_body(messages, thinking=thinking, max_tokens=max_tokens, temperature=temperature, extra=extra)
        body["stream"] = True
        yield from _sse(self.request("/v1/chat/completions", body, stream=True))

    def complete(self, prompt: str, *, stream=False, max_tokens=None, temperature=0.0, **extra):
        body = {"model": self.model or "default", "prompt": prompt, "temperature": temperature, **extra}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if not stream:
            return self.request("/v1/completions", body)["choices"][0]["text"]
        body["stream"] = True
        return (ch.get("text", "") for ev in _sse(self.request("/v1/completions", body, stream=True))
                for ch in ev.get("choices") or [] if ch.get("text"))

    def generate(self, *, input_ids=None, text=None, sampling_params=None, stream=False, **extra):
        """Native endpoint. Streamed events carry the whole text so far, not a delta."""
        body = {"sampling_params": sampling_params or {}, **extra}
        if input_ids is not None:
            body["input_ids"] = list(input_ids)
        else:
            body["text"] = text
        if not stream:
            return self.request("/generate", body)
        body["stream"] = True
        return _sse(self.request("/generate", body, stream=True))
