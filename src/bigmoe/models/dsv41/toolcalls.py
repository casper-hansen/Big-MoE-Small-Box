"""Tool calling for the DeepSeek-V4.1-Flash server (stdlib only, so it is testable without the GPU stack).

The checkpoint's chat template already speaks tools: it renders the request's `tools` into the system message, past
assistant `tool_calls` as DSML blocks and `tool` results as <tool_result> user content. The model answers with

    <｜DSML｜ calls>
    <｜DSML｜ invoke name="bash">
    <｜DSML｜ parameter name="command" string="true">ls -la</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    </｜DSML｜ calls>

which this module turns back into OpenAI `tool_calls` (string="true": raw text, "false": JSON).
"""
import json
import re
import uuid

DSML = "｜DSML｜"
CALLS_OPEN = f"<{DSML} calls>"
CALLS_BLOCK = re.compile(rf"<{DSML} calls>(.*?)(?:</{DSML} calls>|$)", re.S)
INVOKE = re.compile(rf"<{DSML} invoke name=\"([^\"]*)\">(.*?)(?:</{DSML} invoke>|$)", re.S)
PARAM = re.compile(rf"<{DSML} parameter name=\"([^\"]*)\"(?: string=\"(true|false)\")?>(.*?)</{DSML} parameter>", re.S)


def normalize_messages(messages):
    """OpenAI -> DeepSeek template: assistant tool-call arguments as dicts (the template renders a JSON string as one
    parameter called "arguments"), content never None."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function", tc)
                if isinstance(fn.get("arguments"), str):
                    try:
                        args = json.loads(fn["arguments"] or "{}")
                        fn["arguments"] = args if isinstance(args, dict) else {"arguments": args}
                    except ValueError:
                        fn["arguments"] = {"arguments": fn["arguments"]}
                tcs.append(tc)
            m["tool_calls"] = tcs
        out.append(m)
    return out


def _tool_names(tools) -> dict:
    """Name as the template shows it ("ns::name" for namespaced tools) -> the function name the client declared."""
    names = {}
    for t in tools or ():
        fn = t.get("function", t)
        name = fn.get("name")
        if not name:
            continue
        ns = t.get("namespace") or fn.get("namespace")
        if isinstance(ns, dict):
            ns = ns.get("name")
        names[f"{ns}::{name}" if ns else name] = name
        names.setdefault(name, name)
    return names


def parse_tool_calls(content: str, tools):
    """(content without DSML call blocks, [OpenAI tool_calls]). No tools in the request: content is left untouched."""
    if not tools or CALLS_OPEN not in content:
        return content.strip(), []
    names = _tool_names(tools)
    calls = []
    for block in CALLS_BLOCK.findall(content):
        for name, body in INVOKE.findall(block):
            args = {}
            for key, is_string, value in PARAM.findall(body):
                if is_string == "false":
                    try:
                        args[key] = json.loads(value)
                        continue
                    except ValueError:
                        pass
                args[key] = value
            calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                          "function": {"name": names.get(name, name), "arguments": json.dumps(args, ensure_ascii=False)}})
    return CALLS_BLOCK.sub("", content).strip(), calls


CALLS_CLOSE = f"</{DSML} calls>"
INVOKE_CLOSE = f"</{DSML} invoke>"
PARAM_CLOSE = f"</{DSML} parameter>"
INVOKE_TAG = re.compile(rf"<{DSML} invoke name=\"([^\"]*)\">")
PARAM_TAG = re.compile(rf"<{DSML} parameter name=\"([^\"]*)\"(?: string=\"(true|false)\")?>")


def _partial_suffix(buf: str, start: int, marker: str) -> int:
    """Length of the longest tail of buf[start:] that is a proper prefix of marker (0 if none)."""
    for k in range(min(len(marker) - 1, len(buf) - start), 0, -1):
        if marker.startswith(buf[len(buf) - k:]):
            return k
    return 0


def _json_value(raw: str) -> str:
    """A string="false" parameter as JSON text: its own JSON if it parses, else the raw text as a JSON string."""
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except ValueError:
        return json.dumps(raw, ensure_ascii=False)


class ToolStream:
    """Turns the answer text into OpenAI streaming deltas as it is generated: plain text as `content`, DSML call blocks
    as `tool_calls` deltas (the call's id and name when its invoke tag closes, then its JSON arguments piece by piece,
    string parameter values as they are decoded). A long call (a whole file in a `write`) keeps the stream moving, so
    clients see progress and no idle timeout fires: the HTTP stack of Node's fetch drops a response body that is silent
    for 300 s with "terminated", which is what holding a call block back until the end did.

    Held back only: a trailing partial marker or whitespace that might precede a call block, a partial closing tag in a
    string value, a string="false" value until it closes (it is short, and parsed as a whole), and an unfinished tag.
    The concatenated arguments of each call are the JSON object parse_tool_calls would have produced."""

    def __init__(self, tools):
        self.names = _tool_names(tools)
        self.buf, self.pos = "", 0
        self.state = "text"          # text | calls | invoke | value
        self.calls = 0               # invokes started (the open one has index calls - 1)
        self.params = 0              # parameters started in the open invoke
        self.is_string = True        # the open parameter's kind
        self.after_block = False     # strip whitespace between a closed call block and further text

    def feed(self, new: str):
        """-> (content text, [tool_calls deltas]) to send now."""
        self.buf += new
        return self._run(final=False)

    def finish(self):
        """-> (content text, [tool_calls deltas], whether there were tool calls). Closes what is still open (output cut
        off by max_tokens), so every call's arguments are a complete JSON object."""
        content, deltas = self._run(final=True)
        if self.state == "value":
            self._args(deltas, '"' if self.is_string else _json_value(self.buf[self.pos:]))
            self.pos, self.state = len(self.buf), "invoke"
        if self.state == "invoke":
            self._args(deltas, "}")
        self.state = "text"
        return content.strip(), deltas, self.calls > 0

    def _args(self, deltas, piece: str):
        if not piece:
            return
        i = self.calls - 1
        if deltas and deltas[-1]["index"] == i:          # merge with this call's delta from the same feed
            deltas[-1]["function"]["arguments"] += piece
        else:
            deltas.append({"index": i, "function": {"arguments": piece}})

    def _run(self, final: bool):
        content, deltas = [], []
        while self._step(content, deltas, final):
            pass
        return "".join(content), deltas

    def _step(self, content, deltas, final) -> bool:
        """Consume what can be decided from buf[pos:]; True if the state changed and there may be more to do."""
        buf, pos = self.buf, self.pos
        if self.state == "text":
            i = buf.find(CALLS_OPEN, pos)
            end = i if i >= 0 else len(buf)
            if i < 0 and not final:
                end -= _partial_suffix(buf, pos, CALLS_OPEN)
            if i >= 0 or not final:
                while end > pos and buf[end - 1].isspace():
                    end -= 1
            out = buf[pos:end]
            if self.after_block:
                out = out.lstrip()
            if out:
                content.append(out)
                self.after_block = False
            if i >= 0:
                self.pos, self.state = i + len(CALLS_OPEN), "calls"
                return True
            self.pos = end
            return False

        if self.state == "value":
            i = buf.find(PARAM_CLOSE, pos)
            if i < 0 and not self.is_string:
                return False                              # wait for the whole JSON value
            if i < 0:
                end = len(buf) - (0 if final else _partial_suffix(buf, pos, PARAM_CLOSE))
                self._args(deltas, json.dumps(buf[pos:end], ensure_ascii=False)[1:-1])
                self.pos = end
                return False
            raw = buf[pos:i]
            self._args(deltas, json.dumps(raw, ensure_ascii=False)[1:-1] + '"' if self.is_string else _json_value(raw))
            self.pos, self.state = i + len(PARAM_CLOSE), "invoke"
            return True

        # calls / invoke: the next complete tag decides; text between tags (newlines) is dropped
        j = buf.find("<", pos)
        k = buf.find(">", j) if j >= 0 else -1
        if k < 0:
            self.pos = j if j >= 0 else len(buf)
            return False
        tag, self.pos = buf[j:k + 1], k + 1
        if self.state == "calls":
            m = INVOKE_TAG.fullmatch(tag)
            if m:
                self.calls += 1
                self.params = 0
                name = self.names.get(m.group(1), m.group(1))
                deltas.append({"index": self.calls - 1, "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                               "function": {"name": name, "arguments": "{"}})
                self.state = "invoke"
            elif tag == CALLS_CLOSE:
                self.state, self.after_block = "text", True
            return True
        m = PARAM_TAG.fullmatch(tag)
        if m:
            key = json.dumps(m.group(1), ensure_ascii=False)
            self.is_string = m.group(2) != "false"
            self._args(deltas, (", " if self.params else "") + key + ": " + ('"' if self.is_string else ""))
            self.params += 1
            self.state = "value"
        elif tag in (INVOKE_CLOSE, CALLS_CLOSE):         # a calls close here: the invoke was never closed
            self._args(deltas, "}")
            self.state = "calls" if tag == INVOKE_CLOSE else "text"
            self.after_block = tag == CALLS_CLOSE
        return True
