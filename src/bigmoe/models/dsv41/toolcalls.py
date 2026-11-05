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


class ToolStream:
    """Streams the answer text as it comes and holds back what may be a tool-call block: everything from
    "<｜DSML｜ calls>" on, plus a trailing partial marker or whitespace that might precede one. finish() parses the rest."""

    def __init__(self, tools):
        self.tools = tools
        self.buf, self.sent, self.in_calls = "", 0, False

    def feed(self, new: str) -> str:
        self.buf += new
        if self.in_calls:
            return ""
        i = self.buf.find(CALLS_OPEN, self.sent)
        if i >= 0:
            self.in_calls = True
            out = self.buf[self.sent:i].rstrip()
            self.sent = i
            return out
        end = len(self.buf)
        for k in range(min(len(CALLS_OPEN) - 1, end - self.sent), 0, -1):
            if CALLS_OPEN.startswith(self.buf[end - k:]):
                end -= k
                break
        while end > self.sent and self.buf[end - 1].isspace():
            end -= 1
        out = self.buf[self.sent:end]
        self.sent = end
        return out

    def finish(self):
        """(remaining content, tool_calls)"""
        return parse_tool_calls(self.buf[self.sent:], self.tools)
