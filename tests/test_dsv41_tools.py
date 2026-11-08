"""DeepSeek-V4.1-Flash server tool calling (bigmoe.models.dsv41.toolcalls): DSML <-> OpenAI tool_calls. No GPU."""
import json

from bigmoe.models.dsv41.toolcalls import ToolStream, normalize_messages, parse_tool_calls

TOOLS = [
    {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}, "timeoutMs": {"type": "number"}}}}},
    {"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {
        "file_path": {"type": "string"}}}}},
]
# what the model answered in a DSH session when the server did not pass the tools through
SESSION_OUTPUT = ('\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="bash">\n'
                  '<｜DSML｜ parameter name="command" string="true">ls -la</｜DSML｜ parameter>\n'
                  '</｜DSML｜ invoke>\n</｜DSML｜ calls>')


def args_of(call):
    return call["function"]["name"], json.loads(call["function"]["arguments"])


def test_parse_session_output():
    content, calls = parse_tool_calls(SESSION_OUTPUT, TOOLS)
    assert content == ""
    assert [args_of(c) for c in calls] == [("bash", {"command": "ls -la"})]
    assert calls[0]["type"] == "function" and calls[0]["id"].startswith("call_")


def test_parse_json_params_multiple_invokes_and_text():
    out = ('Let me check.\n\n<｜DSML｜ calls>\n'
           '<｜DSML｜ invoke name="bash">\n'
           '<｜DSML｜ parameter name="command" string="true">echo "a <b>"\nwc -l x</｜DSML｜ parameter>\n'
           '<｜DSML｜ parameter name="timeoutMs" string="false">5000</｜DSML｜ parameter>\n'
           '</｜DSML｜ invoke>\n'
           '<｜DSML｜ invoke name="read">\n'
           '<｜DSML｜ parameter name="file_path" string="true">README.md</｜DSML｜ parameter>\n'
           '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    content, calls = parse_tool_calls(out, TOOLS)
    assert content == "Let me check."
    assert [args_of(c) for c in calls] == [("bash", {"command": 'echo "a <b>"\nwc -l x', "timeoutMs": 5000}),
                                           ("read", {"file_path": "README.md"})]


def test_without_tools_the_text_is_left_alone():
    assert parse_tool_calls(SESSION_OUTPUT, None) == (SESSION_OUTPUT.strip(), [])
    assert parse_tool_calls("  plain answer ", TOOLS) == ("plain answer", [])


def test_truncated_block_and_bad_json():
    out = ('<｜DSML｜ calls>\n<｜DSML｜ invoke name="bash">\n'
           '<｜DSML｜ parameter name="timeoutMs" string="false">not json</｜DSML｜ parameter>\n')   # hit max_tokens
    content, calls = parse_tool_calls(out, TOOLS)
    assert content == "" and [args_of(c) for c in calls] == [("bash", {"timeoutMs": "not json"})]


def test_namespaced_tool_maps_back_to_its_function_name():
    tools = [{"type": "function", "namespace": "fs", "function": {"name": "read", "parameters": {}}}]
    out = '<｜DSML｜ calls>\n<｜DSML｜ invoke name="fs::read">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'
    assert [args_of(c) for c in parse_tool_calls(out, tools)[1]] == [("read", {})]


def accumulate(deltas):
    """What an OpenAI client builds from streamed tool_calls deltas: per index, the first id/name and the
    concatenated arguments."""
    calls = {}
    for d in deltas:
        c = calls.setdefault(d["index"], {"id": None, "type": "function", "function": {"name": None, "arguments": ""}})
        c["id"] = c["id"] or d.get("id")
        c["function"]["name"] = c["function"]["name"] or d["function"].get("name")
        c["function"]["arguments"] += d["function"].get("arguments", "")
    return [calls[i] for i in sorted(calls)]


def stream(chunks):
    """-> (content streamed while feeding, content at finish, accumulated calls, per-feed deltas, called)"""
    ts = ToolStream(TOOLS)
    streamed, per_feed = "", []
    for c in chunks:
        out, deltas = ts.feed(c)
        streamed += out
        per_feed.append(deltas)
    content, deltas, called = ts.finish()
    per_feed.append(deltas)
    return streamed, content, accumulate([d for f in per_feed for d in f]), per_feed, called


def test_stream_char_by_char_matches_the_whole_text_parse():
    for out in (SESSION_OUTPUT, 'Let me check.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="bash">\n'
                '<｜DSML｜ parameter name="command" string="true">echo "a <b>" \\ \t é\nwc -l x</｜DSML｜ parameter>\n'
                '<｜DSML｜ parameter name="timeoutMs" string="false">5000</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n'
                '<｜DSML｜ invoke name="read">\n<｜DSML｜ parameter name="file_path" string="true">README.md'
                '</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'):
        want_content, want = parse_tool_calls(out, TOOLS)
        for chunks in (list(out), [out], [out[i:i + 7] for i in range(0, len(out), 7)]):
            streamed, content, calls, _, called = stream(chunks)
            assert (streamed + content).strip() == want_content and called
            assert [args_of(c) for c in calls] == [args_of(c) for c in want]
            assert all(c["id"].startswith("call_") for c in calls)


def test_a_long_string_argument_streams_while_it_is_generated():
    """A `write` of a whole file must not be held back until the call ends (a client drops a silent stream)."""
    head = '<｜DSML｜ calls>\n<｜DSML｜ invoke name="write">\n<｜DSML｜ parameter name="content" string="true">'
    body = ["<html>\n", "  <p>\"chart\"</p>\n", "</ht", "ml>"]
    streamed, content, calls, per_feed, _ = stream([head] + body + ["</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n"])
    assert per_feed[0][0]["function"]["name"] == "write"
    assert all(per_feed[1 + i] for i in range(len(body))), "every body chunk sends an arguments delta"
    assert json.loads(calls[0]["function"]["arguments"]) == {"content": "".join(body)}


def test_a_partial_closing_tag_is_held_back():
    head = '<｜DSML｜ calls>\n<｜DSML｜ invoke name="bash">\n<｜DSML｜ parameter name="command" string="true">ls'
    _, _, calls, per_feed, _ = stream([head, "</｜DS", "ML｜ parameter>", "</｜DSML｜ invoke>"])
    assert per_feed[1] == []
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls"}


def test_stream_cut_off_mid_call_still_gives_complete_json():
    out = ('<｜DSML｜ calls>\n<｜DSML｜ invoke name="bash">\n'
           '<｜DSML｜ parameter name="timeoutMs" string="false">not json</｜DSML｜ parameter>\n'
           '<｜DSML｜ parameter name="command" string="true">echo hal')     # hit max_tokens
    _, content, calls, _, called = stream(list(out))
    assert content == "" and called
    assert [args_of(c) for c in calls] == [("bash", {"timeoutMs": "not json", "command": "echo hal"})]


def test_stream_text_after_a_call_block_and_a_namespaced_name():
    ts = ToolStream([{"type": "function", "namespace": "fs", "function": {"name": "read", "parameters": {}}}])
    out, deltas = ts.feed('<｜DSML｜ calls>\n<｜DSML｜ invoke name="fs::read">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n\nok')
    content, rest, called = ts.finish()
    assert (out + content, called) == ("ok", True)
    assert [args_of(c) for c in accumulate(deltas + rest)] == [("read", {})]


def test_stream_releases_a_false_marker_prefix_and_whitespace():
    streamed, content, calls, _, called = stream(["Use <", "｜DSM", "L tags?", "\n\n", "Done."])
    assert streamed + content == "Use <｜DSML tags?\n\nDone." and calls == [] and not called


def test_stream_plain_answer_is_streamed_whole():
    streamed, content, calls, _, called = stream(["Hello", " there", "!\n"])
    assert (streamed, content, calls, called) == ("Hello there!", "", [], False)


def test_normalize_messages_turns_argument_strings_into_dicts():
    msgs = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}},
        {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": "not json"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "README.md"}]
    out = normalize_messages(msgs)
    assert out[0]["content"] == ""
    assert [tc["function"]["arguments"] for tc in out[0]["tool_calls"]] == [{"command": "ls"}, {"arguments": "not json"}]
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}', "input is not mutated"
