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


def stream(chunks):
    ts = ToolStream(TOOLS)
    streamed = "".join(ts.feed(c) for c in chunks)
    content, calls = ts.finish()
    return streamed, content, calls


def test_stream_holds_back_the_call_block_char_by_char():
    streamed, content, calls = stream(list("Let me look." + SESSION_OUTPUT))
    assert streamed == "Let me look."
    assert content == "" and [args_of(c) for c in calls] == [("bash", {"command": "ls -la"})]


def test_stream_releases_a_false_marker_prefix_and_whitespace():
    streamed, content, calls = stream(["Use <", "｜DSM", "L tags?", "\n\n", "Done."])
    assert streamed + content == "Use <｜DSML tags?\n\nDone." and calls == []


def test_stream_plain_answer_is_streamed_whole():
    streamed, content, calls = stream(["Hello", " there", "!\n"])
    assert (streamed, content, calls) == ("Hello there!", "", [])


def test_normalize_messages_turns_argument_strings_into_dicts():
    msgs = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}},
        {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": "not json"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "README.md"}]
    out = normalize_messages(msgs)
    assert out[0]["content"] == ""
    assert [tc["function"]["arguments"] for tc in out[0]["tool_calls"]] == [{"command": "ls"}, {"arguments": "not json"}]
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}', "input is not mutated"
