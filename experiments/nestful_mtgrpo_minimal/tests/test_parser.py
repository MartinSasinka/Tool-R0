from parser import parse_tool_call


def test_invalid_tag_fails():
    assert parse_tool_call("just some text, no tag").ok is False
    assert parse_tool_call("").reason == "empty_output"
    assert parse_tool_call("<tool_call_answer>not json</tool_call_answer>").ok is False


def test_two_calls_fail():
    text = (
        '<tool_call_answer>[{"name":"a","arguments":{}},'
        '{"name":"b","arguments":{}}]</tool_call_answer>'
    )
    r = parse_tool_call(text)
    assert r.ok is False
    assert r.reason == "multiple_tool_calls"


def test_two_tags_fail():
    text = (
        '<tool_call_answer>[{"name":"a","arguments":{}}]</tool_call_answer>'
        '<tool_call_answer>[{"name":"b","arguments":{}}]</tool_call_answer>'
    )
    assert parse_tool_call(text).reason == "multiple_tags"


def test_exactly_one_valid_call_passes():
    text = '<tool_call_answer>[{"name":"add","arguments":{"arg_0":1,"arg_1":2}}]</tool_call_answer>'
    r = parse_tool_call(text)
    assert r.ok is True
    assert r.is_terminal is False
    assert r.call["name"] == "add"
    assert r.call["arguments"] == {"arg_0": 1, "arg_1": 2}


def test_empty_list_is_terminal():
    r = parse_tool_call("<tool_call_answer>[]</tool_call_answer>")
    assert r.ok is True
    assert r.is_terminal is True


def test_arguments_not_dict_fails():
    text = '<tool_call_answer>[{"name":"add","arguments":5}]</tool_call_answer>'
    assert parse_tool_call(text).ok is False
