"""Tests for claude_sdk."""

import pytest
from claude_sdk import Message, Conversation, ClaudeClient, ToolDef, CostTracker
from claude_sdk.message import ContentBlock, _estimate_tokens
from claude_sdk.cost import CostEntry


# ─── Message ────────────────────────────────────────────────────────────────

class TestMessage:
    def test_user_constructor(self):
        m = Message.user("hello")
        assert m.role == "user"
        assert m.content == "hello"

    def test_assistant_constructor(self):
        m = Message.assistant("hi there")
        assert m.role == "assistant"
        assert m.content == "hi there"

    def test_system_constructor(self):
        m = Message.system("be helpful")
        assert m.role == "system"

    def test_text_content_string(self):
        m = Message.user("plain text")
        assert m.text_content() == "plain text"

    def test_text_content_blocks(self):
        blocks = [
            ContentBlock(type="text", text="hello "),
            ContentBlock(type="text", text="world"),
        ]
        m = Message.from_blocks("user", blocks)
        assert m.text_content() == "hello \nworld"

    def test_content_blocks_from_string(self):
        m = Message.user("abc")
        blocks = m.content_blocks()
        assert len(blocks) == 1
        assert blocks[0].type == "text"
        assert blocks[0].text == "abc"

    def test_token_count_positive(self):
        m = Message.user("a" * 100)
        assert m.token_count() > 0

    def test_token_count_empty(self):
        m = Message(role="user", content="")
        assert m.token_count() >= 0

    def test_to_api_dict_string_content(self):
        m = Message.user("hi")
        d = m.to_api_dict()
        assert d["role"] == "user"
        assert d["content"] == "hi"

    def test_to_api_dict_blocks(self):
        blocks = [ContentBlock(type="text", text="yo")]
        m = Message.from_blocks("assistant", blocks)
        d = m.to_api_dict()
        assert isinstance(d["content"], list)
        assert d["content"][0]["type"] == "text"

    def test_repr_truncation(self):
        m = Message.user("a" * 200)
        r = repr(m)
        assert "…" in r

    def test_unique_ids(self):
        m1 = Message.user("a")
        m2 = Message.user("b")
        assert m1.id != m2.id


class TestContentBlock:
    def test_to_dict_text(self):
        b = ContentBlock(type="text", text="hello")
        d = b.to_dict()
        assert d == {"type": "text", "text": "hello"}

    def test_to_dict_extra_data(self):
        b = ContentBlock(type="tool_use", data={"tool_name": "foo"})
        d = b.to_dict()
        assert d["tool_name"] == "foo"

    def test_token_count(self):
        b = ContentBlock(type="text", text="hello world")
        assert b.token_count() > 0

    def test_token_count_no_text(self):
        b = ContentBlock(type="image")
        assert b.token_count() == 0


# ─── Conversation ───────────────────────────────────────────────────────────

class TestConversation:
    def test_add_and_len(self):
        c = Conversation()
        c.add(Message.user("hi"))
        c.add(Message.assistant("hello"))
        assert len(c) == 2

    def test_add_user_shortcut(self):
        c = Conversation()
        m = c.add_user("test")
        assert m.role == "user"
        assert len(c) == 1

    def test_add_system_sets_prompt(self):
        c = Conversation()
        c.add_system("be helpful")
        assert c.system_prompt == "be helpful"

    def test_pop(self):
        c = Conversation()
        c.add_user("a")
        c.add_user("b")
        popped = c.pop()
        assert popped.content == "b"
        assert len(c) == 1

    def test_pop_empty(self):
        c = Conversation()
        assert c.pop() is None

    def test_clear(self):
        c = Conversation()
        c.add_user("a")
        c.clear()
        assert len(c) == 0

    def test_total_tokens(self):
        c = Conversation()
        c.add_user("hello world")
        assert c.total_tokens() > 0

    def test_to_api_messages_skips_system(self):
        c = Conversation()
        c.add_system("system prompt")
        c.add_user("hello")
        api = c.to_api_messages()
        assert len(api) == 1
        assert api[0]["role"] == "user"

    def test_trim_to_context(self):
        c = Conversation(max_context_tokens=20)
        c.add_user("a" * 100)
        c.add_user("b" * 100)
        c.add_user("short")
        c.trim_to_context(reserve_tokens=0)
        # should have dropped messages to fit
        assert c.total_tokens() <= 20 or len(c) <= 2

    def test_truncate_to_last_n(self):
        c = Conversation()
        c.add_system("sys")
        for i in range(10):
            c.add_user(f"msg {i}")
        c.truncate_to_last_n(3)
        # system + last 3
        assert len(c) == 4

    def test_last_message(self):
        c = Conversation()
        assert c.last_message() is None
        c.add_user("hi")
        assert c.last_message().content == "hi"

    def test_iteration(self):
        c = Conversation()
        c.add_user("a")
        c.add_user("b")
        roles = [m.role for m in c]
        assert roles == ["user", "user"]


# ─── ToolDef ────────────────────────────────────────────────────────────────

class TestToolDef:
    def test_basic_creation(self):
        t = ToolDef(name="search", description="Search the web")
        assert t.name == "search"

    def test_default_schema(self):
        t = ToolDef(name="x", description="x")
        assert t.input_schema["type"] == "object"

    def test_to_api_dict(self):
        t = ToolDef(
            name="calc",
            description="Calculate",
            input_schema={"type": "object", "properties": {"expr": {"type": "string"}}},
        )
        d = t.to_api_dict()
        assert d["name"] == "calc"
        assert "input_schema" in d

    def test_execute_with_handler(self):
        t = ToolDef(name="add", description="Add", handler=lambda a, b: a + b)
        assert t.execute(a=1, b=2) == 3

    def test_execute_no_handler_raises(self):
        t = ToolDef(name="x", description="x")
        with pytest.raises(RuntimeError, match="no handler"):
            t.execute()

    def test_function_decorator(self):
        @ToolDef.function(name="echo", description="Echo input")
        def echo(text: str) -> str:
            return text

        assert isinstance(echo, ToolDef)
        assert echo.name == "echo"
        assert echo.execute(text="hi") == "hi"


# ─── CostTracker ────────────────────────────────────────────────────────────

class TestCostTracker:
    def test_estimate(self):
        ct = CostTracker()
        cost = ct.estimate(1000, 500, "claude-3.5-sonnet")
        assert cost > 0
        # 1000 * 3.0 + 500 * 15.0 = 3000 + 7500 = 10500 / 1M = 0.0105
        assert abs(cost - 0.0105) < 0.0001

    def test_record(self):
        ct = CostTracker()
        entry = ct.record(1000, 500)
        assert isinstance(entry, CostEntry)
        assert len(ct.entries) == 1

    def test_total_cost(self):
        ct = CostTracker()
        ct.record(1000, 500)
        ct.record(2000, 1000)
        assert ct.total_cost() > 0

    def test_total_tokens(self):
        ct = CostTracker()
        ct.record(100, 50)
        ct.record(200, 100)
        inp, out = ct.total_tokens()
        assert inp == 300
        assert out == 150

    def test_by_model(self):
        ct = CostTracker()
        ct.record(100, 50, "claude-3.5-sonnet")
        ct.record(100, 50, "claude-3-opus")
        breakdown = ct.by_model()
        assert "claude-3.5-sonnet" in breakdown
        assert "claude-3-opus" in breakdown

    def test_default_model(self):
        ct = CostTracker(default_model="claude-3-haiku")
        entry = ct.record(1000, 1000)
        assert entry.model == "claude-3-haiku"

    def test_repr(self):
        ct = CostTracker()
        ct.record(100, 50)
        r = repr(ct)
        assert "calls=1" in r


# ─── ClaudeClient ───────────────────────────────────────────────────────────

class TestClaudeClient:
    def test_build_request_basic(self):
        client = ClaudeClient(model="claude-3.5-sonnet")
        conv = Conversation()
        conv.add_user("hello")
        req = client.build_request(conv)
        assert req["model"] == "claude-3.5-sonnet"
        assert req["max_tokens"] == 4096
        assert len(req["messages"]) == 1

    def test_build_request_with_system(self):
        client = ClaudeClient(system="be helpful")
        conv = Conversation()
        conv.add_user("hi")
        req = client.build_request(conv)
        assert req["system"] == "be helpful"

    def test_build_request_with_tools(self):
        tool = ToolDef(name="search", description="Search")
        client = ClaudeClient(tools=[tool])
        conv = Conversation()
        conv.add_user("find info")
        req = client.build_request(conv)
        assert len(req["tools"]) == 1
        assert req["tools"][0]["name"] == "search"

    def test_build_request_custom_params(self):
        client = ClaudeClient()
        conv = Conversation()
        conv.add_user("hi")
        req = client.build_request(conv, max_tokens=2048, temperature=0.5)
        assert req["max_tokens"] == 2048
        assert req["temperature"] == 0.5

    def test_simulate_response(self):
        client = ClaudeClient()
        msg = client.simulate_response("Here is the answer")
        assert msg.role == "assistant"
        assert msg.text_content() == "Here is the answer"
        assert msg.model == client.model
        assert "input_tokens" in msg.usage
        assert client.cost_tracker.total_cost() > 0

    def test_add_tool(self):
        client = ClaudeClient()
        client.add_tool(ToolDef(name="t1", description="test"))
        assert len(client.tools) == 1

    def test_find_tool(self):
        t = ToolDef(name="search", description="Search")
        client = ClaudeClient(tools=[t])
        assert client.find_tool("search") is t
        assert client.find_tool("missing") is None

    def test_system_from_conversation(self):
        client = ClaudeClient()  # no system set on client
        conv = Conversation()
        conv.system_prompt = "conv system"
        conv.add_user("hi")
        req = client.build_request(conv)
        assert req["system"] == "conv system"

    def test_repr(self):
        client = ClaudeClient()
        r = repr(client)
        assert "claude-3.5-sonnet" in r
