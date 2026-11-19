# H4 — local Qwen tool-aware chat adapter and a real stdio MCP smoke test.
import asyncio, uuid
from typing import Any
from pydantic import Field
from langchain_huggingface import ChatHuggingFace
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import timedelta


class LocalQwenChat(ChatHuggingFace):
    """ChatHuggingFace subclass bridging local Qwen text/tool and async protocols."""
    generator: Any = Field(exclude=True)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        schemas = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=schemas, tool_choice=tool_choice, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        schemas = kwargs.get("tools", [])
        observed = sum(isinstance(m, ToolMessage) for m in messages)
        if observed >= MAX_AGENT_ATTEMPTS:
            raise RuntimeError("Prebuilt ReAct tool-call budget exhausted (3); no fabricated success")
        serialized = []
        for message in messages:
            role = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}.get(message.type)
            if role is None:
                raise ValueError(f"Unsupported chat role {message.type}")
            content = message.content
            if isinstance(content, list):
                content = "\n".join(block.get("text", str(block)) if isinstance(block, dict) else str(block) for block in content)
            item = {"role": role, "content": str(content)}
            if isinstance(message, AIMessage) and message.tool_calls:
                item["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["args"]}} for c in message.tool_calls]
            serialized.append(item)
        last_question = next((str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        require_tool = bool(schemas and observed == 0 and re.search(r"spi|cpi|branch change|thesis|exam|academic|iit kanpur|iitk", last_question, re.I))
        available_names = {schema["function"]["name"] for schema in schemas}
        required_names = available_names
        if require_tool:
            if re.search(r"\d\s*:\s*\d", last_question) and re.search(r"spi|cpi", last_question, re.I):
                required_names = {"calculate_spi"} & available_names
            elif re.search(r"thesis", last_question, re.I):
                required_names = {"lookup_thesis_procedure", "search_doaa_knowledge"} & available_names
            elif re.search(r"exam|schedule|calendar", last_question, re.I):
                required_names = {"lookup_schedule", "search_doaa_knowledge"} & available_names
            else:
                required_names = {"search_doaa_knowledge"} & available_names
        call_formats = []
        for schema in schemas:
            function = schema["function"]
            if function["name"] not in required_names:
                continue
            properties = function["parameters"].get("properties", {})
            examples = {key: ("credits:gradepoint pairs from the question" if key == "grades" else "the full relevant search question") for key in properties}
            call_formats.append(shared.stable_json({"name": function["name"], "arguments": examples}))
        format_instruction = ("Select ONE appropriate tool and return ONLY its JSON call, no prose or eligibility answer. "
                              "CPI means Cumulative Performance Index; eligibility policy requires the DOAA search, not the SPI calculator. "
                              "Copy actual search terms or actual credit:grade pairs into arguments. Allowed call formats:\n" + "\n".join(call_formats))
        if require_tool:
            serialized.append({"role": "user", "content": format_instruction})
        elif observed:
            serialized.append({"role": "user", "content": "Use the actual tool observation above to answer the original question concisely. Do not repeat the same call. Preserve exact calculator output; if policy evidence is missing, say so."})
        raw = self.generator(serialized, max_tokens=256, tool_schemas=schemas or None)
        for attempt in range(3):
            try:
                calls = shared.parse_tool_calls(raw, schemas)
                if require_tool and not calls:
                    raise ValueError("A real tool call is required before this answer")
                if require_tool and any(call["name"] not in required_names for call in calls):
                    raise ValueError("Tool is outside this question's scope; choose " + ", ".join(sorted(required_names)))
                break
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"Malformed Qwen tool call after repair: {exc}") from exc
                raw = self.generator(serialized + [{"role": "user", "content": format_instruction + "\nReturn the JSON only. Validation error: " + str(exc)}],
                                     max_tokens=160, tool_schemas=schemas or None)
        raw_model_output = raw
        ai_calls = [{**call, "id": "call_" + uuid.uuid4().hex, "type": "tool_call"} for call in calls]
        if not ai_calls:
            observations = [m for m in messages if isinstance(m, ToolMessage) and m.name == "calculate_spi"]
            if observations:
                observed_text = observations[-1].content
                if isinstance(observed_text, list):
                    observed_text = "\n".join(b.get("text", "") for b in observed_text if isinstance(b, dict))
                if str(observed_text).startswith("SPI ="):
                    trace_event("prebuilt_calculator_grounding", raw=raw, observation=str(observed_text))
                    raw = str(observed_text)
            else:
                policy_observations = [m for m in messages if isinstance(m, ToolMessage)
                                       and m.name in {"search_doaa_knowledge", "lookup_thesis_procedure", "lookup_schedule"}]
                if policy_observations:
                    observed_text = policy_observations[-1].content
                    if isinstance(observed_text, list):
                        observed_text = "\n".join(b.get("text", "") for b in observed_text if isinstance(b, dict))
                    if str(observed_text).startswith("Verified calendar event:"):
                        raw = str(observed_text).split("\n\n", 1)[0]
                    elif re.search(r"(?m)^\[C\d+\]", str(observed_text)):
                        raw = source_constrained_answer(last_question, str(observed_text), raw)
        reply = AIMessage(content="" if ai_calls else raw, tool_calls=ai_calls)
        trace_event("prebuilt_model", raw=raw_model_output, emitted_content=raw, tool_calls=ai_calls)
        return ChatResult(generations=[ChatGeneration(message=reply)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return await asyncio.to_thread(self._generate, messages, stop=stop, run_manager=None, **kwargs)


chat_llm = LocalQwenChat(llm=get_llm_wrapper(), tokenizer=tokenizer, model_id=model_name, generator=_generate_messages)
REACT_SYSTEM_PROMPT = ("You are an IIT Kanpur academic assistant. Use calculate_spi for numeric credits:gradepoint arithmetic; "
                       "use the provided DOAA/thesis tools for policy/eligibility facts. Call tools before answering those questions. "
                       "After a tool result, answer concisely using it, cite its [Cnumber] labels, and preserve exact calculator numbers. "
                       "State missing evidence honestly. General conceptual questions may be answered directly.")


class NotebookMCPClient(MultiServerMCPClient):
    """Same real MCP session with an OS-backed stderr file, not Jupyter OutStream."""
    @asynccontextmanager
    async def session(self, server_name, *, auto_initialize=True):
        connection = self.connections[server_name]
        if connection["transport"] != "stdio":
            async with super().session(server_name, auto_initialize=auto_initialize) as session:
                yield session
            return
        parameters = StdioServerParameters(command=connection["command"], args=connection["args"],
                                           env=connection.get("env"), cwd=str(PROJECT_ROOT))
        # SDK v1's default errlog=sys.stderr cannot fileno() in this ipykernel.
        # Pass a genuine file through the documented SDK argument instead.
        with (ARTIFACT_ROOT / "mcp-server.stderr.log").open("a", encoding="utf-8") as error_log:
            async with stdio_client(parameters, errlog=error_log) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=300)) as session:
                    if auto_initialize:
                        await session.initialize()
                    yield session


def mcp_client(preload=False):
    env = dict(os.environ)
    env["A6_RUNTIME_ROOT"] = str(RUNTIME_ROOT)
    env["A6_MCP_PRELOAD"] = "1" if preload else "0"
    return NotebookMCPClient({"doaa": {"transport": "stdio", "command": sys.executable,
                                        "args": [str(PROJECT_ROOT / "doaa_mcp_server.py")],
                                        "env": env}})


async def _mcp_transport_smoke():
    client = mcp_client()
    async with client.session("doaa") as session:
        remote_tools = await load_mcp_tools(session)
        names = {t.name for t in remote_tools}
        assert names == {"search_doaa_knowledge", "calculate_spi", "lookup_thesis_procedure"}, names
        calculator = next(t for t in remote_tools if t.name == "calculate_spi")
        result = await calculator.ainvoke({"grades": "4:9,3:8,4:10,3:7,3:6"})
        assert "8.18" in str(result), result
        STORE.save("mcp", RUN_MODE + "_transport", {"status": "passed", "names": sorted(names), "observation": str(result), "at": utc_now()})
        return sorted(names)


async def await_mcp(coroutine_factory):
    """Own all sessions on one subprocess-capable loop; no nest_asyncio patch."""
    if sys.platform == "win32":
        def run_worker():
            loop = asyncio.ProactorEventLoop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coroutine_factory())
            finally:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
                loop.close()
        return await asyncio.to_thread(run_worker)
    return await coroutine_factory()


mcp_discovered_names = await asyncio.wait_for(await_mcp(_mcp_transport_smoke), timeout=120)
record_check("mcp_transport", discovered=mcp_discovered_names)
print("Real MCP discovery and calculator observation:", mcp_discovered_names)
