# H3 — genuine tools plus explicitly labeled control-flow test fixtures.
from unittest.mock import patch
assert "8.18" in calculate_spi.invoke("4:9,3:8,4:10,3:7,3:6")
for bad in ("", "0:9", "3:11", "nan:8", "4:inf", "4:9,", "4:9:3"):
    assert calculate_spi.invoke(bad).startswith("Error:"), bad
for selected_tool in tools:
    argument = "grades" if selected_tool.name == "calculate_spi" else "query"
    assert "Error:" in selected_tool.invoke({argument: ""})
    assert "Error:" in selected_tool.invoke({"wrong_argument": "x"})
assert route_after_reasoning({"next_action": "tool"}) == "tool_executor"
assert route_after_reasoning({"next_action": "answer"}) == "answer_generator"
assert route_after_quality_check({"next_action": "retry", "iteration": 2}) == "reasoner"
assert route_after_quality_check({"next_action": "retry", "iteration": 3}) == "end"
assert len(AgentState.__annotations__) == 7

def _control_fixture(prompt, max_tokens=256):
    if prompt.startswith("Select the next action"):
        return '{"action":"answer","tool_name":"","tool_input":""}'
    if prompt.startswith("Review answer"):
        return '{"accept":false,"reason":"Deliberately forced rejection in a control-flow unit fixture"}'
    return "Fixture answer used only to verify graph termination, not a model result."

_trace_before_fixture = len(RUN_EVENTS)
_original_run_qwen = run_qwen
try:
    run_qwen = _control_fixture
    _forced = agent.invoke(new_turn_state("Control-flow unit fixture"), config={"recursion_limit": 24})
finally:
    run_qwen = _original_run_qwen
assert _forced["iteration"] == 3 and _forced["next_action"] == "accept"
assert "Limitation:" in _forced["final_answer"]
fixture_events = RUN_EVENTS[_trace_before_fixture:]
STORE.save("test_fixtures", RUN_MODE + "_retry_exhaustion", {"synthetic": True, "events": fixture_events})
del RUN_EVENTS[_trace_before_fixture:]
print("Tool/input and routing checks passed. Forced retry fixture terminated after exactly three attempts; it is not an empirical agent answer.")
record_check("agent_control_flow", fixture=True, attempts=3)
