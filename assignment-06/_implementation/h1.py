# H1 — shared implementation has been materialized above from the embedded source.
import importlib, re
importlib.invalidate_caches()
import doaa_shared as shared
shared = importlib.reload(shared)
from doaa_shared import (ArtifactStore, atomic_json, atomic_bytes, digest, utc_now,
                         normalized_f32, format_records, parse_json_object, parse_judge_response)
STORE = ArtifactStore(ARTIFACT_ROOT)
STORE.save("environment", "latest", ENVIRONMENT)


def record_check(name, **details):
    CHECKS[name] = {"status": "passed", "at": utc_now(), **details}
    STORE.save("checks", RUN_MODE + "_" + name, CHECKS[name])


def trace_event(node, state=None, **details):
    event = {"node": node, "at": utc_now(), "iteration": (state or {}).get("iteration", 0), **details}
    RUN_EVENTS.append(event)
    STORE.save("traces", RUN_MODE + "_current", RUN_EVENTS)
    return event


def model_fingerprint():
    import torch
    tok, net = get_qwen()
    return {"id": GENERATION_MODEL_ID, "revision": _MODEL_RESOURCES["qwen_revision"],
            "dtype": str(net.dtype), "torch": torch.__version__, "implementation": GENERATION_IMPLEMENTATION_KEY}


def embedding_fingerprint():
    encoder = get_embedding_model()
    config = encoder[0].auto_model.config
    return {"id": EMBEDDING_MODEL_ID, "revision": _MODEL_RESOURCES["bge_revision"],
            "max_seq_length": encoder.max_seq_length, "normalized": True, "prefix": ""}


def generation_record(user_message, max_tokens):
    """Inspect the real generation budget metadata, including cache hits."""
    config = {"model": model_fingerprint(), "messages": [{"role": "user", "content": user_message}],
              "tools": None, "max_tokens": max_tokens, "do_sample": False}
    return STORE.load("generations", digest(config))


def _generate_messages(messages, max_tokens=256, tool_schemas=None, cache=True):
    import torch
    if type(max_tokens) is not int or max_tokens <= 0 or max_tokens > 8192:
        raise ValueError("max_tokens must be an integer in [1,8192]")
    if not messages or any(not isinstance(m.get("content", ""), str) for m in messages):
        raise ValueError("Expected a nonempty list of text messages")
    tok, net = get_qwen()
    config = {"model": model_fingerprint(), "messages": messages, "tools": tool_schemas,
              "max_tokens": max_tokens, "do_sample": False}
    key = digest(config)
    saved = STORE.load("generations", key) if cache else None
    if saved and saved.get("status") == "complete" and isinstance(saved.get("text"), str) and saved["text"].strip():
        print(f"Generation cache hit: {key[:10]}")
        return saved["text"]
    formatted = tok.apply_chat_template(messages, tools=tool_schemas, tokenize=False, add_generation_prompt=True)
    encoded = tok(formatted, return_tensors="pt", add_special_tokens=False)
    limit = min(int(getattr(net.config, "max_position_embeddings", 32768)), 32768)
    if encoded.input_ids.shape[1] + max_tokens > limit:
        raise ValueError(f"Prompt plus output exceeds {limit} tokens; reduce history/context explicitly")
    device = net.get_input_embeddings().weight.device
    inputs_local = {name: tensor.to(device) for name, tensor in encoded.items()}
    started = time.perf_counter()
    with _INFERENCE_LOCK, torch.inference_mode():
        generated = net.generate(**inputs_local, max_new_tokens=max_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id,
                                 eos_token_id=tok.eos_token_id)
    suffix = generated[0, encoded.input_ids.shape[1]:]
    result = tok.decode(suffix, skip_special_tokens=True).strip()
    if not result:
        raise RuntimeError("Qwen generated an empty answer; no successful checkpoint saved")
    STORE.save("generations", key, {"status": "complete", "text": result, "config": config,
                                   "input_tokens": int(encoded.input_ids.shape[1]), "output_tokens": len(suffix),
                                   "seconds": time.perf_counter() - started})
    return result


def _run_qwen_impl(user_message, max_tokens):
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("user_message must be a nonempty string")
    return _generate_messages([{"role": "user", "content": user_message}], max_tokens)


def get_llm_wrapper():
    if "langchain_llm" not in _MODEL_RESOURCES:
        from transformers import pipeline
        from langchain_huggingface import HuggingFacePipeline
        from langchain_core.caches import BaseCache
        from langchain_core.outputs import Generation
        class CheckedPipelineCache(BaseCache):
            def lookup(self, prompt, llm_string):
                key = digest({"model": model_fingerprint(), "prompt": prompt, "llm": llm_string})
                saved = STORE.load("direct_pipeline", key)
                if saved:
                    print("Direct pipeline cache hit:", key[:10])
                    return [Generation(**row) for row in saved["generations"]]
                return None
            def update(self, prompt, llm_string, return_val):
                key = digest({"model": model_fingerprint(), "prompt": prompt, "llm": llm_string})
                STORE.save("direct_pipeline", key, {"generations": [row.model_dump(mode="json") for row in return_val]})
            def clear(self, **kwargs):
                raise RuntimeError("Use a changed configuration key to invalidate cached pipeline results; existing artifacts are preserved.")
        tok, net = get_qwen()
        text_pipe = pipeline("text-generation", model=net, tokenizer=tok, max_new_tokens=256,
                             do_sample=False, return_full_text=False)
        _MODEL_RESOURCES["langchain_llm"] = HuggingFacePipeline(pipeline=text_pipe, model_id=GENERATION_MODEL_ID,
                                                              cache=CheckedPipelineCache())
    return _MODEL_RESOURCES["langchain_llm"]


def retrieve_shared(query, kind=None, k=2):
    metadata = globals().get("chunk_metadata", [])
    records = shared.retrieve_records(query, embedding_model, index, chunks, metadata, k=k, kind=kind)
    query_key = digest({"query": query, "kind": kind, "k": k,
                        "chunks": [digest(c) for c in chunks], "embedding": embedding_fingerprint(),
                        "retrieval_version": shared.RETRIEVAL_VERSION})
    STORE.save("retrieval", query_key, {"query": query, "records": records, "index_key": globals().get("index_key")})
    return records


def safe_search(query, kind=None):
    try:
        records = retrieve_shared(query, kind)
        return shared.format_schedule_records(query, records) if kind == "schedule" else format_records(records)
    except Exception as exc:
        return f"Error: DOAA retrieval failed ({type(exc).__name__}): {exc}"


def protect_tool(original):
    from langchain_core.tools import StructuredTool
    class InputSafeTool(StructuredTool):
        def invoke(self, input, config=None, **kwargs):
            try:
                return super().invoke(input, config=config, **kwargs)
            except Exception as exc:
                return f"Error: invalid input or tool failure ({type(exc).__name__}): {exc}"
        async def ainvoke(self, input, config=None, **kwargs):
            try:
                return await super().ainvoke(input, config=config, **kwargs)
            except Exception as exc:
                return f"Error: invalid input or tool failure ({type(exc).__name__}): {exc}"
    return InputSafeTool(name=original.name, description=original.description, args_schema=original.args_schema,
                         func=original.func, coroutine=original.coroutine, handle_validation_error=lambda exc: "Error: invalid tool input: " + str(exc),
                         handle_tool_error=True)


def build_rag_prompt(question, context, guidance=""):
    rules = ("Use only these excerpts to answer. Return at most FOUR short bullet points, one sentence each, "
             "under 110 words total. Cite [Cnumber]. No title, nested lists, explanations of your writing, "
             "or invented acronym expansions. Include relevant conditions/exceptions. State what the excerpts do not establish. ")
    if re.search(r"SPGC[-_]\d|extension.*SOTA|SOTA.*extension", context, re.I):
        rules += "Individual students' historical extensions are NOT a general deadline or a definition. "
    if re.search(r"\bCPI\b", question, re.I) and re.search(r"processing fee|process ing fee|conversion certificate", context, re.I):
        rules += "Certificate-processing fees do NOT define the academic CPI calculation. Do not calculate with fees; say if the requested academic formula is absent. "
    return (rules + ("\nREPAIR REQUIREMENTS: " + guidance if guidance else "")
            + f"\n\nEXCERPTS:\n{context}\n\nQUESTION: {question}\n"
            + "Write ONLY the factual answer to this question in short bullets now:\n")


def generate_grounded_rag(question, context, max_tokens=384):
    """Generate with Qwen, then publish only source-verifiable claims from the same top-2."""
    prompt = build_rag_prompt(question, context)
    raw = run_qwen(prompt, max_tokens=max_tokens)
    return source_constrained_answer(question, context, raw)


def source_constrained_answer(question, context, raw_answer):
    """Qwen selects actual source sentences; no unverified paraphrase becomes a policy fact."""
    identifiers = list(dict.fromkeys(int(value) for value in re.findall(r"(?m)^\[C(\d+)\]", context)))
    records = [{"chunk_id": number, "text": chunks[number], "sources": chunk_metadata[number]["sources"]}
               for number in identifiers if 0 <= number < len(chunks) and chunks[number] in context]
    labels = " ".join(f"[C{row['chunk_id']}]" for row in records)
    limit = shared.evidence_scope_limit(question, records)
    catalog = shared.prioritize_evidence_sentences(question, shared.evidence_sentence_catalog(records))
    key = digest({"question": question, "context": context, "guard": "source-sentences-3-bounded-catalog"})
    if limit or not catalog:
        answer = (limit or "No complete supporting statement can be verified from these retrieved excerpts.") + " " + labels
        STORE.save("source_guards", key, {"raw_model_answer": raw_answer, "answer": answer, "limitation": limit,
                   "context_hash": digest(context), "model": model_fingerprint(), "status": "verified_abstention"})
        return answer
    selection_prompt = ("Choose source sentence IDs that answer the QUESTION. Include the important conditions, exceptions, "
                        "approvals and limits. Select up to TEN useful distinct IDs; omit unrelated material. "
                        "Return ONLY one JSON array of integer IDs, for example [1,3]. Never invent an ID.\n"
                        + "\n".join(f"{row['id']}: {row['text']}" for row in catalog)
                        + f"\nQUESTION: {question}\nJSON array of selected IDs:")
    attempts = []
    for attempt in range(3):
        raw_selection = run_qwen(selection_prompt + ("\nReturn only a JSON array. Previous invalid format: " + attempts[-1]["raw"] if attempts else ""), max_tokens=128)
        try:
            chosen = shared.parse_sentence_selection(raw_selection, catalog)
            break
        except (ValueError, TypeError) as exc:
            attempts.append({"raw": raw_selection, "error": str(exc)})
    else:
        STORE.save("source_guards", key, {"status": "selection_failed", "attempts": attempts, "raw_model_answer": raw_answer})
        raise RuntimeError("Qwen did not produce a valid source-sentence selection after three attempts")
    model_selected = list(chosen)
    coverage_added = []
    if re.search(r"rules|procedure|process|eligib|requirements|conditions|withdrawal|drop", question, re.I):
        coverage_added = [row["id"] for row in catalog if row["id"] not in chosen]
        chosen = [row["id"] for row in catalog]
    selected = [next(row for row in catalog if row["id"] == number) for number in chosen]
    answer = "\n".join(f"- {row['text']} [C{row['chunk_id']}]" for row in selected)
    if not selected:
        answer = "The model did not identify a supporting complete statement in these excerpts. " + labels
    if selected and re.search(r"\bam i\b|\bcan i\b", question, re.I):
        answer += "\n\nThese source conditions do not establish your individual eligibility."
    STORE.save("source_guards", key, {"status": "verified_source_sentences", "raw_model_answer": raw_answer,
               "selection_raw": raw_selection, "model_selected_ids": model_selected, "coverage_added_ids": coverage_added,
               "selected": selected, "attempts": attempts, "answer": answer,
               "context_hash": digest(context), "model": model_fingerprint()})
    return answer


def latest_human_question(messages):
    from langchain_core.messages import HumanMessage
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError("Agent state has no HumanMessage")


def prior_personal_statements(messages, question):
    """Retrieve user-supplied evidence from this thread, never fixed persona facts."""
    from langchain_core.messages import HumanMessage
    if not re.search(r"\bmy\s+(?:name|department|degree|program)\b", question, re.I):
        return []
    user_turns = [str(message.content) for message in messages if isinstance(message, HumanMessage)]
    return [turn for turn in user_turns[:-1]
            if re.search(r"\bmy name is\b|\bmy department is\b|\bi am\b.*\bstudent\b|\bi study\b", turn, re.I)][-3:]


def new_turn_state(question):
    from langchain_core.messages import HumanMessage
    return {"messages": [HumanMessage(content=question)], "next_action": "", "tool_name": "", "tool_input": "",
            "tool_result": "", "final_answer": "", "iteration": 0}


def validate_action(obj):
    if obj.get("action") not in {"tool", "answer"}:
        raise ValueError("action must be tool or answer")
    if obj["action"] == "tool":
        if obj.get("tool_name") not in {t.name for t in tools}:
            raise ValueError("Unknown tool name")
        if not isinstance(obj.get("tool_input"), str) or not obj["tool_input"].strip():
            raise ValueError("tool_input must be a nonempty string")
    return obj


def conversation_key(thread_id):
    return digest({"thread_id": thread_id, "mode": RUN_MODE, "corpus": corpus["corpus_hash"],
                   "graph": GRAPH_IMPLEMENTATION_KEY})


def save_conversation(result, thread_id):
    from langchain_core.messages import messages_to_dict
    payload = {**result, "messages": messages_to_dict(result["messages"]), "graph_version": "a6-graph-1",
               "corpus_hash": corpus["corpus_hash"]}
    payload["graph_version"] = GRAPH_IMPLEMENTATION_KEY
    STORE.save("conversations", conversation_key(thread_id), payload)


def restore_conversation(graph, config):
    from langchain_core.messages import messages_from_dict
    if graph.get_state(config).values:
        return
    key = conversation_key(config["configurable"]["thread_id"])
    payload = STORE.load("conversations", key)
    if payload is None:
        legacy = STORE.load("conversations", digest(config["configurable"]["thread_id"]))
        if legacy and legacy.get("graph_version") == GRAPH_IMPLEMENTATION_KEY and legacy.get("corpus_hash") == corpus["corpus_hash"]:
            payload = legacy
            STORE.save("conversations", key, legacy)  # Copy a compatible legacy archive; preserve the original.
    if payload and payload.get("graph_version") == GRAPH_IMPLEMENTATION_KEY and payload.get("corpus_hash") == corpus["corpus_hash"]:
        state = {k: payload[k] for k in AgentState.__annotations__}
        state["messages"] = messages_from_dict(state["messages"])
        graph.update_state(config, state, as_node="quality_checker")


def require_valid_rag():
    global chunks, chunk_metadata, index, embedding_model
    if "embedding_model" not in globals():
        embedding_model = get_embedding_model()
    if "index" not in globals() or "chunks" not in globals():
        chunks, chunk_metadata, index, _ = shared.load_runtime_bundle(RUNTIME_ROOT)
    if not chunks or index.ntotal != len(chunks) or index.d != 384:
        raise RuntimeError("Invalid RAG state. Run Q1 from the beginning; no dummy data is substituted.")


_roundtrip_key = digest("H1 checkpoint smoke")
GENERATION_IMPLEMENTATION_KEY = shared.notebook_function_key(NOTEBOOK_PATH, {"_generate_messages", "_run_qwen_impl"})
STORE.save("smoke", _roundtrip_key, {"value": "atomic checkpoint"})
assert STORE.load("smoke", _roundtrip_key) == {"value": "atomic checkpoint"}
record_check("checkpoint_roundtrip")
