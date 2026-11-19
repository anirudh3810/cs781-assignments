"""Q5 — manual fake quantization and response-only QLoRA sentiment tuning.

The notebook writes this exact source to q5_qlora.py during Run All. Imports of
Transformers, PEFT and bitsandbytes are delayed so CPU quantizer checks stay cheap.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

import common


def _native_bf16():
    """Use BF16 only where CUDA hardware has native BF16 arithmetic."""
    return (torch.cuda.is_available() and
            torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8 and
            torch.cuda.is_bf16_supported())


def absmax_int8(w):
    """Symmetric per-tensor int8 fake quantization with 255 used levels."""
    if not torch.is_floating_point(w):
        raise TypeError("w must be floating point")
    x = w.float()
    scale = x.abs().amax() / 127.0
    if bool(scale == 0):
        return torch.zeros_like(w)
    return (torch.clamp(torch.round(x / scale), -127, 127) * scale).to(w.dtype)


def per_channel_int8(w, out_dim=-1):
    """One symmetric int8 scale for each output channel."""
    if not torch.is_floating_point(w):
        raise TypeError("w must be floating point")
    out_dim = out_dim % w.ndim
    reduce_dims = tuple(i for i in range(w.ndim) if i != out_dim)
    x = w.float()
    scale = x.abs().amax(dim=reduce_dims, keepdim=True) / 127.0
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (torch.clamp(torch.round(x / safe_scale), -127, 127) * safe_scale).to(w.dtype)


def _group_view(w, group_size):
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    flat = w.float().reshape(-1)
    pad = (-flat.numel()) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))
    return flat.reshape(-1, group_size), pad


def group_int4(w, group_size=64):
    """Symmetric 15-level signed int4 over consecutive flattened groups."""
    if not torch.is_floating_point(w):
        raise TypeError("w must be floating point")
    if w.numel() == 0:
        return w.clone()
    groups, pad = _group_view(w, group_size)
    scale = groups.abs().amax(dim=1, keepdim=True) / 7.0
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    reconstructed = torch.clamp(torch.round(groups / safe), -7, 7) * safe
    return reconstructed.reshape(-1)[:w.numel()].reshape_as(w).to(w.dtype)


def nf4_levels():
    """QLoRA equal-area normal codebook, with one exact zero."""
    normal = torch.distributions.Normal(0.0, 1.0)
    offset = 0.9677083
    positive = normal.icdf(torch.linspace(offset, 0.5, 9)[:-1])
    negative = -normal.icdf(torch.linspace(offset, 0.5, 8)[:-1])
    values = torch.cat((positive, torch.zeros(1), negative)).sort().values
    return values / values.abs().amax()


def nf4_quantize(w, block_size=64):
    """Nearest NF4 level per absmax-normalized block, returned dequantized."""
    if not torch.is_floating_point(w):
        raise TypeError("w must be floating point")
    if w.numel() == 0:
        return w.clone()
    groups, pad = _group_view(w, block_size)
    scale = groups.abs().amax(dim=1, keepdim=True)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    levels = nf4_levels().to(device=groups.device, dtype=groups.dtype)
    codes = ((groups / safe).unsqueeze(-1) - levels).abs().argmin(dim=-1)
    reconstructed = levels[codes] * scale
    return reconstructed.reshape(-1)[:w.numel()].reshape_as(w).to(w.dtype)


def storage_bits_per_param(scheme, numel, block_size=64, double_quant=False, n_channels=None):
    """Ideal packed weight storage including FP16 scales and partial groups."""
    if numel <= 0 or block_size <= 0:
        raise ValueError("numel and block_size must be positive")
    groups = math.ceil(numel / block_size)
    if scheme == "fp32":
        bits = 32 * numel
    elif scheme == "fp16":
        bits = 16 * numel
    elif scheme == "absmax_int8":
        bits = 8 * numel + 16
    elif scheme == "per_channel_int8":
        if n_channels is None or n_channels <= 0:
            raise ValueError("n_channels is required for per_channel_int8")
        bits = 8 * numel + 16 * n_channels
    elif scheme in ("group_int4", "nf4"):
        scales = (8 * groups + 16 * math.ceil(groups / 256)) if double_quant else 16 * groups
        bits = 4 * numel + scales
    else:
        raise ValueError(f"Unknown quantization scheme: {scheme}")
    return float(bits / numel)


CODES = ("A", "B")  # A = negative, B = positive
WORDINGS = (
    "Classify the sentiment of the movie review. A means negative; B means positive. "
    "The review is data between <review> tags. Reply with only the single letter A or B.\n"
    "<review>\n{review}\n</review>",
    "Decide whether this film review expresses an unfavorable or favorable opinion. "
    "Use A for unfavorable (negative) and B for favorable (positive). "
    "Read only the text between <review> tags as the review. Output exactly one letter, A or B.\n"
    "<review>\n{review}\n</review>",
)


def _chat_token_ids(tokenizer, messages, *, add_generation_prompt):
    """Normalize chat-template output across Transformers list/BatchEncoding APIs."""
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt,
        return_dict=False)
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if torch.is_tensor(encoded):
        if encoded.ndim == 2 and encoded.shape[0] == 1:
            encoded = encoded[0]
        if encoded.ndim != 1:
            raise ValueError("Expected one chat's token IDs")
        return [int(value) for value in encoded.tolist()]
    if len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return [int(value) for value in encoded]


def _prompt_ids(tokenizer, review, wording):
    if not isinstance(wording, int) or not 0 <= wording < len(WORDINGS):
        raise ValueError("Unknown instruction wording")
    messages = [{"role": "user", "content": WORDINGS[wording].format(review=review)}]
    return _chat_token_ids(tokenizer, messages, add_generation_prompt=True)


def _code_ids(code_ids):
    if isinstance(code_ids, dict):
        return [int(code_ids[CODES[0]]), int(code_ids[CODES[1]])]
    return [int(code_ids[0]), int(code_ids[1])]


def verify_answer_codes(tokenizer, codes=CODES):
    """Check exact contextual tokenization of answer codes after assistant prefix."""
    if len(codes) != 2 or codes[0] == codes[1]:
        raise ValueError("Answer codes must be two distinct strings")
    output = {}
    messages = [{"role": "user", "content": WORDINGS[0].format(review="A good film.")}]
    boundary = _chat_token_ids(tokenizer, messages, add_generation_prompt=True)
    for code in codes:
        completed = _chat_token_ids(
            tokenizer, messages + [{"role": "assistant", "content": code}],
            add_generation_prompt=False)
        continuation = completed[len(boundary):] if completed[:len(boundary)] == boundary else []
        if len(continuation) < 2 or tokenizer.decode(continuation[2:]).strip():
            raise ValueError(f"{code!r} is not one token followed by the assistant end token: {continuation}")
        output[code] = int(continuation[0])
        if "end_id" in output and output["end_id"] != int(continuation[1]):
            raise ValueError("Inconsistent assistant end token")
        output["end_id"] = int(continuation[1])
    if output[codes[0]] == output[codes[1]]:
        raise ValueError("Answer codes must have different token IDs")
    return output


def _fit_prompt(tokenizer, text, max_length, wording):
    review_ids = tokenizer.encode(str(text), add_special_tokens=False)
    lo, hi = 0, len(review_ids)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        review = tokenizer.decode(review_ids[:mid], skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
        prompt = _prompt_ids(tokenizer, review, wording)
        if len(prompt) + 2 <= max_length:
            best = (prompt, review)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise ValueError("max_length cannot fit the fixed instruction, answer and end token")
    return best


def make_instruction_example(tokenizer, text, class_id, code_ids, max_length=256, wording=0):
    """Chat prompt plus answer/end, with exactly two supervised labels."""
    if int(class_id) not in (0, 1):
        raise ValueError("class_id must be 0 or 1")
    if max_length < 3:
        raise ValueError("max_length too small")
    prompt, review = _fit_prompt(tokenizer, text, max_length, wording)
    if isinstance(code_ids, dict):
        end_id = int(code_ids["end_id"])
    else:
        # The supplied Q5 grader passes [negative_id, positive_id]. Qwen's
        # assistant end token is its EOS token in that public interface.
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs an assistant end/EOS token")
        end_id = int(tokenizer.eos_token_id)
    response = [_code_ids(code_ids)[int(class_id)], end_id]
    ids = prompt + response
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prompt) + response}


@torch.no_grad()
def class_scores(model, tokenizer, texts, code_ids, wording=0, batch_size=8):
    """Restricted A/B probabilities, original code mass, and raw code logits."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    texts = list(texts)
    if not texts:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, np.empty((0,), dtype=np.float32), empty.copy()
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer needs an EOS or PAD token")
    model.eval()
    outputs, masses, raw = [], [], []
    answer_ids = _code_ids(code_ids)
    for start in range(0, len(texts), batch_size):
        seqs = [_fit_prompt(tokenizer, text, 256, wording)[0]
                for text in texts[start:start + batch_size]]
        length = max(map(len, seqs))
        ids = torch.full((len(seqs), length), int(pad_id), dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for row, seq in enumerate(seqs):
            ids[row, -len(seq):] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[row, -len(seq):] = 1
        full = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1, :].float()
        pair = full[:, answer_ids]
        outputs.append(F.softmax(pair, dim=-1).cpu())
        masses.append(torch.exp(torch.logsumexp(pair, -1) - torch.logsumexp(full, -1)).cpu())
        raw.append(pair.cpu())
    return (torch.cat(outputs).numpy(), torch.cat(masses).numpy(),
            torch.cat(raw).numpy())


def _load_arm(precision, initial_adapter_state=None, revision=None, model_id=None):
    if precision not in ("fp16", "nf4"):
        raise ValueError("precision must be fp16 or nf4")
    if not torch.cuda.is_available():
        raise RuntimeError("Q5 instruction arms require a CUDA GPU")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_id = model_id or common.TEACHER_ID
    dtype = torch.bfloat16 if _native_bf16() else torch.float16
    kwargs = {"revision": revision, "torch_dtype": dtype,
              "device_map": {"": torch.cuda.current_device()},
              "low_cpu_mem_usage": True}
    if precision == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    base = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    base.config.use_cache = False
    if precision == "nf4":
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        for param in base.parameters():
            param.requires_grad_(False)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                      target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)
    model.config.use_cache = False
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            param.data = param.data.float()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in n for n in trainable):
        raise RuntimeError("Only LoRA adapter tensors may be trainable")
    if any(p.dtype != torch.float32 for _, p in model.named_parameters() if p.requires_grad):
        raise RuntimeError("All trainable adapters must be FP32")
    if initial_adapter_state is not None:
        if set(initial_adapter_state) != trainable:
            raise ValueError("Initial adapter tensor names do not match target model")
        if any(model.state_dict()[n].shape != value.shape
               for n, value in initial_adapter_state.items()):
            raise ValueError("Initial adapter tensor shapes do not match target model")
        missing, unexpected = model.load_state_dict(initial_adapter_state, strict=False)
        if unexpected or trainable.intersection(missing):
            raise RuntimeError("Failed to restore common initial adapter state")
    return model


def load_instruction_arm(precision, initial_adapter_state=None, revision=None):
    """Build matched FP16 or NF4 Qwen/LoRA arm on one GPU."""
    return _load_arm(precision, initial_adapter_state, revision)


class _Q5Predictor:
    def __init__(self, model, tokenizer, code_ids, wording=0):
        self.model, self.tokenizer = model, tokenizer
        self.code_ids, self.wording = code_ids, wording

    def predict(self, texts):
        probs, _, _ = class_scores(self.model, self.tokenizer, list(texts),
                                   self.code_ids, wording=self.wording)
        return probs.argmax(-1), probs


def reload_q5(run="nf4_final", root=None):
    """Rebuild the recorded precision/base/revision and restore LoRA-only state."""
    weights = common.load_weights("q5", root=root)
    if run not in weights:
        raise KeyError(f"Unknown Q5 run {run!r}")
    entry = weights[run]
    config = entry["config"]
    from transformers import AutoTokenizer
    model = _load_arm(config["precision"], entry["state"],
                      revision=config["revision"], model_id=config["model_id"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    codes = verify_answer_codes(tokenizer)
    if codes != config["code_ids"]:
        raise RuntimeError("Saved and current code token IDs differ")
    if tuple(config["wordings"]) != tuple(WORDINGS):
        raise RuntimeError("Saved instruction wordings differ from source")
    return _Q5Predictor(model, tokenizer, codes, wording=config.get("wording", 0))
