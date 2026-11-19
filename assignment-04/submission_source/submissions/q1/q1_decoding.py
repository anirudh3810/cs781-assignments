"""Q1: decoding, exact KV caching, and exact speculative sampling.

The functions accept a decoder model returning [batch, time, vocabulary] logits.
Generated token lists exclude the prefix. No notebook globals are used here.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def _model_device(model):
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


def _ids(ids):
    if isinstance(ids, torch.Tensor):
        if ids.ndim != 1:
            raise ValueError("Token IDs must be one-dimensional.")
        ids = ids.tolist()
    result = [int(token) for token in ids]
    if not result:
        raise ValueError("A nonempty prefix is required.")
    return result


@torch.no_grad()
def next_token_logits(model, prefix_ids):
    """Return the unpadded prefix's final-position raw logits as [vocabulary]."""
    ids = _ids(prefix_ids)
    tensor = torch.tensor(ids, dtype=torch.long, device=_model_device(model))[None, :]
    logits = model(tensor)
    if logits.ndim != 3 or logits.shape[:2] != tensor.shape:
        raise ValueError("The model must return logits [batch, time, vocabulary].")
    return logits[0, -1, :]


def prepare_prefix(prefix_ids, max_new_tokens, context_limit, overflow_policy="reject"):
    """Reserve continuation capacity without wrapping absolute positions."""
    ids = _ids(prefix_ids)
    if overflow_policy not in ("reject", "left_truncate"):
        raise ValueError("overflow_policy must be reject or left_truncate.")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer.")
    if not isinstance(context_limit, int) or context_limit < 1:
        raise ValueError("context_limit must be a positive integer.")
    room = context_limit - max_new_tokens
    if room < 1:
        raise ValueError("The context limit leaves no room for a nonempty prefix.")
    if len(ids) > room:
        if overflow_policy == "reject":
            raise ValueError("Prefix plus requested output exceeds the context limit.")
        ids = ids[-room:]
    return ids


def filtered_probs(logits, temperature=1.0, top_k=None, top_p=1.0):
    """Apply temperature, top-k and top-p in that order, then renormalize."""
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be positive and finite.")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k must be a positive integer or None.")
    if not math.isfinite(float(top_p)) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1].")
    if logits.ndim != 1 or not logits.numel():
        raise ValueError("logits must be a nonempty one-dimensional vector.")
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
        raise ValueError("logits do not define a finite probability distribution.")
    # Stable sorting gives the lower token ID priority when scores tie.
    order = torch.argsort(probabilities, descending=True, stable=True)
    keep = torch.ones_like(probabilities, dtype=torch.bool)
    if top_k is not None:
        keep[:] = False
        keep[order[:min(top_k, probabilities.numel())]] = True
    if top_p < 1:
        ordered = probabilities[order] * keep[order]
        ordered = ordered / ordered.sum()
        crossing = int(torch.searchsorted(ordered.cumsum(0),
                                          torch.tensor(top_p, device=ordered.device)).item())
        nucleus = torch.zeros_like(keep)
        nucleus[order[:min(crossing + 1, probabilities.numel())]] = True
        keep &= nucleus
    probabilities = probabilities * keep
    return probabilities / probabilities.sum()


@torch.no_grad()
def generate(model, prefix_ids, max_new_tokens, *, strategy="greedy", eos_id=None,
             temperature=1.0, top_k=None, top_p=1.0, generator=None):
    """Autoregressive generation with sampled-token probabilities and EOS trace."""
    if strategy not in ("greedy", "sample"):
        raise ValueError("strategy must be greedy or sample.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative.")
    context = _ids(prefix_ids)
    output, chosen_probs = [], []
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        logits = next_token_logits(model, context)
        if strategy == "greedy":
            # torch.argmax resolves ties to the first (lowest-ID) maximum.
            token = int(torch.argmax(logits).item())
            probability = float(torch.softmax(logits.float(), -1)[token].item())
        else:
            probs = filtered_probs(logits, temperature, top_k, top_p)
            token = int(torch.multinomial(probs, 1, generator=generator).item())
            probability = float(probs[token].item())
        output.append(token)
        chosen_probs.append(probability)
        context.append(token)
        if eos_id is not None and token == eos_id:
            stop_reason = "eos"
            break
    return output, {"chosen_ids": output.copy(), "token_probs": chosen_probs,
                    "stop_reason": stop_reason}


@torch.no_grad()
def beam_search(model, prefix_ids, max_new_tokens, beam_width=4,
                length_penalty=0.0, eos_id=None):
    """Beam search with optional GNMT final length normalisation.

    Expansion and admission retain the assignment's raw-logprob ranking. The
    optional penalty affects final candidate scores, including early EOS.
    """
    if beam_width < 1 or max_new_tokens < 0:
        raise ValueError("beam_width must be >= 1 and max_new_tokens >= 0.")
    if not math.isfinite(float(length_penalty)) or length_penalty < 0:
        raise ValueError("length_penalty must be finite and nonnegative.")
    prefix = _ids(prefix_ids)
    live = [{"ids": [], "raw_logprob": 0.0, "score": 0.0, "finished": False}]
    finished = []
    for _ in range(max_new_tokens):
        if not live:
            break
        children = []
        for parent in live:
            log_probs = torch.log_softmax(next_token_logits(model, prefix + parent["ids"]).float(), -1)
            order = torch.argsort(log_probs, descending=True, stable=True)
            for token in order[:min(beam_width, log_probs.numel())].tolist():
                ids = parent["ids"] + [int(token)]
                raw = parent["raw_logprob"] + float(log_probs[token].item())
                children.append({"ids": ids, "raw_logprob": raw, "score": raw,
                                 "finished": eos_id is not None and token == eos_id})
        children.sort(key=lambda record: (-record["raw_logprob"], record["ids"]))
        admitted = children[:beam_width]
        finished.extend(record for record in admitted if record["finished"])
        live = [record for record in admitted if not record["finished"]]
    candidates = finished + live
    for record in candidates:
        length = max(1, len(record["ids"]))
        gnmt_factor = ((5.0 + length) / 6.0) ** length_penalty
        record["score"] = record["raw_logprob"] / gnmt_factor
    candidates.sort(key=lambda record: (-record["score"], record["ids"]))
    return candidates[0]["ids"], candidates


class CachedA3LM(nn.Module):
    """Incremental inference using references to the trained A3 decoder weights."""

    def __init__(self, trained_builtin_model):
        super().__init__()
        self.trained = trained_builtin_model.eval()
        self.layers = self.trained.encoder.layers
        self.num_heads = self.layers[0].self_attn.num_heads
        self.head_dim = self.layers[0].self_attn.head_dim
        self.context_limit = self.trained.cfg.max_seq_len

    def _attention(self, layer, x, old_kv, past_len):
        attn = layer.self_attn
        q_weight, k_weight, v_weight = attn.in_proj_weight.chunk(3, dim=0)
        if attn.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = attn.in_proj_bias.chunk(3, dim=0)

        def project(weight, bias):
            value = F.linear(x, weight, bias)
            batch, length, _ = value.shape
            return value.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        q = project(q_weight, q_bias)
        k_new = project(k_weight, k_bias)
        v_new = project(v_weight, v_bias)
        if old_kv is None:
            k, v = k_new, v_new
        else:
            old_k, old_v = old_kv
            if old_k.shape != old_v.shape or old_k.shape[:2] != k_new.shape[:2] or old_k.shape[2] != past_len:
                raise ValueError("Malformed past_key_values.")
            k = torch.cat((old_k, k_new), dim=2)
            v = torch.cat((old_v, v_new), dim=2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        absolute_q = past_len + torch.arange(q.shape[2], device=x.device)
        absolute_k = torch.arange(k.shape[2], device=x.device)
        causal = absolute_k[None, :] <= absolute_q[:, None]
        scores = scores.masked_fill(~causal[None, None], torch.finfo(scores.dtype).min)
        values = torch.softmax(scores, dim=-1) @ v
        joined = values.transpose(1, 2).contiguous().reshape(x.shape)
        return attn.out_proj(joined), (k, v)

    @torch.no_grad()
    def forward(self, input_ids, past_key_values=None):
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have shape [batch, positive time].")
        if past_key_values is None:
            previous = [None] * len(self.layers)
            past_len = 0
        else:
            previous = list(past_key_values)
            if len(previous) != len(self.layers):
                raise ValueError("One K/V pair per layer is required.")
            past_len = previous[0][0].shape[2]
        length = input_ids.shape[1]
        if past_len + length > self.context_limit:
            raise ValueError("Cached request exceeds max_seq_len.")
        position = self.trained.pos_emb[past_len:past_len + length]
        x = self.trained.tok_emb(input_ids) + position[None]
        new_cache = []
        for layer, old_kv in zip(self.layers, previous):
            if layer.norm_first:
                attended, kv = self._attention(layer, layer.norm1(x), old_kv, past_len)
                x = x + layer.dropout1(attended)
                x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x))))))
            else:
                attended, kv = self._attention(layer, x, old_kv, past_len)
                x = layer.norm1(x + layer.dropout1(attended))
                x = layer.norm2(x + layer.dropout2(layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(x))))))
            new_cache.append(kv)
        return self.trained.head(self.trained.ln_f(x)), new_cache

    @staticmethod
    def cache_nbytes(past_key_values):
        if past_key_values is None:
            return 0
        return sum(t.numel() * t.element_size() for pair in past_key_values for t in pair)


class BigramDraft:
    """Sparse add-one-smoothed bigram draft over target token IDs."""

    def __init__(self, vocab_size):
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive.")
        self.vocab_size = vocab_size
        self.counts = {}
        self.row_totals = {}

    def fit(self, training_token_ids):
        self.counts.clear()
        self.row_totals.clear()
        iterator = iter(training_token_ids)
        previous = next(iterator, None)
        for token in iterator:
            previous, token = int(previous), int(token)
            if not (0 <= previous < self.vocab_size and 0 <= token < self.vocab_size):
                raise ValueError("Training token outside the target vocabulary.")
            row = self.counts.setdefault(previous, {})
            row[token] = row.get(token, 0) + 1
            self.row_totals[previous] = self.row_totals.get(previous, 0) + 1
            previous = token
        return self

    def next_probs(self, prefix_ids):
        previous = _ids(prefix_ids)[-1]
        if not 0 <= previous < self.vocab_size:
            raise ValueError("Prefix token outside the target vocabulary.")
        counts = self.counts.get(previous, {})
        denominator = self.row_totals.get(previous, 0) + self.vocab_size
        result = torch.full((self.vocab_size,), 1.0 / denominator, dtype=torch.float32)
        for token, count in counts.items():
            result[token] = (count + 1) / denominator
        return result


@torch.no_grad()
def speculative_sample(target, draft, prefix_ids, max_new_tokens, gamma=4, eos_id=None, generator=None):
    """Exact draft acceptance/correction with one target call per proposed block."""
    if gamma < 1 or max_new_tokens < 0:
        raise ValueError("gamma must be >= 1 and max_new_tokens >= 0.")
    prefix = _ids(prefix_ids)
    output = []
    stats = {"proposed": 0, "accepted": 0, "target_calls": 0, "blocks": 0}
    device = _model_device(target)
    while len(output) < max_new_tokens:
        prefix_now = prefix + output
        block_len = min(gamma, max_new_tokens - len(output))
        proposed, draft_probs = [], []
        for _ in range(block_len):
            q = torch.as_tensor(draft.next_probs(prefix_now + proposed), dtype=torch.float32, device=device)
            if q.ndim != 1 or not torch.isfinite(q).all() or q.min() < 0 or q.sum() <= 0:
                raise ValueError("The draft must return a valid probability vector.")
            q = q / q.sum()
            token = int(torch.multinomial(q, 1, generator=generator).item())
            proposed.append(token)
            draft_probs.append(q)
            if eos_id is not None and token == eos_id:
                break
        stats["proposed"] += len(proposed)
        stats["blocks"] += 1
        ids = torch.tensor(prefix_now + proposed, dtype=torch.long, device=device)[None]
        target_logits = target(ids)[0]
        stats["target_calls"] += 1
        offset = len(prefix_now) - 1
        rejected = False
        for j, token in enumerate(proposed):
            p = torch.softmax(target_logits[offset + j].float(), -1)
            q = draft_probs[j]
            if p.shape != q.shape:
                raise ValueError("Draft and target vocabulary sizes differ.")
            acceptance = min(1.0, float((p[token] / q[token]).item()))
            if torch.rand((), device=device, generator=generator).item() < acceptance:
                output.append(token)
                stats["accepted"] += 1
                if eos_id is not None and token == eos_id:
                    return output, stats
                if len(output) >= max_new_tokens:
                    return output, stats
                continue
            correction = torch.clamp(p - q, min=0)
            # If numerical cancellation leaves no correction mass, p is a safe fallback.
            correction = correction / correction.sum() if correction.sum() > 1e-12 else p
            replacement = int(torch.multinomial(correction, 1, generator=generator).item())
            output.append(replacement)
            rejected = True
            if eos_id is not None and replacement == eos_id:
                return output, stats
            break
        if not rejected and len(output) < max_new_tokens:
            bonus_p = torch.softmax(target_logits[offset + len(proposed)].float(), -1)
            bonus = int(torch.multinomial(bonus_p, 1, generator=generator).item())
            output.append(bonus)
            if eos_id is not None and bonus == eos_id:
                return output, stats
    return output, stats
