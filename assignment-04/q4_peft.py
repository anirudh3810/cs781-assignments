"""Question 4: prompt tuning, adapters, LoRA, and LoRMA+ for GPT-2."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

import common
from q2_classifier import SentimentClassifier, make_classifier, last_nonpad


class SoftPrompt(nn.Module):
    def __init__(self, init_embeddings):
        super().__init__()
        if init_embeddings.ndim != 2:
            raise ValueError("prompt initialization must have shape [m, hidden]")
        self.prompt = nn.Parameter(init_embeddings.detach().clone())

    def forward(self, token_embeddings, attention_mask):
        if token_embeddings.ndim != 3 or attention_mask.shape != token_embeddings.shape[:2]:
            raise ValueError("embedding and mask shapes disagree")
        batch = token_embeddings.shape[0]
        prompt = self.prompt.to(dtype=token_embeddings.dtype).unsqueeze(0).expand(batch, -1, -1)
        prompt_mask = attention_mask.new_ones((batch, self.prompt.shape[0]))
        return (torch.cat((prompt, token_embeddings), dim=1),
                torch.cat((prompt_mask, attention_mask), dim=1))


class SoftPromptClassifier(nn.Module):
    """Pool the last real review token after prepending trainable embeddings."""
    def __init__(self, clf, soft_prompt):
        super().__init__()
        self.clf, self.soft = clf, soft_prompt

    def forward(self, input_ids=None, attention_mask=None):
        if input_ids is None:
            raise ValueError("input_ids required")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        embeddings = self.clf.backbone.get_input_embeddings()(input_ids)
        full_embeddings, full_mask = self.soft(embeddings, attention_mask)
        max_positions = getattr(self.clf.backbone.config, "n_positions", None)
        if max_positions is not None and full_embeddings.shape[1] > max_positions:
            raise ValueError(f"prompt + text exceeds {max_positions} positions")
        output = self.clf.backbone(inputs_embeds=full_embeddings,
                                   attention_mask=full_mask, return_dict=True)
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        pooled = last_nonpad(hidden[:, self.soft.prompt.shape[0]:], attention_mask)
        return self.clf.classifier(pooled)


class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_size, bottleneck=16, init_std=None):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.enabled = True
        if init_std is None:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        else:
            if init_std < 0:
                raise ValueError("init_std must be nonnegative")
            nn.init.normal_(self.down.weight, std=float(init_std))
            nn.init.normal_(self.up.weight, std=float(init_std))
            nn.init.zeros_(self.down.bias)
            nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states):
        if not self.enabled:
            return hidden_states
        return hidden_states + self.up(F.gelu(self.down(hidden_states)))


def _check_targets(targets, allowed):
    if targets is None:
        return None
    result = tuple(targets)
    if not result or len(set(result)) != len(result) or not set(result) <= set(allowed):
        raise ValueError(f"targets must be a nonempty subset of {allowed}")
    return tuple(t for t in allowed if t in result)


def _random_factor(shape, fan_in, *, device, dtype, generator=None):
    bound = 1.0 / math.sqrt(fan_in)
    return (torch.rand(shape, generator=generator, device=device, dtype=dtype) * 2 - 1) * bound


class LoRAConv1D(nn.Module):
    """Frozen GPT-2 Conv1D plus A/B in mathematical [out,in] orientation."""
    def __init__(self, base_conv1d, rank=8, alpha=16, targets=None, swap_ab=False):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base_conv1d
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank, self.alpha = int(rank), float(alpha)
        self.scaling = self.alpha / self.rank
        self.targets = _check_targets(targets, ("q", "k", "v"))
        self.swap_ab = bool(swap_ab)
        d_in, d_out = self.base.weight.shape
        if self.targets is None:
            a = _random_factor((rank, d_in), d_in, device=self.base.weight.device,
                               dtype=self.base.weight.dtype)
            b = torch.zeros((d_out, rank), device=a.device, dtype=a.dtype)
            if self.swap_ab:
                a.zero_()
                b = _random_factor((d_out, rank), rank, device=a.device, dtype=a.dtype)
            self.lora_A, self.lora_B = nn.Parameter(a), nn.Parameter(b)
        else:
            if d_out % 3:
                raise ValueError("Q/K/V target requires fused c_attn output")
            block = d_out // 3
            self.lora_As, self.lora_Bs = nn.ParameterDict(), nn.ParameterDict()
            for name in self.targets:
                a = _random_factor((rank, d_in), d_in, device=self.base.weight.device,
                                   dtype=self.base.weight.dtype)
                b = torch.zeros((block, rank), device=a.device, dtype=a.dtype)
                if self.swap_ab:
                    a.zero_()
                    b = _random_factor((block, rank), rank, device=a.device, dtype=a.dtype)
                self.lora_As[name], self.lora_Bs[name] = nn.Parameter(a), nn.Parameter(b)

    def forward(self, x):
        base = self.base(x)
        if self.targets is None:
            return base + self.scaling * F.linear(F.linear(x, self.lora_A), self.lora_B)
        block = base.shape[-1] // 3
        pieces = []
        for i, name in enumerate(("q", "k", "v")):
            piece = base[..., i * block:(i + 1) * block]
            if name in self.targets:
                piece = piece + self.scaling * F.linear(F.linear(x, self.lora_As[name]),
                                                         self.lora_Bs[name])
            pieces.append(piece)
        return torch.cat(pieces, dim=-1)

    def merged_copy(self):
        d_in, d_out = self.base.weight.shape
        merged = Conv1D(d_out, d_in).to(device=self.base.weight.device,
                                        dtype=self.base.weight.dtype)
        with torch.no_grad():
            merged.weight.copy_(self.base.weight)
            merged.bias.copy_(self.base.bias)
            if self.targets is None:
                merged.weight.add_(self.scaling * (self.lora_B @ self.lora_A).T)
            else:
                block = d_out // 3
                for name in self.targets:
                    i = ("q", "k", "v").index(name)
                    merged.weight[:, i * block:(i + 1) * block].add_(
                        self.scaling * (self.lora_Bs[name] @ self.lora_As[name]).T)
        return merged


LORMA_VARIANT = {
    "name": "LoRMA+",
    "paper_version": "Bihany et al., arXiv:2506.07621v1 (2025), Section 3.2.2, Equation (8)",
    "equation": "W'=(I_d+(alpha/r)BA)W_0; A:[r,d], B:[d,r]",
    "alpha_over_rank": 1.0,
}


def make_lorma_factors(d_in, d_out, rank, init=False, generator=None,
                       dtype=torch.float64, swap_ab=False):
    """Return (A,B) for LoRMA+; d_in is W's input width, d_out its output width."""
    if min(d_in, d_out, rank) <= 0:
        raise ValueError("dimensions and rank must be positive")
    a = _random_factor((rank, d_out), d_out, device="cpu", dtype=dtype,
                       generator=generator)
    b = _random_factor((d_out, rank), rank, device="cpu", dtype=dtype,
                       generator=generator)
    if init:
        if swap_ab:
            a.zero_()
        else:
            b.zero_()
    return a, b


def _lorma_scale(factors, variant_config):
    if variant_config is None:
        return 1.0
    if "alpha_over_rank" in variant_config:
        return float(variant_config["alpha_over_rank"])
    return float(variant_config.get("alpha", factors[0].shape[0])) / factors[0].shape[0]


def lorma_dense(x, frozen_weight, factors, variant_config=None):
    """Materialize the identity-inflated output-side transform and adapted weight."""
    a, b = factors
    d_out = frozen_weight.shape[0]
    if a.shape[1] != d_out or b.shape != (d_out, a.shape[0]):
        raise ValueError("LoRMA factor shapes disagree with weight")
    inflated = torch.eye(d_out, device=frozen_weight.device, dtype=frozen_weight.dtype)
    inflated = inflated + _lorma_scale(factors, variant_config) * (b @ a)
    return F.linear(x, inflated @ frozen_weight)


def lorma_reordered(x, frozen_weight, factors, variant_config=None):
    """Apply W0, then A, then B; no square product is materialized."""
    a, b = factors
    h = F.linear(x, frozen_weight)
    return h + _lorma_scale(factors, variant_config) * F.linear(F.linear(h, a), b)


class LoRMAConv1D(nn.Module):
    def __init__(self, base_conv1d, rank=8, variant_config=None,
                 targets=None, swap_ab=False):
        super().__init__()
        self.base = base_conv1d
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.variant_config = dict(variant_config or {})
        self.targets = _check_targets(targets, ("q", "k", "v"))
        self.swap_ab = bool(swap_ab)
        d_in, d_out = self.base.weight.shape
        if self.targets is None:
            a, b = make_lorma_factors(d_in, d_out, rank, init=True,
                                      dtype=self.base.weight.dtype, swap_ab=swap_ab)
            self.lorma_A = nn.Parameter(a.to(self.base.weight.device))
            self.lorma_B = nn.Parameter(b.to(self.base.weight.device))
        else:
            if d_out % 3:
                raise ValueError("Q/K/V target requires fused c_attn output")
            self.lorma_As, self.lorma_Bs = nn.ParameterDict(), nn.ParameterDict()
            for name in self.targets:
                a, b = make_lorma_factors(d_in, d_out // 3, rank, init=True,
                                          dtype=self.base.weight.dtype, swap_ab=swap_ab)
                self.lorma_As[name] = nn.Parameter(a.to(self.base.weight.device))
                self.lorma_Bs[name] = nn.Parameter(b.to(self.base.weight.device))

    def forward(self, x):
        base = self.base(x)
        scale = _lorma_scale((self.lorma_A, self.lorma_B), self.variant_config) if self.targets is None \
            else float(self.variant_config.get("alpha", self.rank)) / self.rank
        if self.targets is None:
            weight_output = base - self.base.bias
            return base + scale * F.linear(F.linear(weight_output, self.lorma_A), self.lorma_B)
        block = base.shape[-1] // 3
        pieces = []
        for i, name in enumerate(("q", "k", "v")):
            piece = base[..., i * block:(i + 1) * block]
            if name in self.targets:
                unbiased = piece - self.base.bias[i * block:(i + 1) * block]
                piece = piece + scale * F.linear(F.linear(unbiased, self.lorma_As[name]),
                                                 self.lorma_Bs[name])
            pieces.append(piece)
        return torch.cat(pieces, dim=-1)


def _adapter_output_hook(block, inputs, output):
    if isinstance(output, tuple):
        return (block.adapter(output[0]),) + output[1:]
    return block.adapter(output)


def attach_peft(classifier, method, config=None):
    """Attach the requested PEFT method and train only PEFT tensors plus the head."""
    config = dict(config or {})
    if method not in ("soft_prompt", "adapter", "lora", "lorma"):
        raise ValueError(f"unknown PEFT method {method!r}")
    for p in classifier.parameters():
        p.requires_grad_(False)
    for p in classifier.classifier.parameters():
        p.requires_grad_(True)
    classifier.backbone_eval = True
    backbone = classifier.backbone
    if method == "soft_prompt":
        m = int(config.get("num_virtual_tokens", 16))
        if m <= 0:
            raise ValueError("num_virtual_tokens must be positive")
        embedding = backbone.get_input_embeddings()
        init_kind = config.get("init", "vocab")
        if init_kind == "random":
            initial = torch.empty((m, embedding.embedding_dim), device=embedding.weight.device,
                                  dtype=embedding.weight.dtype).uniform_(-0.5, 0.5)
        else:
            ids = config.get("init_token_ids")
            if init_kind == "class_label":
                from transformers import AutoTokenizer
                kwargs = {"revision": config["revision"]} if config.get("revision") else {}
                tok = AutoTokenizer.from_pretrained(config.get("backbone", common.BACKBONE_ID), **kwargs)
                ids = [tok.encode(" negative", add_special_tokens=False)[0],
                       tok.encode(" positive", add_special_tokens=False)[0]]
            elif init_kind != "vocab":
                raise ValueError(f"unknown soft-prompt init {init_kind}")
            ids = list(range(m)) if ids is None else list(ids)
            if not ids:
                raise ValueError("init_token_ids cannot be empty")
            ids = (ids * ((m + len(ids) - 1) // len(ids)))[:m]
            with torch.no_grad():
                initial = embedding(torch.tensor(ids, device=embedding.weight.device)).detach().clone()
        model = SoftPromptClassifier(classifier, SoftPrompt(initial))
    elif method == "adapter":
        model = classifier
        width = int(config.get("bottleneck", 16))
        for block in backbone.h:
            block.adapter = BottleneckAdapter(backbone.config.hidden_size, width,
                                              init_std=config.get("adapter_init_std"))
            block.register_forward_hook(_adapter_output_hook)
    else:
        model = classifier
        targets = config.get("targets")
        rank = int(config.get("rank", 8))
        alpha = float(config.get("alpha", 16 if method == "lora" else rank))
        swap_ab = bool(config.get("swap_ab", False))
        if method == "lora":
            if targets is not None:
                targets = _check_targets(targets, ("q", "k", "v", "o"))
            for block in backbone.h:
                if targets is None:
                    block.attn.c_attn = LoRAConv1D(block.attn.c_attn, rank, alpha,
                                                  swap_ab=swap_ab)
                else:
                    qkv = tuple(t for t in targets if t in ("q", "k", "v"))
                    if qkv:
                        block.attn.c_attn = LoRAConv1D(block.attn.c_attn, rank, alpha,
                                                      targets=qkv, swap_ab=swap_ab)
                    if "o" in targets:
                        block.attn.c_proj = LoRAConv1D(block.attn.c_proj, rank, alpha,
                                                      swap_ab=swap_ab)
        else:
            targets = _check_targets(targets, ("q", "k", "v")) if targets is not None else None
            for block in backbone.h:
                block.attn.c_attn = LoRMAConv1D(block.attn.c_attn, rank,
                                               {"alpha": alpha}, targets=targets,
                                               swap_ab=swap_ab)
    for name, p in model.named_parameters():
        if "classifier." in name or ".adapter." in name or ".lora_A" in name or ".lora_B" in name \
                or ".lorma_A" in name or ".lorma_B" in name or name == "soft.prompt":
            p.requires_grad_(True)
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    return model.to(common.DEVICE), names


def reload_q4(method, root=None):
    """Recreate frozen GPT-2, PEFT tensors, tokenizer, and classifier predictor."""
    if method not in ("soft_prompt", "adapter", "lora", "lorma"):
        raise ValueError(method)
    saved = common.load_weights("q4", root)
    entry = saved[method]
    config = entry["config"]
    model = make_classifier(config["backbone"], mode="frozen",
                            revision=config.get("revision"))
    model, names = attach_peft(model, method, config)
    missing, unexpected = model.load_state_dict(entry["state"], strict=False)
    if unexpected or (set(names) & set(missing)):
        raise RuntimeError(f"invalid Q4 saved state: missing={missing}, unexpected={unexpected}")
    model.eval()
    from transformers import AutoTokenizer
    kwargs = {"revision": config["revision"]} if config.get("revision") else {}
    tokenizer = AutoTokenizer.from_pretrained(config["backbone"], **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = config.get("padding_side", "right")
    collate = common.make_classifier_collate(tokenizer,
                                             max_length=int(config.get("max_length", 128)))
    return common.ClassifierPredictor(model, collate)
