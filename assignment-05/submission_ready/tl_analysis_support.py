"""Analysis helpers shared by student-written TransformerLens ports.

Students implement the weight/configuration mapping in ``to_hooked``. This module
adds only the common readout/DLA methods and the A3 post-norm block semantics;
it does not copy any weights or construct the model for them.
"""
from types import MethodType

import torch


def _post_norm_forward(block, resid_pre, shortformer_pos_embed=None,
                       past_kv_cache_entry=None, attention_mask=None):
    """A3 post-LN order: LN(resid + attention), then LN(resid + MLP)."""
    resid_pre = block.hook_resid_pre(resid_pre)
    attn_out = block.attn(
        query_input=resid_pre, key_input=resid_pre, value_input=resid_pre,
        past_kv_cache_entry=past_kv_cache_entry, attention_mask=attention_mask,
    )
    attn_out = block.hook_attn_out(attn_out)
    resid_mid = block.hook_resid_mid(block.ln1(resid_pre + attn_out))
    mlp_out = block.hook_mlp_out(block.mlp(resid_mid))
    return block.hook_resid_post(block.ln2(resid_mid + mlp_out))


def install_analysis_support(model, norm_first=True):
    """Attach the common Q5/Q6 methods to an actual HookedTransformer instance.

    ``model`` must already be a TransformerLens model with the student's mapped
    parameters. The helpers intentionally operate on TransformerLens cache names.
    """
    model._a3_norm_first = bool(norm_first)
    if not norm_first:
        for block in model.blocks:
            block.forward = MethodType(_post_norm_forward, block)

    def head_writes(self, cache, layer):
        return cache[f"blocks.{layer}.attn.hook_result"]

    def component_writes(self, cache):
        writes = {"embed": cache["hook_embed"]}
        if "hook_pos_embed" in cache.cache_dict:
            writes["pos_embed"] = cache["hook_pos_embed"]
        for layer, block in enumerate(self.blocks):
            resid_pre = cache[f"blocks.{layer}.hook_resid_pre"]
            resid_mid = cache[f"blocks.{layer}.hook_resid_mid"]
            heads = self.head_writes(cache, layer)
            out_bias = block.attn.b_O
            attn_bias = out_bias.view(1, 1, -1).expand_as(resid_pre)
            if self._a3_norm_first:
                for head in range(self.cfg.n_heads):
                    writes[f"L{layer}H{head}"] = heads[:, :, head]
                writes[f"attn_bias_{layer}"] = attn_bias
                writes[f"mlp_{layer}"] = cache[f"blocks.{layer}.hook_mlp_out"]
            else:
                previous = block.ln1(resid_pre + attn_bias)
                writes[f"attn_bias_{layer}"] = previous - resid_pre
                cumulative = torch.zeros_like(resid_pre)
                for head in range(self.cfg.n_heads):
                    cumulative = cumulative + heads[:, :, head]
                    current = block.ln1(resid_pre + attn_bias + cumulative)
                    writes[f"L{layer}H{head}"] = current - previous
                    previous = current
                writes[f"mlp_{layer}"] = cache[f"blocks.{layer}.hook_resid_post"] - resid_mid
        return writes

    def residual_to_logits(self, resid, cache=None):
        return self.unembed(self.ln_final(resid))

    def project_component_to_true_logit(self, component, cache, target_ids):
        scale = cache["ln_final.hook_scale"]
        centered = component - component.mean(dim=-1, keepdim=True)
        direction = self.ln_final.w[:, None] * self.unembed.W_U
        selected = direction[:, target_ids].permute(1, 2, 0)
        return ((centered / scale) * selected).sum(-1)

    def readout_bias_to_true_logit(self, cache, target_ids):
        bias = self.ln_final.b @ self.unembed.W_U
        if self.unembed.b_U is not None:
            bias = bias + self.unembed.b_U
        return bias[target_ids]

    model.head_writes = MethodType(head_writes, model)
    model.component_writes = MethodType(component_writes, model)
    model.residual_to_logits = MethodType(residual_to_logits, model)
    model.project_component_to_true_logit = MethodType(project_component_to_true_logit, model)
    model.readout_bias_to_true_logit = MethodType(readout_bias_to_true_logit, model)
    return model


@torch.no_grad()
def logit_lens(model, cache, tokens, passage_length):
    repeat = slice(passage_length + 1, 2 * passage_length)
    def accuracy(logits):
        return (logits[:, repeat].argmax(-1) == tokens[:, 1:][:, repeat]).float().mean().item()
    residuals = [cache["blocks.0.hook_resid_pre"]]
    residuals.extend(cache[f"blocks.{layer}.hook_resid_post"] for layer in range(model.cfg.n_layers))
    return [{"depth": depth, "acc_repeat": accuracy(model.residual_to_logits(resid, cache))}
            for depth, resid in enumerate(residuals)]


@torch.no_grad()
def decompose_correct_logit(model, cache, tokens):
    targets = tokens[:, 1:]
    components = model.component_writes(cache)
    names = list(components)
    contributions = [model.project_component_to_true_logit(components[n], cache, targets)
                     for n in names]
    bias = model.readout_bias_to_true_logit(cache, targets)
    residual = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"]
    logits = model.residual_to_logits(residual, cache)
    true_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return {"names": names, "contrib": torch.stack(contributions), "bias": bias, "logit": true_logit}


def summarize_dla(decomp, passage_length):
    repeat = slice(passage_length + 1, 2 * passage_length)
    return {name: decomp["contrib"][i][:, repeat].mean().item()
            for i, name in enumerate(decomp["names"])}
