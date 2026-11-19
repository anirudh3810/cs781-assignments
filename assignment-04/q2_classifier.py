# [SUBMITTED IMPLEMENTATION] Independent Q2 module; the grader imports this file.
"""GPT-2 sentiment classifier with attention-mask-based final-token pooling."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import common


def last_nonpad(hidden_states, attention_mask):
    """Select the final unmasked state in each row, regardless of padding side."""
    if hidden_states.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("expected hidden states [B,T,D] and attention mask [B,T]")
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("hidden-state and mask batch/sequence dimensions differ")
    if hidden_states.shape[1] == 0:
        raise ValueError("cannot pool an empty sequence")
    valid = attention_mask.to(device=hidden_states.device).bool()
    if not bool(valid.any(dim=1).all()):
        raise ValueError("cannot pool an all-padding row")
    positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    final_positions = torch.where(valid, positions.unsqueeze(0), -1).amax(dim=1)
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, final_positions]


class SentimentClassifier(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.backbone_eval = False

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_eval:
            self.backbone.eval()
        return self

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        """Return [B,2] class logits; accept token IDs or Q4 soft-prompt embeds."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids or inputs_embeds")
        if attention_mask is None:
            source = input_ids if input_ids is not None else inputs_embeds
            attention_mask = torch.ones(source.shape[:2], device=source.device, dtype=torch.long)
        outputs = self.backbone(input_ids=input_ids, inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        return self.classifier(last_nonpad(hidden, attention_mask))


def classification_loss(model, batch):
    """Cross-entropy loss expected by common.train_updates."""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return F.cross_entropy(logits.float(), batch["labels"])


def make_classifier(backbone_id=common.BACKBONE_ID, initial_head_state=None,
                    mode="full", revision=None):
    """Create a fresh pretrained backbone with a matched optional head state."""
    if mode not in ("full", "linear_probe", "frozen"):
        raise ValueError("mode must be full, linear_probe, or frozen")
    kwargs = {"revision": revision} if revision is not None else {}
    backbone = AutoModel.from_pretrained(backbone_id, **kwargs)
    hidden_size = backbone.config.hidden_size
    model = SentimentClassifier(backbone, hidden_size, 2)
    if initial_head_state is not None:
        model.classifier.load_state_dict(initial_head_state, strict=True)
    if mode in ("linear_probe", "frozen"):
        for param in model.backbone.parameters():
            param.requires_grad_(False)
        model.backbone_eval = True
        model.backbone.eval()
    else:
        for param in model.backbone.parameters():
            param.requires_grad_(True)
    return model.to(common.DEVICE)


def reload_q2(method, root=None):
    """Reconstruct the saved text-to-probabilities predictor without notebook state."""
    if method not in ("linear_probe", "full"):
        raise ValueError("unknown Q2 method")
    saved = common.load_weights("q2", root)
    if "states" in saved and method not in saved:
        saved = saved["states"]
    entry = saved[method]
    config = entry["config"]
    model = make_classifier(backbone_id=config["backbone"], mode=method,
                            revision=config.get("revision"))
    state = entry["state"]
    incompatible = model.load_state_dict(state, strict=(method == "full"))
    if method == "linear_probe":
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if unexpected or not all(name.startswith("backbone.") for name in missing):
            raise RuntimeError(f"invalid linear-probe state: {incompatible}")
        if not {"classifier.weight", "classifier.bias"}.issubset(state):
            raise RuntimeError("saved linear probe has no complete classification head")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config["backbone"],
                                               **({"revision": config["revision"]}
                                                  if config.get("revision") else {}))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = config.get("padding_side", "right")
    collate = common.make_classifier_collate(tokenizer,
                                             max_length=int(config["max_length"]))
    return common.ClassifierPredictor(model, collate)
