"""Q6 — class-score distillation into a DistilGPT-2 classifier.

This file is emitted by the notebook's ``%%writefile q6_distill.py`` cell.
It deliberately has no dependency on notebook globals.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import common
from q2_classifier import SentimentClassifier


def distillation_loss(student_logits, teacher_logits, labels, tau=2.0, kd_weight=0.5):
    """Return (1-lambda)*CE + lambda*tau**2*KL(p_teacher || p_student).

    CE uses the unscaled student logits; KL is summed across classes and
    averaged across the batch.  Teacher class scores are fixed targets.
    """
    if not 0.0 <= float(kd_weight) <= 1.0 or float(tau) <= 0:
        raise ValueError("kd_weight must be in [0,1] and tau must be positive")
    if student_logits.ndim != 2 or student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same [batch,classes] shape")
    student = student_logits.float()
    teacher = teacher_logits.detach().float()
    ce = F.cross_entropy(student, labels.long())
    log_student = F.log_softmax(student / float(tau), dim=-1)
    log_teacher = F.log_softmax(teacher / float(tau), dim=-1)
    kd = (log_teacher.exp() * (log_teacher - log_student)).sum(dim=-1).mean() * float(tau) ** 2
    total = (1.0 - float(kd_weight)) * ce + float(kd_weight) * kd
    return total, {"ce": float(ce.detach()), "kd": float(kd.detach())}


def reload_q6(mode, root=None):
    """Load the selected full student and its tokenizer as a text predictor."""
    if mode not in ("ce", "kd"):
        raise ValueError("mode must be 'ce' or 'kd'")
    saved = common.load_weights("q6", root)
    entry = saved[mode]
    cfg = entry["config"]
    model_id = cfg["student"]
    revision = cfg.get("student_revision") or cfg.get("revision")
    kwargs = {"revision": revision} if revision else {}
    backbone = AutoModel.from_pretrained(model_id, **kwargs)
    model = SentimentClassifier(backbone, backbone.config.hidden_size, num_classes=2)
    model.load_state_dict(entry["state"], strict=True)
    model.to(common.DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.get("tokenizer", model_id), **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = cfg.get("padding_side", "right")
    collate = common.make_classifier_collate(tokenizer, max_length=int(cfg["max_length"]))
    return common.ClassifierPredictor(model, collate, batch_size=int(cfg.get("inference_batch", 16)))
