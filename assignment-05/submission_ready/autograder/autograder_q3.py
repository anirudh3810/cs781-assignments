"""autograder_q3.py — Safety alignment: RLHF / DPO / ORPO / GRPO (CPU only).

    python autograder/autograder_q3.py --student_dir .

1. Unit tests: every loss is evaluated on FIXED tensors and compared with known values.
2. Behaviour: base.pt and aligned.pt are forward-passed on CPU over the HP safety preference set;
   alignment must raise the (safe - harmful) log-probability margin.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_grading import (GradeReport, cli_main, load_student_module, load_model_arch, load_decoder,
                            load_metrics, require, defines_function, is_implemented, approx, REPO_ROOT)

PC = torch.tensor([-2.0, -1.5]); PR = torch.tensor([-3.0, -2.0])
RC = torch.tensor([-2.5, -1.8]); RR = torch.tensor([-2.6, -2.1])
REF = {"dpo": 0.666178, "orpo": 1.036187, "rm": 0.340642, "grpo_loss": -0.047415,
       "reinforce": -0.25, "seqlp": -3.697186}
FUNCS = ["sequence_logprobs", "reward_model_loss", "reinforce_policy_loss", "dpo_loss",
         "orpo_loss", "grpo_advantages", "grpo_loss"]


def main(sd):
    rep = GradeReport("Q3 — Safety alignment")
    apy = os.path.join(sd, "alignment.py"); q3 = os.path.join(sd, "submissions", "q3")
    if not require(rep, [apy]):
        return rep.summary()
    for f in FUNCS:
        rep.check(f"implements {f}", defines_function(apy, f) and is_implemented(apy, f))
    AL = load_student_module(apy, "student_alignment")

    rep.guard("dpo_loss value", lambda: (approx(AL.dpo_loss(PC, PR, RC, RR, 0.1).item(), REF["dpo"], 1e-3), ""))
    rep.guard("dpo_loss: reference log-probs receive no gradient", lambda: _dpo_detach(AL))
    rep.guard("reward_model_loss value", lambda: (approx(AL.reward_model_loss(
        torch.tensor([1.0, 0.5]), torch.tensor([-1.0, 0.2])).item(), REF["rm"], 1e-3), ""))
    rep.guard("reinforce_policy_loss value", lambda: (approx(AL.reinforce_policy_loss(
        torch.tensor([-1.0, -2.0]), torch.tensor([1.0, 0.0])).item(), REF["reinforce"], 1e-3), ""))
    rep.guard("orpo_loss value", lambda: (approx(AL.orpo_loss(
        PC, PR, torch.tensor([1.2, 0.8]), 0.1).item(), REF["orpo"], 1e-3), ""))
    rep.guard("grpo_advantages: zero mean, unit std", lambda: _adv(AL))
    rep.guard("grpo_loss value (clipped surrogate)", lambda: (approx(AL.grpo_loss(
        torch.tensor([0.0, 0.1]), torch.tensor([-0.2, 0.0]), torch.tensor([1.0, -1.0]), 0.2).item(),
        REF["grpo_loss"], 1e-3), ""))

    def t_seqlp():
        torch.manual_seed(0)
        lg = torch.randn(1, 4, 5); ids = torch.tensor([[1, 2, 3, 4]]); mask = torch.tensor([[0., 0., 1., 1.]])
        return approx(AL.sequence_logprobs(lg, ids, mask).item(), REF["seqlp"], 1e-3), ""
    rep.guard("sequence_logprobs value (causal shift + response mask)", t_seqlp)

    base_pt, al_pt = os.path.join(q3, "base.pt"), os.path.join(q3, "aligned.pt")
    if not (os.path.exists(base_pt) and os.path.exists(al_pt)):
        rep.check("submissions/q3/base.pt and aligned.pt present", False)
        return rep.summary()
    import common_a5 as C
    MA = load_model_arch(sd)
    pref = os.path.join(sd, "data", "preferences.jsonl")
    recs = C.load_preferences(pref if os.path.exists(pref) else os.path.join(REPO_ROOT, "data", "preferences.jsonl"))
    tp = os.path.join(q3, "tokenizer.json")
    tok = C.CharTokenizer.load(tp) if os.path.exists(tp) else C.build_char_tokenizer_from_preferences(recs)
    base, _ = load_decoder(base_pt, MA); aligned, _ = load_decoder(al_pt, MA)
    batch = C.make_pref_batch(recs, tok, max_len=base.cfg.max_seq_len)

    def margin(m):
        cl = AL.sequence_logprobs(m(batch["chosen_ids"]), batch["chosen_ids"], batch["chosen_mask"])
        rl = AL.sequence_logprobs(m(batch["rejected_ids"]), batch["rejected_ids"], batch["rejected_mask"])
        return (cl - rl), (cl - rl).mean().item()
    try:
        per_b, bm = margin(base); per_a, am = margin(aligned)
    except Exception as e:  # noqa: BLE001
        rep.check("behaviour checks (need a working sequence_logprobs)", False, f"{type(e).__name__}: {e}")
        return rep.summary()
    rep.check("aligned margin > base margin", am > bm, f"base {bm:.2f} -> aligned {am:.2f}")
    rep.check("aligned model prefers the safe response on average (margin > 0)", am > 0, f"{am:.2f}")
    rep.check("aligned model prefers the safe response on >= 80% of prompts",
              (per_a > 0).float().mean().item() >= 0.8, f"{(per_a > 0).float().mean().item():.0%}")
    if os.path.exists(os.path.join(q3, "metrics.json")):
        m = load_metrics(q3)
        rep.check("logged aligned_margin matches the saved model",
                  approx(m.get("aligned_margin", 1e9), am, max(1.0, 0.05 * abs(am))),
                  f"logged {m.get('aligned_margin')} vs {am:.2f}")
    return rep.summary()


@torch.enable_grad()
def _dpo_detach(AL):
    pc, pr = PC.clone().requires_grad_(), PR.clone().requires_grad_()
    rc, rr = RC.clone().requires_grad_(), RR.clone().requires_grad_()
    AL.dpo_loss(pc, pr, rc, rr, 0.1).backward()
    ok = pc.grad is not None and (rc.grad is None or rc.grad.abs().sum() == 0)
    return ok, ""


def _adv(AL):
    a = AL.grpo_advantages(torch.tensor([1.0, 2.0, 3.0, 0.0, 7.0]))
    return abs(a.mean().item()) < 1e-4 and abs(a.std().item() - 1) < 1e-3, f"mean {a.mean():.4f} std {a.std():.4f}"


if __name__ == "__main__":
    cli_main(main)
