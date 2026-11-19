"""Assignment Q4 simplified Muon: matrix momentum + quintic Newton–Schulz."""
import math
import torch


@torch.no_grad()
def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5,
                                eps: float = 1e-7) -> torch.Tensor:
    if G.ndim != 2:
        raise ValueError("Newton–Schulz requires a 2D matrix")
    if steps < 0 or eps <= 0:
        raise ValueError("steps must be nonnegative and eps positive")
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ A @ X)
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 ns_steps: int = 5):
        if lr <= 0 or not 0 <= momentum < 1 or ns_steps < 0:
            raise ValueError("Invalid Muon hyperparameters")
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon is defined only for 2D weight matrices")
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                update = newton_schulz_orthogonalize(buf, steps=group["ns_steps"])
                update = update * math.sqrt(max(1.0, p.size(0) / p.size(1)))
                p.add_(update, alpha=-group["lr"])
        return loss
