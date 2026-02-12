# -----------------------------
# Ridge solver: α* = (Φ^TΦ + λI)^{-1} Φ^T f
# -----------------------------

import torch

def ridge_solve(Phi: torch.Tensor, f: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Phi: (B, p)
    f:   (B,) or (B,1)
    returns α*: (p,)
    """
    if f.ndim == 2 and f.shape[1] == 1:
        f = f[:, 0]
    _, p = Phi.shape
    A = Phi.T @ Phi + lam * torch.eye(p, device=Phi.device, dtype=Phi.dtype)  # (p, p)
    b = Phi.T @ f  # (p,)
    alpha = torch.linalg.solve(A, b)  # (p,)
    return alpha
