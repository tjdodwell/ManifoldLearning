import torch

from typing import Tuple, Optional

from manifold_learning.ridge import ridge_solve
from manifold_learning.manifolds import qr_retract, grassmann_tangent_projection
from manifold_learning.features import RFFFeatures


@torch.no_grad()
def stochastic_grassmann_ridge(
    X: torch.Tensor,                      # (N, d) rows are x_i^T
    f: torch.Tensor,                      # (N,) or (N,1) targets f_i
    r: int,                               # reduced dimension
    feature_model: RFFFeatures,           # provides φ and J_φ^T α
    lam: float = 1e-2,                    # λ
    eta: float = 1e-1,                    # η_k (constant here)
    batch_size: int = 256,                # B
    steps: int = 2000,                    # budget
    tol: float = 1e-6,                    # stabilisation tolerance
    patience: int = 25,                   # stabilisation patience
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    Returns:
      M_star: (d, r)
      alpha_final: (p,) ridge weights trained on full data in reduced space
    """
    torch.manual_seed(seed)
    device = X.device
    dtype = X.dtype

    N, d = X.shape
    assert feature_model.r == r, "feature_model.r must match r"
    p = feature_model.p

    # Initialise M0 with orthonormal columns
    M0 = torch.randn(d, r, device=device, dtype=dtype)
    M = qr_retract(M0)

    # Stabilisation metric: change in projection matrix ||P(M_{k+1}) - P(M_k)||_F
    def proj(M_: torch.Tensor) -> torch.Tensor:
        return M_ @ M_.T

    stable = 0
    prev_metric: Optional[float] = None

    for k in range(steps):
        # 1) Sample mini-batch B_k
        idx = torch.randint(0, N, (batch_size,), device=device)
        Xb = X[idx]  # (B, d)
        fb = f[idx]  # (B,) or (B,1)

        if fb.ndim == 2 and fb.shape[1] == 1:
            fb = fb[:, 0]

        # 2) Projected coordinates z_i = M^T x_i
        # Xb rows are x_i^T so Z = Xb M gives z_i^T; shape (B, r)
        Z = Xb @ M  # (B, r)

        # Construct Φ_Bk(M_k)
        Phi = feature_model.phi(Z)  # (B, p)

        # 3) Solve ridge regression for α*
        alpha_star = ridge_solve(Phi, fb, lam)  # (p,)

        # 4) Residuals e = f_Bk - Φ α*
        fhat = Phi @ alpha_star         # (B,)
        e = fb - fhat                   # (B,)

        # 5) Euclidean gradient:
        #   G = -2 Σ_{i=1}^B e_i x_i (J_φ(z_i)^T α*)^T
        # Compute g_i = J_φ(z_i)^T α*  ∈ R^r for each i
        g = feature_model.Jphi_times_v(Z, alpha_star)  # (B, r)

        # Form G in a vectorised way:
        # For each i, contribution is (-2 e_i) * x_i[:,None] @ g_i[None,:]
        # Sum_i -> Xb^T @ ((-2 e)[:,None] * g)
        W = (-2.0 * e)[:, None] * g   # (B, r)
        G = Xb.T @ W                  # (d, r)

        # 6) Project to Grassmann tangent space Δ = (I - M M^T) G
        Delta = grassmann_tangent_projection(M, G)  # (d, r)

        # 7) Retract (QR):  M~ = M - η Δ ;  M_{k+1} = qf(M~)
        M_new = qr_retract(M - eta * Delta)

        # stop when J_B(M) stabilises (proxy: subspace change)
        metric = torch.linalg.norm(proj(M_new) - proj(M), ord="fro").item()
        M = M_new

        if prev_metric is not None and abs(prev_metric - metric) < tol:
            stable += 1
        else:
            stable = 0
        prev_metric = metric

        if stable >= patience:
            break

    # Final surrogate trained on full data in reduced space
    Z_full = X @ M
    Phi_full = feature_model.phi(Z_full)
    alpha_final = ridge_solve(Phi_full, f if f.ndim == 1 else f[:, 0], lam)

    return M, alpha_final