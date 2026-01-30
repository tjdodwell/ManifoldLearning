import torch

def grassmann_tangent_projection(M: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """
    Project Euclidean gradient G to the Grassmann tangent space at M:
        Δ = (I - M M^T) G
    M, G: (d, r)
    """
    return G - M @ (M.T @ G)

def qr_retract(M: torch.Tensor) -> torch.Tensor:
    """
    QR retraction onto Stiefel (orthonormal columns). On the Grassmann,
    this is a valid retraction (subspace is what matters).
    M: (d, r)
    """
    Q, R = torch.linalg.qr(M, mode="reduced")
    # Optional sign-fix for stability (keeps QR deterministic-ish)
    s = torch.sign(torch.diag(R))
    s[s == 0] = 1.0
    Q = Q * s
    return Q


