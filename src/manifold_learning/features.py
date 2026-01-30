import torch

class RFFFeatures(torch.nn.Module):
    """
    Random Fourier Features for an RBF kernel.
    φ: R^r -> R^p,   p = 2m
      φ(z) = sqrt(2/m) [cos(Ωz + b), sin(Ωz + b)]
    Jacobian available in closed form.

    This is a convenient concrete φ that matches the algorithm's need for J_φ(z).
    """
    def __init__(self, r: int, m: int, sigma: float = 1.0, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.r = r
        self.m = m
        self.p = 2 * m
        self.sigma = sigma

        # Ω ~ N(0, 1/sigma^2)
        Omega = torch.randn(m, r, **factory_kwargs) / sigma
        b = 2 * torch.pi * torch.rand(m, **factory_kwargs)
        self.register_buffer("Omega", Omega)  # (m, r)
        self.register_buffer("b", b)          # (m,)

        self.scale = (2.0 / m) ** 0.5

    def phi(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: (B, r)
        returns Φ: (B, p)
        """
        T = Z @ self.Omega.T + self.b  # (B, m)
        return self.scale * torch.cat([torch.cos(T), torch.sin(T)], dim=1)  # (B, 2m)

    def Jphi_times_v(self, Z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute (J_φ(z)^T v) for each row in Z, where:
          Z: (B, r)
          v: (p,) or (B, p)

        Returns:
          g: (B, r) where g[i] = J_φ(z_i)^T v_i
        This matches the algorithm term: J_φ(z_i)^T α*
        """
        if v.ndim == 1:
            v = v[None, :].expand(Z.shape[0], -1)  # (B, p)

        m = self.m
        v_cos = v[:, :m]   # (B, m)
        v_sin = v[:, m:]   # (B, m)

        T = Z @ self.Omega.T + self.b            # (B, m)
        sinT = torch.sin(T)
        cosT = torch.cos(T)

        # d/dz cos(T) = -sin(T) * Ω
        # d/dz sin(T) =  cos(T) * Ω
        # φ has extra scale factor
        # For each sample i:
        # J_φ(z_i)^T v_i = scale * Σ_j [ vcos_ij * (-sinT_ij) + vsin_ij * cosT_ij ] * Ω_j
        weights = self.scale * (-v_cos * sinT + v_sin * cosT)  # (B, m)
        # weights @ Ω gives (B, r)
        return weights @ self.Omega  # (B, r)
