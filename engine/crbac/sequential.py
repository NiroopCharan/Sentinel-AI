"""Mixture-free product e-process for conditionally super-uniform p-values."""
import math

class ProductEProcess:
    """E_t=prod(kappa*p_t**(kappa-1)); valid only if each p is conditionally super-uniform."""
    def __init__(self, kappa: float = 0.5, alpha: float = 0.05):
        if not 0 < kappa < 1: raise ValueError("kappa must be strictly between 0 and 1")
        if not 0 < alpha < 1: raise ValueError("alpha must be strictly between 0 and 1")
        self.kappa, self.alpha, self.log_e = kappa, alpha, 0.0
        self.n = 0

    def update(self, p_value: float) -> tuple[float, bool]:
        if not 0 < p_value <= 1: raise ValueError("p_value must be in (0, 1]")
        self.log_e += math.log(self.kappa) + (self.kappa - 1.0) * math.log(p_value)
        self.n += 1
        # Compare in log space to avoid overflow.
        return self.log_e, self.log_e >= math.log(1.0 / self.alpha)

    @property
    def e_value(self) -> float:
        return math.exp(min(self.log_e, 700.0))
