import torch
from abc import ABC, abstractmethod


class BaseSchedule(ABC):
    """
    Abstract base class for diffusion schedules.
    Defines the coefficients alpha(t) and sigma(t) for q(x_t | x_0).
    x_t = alpha(t) * x_0 + sigma(t) * epsilon
    """

    def __init__(self, device="cpu"):
        self.device = device

    @abstractmethod
    def get_coefficients(self, t: torch.Tensor):
        """
        Returns:
            alpha: Signal scale
            sigma: Noise scale
            d_alpha: Derivative of alpha w.r.t t (velocity component)
            d_sigma: Derivative of sigma w.r.t t (velocity component)
        """
        pass


class LinearSchedule(BaseSchedule):
    """
    Rectified Flow / Flow Matching Schedule.
    alpha(t) = 1 - t
    sigma(t) = t
    (Assuming t=0 is Data, t=1 is Noise)
    """

    def __init__(self, device="cpu", is_inverted: bool = False):
        super().__init__(device)
        self.is_inverted = is_inverted

    def get_coefficients(self, t: torch.Tensor):
        if not self.is_inverted:
            # alpha = t, sigma = 1-t
            alpha = t
            sigma = 1.0 - t
        else:
            alpha = 1.0 - t
            sigma = t

        d_alpha = -torch.ones_like(t)
        d_sigma = torch.ones_like(t)
        return alpha, sigma, d_alpha, d_sigma


class DDPMSchedule(BaseSchedule):
    """
    Standard SD 1.5 Discrete Beta Schedule mapped to continuous time.
    """

    def __init__(
        self,
        device="cpu",
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
    ):
        super().__init__(device)
        self.num_train_timesteps = num_train_timesteps

        # SD 1.5 Betas
        betas = (
            torch.linspace(
                beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32
            )
            ** 2
        )
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def get_coefficients(self, t: torch.Tensor):
        """
        Interpolates discrete schedule to continuous t [0, 1].
        t=0 -> Data (Step 0), t=1 -> Noise (Step T)
        """
        # Map t [0, 1] to indices [0, T-1]
        t_idx = (
            (t * (self.num_train_timesteps - 1))
            .long()
            .clamp(0, self.num_train_timesteps - 1)
        )

        alpha = self.sqrt_alphas_cumprod[t_idx]
        sigma = self.sqrt_one_minus_alphas_cumprod[t_idx]

        # Derivatives are set to zero
        d_alpha = torch.zeros_like(t)
        d_sigma = torch.zeros_like(t)

        return alpha, sigma, d_alpha, d_sigma
