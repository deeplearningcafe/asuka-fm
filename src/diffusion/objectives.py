import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from src.diffusion.schedules import BaseSchedule


# based on https://github.com/bluvoll/sd-scripts-f2vae/blob/main/library/train_util.py
def euclidean_optimal_transport(
    X: torch.Tensor, Y: torch.Tensor, backend: str = "auto"
):
    """Compute an optimal assignment under Euclidean (L2) distance."""
    # X and Y are shape (B, D)
    cost = torch.cdist(X, Y, p=2.0)

    if backend == "cuda":
        return _cuda_assignment(cost)
    if backend == "scipy":
        return _scipy_assignment(cost)

    try:
        return _cuda_assignment(cost)
    except (ImportError, RuntimeError):
        return _scipy_assignment(cost)


def _cuda_assignment(cost: torch.Tensor):
    from torch_linear_assignment import (
        assignment_to_indices,
        batch_linear_assignment,
    )

    assignment = batch_linear_assignment(cost.unsqueeze(0))
    row_idx, col_idx = assignment_to_indices(assignment)
    return cost, (row_idx.squeeze(0), col_idx.squeeze(0))


def _scipy_assignment(cost: torch.Tensor):
    from scipy.optimize import linear_sum_assignment

    cost_np = cost.to(torch.float32).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    row = torch.from_numpy(row_ind).to(cost.device, torch.long)
    col = torch.from_numpy(col_ind).to(cost.device, torch.long)
    return cost, (row, col)


def time_snr_shift(t: torch.Tensor, shift: float = 1.0):
    """
    Shifts the time distribution.
    shift > 1 focuses more on the noise (t=1) end (higher SNR in some formulations).
    For Flow Matching (t=0 Data, t=1 Noise):
    Low t is data, High t is noise.
    """
    if shift == 1.0:
        return t
    return (t * shift) / (1 + (shift - 1) * t)


class DiffusionObjective(ABC):
    def __init__(self, schedule: BaseSchedule):
        self.schedule = schedule

    @abstractmethod
    def forward(self, model, x_start, condition, weights=None):
        pass


class FlowMatchingObjective(DiffusionObjective):
    """
    Conditional Flow Matching Loss.
    Target: Velocity v = dx/dt.
    """

    def __init__(
        self, schedule: BaseSchedule, shift: float = 1.0, use_ot: bool = False
    ):
        super().__init__(schedule)
        self.shift = shift
        self.use_ot = use_ot

    def forward(self, model, x_start, condition, weights=None, attention_mask=None):
        b, c, h, w = x_start.shape
        device = x_start.device

        nt = torch.randn((b,), device=device)
        t = torch.sigmoid(nt)
        t = time_snr_shift(t, self.shift)

        alpha, sigma, d_alpha, d_sigma = self.schedule.get_coefficients(t)

        alpha = alpha.view(b, 1, 1, 1)
        sigma = sigma.view(b, 1, 1, 1)

        epsilon = torch.randn_like(x_start)

        # Minibatch Optimal Transport
        if self.use_ot:
            data_flat = x_start.view(b, -1)
            eps_flat = epsilon.view(b, -1)

            _, (row_idx, col_idx) = euclidean_optimal_transport(data_flat, eps_flat)

            # Reorder eps
            eps_sorted = torch.empty_like(epsilon)
            eps_sorted[row_idx] = epsilon[col_idx]
            epsilon = eps_sorted

        x_t = alpha * x_start + sigma * epsilon

        # v = d_alpha * x_start + d_sigma * epsilon
        d_alpha = d_alpha.view(b, 1, 1, 1)
        d_sigma = d_sigma.view(b, 1, 1, 1)
        v_target = d_alpha * x_start + d_sigma * epsilon

        # SD1.5 UNet expects timesteps [0, 1000].
        t_input = t * 1000.0

        model_output = model(
            x_t, t_input, encoder_hidden_states=condition, attention_mask=attention_mask
        )

        v_pred_metrics = []
        v_true_metrics = []
        loss = F.mse_loss(model_output, v_target, reduction="none")
        with torch.no_grad():
            v_pred_metrics.append(torch.norm(model_output.detach()))
            v_true_metrics.append(torch.norm(v_target.detach()))
            v_pred_metrics.append(torch.mean(torch.abs(model_output.detach())))
            v_true_metrics.append(torch.mean(torch.abs(v_target.detach())))

        raw_loss = loss.mean(dim=[1, 2, 3])

        loss = raw_loss
        if weights is not None:
            loss = loss * weights
        loss = loss.mean()

        return loss, {
            "loss": loss.detach(),
            "raw_loss": raw_loss.mean().detach(),
            "pred_norm": v_pred_metrics[0],
            "pred_mean_abs": v_pred_metrics[1],
            "target_norm": v_true_metrics[0],
            "target_mean_abs": v_true_metrics[1],
        }


class DDPMObjective(DiffusionObjective):
    """
    Standard DDPM Epsilon Prediction.
    """

    def __init__(
        self,
        schedule: BaseSchedule,
        min_snr_gamma: float = 5.0,
        input_perturb: float = 0.0,
    ):
        super().__init__(schedule)
        self.min_snr_gamma = min_snr_gamma
        self.input_perturb = input_perturb

    def forward(self, model, x_start, condition, weights=None, attention_mask=None):
        b, c, h, w = x_start.shape
        device = x_start.device

        t_idx = torch.randint(0, 1000, (b,), device=device).long()

        # Normalize t for schedule query
        t_norm = t_idx.float() / 1000.0

        alpha, sigma, _, _ = self.schedule.get_coefficients(t_norm)
        alpha = alpha.view(b, 1, 1, 1)
        sigma = sigma.view(b, 1, 1, 1)

        # Noise with Perturbation
        noise = torch.randn_like(x_start)
        if self.input_perturb > 0:
            noise = noise + self.input_perturb * torch.rand_like(x_start)

        x_t = alpha * x_start + sigma * noise

        model_output = model(
            x_t, t_idx, encoder_hidden_states=condition, attention_mask=attention_mask
        )

        loss = F.mse_loss(model_output, noise, reduction="none")
        raw_loss = loss.mean(dim=[1, 2, 3])

        v_pred_metrics = []
        v_true_metrics = []
        with torch.no_grad():
            v_pred_metrics.append(torch.norm(model_output.detach()))
            v_true_metrics.append(torch.norm(noise.detach()))
            v_pred_metrics.append(torch.mean(torch.abs(model_output.detach())))
            v_true_metrics.append(torch.mean(torch.abs(noise.detach())))

        # Min-SNR Weighting
        snr_weights = torch.ones_like(raw_loss)
        if self.min_snr_gamma > 0.0:
            snr = (alpha / sigma) ** 2
            snr_weights = torch.clamp(self.min_snr_gamma / snr, max=1.0).squeeze()

        loss = raw_loss * snr_weights

        if weights is not None:
            loss = loss * weights

        return loss.mean(), {
            "loss": loss.mean().detach(),
            "raw_loss": raw_loss.mean().detach(),
            "pred_norm": v_pred_metrics[0],
            "pred_mean_abs": v_pred_metrics[1],
            "target_norm": v_true_metrics[0],
            "target_mean_abs": v_true_metrics[1],
        }
