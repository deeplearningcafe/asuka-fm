import math
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from src.diffusion.schedules import BaseSchedule


def logit_normal_sample(n, device, mean=0.0, std=1.0, shift=1.0):
    """
    Samples t from a Logit-Normal distribution.
    Applies timeshift mathematically by adding log(shift) to the mean.
    """
    # For t=0 (Data), FLUX shifts logit by log(s).
    mean = math.log(shift) if shift != 1.0 else 0.0
    s = torch.randn(n, device=device) * std + mean
    return torch.sigmoid(s)


def log_normal_sigma(n, device, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
    rnd_normal = torch.randn(n, device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    return sigma


def uniform_timesteps(n, device):
    """Standard Uniform sampling t ~ U[0, 1]"""
    timesteps = torch.rand((n,), device=device)
    return timesteps


def get_timestep_sampling_fn(timestep_sampling):
    if timestep_sampling == "logit-normal":

        def sample_fn(n, device, shift=1.0):
            return logit_normal_sample(n, device, mean=0.0, std=1.0, shift=shift)

        return sample_fn
    elif timestep_sampling == "uniform":

        def sample_fn(n, device, shift=1.0):
            return uniform_timesteps(n, device)

        return sample_fn


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
    Conditional Flow Matching Loss supporting torch.compile optimization.
    Target: Velocity v = dx/dt = d_alpha * x_start + d_sigma * epsilon.
    """

    def __init__(
        self,
        schedule: BaseSchedule,
        timestep_sampling: str = "logit-normal",
        shift: float = 1.0,
        use_ot: bool = False,
        use_unet_mult: bool = True,
    ):
        super().__init__(schedule)
        self.shift = shift
        self.use_ot = use_ot
        self.use_unet_mult = use_unet_mult
        self.timestep_sampling_fn = get_timestep_sampling_fn(timestep_sampling)

    def _compute_ot_eps(self, x_start: torch.Tensor, epsilon: torch.Tensor):
        """Computes Optimal Transport assignment in eager mode."""
        b = x_start.shape[0]
        data_flat = x_start.view(b, -1)
        eps_flat = epsilon.view(b, -1)
        _, (row_idx, col_idx) = euclidean_optimal_transport(data_flat, eps_flat)
        eps_sorted = torch.empty_like(epsilon)
        eps_sorted[row_idx] = epsilon[col_idx]
        return eps_sorted

    def _compiled_loss_step(
        self,
        model,
        x_start,
        condition,
        epsilon,
        weights=None,
        attention_mask=None,
        pos_map=None,
    ):
        """Compilable forward loss step free of graph breaks."""
        b = x_start.shape[0]
        device = x_start.device

        t = self.timestep_sampling_fn(b, device, shift=self.shift)
        t_view = t.view(-1, *([1] * (x_start.ndim - 1)))
        alpha, sigma, d_alpha, d_sigma = self.schedule.get_coefficients(t_view)

        x_t = alpha * x_start + sigma * epsilon
        v_target = d_alpha * x_start + d_sigma * epsilon

        t_input = t
        if self.use_unet_mult:
            t_input = t_input * 1000

        model_kwargs = {
            "encoder_hidden_states": condition,
            "attention_mask": attention_mask,
        }
        if pos_map is not None:
            model_kwargs["pos_map"] = pos_map

        model_output = model(x_t, t_input, **model_kwargs)

        loss = F.mse_loss(
            model_output.to(torch.float32),
            v_target.to(torch.float32),
            reduction="none",
        )
        raw_loss = loss.mean(dim=[1, 2, 3])

        final_loss = raw_loss
        if weights is not None:
            final_loss = final_loss * weights
        final_loss = final_loss.mean()

        pred_norm = torch.norm(model_output.detach())
        target_norm = torch.norm(v_target.detach())
        pred_abs = torch.mean(torch.abs(model_output.detach()))
        target_abs = torch.mean(torch.abs(v_target.detach()))

        metrics = {
            "loss": final_loss.detach(),
            "raw_loss": raw_loss.mean().detach(),
            "pred_norm": pred_norm,
            "pred_mean_abs": pred_abs,
            "target_norm": target_norm,
            "target_mean_abs": target_abs,
        }

        return final_loss, metrics

    def forward(
        self,
        model,
        x_start,
        condition,
        weights=None,
        attention_mask=None,
        pos_map=None,
    ):
        epsilon = torch.randn_like(x_start)
        if self.use_ot:
            epsilon = self._compute_ot_eps(x_start, epsilon)

        return self._compiled_loss_step(
            model,
            x_start,
            condition,
            epsilon,
            weights=weights,
            attention_mask=attention_mask,
            pos_map=pos_map,
        )


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
