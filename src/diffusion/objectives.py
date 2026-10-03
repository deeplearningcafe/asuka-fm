import math
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from src.diffusion.schedules import BaseSchedule


def logit_normal_sample(n, device, mean=0.0, std=1.0, shift=1.0):
    """
    Samples t from a Logit-Normal distribution for t=0 (Noise) -> t=1 (Data).
    Shifts the mean by -log(shift) to concentrate samples towards noise (t=0).
    """
    # For t=0 (Data), FLUX shifts logit by log(s).
    mean = -math.log(shift) if shift != 1.0 else 0.0
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
        prediction_target: str = "v",
        loss_target: str = "v",
        noise_scale: float = 1.0,
        timestep_sampling: str = "logit-normal",
        shift: float = 1.0,
        use_ot: bool = False,
        use_unet_mult: bool = True,
    ):
        super().__init__(schedule)
        self.prediction_target = prediction_target
        self.loss_target = loss_target
        self.noise_scale = float(noise_scale)
        self.shift = shift
        self.use_ot = use_ot
        self.use_unet_mult = use_unet_mult
        self.timestep_sampling_fn = get_timestep_sampling_fn(timestep_sampling)
        self.clip_denom = self.prediction_target == "x" and self.loss_target == "v"

    def _solve_linear_system(
        self,
        pred: torch.Tensor,
        z_t: torch.Tensor,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
        d_alpha: torch.Tensor,
        d_sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solves linear system to convert model prediction to (x, eps, v)."""
        alpha = alpha.clamp(min=1e-5)
        min_sigma = 0.05 if self.clip_denom else 1e-4
        sigma_safe = sigma.clamp(min=min_sigma)

        det = alpha * d_sigma - sigma * d_alpha
        det_safe = torch.where(det.abs() < 1e-5, 1e-5 * torch.sign(det + 1e-35), det)

        if self.prediction_target == "x":
            x_pred = pred
            eps_pred = (z_t - alpha * x_pred) / sigma_safe
            v_pred = d_alpha * x_pred + d_sigma * eps_pred
        elif self.prediction_target == "eps":
            eps_pred = pred
            x_pred = (z_t - sigma * eps_pred) / alpha
            v_pred = d_alpha * x_pred + d_sigma * eps_pred
        elif self.prediction_target == "v":
            v_pred = pred
            x_pred = (d_sigma * z_t - sigma * v_pred) / det_safe
            eps_pred = (alpha * v_pred - d_alpha * z_t) / det_safe
        else:
            raise ValueError(f"Unknown prediction target: {self.prediction_target}")

        return x_pred, eps_pred, v_pred

    def _compute_ot_eps(self, data: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Solves optimal transport assignment under L2 distance in eager mode."""
        b = data.shape[0]
        data_flat = data.view(b, -1)
        eps_flat = eps.view(b, -1)
        _, (row_idx, col_idx) = euclidean_optimal_transport(data_flat, eps_flat)
        eps_sorted = torch.empty_like(eps)
        eps_sorted[row_idx] = eps[col_idx]
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
        b = x_start.shape[0]
        """Compilable forward loss step free of graph breaks."""
        device = x_start.device

        t = self.timestep_sampling_fn(b, device, shift=self.shift)
        if self.clip_denom:
            t = t.clamp(min=1e-3, max=0.98)
        t_view = t.view(-1, *([1] * (x_start.ndim - 1)))
        alpha, sigma, d_alpha, d_sigma = self.schedule.get_coefficients(t_view)

        if self.noise_scale != 1.0:
            sigma = sigma * self.noise_scale
            d_sigma = d_sigma * self.noise_scale

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

        x_pred, eps_pred, v_pred = self._solve_linear_system(
            model_output, x_t, alpha, sigma, d_alpha, d_sigma
        )

        if self.loss_target == "x":
            pred = x_pred
            target = x_start
        elif self.loss_target == "eps":
            pred = eps_pred
            target = epsilon
        elif self.loss_target == "v":
            pred = v_pred
            target = v_target
        else:
            raise ValueError(f"Unknown loss target: {self.loss_target}")

        loss = F.mse_loss(
            pred.to(torch.float32),
            target.to(torch.float32),
            reduction="none",
        )
        raw_loss = loss.mean(dim=[1, 2, 3])

        final_loss = raw_loss
        if weights is not None:
            final_loss = final_loss * weights
        final_loss = final_loss.mean()

        pred_norm = torch.norm(model_output.detach())
        target_norm = torch.norm(target.detach())
        pred_abs = torch.mean(torch.abs(model_output.detach()))
        target_abs = torch.mean(torch.abs(target.detach()))

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
        model: torch.nn.Module,
        x_start: torch.Tensor,
        condition: torch.Tensor,
        weights=None,
        attention_mask=None,
        pos_map=None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Public entrypoint generating Gaussian noise and executing loss step.
        """
        epsilon = torch.randn_like(x_start)
        if self.use_ot:
            epsilon = self._compute_ot_eps(x_start, epsilon)

        return self._compiled_loss_step(
            model=model,
            x_start=x_start,
            condition=condition,
            epsilon=epsilon,
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
