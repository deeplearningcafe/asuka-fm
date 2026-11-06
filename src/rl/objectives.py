import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any


# from https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py#L11
def cps_step_with_logprob(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coefficients-Preserving Sampling (CPS) step for Flow Matching.

    Interpolates deterministically along probability flow while preserving
    exact schedule variance when injecting stochasticity.
    Convention: t=0 is pure noise, t=1 is clean data; dt > 0 integrates
    forward towards clean data.
    """
    sample_f32 = sample.float()
    v_f32 = model_output.float()
    prev_f32 = prev_sample.float() if prev_sample is not None else None

    if isinstance(t, torch.Tensor):
        t_val = t.view(-1, *([1] * (sample.ndim - 1))).float()
    else:
        t_val = float(t)

    t_next = t_val + dt
    sigma_curr = 1.0 - t_val
    sigma_next = (1.0 - t_next).clamp(min=0.0)
    alpha_next = t_next.clamp(max=1.0)

    # Predicted clean data (x1) and noise (x0) from velocity v = x1 - x0
    pred_x1 = sample_f32 + sigma_curr * v_f32
    pred_x0 = sample_f32 - t_val * v_f32

    # CPS angle: eta in [0, 1] maps to [0, pi / 2]
    angle = float(noise_level) * (math.pi / 2.0)
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)

    # Transition mean: linear blend of predicted data and noise
    prev_mean = alpha_next * pred_x1 + sigma_next * cos_theta * pred_x0
    std_dev_step = sigma_next * sin_theta

    if prev_f32 is None:
        if sin_theta > 1e-5 and std_dev_step.max() > 1e-6:
            eps = torch.randn(
                sample_f32.shape,
                generator=generator,
                device=sample_f32.device,
                dtype=torch.float32,
            )
            prev_sample_out = prev_mean + std_dev_step * eps
        else:
            prev_sample_out = prev_mean
    else:
        prev_sample_out = prev_f32

    # Gaussian log-probability density calculation
    variance = (std_dev_step**2).clamp(min=1e-7)
    diff_sq = (prev_sample_out.detach() - prev_mean) ** 2
    log_prob = (
        -0.5 * diff_sq / variance
        - torch.log(std_dev_step.clamp(min=1e-5))
        - 0.5 * math.log(2.0 * math.pi)
    )

    dim_indices = tuple(range(1, log_prob.ndim))
    log_prob = log_prob.mean(dim=dim_indices)

    return (
        prev_sample_out.to(sample.dtype),
        log_prob,
        prev_mean.to(sample.dtype),
        std_dev_step,
    )


def sde_step_with_logprob(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reverse SDE step for Flow Matching converting velocity to mean drift
    and calculating transition log-probabilities.

    Time convention in asuka-fm:
        t = 0.0 is pure Gaussian noise (alpha=0, sigma=1).
        t = 1.0 is clean data (alpha=1, sigma=0).
        dt > 0 integrates forward from noise to data.
    """
    sample_f32 = sample.float()
    model_f32 = model_output.float()
    prev_f32 = prev_sample.float() if prev_sample is not None else None

    if isinstance(t, torch.Tensor):
        t_val = t.view(-1, *([1] * (sample.ndim - 1))).float()
    else:
        t_val = float(t)

    sigma = 1.0 - t_val

    # At t=0 (sigma=1), 1 - sigma is 0.
    denom_1_minus_sigma = torch.where(
        sigma >= 1.0 - 1e-5,
        torch.tensor(dt, device=sample.device, dtype=torch.float32),
        (1.0 - sigma).clamp(min=dt),
    )

    sigma_safe = sigma.clamp(min=dt)

    # SDE diffusion coefficient g(t) = std_dev_t
    std_dev_t = torch.sqrt(sigma / denom_1_minus_sigma) * noise_level
    std_dev_step = std_dev_t * math.sqrt(dt)

    # Mean drift towards clean data (dt > 0, v = dx/dt = x0 - eps)
    decay_factor = 1.0 - ((std_dev_t**2) / (2.0 * sigma_safe)) * dt
    drift_factor = 1.0 + (std_dev_t**2) * (1.0 - sigma) / (2.0 * sigma_safe)
    prev_mean = sample_f32 * decay_factor + model_f32 * drift_factor * dt

    if prev_f32 is None:
        noise = torch.randn(
            model_f32.shape,
            generator=generator,
            device=model_f32.device,
            dtype=torch.float32,
        )
        prev_sample_out = prev_mean + std_dev_step * noise
    else:
        prev_sample_out = prev_f32

    # Transition log-probability under N(prev_mean, std_dev_step^2 * I)
    variance = (std_dev_step**2).clamp(min=1e-8)
    log_prob = (
        -0.5 * ((prev_sample_out.detach() - prev_mean) ** 2) / variance
        - torch.log(std_dev_step.clamp(min=1e-8))
        - 0.5 * math.log(2.0 * math.pi)
    )

    dim_indices = tuple(range(1, log_prob.ndim))
    log_prob = log_prob.mean(dim=dim_indices)

    return (
        prev_sample_out.to(sample.dtype),
        log_prob,
        prev_mean.to(sample.dtype),
        std_dev_step,
    )


class FlowGRPOObjective(nn.Module):
    """
    Flow-GRPO policy gradient loss with PPO clipping and optional KL loss.
    """

    def __init__(
        self,
        clip_range: float = 0.2,
        adv_clip_max: float = 5.0,
        beta: float = 0.0,
        noise_level: float = 0.7,
        sde_type: str = "sde",
        use_unet_mult: bool = True,
    ):
        super().__init__()
        self.clip_range = float(clip_range)
        self.adv_clip_max = float(adv_clip_max)
        self.beta = float(beta)
        self.noise_level = float(noise_level)
        self.sde_type = str(sde_type).lower()
        self.use_unet_mult = use_unet_mult

    def _step_fn(self, **kwargs):
        if self.sde_type == "cps":
            return cps_step_with_logprob(**kwargs)
        return sde_step_with_logprob(**kwargs)

    def forward(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        next_latents: torch.Tensor,
        timesteps: torch.Tensor,
        dt: float,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        condition: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pos_map: Optional[torch.Tensor] = None,
        ref_model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        t_model = timesteps * 1000.0 if self.use_unet_mult else timesteps
        # this is the te output
        model_kwargs = {
            "encoder_hidden_states": condition,
            "attention_mask": attention_mask,
        }
        if pos_map is not None:
            model_kwargs["pos_map"] = pos_map

        v_pred = model(latents, t_model, **model_kwargs)

        # Transition log-prob under current policy
        _, log_prob, prev_mean, std_dev_step = self._step_fn(
            sample=latents,
            model_output=v_pred,
            t=timesteps,
            dt=dt,
            noise_level=self.noise_level,
            prev_sample=next_latents,
        )

        # GRPO clipped surrogate policy loss
        adv = torch.clamp(advantages, -self.adv_clip_max, self.adv_clip_max)
        ratio = torch.exp(log_prob - old_log_probs)
        unclipped_loss = -adv * ratio
        clipped_loss = -adv * torch.clamp(
            ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
        )
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

        # reference model KL penalty
        kl_loss = torch.tensor(0.0, device=latents.device)
        if self.beta > 0.0 and ref_model is not None:
            with torch.no_grad():
                v_ref = ref_model(latents, t_model, **model_kwargs)
                _, _, mean_ref, _ = self._step_fn(
                    sample=latents,
                    model_output=v_ref,
                    t=timesteps,
                    dt=dt,
                    noise_level=self.noise_level,
                    prev_sample=next_latents,
                )
            var = (std_dev_step**2).clamp(min=1e-8)
            kl = ((prev_mean - mean_ref) ** 2) / (2.0 * var)
            kl_loss = kl.mean()
            total_loss = policy_loss + self.beta * kl_loss
        else:
            total_loss = policy_loss

        with torch.no_grad():
            approx_kl = 0.5 * torch.mean((log_prob - old_log_probs) ** 2)
            clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.clip_range).float())
            metrics = {
                "loss": total_loss.detach(),
                "policy_loss": policy_loss.detach(),
                "kl_loss": kl_loss.detach(),
                "approx_kl": approx_kl.detach(),
                "clipfrac": clipfrac.detach(),
                "ratio_mean": ratio.mean().detach(),
                "adv_mean": adv.mean().detach(),
            }

        return total_loss, metrics
