import torch
import torch.nn as nn
import copy
import contextlib


class EMAModel:
    """
    Maintains an Exponential Moving Average of the model's parameters.
    Supports offloading the EMA weights to CPU to save VRAM.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.99,
        use_ema: bool = True,
        device: torch.device = None,
    ):
        self.decay = decay
        self.use_ema = use_ema
        self.device = device or next(model.parameters()).device
        self.step = 0

        if self.use_ema:
            # Deepcopy and move to the target device (CPU) in fp32
            self.ema_model = copy.deepcopy(model).to(
                device=self.device, dtype=torch.float32
            )
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()
        else:
            self.ema_model = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        if not self.use_ema:
            return

        self.step += 1
        # Dynamic decay: starts fast, converges to self.decay
        current_decay = min(self.decay, (1 + self.step) / (10 + self.step))

        ema_state_dict = self.ema_model.state_dict()
        model_state_dict = model.state_dict()

        for key, param in model_state_dict.items():
            if key in ema_state_dict:
                ema_param = ema_state_dict[key]
                if ema_param.is_floating_point():
                    param_data = param.data.to(
                        device=self.device, dtype=torch.float32, non_blocking=False
                    )
                    ema_param.data.mul_(current_decay).add_(
                        param_data, alpha=1.0 - current_decay
                    )
                else:
                    ema_param.data.copy_(param.data.to(self.device, non_blocking=True))

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module):
        """
        Temporarily replaces model's parameters with EMA parameters.
        """
        if not self.use_ema:
            yield
            return

        original_state = {k: v.clone().detach() for k, v in model.state_dict().items()}

        ema_state_dict = self.ema_model.state_dict()
        for key, param in model.state_dict().items():
            if key in ema_state_dict:
                ema_param = ema_state_dict[key]
                param.data.copy_(
                    ema_param.data.to(
                        device=param.device, dtype=param.dtype, non_blocking=False
                    )
                )

        try:
            yield
        finally:
            for key, param in model.state_dict().items():
                if key in original_state:
                    param.data.copy_(original_state[key].data)
