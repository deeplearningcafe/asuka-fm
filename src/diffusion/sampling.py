import torch
from typing import List, Dict, Any, Optional
from PIL import Image
from src.diffusion.schedules import BaseSchedule
from src.diffusion.objectives import time_snr_shift


def encode_tokens_batch(
    input_ids: torch.Tensor,
    text_encoder: torch.nn.Module,
    tokenizer: Any,
    max_length: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Encodes a batch of token IDs using the text encoder, handling long prompts
    by chunking if necessary.
    """
    tokenizer_max_length = tokenizer.model_max_length

    # chunking
    if input_ids.shape[-1] > tokenizer_max_length:
        processed_ids = []
        for iids_single in input_ids:
            chunks = []
            # Chunk with overlap/special tokens
            for i in range(
                1, max_length - tokenizer_max_length + 2, tokenizer_max_length - 2
            ):
                ids_chunk = torch.cat(
                    [
                        iids_single[0].unsqueeze(0),  # BOS
                        iids_single[i : i + tokenizer_max_length - 2],
                        iids_single[-1].unsqueeze(0),  # EOS
                    ]
                )
                # padding/EOS for chunks
                if (
                    ids_chunk[-2] != tokenizer.eos_token_id
                    and ids_chunk[-2] != tokenizer.pad_token_id
                ):
                    ids_chunk[-1] = tokenizer.eos_token_id
                if ids_chunk[1] == tokenizer.pad_token_id:
                    ids_chunk[1] = tokenizer.eos_token_id
                chunks.append(ids_chunk)
            processed_ids.append(torch.stack(chunks))

        # [B, num_chunks, 77]
        processed_ids = torch.stack(processed_ids)
        batch_size, num_chunks, _ = processed_ids.shape
        input_ids_for_encoder = processed_ids.reshape(-1, tokenizer_max_length)
    else:
        input_ids_for_encoder = input_ids
        batch_size = input_ids.size(0)

    encoder_output = text_encoder(input_ids_for_encoder.to(device))
    # [B*chunks, 77, Dim]
    text_embeddings = encoder_output[-1][-2]
    text_embeddings = text_encoder.text_model.final_layer_norm(text_embeddings)

    # Reshape and concatenate
    if input_ids.shape[-1] > tokenizer_max_length:
        text_embeddings = text_embeddings.reshape(
            batch_size, -1, text_embeddings.shape[-1]
        )
        states_list = [text_embeddings[:, 0].unsqueeze(1)]  # BOS
        for i in range(1, max_length, tokenizer_max_length):
            states_list.append(text_embeddings[:, i : i + tokenizer_max_length - 2])
        states_list.append(text_embeddings[:, -1].unsqueeze(1))  # EOS
        text_embeddings = torch.cat(states_list, dim=1)
        text_embeddings = text_embeddings[:, :max_length, :]

    return text_embeddings


class CFGModelWrapper:
    """
    Wraps the UNet to handle Classifier-Free Guidance (CFG) and timestep formatting.
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        combined_embeddings: torch.Tensor,
        cfg_scale: float,
        device: torch.device,
        autocast_dtype: torch.dtype,
        is_ddpm: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        self.unet = unet
        self.combined_embeddings = combined_embeddings
        self.cfg_scale = cfg_scale
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.is_ddpm = is_ddpm
        self.attention_mask = attention_mask

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latents [B, C, H, W]
            t: Timesteps [B], in range [0, 1] (0=Data, 1=Noise)
        """
        # inputs for CFG: uncond, cond
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)

        if self.is_ddpm:
            # DDPM expects discrete indices [0, 999]
            # t=0 -> 0, t=1 -> 999
            t_model = (t_in * 999.0).long().clamp(0, 999)
        else:
            t_model = t_in * 1000.0

        with torch.autocast(
            device_type="cuda", dtype=self.autocast_dtype, enabled=True
        ):
            out = self.unet(
                x_in,
                t_model,
                encoder_hidden_states=self.combined_embeddings,
                attention_mask=self.attention_mask,
            )

        out_uncond, out_cond = out.chunk(2)
        return out_uncond + self.cfg_scale * (out_cond - out_uncond)


def linear_shift_schedule(steps, shift=1.0):
    sigmas = torch.linspace(0, 1, steps + 1)
    if shift > 1.0:
        sigmas = time_snr_shift(sigmas, shift)

    sigmas = torch.flip(sigmas, dims=[0])
    return sigmas


@torch.no_grad()
def sample_res_multistep(
    model_wrapper: CFGModelWrapper,
    x: torch.Tensor,
    sigmas: torch.Tensor,
):
    """
    A second-order multistep sampler (Adams-Bashforth / Heun variant).
    """
    # x is starting noise
    d_prev = None  # Previous derivative (velocity)

    for i in range(len(sigmas) - 1):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]

        t_batch = torch.full((x.shape[0],), sigma_cur, device=x.device)

        v_cur = model_wrapper(x, t_batch)
        d_cur = v_cur

        if d_prev is None or sigma_next == 0:
            # First step uses the Euler method.
            dt = sigma_next - sigma_cur
            x = x + d_cur * dt
        else:
            # Second-order Adams-Bashforth method.
            h = sigma_next - sigma_cur
            h_prev = sigmas[i] - sigmas[i - 1]
            x = x + (1.5 * d_cur - 0.5 * d_prev * (h / h_prev)) * h

        d_prev = d_cur
    return x


@torch.no_grad()
def sample_ddpm(
    model_wrapper: CFGModelWrapper,
    schedule: BaseSchedule,
    x: torch.Tensor,
    steps: int,
    eta: float = 0.0,
):
    """
    Sampling for DDPM/DDIM models (epsilon-prediction).
    Uses the unified view via schedule coefficients.
    """
    timesteps = torch.linspace(1, 0, steps + 1, device=x.device)

    # Skip the very last 0.0 in the loop
    seq = timesteps[:-1]

    for i, t_val in enumerate(seq):
        t_curr = t_val
        t_prev = timesteps[i + 1]

        # alpha (signal scale), sigma (noise scale)
        alpha_t, sigma_t, _, _ = schedule.get_coefficients(t_curr)

        # Boundary Condition for Final Step
        if i == steps - 1:
            alpha_prev = torch.ones_like(alpha_t)
            sigma_prev = torch.zeros_like(sigma_t)
        else:
            alpha_prev, sigma_prev, _, _ = schedule.get_coefficients(t_prev)

        # Broadcast
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1)
        alpha_prev = alpha_prev.view(-1, 1, 1, 1)
        sigma_prev = sigma_prev.view(-1, 1, 1, 1)

        t_batch = torch.full((x.shape[0],), t_curr, device=x.device)
        eps_pred = model_wrapper(x, t_batch)

        # 2. DDIM Step
        # Estimate x0 (Data)
        # x_t = alpha_t * x0 + sigma_t * eps
        # -> x0 = (x_t - sigma_t * eps) / alpha_t
        pred_x0 = (x - sigma_t * eps_pred) / alpha_t

        sigma_d = eta * 0.0
        if eta > 0:
            pass

        dir_xt = torch.sqrt(torch.clamp(sigma_prev**2 - sigma_d**2, min=0)) * eps_pred

        noise = torch.randn_like(x) if eta > 0 else 0.0

        x = alpha_prev * pred_x0 + dir_xt + sigma_d * noise

    return x


# TODO: implement dpm solver
def generate_samples(
    unet: torch.nn.Module,
    text_encoder: torch.nn.Module,
    tokenizer: Any,
    vae: torch.nn.Module,
    schedule: BaseSchedule,
    sample_configs: List[Dict[str, Any]],
    global_batch_size: int = 4,
    diffusion_type: str = "flow_matching",
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> List[Image.Image]:
    """
    Main entry point for generating samples during training.
    """
    if not sample_configs:
        return []

    unet.eval()
    text_encoder.eval()
    vae.eval()
    vae.to(device)
    torch.cuda.empty_cache()

    all_images = {}

    total_samples = len(sample_configs)

    for i in range(0, total_samples, global_batch_size):
        batch_configs = sample_configs[i : i + global_batch_size]
        batch_prompts = [cfg.get("prompt", "") for cfg in batch_configs]
        batch_neg = [cfg.get("negative_prompt", "") for cfg in batch_configs]

        H = batch_configs[0].get("height", 512)
        W = batch_configs[0].get("width", 512)
        steps = batch_configs[0].get("sample_steps", 25)
        cfg_scale = batch_configs[0].get("cfg_scale", 7.0)
        seed = batch_configs[0].get("seed", 42)
        shift = batch_configs[0].get("shift", 1.0)

        curr_bs = len(batch_configs)

        with torch.no_grad():
            # 1. Prepare Latents (Noise)
            gen = torch.Generator(device=device).manual_seed(seed)
            latents = torch.randn(
                (curr_bs, 4, H // 8, W // 8), device=device, generator=gen, dtype=dtype
            )

            # 2. Encode Text
            full_prompts = batch_neg + batch_prompts
            tokens = tokenizer(
                full_prompts, padding="longest", truncation=False, return_tensors="pt"
            ).input_ids

            seq_len = tokens.shape[-1]
            target_len = 77
            for tier_len in [77, 152, 227]:
                if seq_len <= tier_len:
                    target_len = tier_len
                    break
            else:
                target_len = 227
                tokens = tokens[:, :227]
                # Ensure the last token is EOS
                tokens[:, -1] = tokenizer.eos_token_id

            if tokens.shape[-1] < target_len:
                padding_length = target_len - tokens.shape[-1]
                padding_tensor = torch.full(
                    (tokens.shape[0], padding_length),
                    tokenizer.eos_token_id,
                    dtype=tokens.dtype,
                    device=tokens.device,
                )
                tokens = torch.cat([tokens, padding_tensor], dim=-1)

            # Generate attention mask
            not_pad_mask = tokens != tokenizer.eos_token_id
            shifted_mask = torch.roll(not_pad_mask, shifts=1, dims=1)
            shifted_mask[:, 0] = True

            attention_mask = not_pad_mask | shifted_mask
            attention_mask = attention_mask.to(device)

            embeddings = encode_tokens_batch(
                tokens,
                text_encoder,
                tokenizer,
                max_length=tokens.shape[-1],
                device=device,
            )

            model_wrapper = CFGModelWrapper(
                unet,
                embeddings,
                cfg_scale,
                device,
                autocast_dtype,
                is_ddpm=(diffusion_type == "ddpm"),
                attention_mask=attention_mask,
            )

            if diffusion_type == "ddpm":
                latents = sample_ddpm(model_wrapper, schedule, latents, steps)
            else:
                sigmas = linear_shift_schedule(steps, shift=shift).to(device)
                latents = sample_res_multistep(model_wrapper, latents, sigmas)

            # Decode one by one to save VRAM
            for j, latent in enumerate(latents):
                latent = latent.unsqueeze(0).to(torch.float32) / 0.18215
                image = vae.decode(latent)
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                image = (image * 255).round().astype("uint8")

                all_images[i + j] = Image.fromarray(image)

    vae.to("cpu")

    return [all_images[k] for k in range(total_samples)]
