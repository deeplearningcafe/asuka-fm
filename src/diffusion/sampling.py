import math
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


def compute_inference_pos_map(
    height: int,
    width: int,
    patch_size: int = 2,
    vae_scale: int = 8,
    zoom: float = 1.0,
    x_shift: float = 0.0,
    y_shift: float = 0.0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Computes patch-unit 2D RoPE position map for arbitrary aspect ratio

    and viewport camera controls.
    """
    num_h = (height // vae_scale) // patch_size
    num_w = (width // vae_scale) // patch_size

    y_pos = torch.arange(num_h, dtype=torch.float32, device=device)
    x_pos = torch.arange(num_w, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing="ij")

    # Center-anchored zoom and normalized shift scaled to patch count
    center_y = (num_h - 1.0) / 2.0
    center_x = (num_w - 1.0) / 2.0
    grid_y = (grid_y - center_y) / zoom + center_y + y_shift * num_h
    grid_x = (grid_x - center_x) / zoom + center_x + x_shift * num_w

    pos_map = torch.stack((grid_y, grid_x), dim=-1).flatten(0, 1)
    return pos_map.to(dtype=dtype)


class CFGModelWrapper:
    """
    Wraps the model for CFG. When cfg_scale <= 1.0 or unconditional
    embeddings are uninitialized, it executes a single conditional pass.
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
        use_unet_mult: bool = True,
        pos_map: Optional[torch.Tensor] = None,
        is_conditional: bool = True,
    ):
        self.unet = unet
        self.combined_embeddings = combined_embeddings
        self.cfg_scale = cfg_scale
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.is_ddpm = is_ddpm
        self.attention_mask = attention_mask
        self.use_unet_mult = use_unet_mult
        self.pos_map = pos_map
        self.is_conditional = is_conditional

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        if self.is_conditional and self.cfg_scale > 1.0:
            x_in = torch.cat([x] * 2, dim=0)
            t_in = torch.cat([t] * 2, dim=0)
            t_model = t_in * 1000.0 if self.use_unet_mult else t_in

            model_kwargs = {}
            if self.pos_map is not None:
                model_kwargs["pos_map"] = torch.cat([self.pos_map] * 2, dim=0)

            with torch.autocast(
                device_type="cuda", dtype=self.autocast_dtype, enabled=True
            ):
                out = self.unet(
                    x_in,
                    t_model,
                    encoder_hidden_states=self.combined_embeddings,
                    attention_mask=self.attention_mask,
                    **model_kwargs,
                )
            out_uncond, out_cond = out.chunk(2, dim=0)
            return out_uncond + self.cfg_scale * (out_cond - out_uncond)
        else:
            t_model = t * 1000.0 if self.use_unet_mult else t
            model_kwargs = {}
            if self.pos_map is not None:
                model_kwargs["pos_map"] = self.pos_map

            # If combined embeddings contain both [uncond, cond], pick cond
            emb = self.combined_embeddings
            mask = self.attention_mask
            if emb is not None and emb.shape[0] == batch_size * 2:
                emb = emb[batch_size:]
                if mask is not None:
                    mask = mask[batch_size:]

            with torch.autocast(
                device_type="cuda", dtype=self.autocast_dtype, enabled=True
            ):
                return self.unet(
                    x,
                    t_model,
                    encoder_hidden_states=emb,
                    attention_mask=mask,
                    **model_kwargs,
                )


def linear_shift_schedule(steps, shift=1.0):
    sigmas = torch.linspace(0, 1, steps + 1)
    if shift > 1.0:
        sigmas = time_snr_shift(sigmas, shift)

    sigmas = torch.flip(sigmas, dims=[0])
    return sigmas


@torch.no_grad()
def sample_euler(
    model_wrapper: CFGModelWrapper,
    x: torch.Tensor,
    num_steps: int = 25,
    shift: float = 1.0,
) -> torch.Tensor:
    """
    Standard Forward Euler ODE Solver integrating t=0 (Noise) -> t=1 (Data).
    """
    device = x.device
    z = x.clone()

    t_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    if shift != 1.0:
        t_grid = t_grid / (shift - (shift - 1.0) * t_grid)

    dt_steps = t_grid[1:] - t_grid[:-1]

    for i in range(num_steps):
        t_curr = t_grid[i]
        dt = dt_steps[i]

        t_input = torch.full((z.shape[0],), t_curr, device=device)
        v_pred = model_wrapper(z, t_input)
        z = z + v_pred * dt

    return z


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
    use_unet_mult: bool = True,
    vae_mean: float = 0.0,
    vae_std: float = 0.18215,
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

        # Viewport camera manipulation parameters
        zoom = batch_configs[0].get("zoom", 1.0)
        x_shift = batch_configs[0].get("x_shift", 0.0)
        y_shift = batch_configs[0].get("y_shift", 0.0)

        curr_bs = len(batch_configs)

        with torch.no_grad():
            # 1. Prepare Latents (Noise)
            gen = torch.Generator(device=device).manual_seed(seed)
            latents = torch.randn(
                (curr_bs, 4, H // 8, W // 8), device=device, generator=gen, dtype=dtype
            )

            # 2. Polymorphic Text Tokenization & Encoding
            full_prompts = batch_neg + batch_prompts

            lengths = [tokenizer.get_length(p) for p in full_prompts]
            max_p_len = max(lengths) if lengths else 77
            target_len = 77
            for tier_len in [77, 152, 227, 256]:
                if max_p_len <= tier_len:
                    target_len = tier_len
                    break
            else:
                target_len = max(256, max_p_len)

            tokens_list, mask_list = [], []
            for p in full_prompts:
                tok, m = tokenizer.encode(
                    p,
                    max_len=target_len,
                    cfg_dropout_prob=0.0,
                    tag_dropout_prob=0.0,
                    shuffle_tags=False,
                )
                tokens_list.append(tok)
                mask_list.append(m)
            tokens = torch.stack(tokens_list).to(device)
            attention_mask = torch.stack(mask_list).to(device)
            embeddings, attention_mask = text_encoder(tokens, mask=attention_mask)

            # 3. Build Continuous 2D RoPE Position Map for DiT
            pos_map = None
            if "pos_map" in batch_configs[0] and (
                batch_configs[0]["pos_map"] is not None
            ):
                pos_maps = [c["pos_map"] for c in batch_configs]
                pos_map = torch.cat(pos_maps, dim=0).to(device=device)
                print("Using Pos map with camera control")
            else:
                patch_size = getattr(unet, "patch_size", None)
                if patch_size is None and hasattr(unet, "module"):
                    patch_size = getattr(unet.module, "patch_size", None)

                has_camera = zoom != 1.0 or x_shift != 0.0 or y_shift != 0.0
                if patch_size is not None or has_camera:
                    print("Using Pos map with camera control")
                    p_size = patch_size if patch_size is not None else 2
                    single_pos = compute_inference_pos_map(
                        height=H,
                        width=W,
                        patch_size=p_size,
                        vae_scale=8,
                        zoom=zoom,
                        x_shift=x_shift,
                        y_shift=y_shift,
                        device=device,
                        dtype=dtype,
                    )
                    pos_map = single_pos.unsqueeze(0).expand(curr_bs, -1, -1)

            model_wrapper = CFGModelWrapper(
                unet=unet,
                combined_embeddings=embeddings,
                cfg_scale=cfg_scale,
                device=device,
                autocast_dtype=autocast_dtype,
                is_ddpm=(diffusion_type == "ddpm"),
                attention_mask=attention_mask,
                use_unet_mult=use_unet_mult,
                pos_map=pos_map,
            )

            if diffusion_type == "ddpm":
                latents = sample_ddpm(model_wrapper, schedule, latents, steps)
            else:
                sigmas = linear_shift_schedule(steps, shift=shift).to(device)
                # latents = sample_res_multistep(model_wrapper, latents, sigmas)
                latents = sample_euler(
                    model_wrapper, latents, num_steps=steps, shift=shift
                )

            # Decode one by one to save VRAM
            for j, latent in enumerate(latents):
                latent = (latent.unsqueeze(0).to(torch.float32) - vae_mean) / vae_std
                torch._dynamo.maybe_mark_dynamic(latent, 1)
                torch._dynamo.maybe_mark_dynamic(latent, 2)
                image = vae.decode(latent).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                image = (image * 255).round().astype("uint8")

                all_images[i + j] = Image.fromarray(image)

    # vae.to("cpu")

    return [all_images[k] for k in range(total_samples)]
