import torch
import torch.nn.functional as F
import time
import sys
import os

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet import Unet, UnetConfig


def benchmark():
    device = "cuda"
    dtype = torch.bfloat16

    # Create dummy UNet
    config = UnetConfig(use_checkpointing=False)
    unet = Unet(config).to(device, dtype)
    unet.train()

    # Define buckets (H, W) and prompt lengths
    buckets = [(512, 512), (512, 768), (640, 896)]
    prompt_lengths = [77, 152, 227]
    batch_size = 8

    print("Benchmarking Attention Masking Approaches (Forward + Backward)")
    print("-" * 80)

    for H, W in buckets:
        if W == 896:
            batch_size = 4
        for seq_len in prompt_lengths:
            print(f"\nBucket: {H}x{W}, Prompt: {seq_len}, Batch: {batch_size}")

            # Dummy data
            x = torch.randn(batch_size, 4, H // 8, W // 8, device=device, dtype=dtype)
            timestep = torch.randint(
                0, 1000, (batch_size,), device=device, dtype=torch.float32
            )
            encoder_hidden_states = torch.randn(
                batch_size, seq_len, 768, device=device, dtype=dtype
            )
            target = torch.randn_like(x)

            # Dummy mask (simulate 50% to 100% valid tokens)
            valid_lens = torch.randint(
                seq_len // 2, seq_len + 1, (batch_size,), device=device
            )
            attention_mask = torch.arange(seq_len, device=device).expand(
                batch_size, seq_len
            ) < valid_lens.unsqueeze(1)

            def run_test(name, use_fa, mask):
                for module in unet.modules():
                    if module.__class__.__name__ == "Attention":
                        module.use_flash_attention = use_fa
                        module.forward = (
                            module.forward_flash_attention
                            if use_fa
                            else module.forward_sdpa
                        )

                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    with torch.autocast(device_type="cuda", dtype=dtype):
                        for _ in range(3):
                            unet.zero_grad(set_to_none=True)
                            out = unet(
                                x, timestep, encoder_hidden_states, attention_mask=mask
                            )
                            loss = F.mse_loss(out, target)
                            loss.backward()

                    torch.cuda.synchronize()

                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()
                    for _ in range(10):
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            unet.zero_grad(set_to_none=True)
                            out = unet(
                                x, timestep, encoder_hidden_states, attention_mask=mask
                            )
                            loss = F.mse_loss(out, target)

                        loss.backward()

                    end_event.record()
                    torch.cuda.synchronize()

                    elapsed = start_event.elapsed_time(end_event) / 10.0
                    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

                    print(
                        f"{name:.<30} {elapsed:>6.2f} ms | "
                        f"Peak VRAM: {peak_vram:>7.2f} MB"
                    )
                except Exception as e:
                    print(f"{name:.<30} Failed: {e}")

            run_test("Baseline (No Mask, SDPA)", use_fa=False, mask=None)
            run_test("Baseline (No Mask, FA2)", use_fa=True, mask=None)
            run_test("PyTorch SDPA (With Mask)", use_fa=False, mask=attention_mask)
            run_test("Flash Attention 2 (Varlen)", use_fa=True, mask=attention_mask)


if __name__ == "__main__":
    benchmark()

