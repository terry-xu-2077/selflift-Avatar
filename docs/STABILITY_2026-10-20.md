# H3 consecutive-run stability validation — 2026-10-20

This maintained fork fixes the external H3 latent upscaler's memory handover between low- and high-resolution sampling. It also retains a local fix for mixed keyframe/reference conditions during spatial tiling.

## Consecutive GPU runs

Hardware: NVIDIA RTX 3090, 24GB VRAM; Windows; approximately 64GB system RAM. ComfyUI v0.40.0, PyTorch 2.11.0+cu130, DynamicVRAM enabled for H3 sampling.

One ComfyUI process ran all three workflows without a restart between runs. Only the seed changed among generation parameters; output filenames were numbered separately. Run 3 was submitted only after the difference between runs 1 and 2 was confirmed to be within 60 seconds.

| Run | Seed | Total seconds | Total, rounded | Low-resolution sampling | Learned latent lift | High-resolution sampling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 666 | 1904.985 | 31m45s | 432.412s | 13.583s | 1325.570s |
| 2 | 677 | 1897.955 | 31m38s | 427.300s | 13.262s | 1339.067s |
| 3 | 678 | 1883.850 | 31m24s | 426.488s | 13.278s | 1325.896s |

Total time is measured from ComfyUI history's `execution_start` to `execution_success`, including conditioning when required, sampling, decoding, encoding and file saving. It excludes queue waiting. Stage columns do not include all of these operations and therefore do not sum to the total.

- Run 1 vs 2: **7.030s** difference.
- Run 2 vs 3: **14.105s** difference.
- All runs completed successfully with 1920×1088 output, 362 frames, 24fps (15.0833 seconds).
- The previous 418.513s latent-lift stall did not recur. Runs 2 and 3 differed by only 0.016s in the learned lift.

## Workflow configuration

- Single H3 model reused in both sampling stages: `minimax_h3_fl2va_bf16.safetensors` with `dmad_minimax_h3_4step_full_critic_comfyui.safetensors` at strength 1.
- Comfy Kitchen attention; KJNodes Low VRAM Attention `head_chunks=4`; Chunk FeedForward `chunks=2`, `seq_threshold=4096`.
- Standard Euler, CFG 1; a simple 8-step schedule with one extra H3 sigma-refinement step, giving 6 low-resolution and 3 high-resolution model evaluations.
- Spatial scale 0.5: 960×544 low-resolution prefix, followed by 1920×1088 refinement.
- External `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`; `rho=0` (no pixel-VAE anchor), no input masks.
- Manual high-resolution spatial tiling: 4 width-axis tiles, automatic overlap.
- Learned lifter temporal chunk 32, overlap 5, 4 windows.
- H.264 MP4 with audio, CRF 12, 24fps.

The original prompts, reference images, videos and machine-specific logs are not distributed with this report. This is a recorded stability observation, not a standalone reproduction bundle.

## Memory handover

The learned lifter uses ordinary torch convolution layers. It is now fully loaded through the native non-dynamic `ModelPatcher`, allowing the model manager to reserve workspace before inference. H3's existing dynamic memory policy is retained. The lifter is unloaded after use, including after a failed call, through the normal ComfyUI cleanup helpers.

Before loading the lifter, runs 2 and 3 had only 293.25MiB and 288.56MiB of free CUDA memory. After reserving workspace and loading its approximately 659MiB weights, free CUDA memory was 3814.31MiB and 3904.56MiB. The log now separates loading and inference times and records local memory snapshots.

The changes preserve the node interfaces, upscaling math, dtype and tiling algorithm. No additional workflow node is needed.

## Tests and scope

The 58-test sampler/tiling suite and 2-test upscaler suite passed. The added cases cover mixed keyframe/reference condition rows, exact repeated-lift equality on CPU, and model cleanup after failure.

Run the suites separately with ComfyUI's Python environment from the plugin directory:

```powershell
python tests/test_avatar.py
python tests/test_upscaler_memory.py
```

The GPU measurements cover one machine, one single-model workflow and three seeds. They do not establish universal timing bounds, lip-sync accuracy, equivalence to full-frame attention, or image quality versus native full-resolution sampling. No additional GPU benchmark was run for the repository fork setup.
