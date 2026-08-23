---
license: apache-2.0
base_model: MiniMaxAI/MiniMax-H3
tags:
  - quantized
  - awq
  - w4a16
  - vllm
  - sglang
  - compressed-tensors
  - multimodal
  - vision-language
  - minimax-h3
  - text-to-video
pipeline_tag: image-text-to-text
---

# Qwen3-VL-32B-W4A16-AWQ-H3

The full 64-layer Qwen3-VL 32B encoder from MiniMax H3, compressed as
symmetric group-128 W4A16 AWQ and calibrated on H3 text, keyframe, and
multimodal-reference prompts. Language linears are W4; the vision tower,
DeepStack projections, token embedding, and normalization weights remain BF16.
The single-file checkpoint is 18.99 GB, down from 66.7 GB BF16.

## ComfyUI quickstart

This repository includes a self-contained ComfyUI node. It embeds the four
small runtime configs, so there is no config folder to install and no custom
node repository to clone.

```bash
COMFYUI_DIR=/path/to/ComfyUI

uvx --from huggingface-hub hf download \
  fbjr/qwen3-vl-32b-W4A16-AWQ-H3 \
  comfyui_minimax_h3_awq_loader.py \
  --local-dir "$COMFYUI_DIR/custom_nodes"

uvx --from huggingface-hub hf download \
  fbjr/qwen3-vl-32b-W4A16-AWQ-H3 \
  qwen3vl_32b_minimax_h3_w4a16_awq.safetensors \
  --local-dir "$COMFYUI_DIR/models/text_encoders"
```

Restart ComfyUI, then add **Load MiniMax H3 Compressed-Tensors AWQ Encoder**
(`MiniMaxH3AWQEncoderLoader`). The node lists real `.safetensors` files and
accepts by metadata, packing, config, and complete tensor inventory—not by a
hardcoded filename.

Do not install the standalone `.py` beside the full
[ComfyUI-h3-explorations](https://github.com/fblissjr/ComfyUI-h3-explorations)
repo. Both register the same loader node ID.

## Example workflows

```bash
uvx --from huggingface-hub hf download \
  fbjr/qwen3-vl-32b-W4A16-AWQ-H3 \
  comfyui_minimax_h3_awq_text_to_video.json \
  comfyui_minimax_h3_awq_image_reference.json \
  comfyui_minimax_h3_awq_first_frame.json \
  --local-dir "$COMFYUI_DIR/user/default/workflows"
```

- [Text to video](./comfyui_minimax_h3_awq_text_to_video.json)
- [Image references plus text](./comfyui_minimax_h3_awq_image_reference.json)
- [First frame to video](./comfyui_minimax_h3_awq_first_frame.json)

They are derived from ComfyUI's official H3 templates. Other than this
encoder loader, their H3 conditioning, sampling, AV decode, and save nodes are
native ComfyUI. Select your own images in the reference and first-frame files.

The text and first-frame examples use the current provisional owner recipe:
the v1.1 768p FL2VA LoRA at strength 0.75, six render steps, and shift 6/3.
That is a working recipe, not a vendor-attested v1.1 schedule. Install its
LoRA at the path stored by those workflows:

```bash
uvx --from huggingface-hub hf download \
  lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors \
  --local-dir "$COMFYUI_DIR/models/loras/h3/lightx2v_Minimax-h3-Turbo"
```

The image-reference example uses the separate native ref2va base recipe. The
workflows also need the diffusion model and video/audio VAEs named in their
nodes; their embedded template metadata points to Comfy-Org/MiniMax-H3.

## What the loader adds

Stock `CLIPLoader` correctly loads Comfy-formatted INT8 ConvRot and NVFP4-AWQ
H3 encoders as the native 5120-wide, 50-layer model. It does not load this
checkpoint's full Hugging Face namespace and compressed-tensors records.

The custom node strictly validates and adapts that representation in memory,
keeps only the 50 H3 language layers, uses the artifact's image/video processor
configs, and executes W4A16 through comfy-kitchen. ComfyUI still owns the H3
architecture, tokenizer, unnormalized layer-50 output, modality tags,
conditioning, offload, and model patching.

See the
[technical note](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/docs/h3_awq_encoder.md)
for the exact native/local boundary and current CUDA routing. Text-only and
two-image conditioning were validated on an RTX 4090; comparative
BF16/INT8/NVFP4/W4 speed and fidelity remain to be measured.

## Serving

```bash
uvx --from vllm vllm serve fbjr/qwen3-vl-32b-W4A16-AWQ-H3 \
  --quantization compressed-tensors \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```
