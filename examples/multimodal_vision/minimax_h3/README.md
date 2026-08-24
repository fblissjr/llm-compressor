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
symmetric group-128 W4A16 AWQ and calibrated on 96 multimodal prompts and
reference image pairs from H3-IR, local generation frame-0 extracts, and H3
avatar dialogue datasets. Language linears are W4; the vision tower, DeepStack
projections, token embedding, and normalization weights remain BF16. The
single-file checkpoint is 18.99 GB, down from 66.7 GB BF16.

## ComfyUI quickstart

The included loader is one self-contained `.py` file; there is no config folder
or custom-node repository to clone.

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
([`MiniMaxH3AWQEncoderLoader`](./comfyui_minimax_h3_awq_loader.py)). Selection
is validated from checkpoint metadata and tensor inventory, not a hardcoded
filename.

Do not install the standalone [`comfyui_minimax_h3_awq_loader.py`](./comfyui_minimax_h3_awq_loader.py) beside the full
[ComfyUI-h3-explorations](https://github.com/fblissjr/ComfyUI-h3-explorations)
repo. Both register the same loader node ID.

## Example workflows

```bash
uvx --from huggingface-hub hf download \
  fbjr/qwen3-vl-32b-W4A16-AWQ-H3 \
  comfyui_minimax_h3_awq_text_to_video.json \
  comfyui_minimax_h3_awq_image_reference.json \
  comfyui_minimax_h3_awq_first_frame.json \
  comfyui_minimax_h3_awq_first_last_frame.json \
  comfyui_minimax_h3_encoder_ab_compare.json \
  --local-dir "$COMFYUI_DIR/user/default/workflows"
```

- [Text to video](./comfyui_minimax_h3_awq_text_to_video.json)
- [Image references plus text](./comfyui_minimax_h3_awq_image_reference.json)
- [First frame to video](./comfyui_minimax_h3_awq_first_frame.json)
- [First and last frames](./comfyui_minimax_h3_awq_first_last_frame.json)
- [Two-clip side-by-side viewer](./comfyui_minimax_h3_encoder_ab_compare.json)

The four generation examples are derived from ComfyUI's official H3 templates;
only the text-encoder loader is custom. Select your own images. The comparison
viewer requires VideoHelperSuite and ComfyUI-KJNodes and intentionally omits
audio.

The text and keyframe examples use the current owner recipe: v1.1 768p FL2VA
LoRA, strength 0.75, six render steps, shift 6/3. This is a working recipe, not
a vendor-attested v1.1 schedule:

```bash
uvx --from huggingface-hub hf download \
  lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors \
  --local-dir "$COMFYUI_DIR/models/loras/h3/lightx2v_Minimax-h3-Turbo"
```

The image-reference example uses the separate native ref2va base recipe. Model
links for the diffusion weights and VAEs are embedded in each workflow.

## What the loader adds

Stock `CLIPLoader` correctly loads Comfy-formatted INT8 ConvRot and NVFP4-AWQ
H3 encoders as the native 5120-wide, 50-layer model. It does not load this
checkpoint's full Hugging Face namespace and compressed-tensors records.

The custom node validates and adapts the full-HF compressed-tensors checkpoint
in memory, retains H3's first 50 language layers, applies the artifact's
processor configs, and executes W4A16 through comfy-kitchen. ComfyUI still owns
the H3 architecture, tokenizer, unnormalized layer-50 output, modality tags,
conditioning, offload, and model patching.

See the
[technical note](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/docs/h3_awq_encoder.md)
for the exact native/local boundary and current CUDA routing.

## Initial RTX 4090 timing

One cold-unload run per encoder, 362 frames / 15.08 seconds, identical graph,
prompt, seed, and generation settings within each row:

| workflow | W4A16 | INT8 ConvRot | NVFP4 |
|---|---:|---:|---:|
| first frame, 640×640, 6 steps | **171.9s** | 182.5s | 173.5s |
| two-image reference, 864×480, 20 steps | **537.4s** | 551.3s | 544.9s |

This is provisional `n=1` end-to-end timing, not a fidelity ranking. The W4
loader also uses its artifact-owned image processor, so these runs do not
isolate quantization error from preprocessing. Use the included comparison
viewer for blind clip review; the complete methodology and raw rows live in
[ComfyUI-h3-explorations](https://github.com/fblissjr/ComfyUI-h3-explorations).

## Serving

```bash
uvx --from vllm vllm serve fbjr/qwen3-vl-32b-W4A16-AWQ-H3 \
  --quantization compressed-tensors \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```
