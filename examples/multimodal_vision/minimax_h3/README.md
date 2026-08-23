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

This is the full 64-layer **Qwen3-VL 32B** encoder from
[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/text_encoder),
compressed as symmetric group-128 **W4A16 AWQ**. It was calibrated on MiniMax
H3 text, keyframe, and multimodal-reference prompt distributions.

The language linears are W4; the vision tower, DeepStack projections, token
embedding, and normalization weights remain BF16. The single-file checkpoint
is 18.99 GB, down from 66.7 GB for the BF16 source.

## ComfyUI quickstart

Use the `MiniMaxH3AWQEncoderLoader` from
[ComfyUI-h3-explorations](https://github.com/fblissjr/ComfyUI-h3-explorations),
not stock `CLIPLoader`. This file retains the full Hugging Face namespace and
`compressed-tensors` packing; the custom node adapts it in memory to ComfyUI's
native 50-layer MiniMax H3 architecture.

This is a storage-format requirement, not a claim that ComfyUI lacks AWQ
support: its separately converted INT8 ConvRot and NVFP4-AWQ H3 checkpoints
load natively and are correctly detected as Qwen3-VL-32B.

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/fblissjr/ComfyUI-h3-explorations.git

cd /path/to/ComfyUI/models/text_encoders
uvx --from huggingface-hub hf download \
  fbjr/qwen3-vl-32b-W4A16-AWQ-H3 \
  qwen3vl_32b_minimax_h3_w4a16_awq.safetensors \
  --local-dir .
```

In ComfyUI, add **Load MiniMax H3 Compressed-Tensors AWQ Encoder**
(`MiniMaxH3AWQEncoderLoader`) and select the downloaded file. The loader is not
hardcoded to its basename; it validates the selected file's embedded config and
complete adapted tensor inventory.

Ready-to-run UI workflows:

- [Text to video](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/workflows/h3_text_to_video.json)
- [Image reference plus text](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/workflows/h3_image_ref_plus_text_to_video.json)
- [All generated workflows and API variants](https://github.com/fblissjr/ComfyUI-h3-explorations/tree/main/workflows)

The adapter keeps ComfyUI's native tokenizer, 5120-wide H3 architecture, and
unnormalized layer-50 output. It adds the compressed-tensors namespace/packing
bridge, strict validation, source-config preprocessing, and comfy-kitchen
W4A16 execution. See the
[technical note](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/docs/h3_awq_encoder.md)
for the exact native/local boundary and CUDA routing behavior.

Text-only and two-image conditioning were validated on an RTX 4090 with about
14.97 GB of H3-relevant weights staged. This is not a fixed peak-VRAM or
performance claim; comparative BF16/INT8/NVFP4/W4 latency and fidelity remain
to be benchmarked.

## Serving

```bash
# vLLM
vllm serve fbjr/qwen3-vl-32b-W4A16-AWQ-H3 \
  --quantization compressed-tensors \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90

# SGLang
python3 -m sglang.launch_server \
  --model-path fbjr/qwen3-vl-32b-W4A16-AWQ-H3 \
  --quantization compressed-tensors \
  --port 30000
```
