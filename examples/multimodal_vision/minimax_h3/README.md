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

This is a **W4A16 AWQ (Activation-Aware Quantized)** release of **Qwen3-VL 32B** optimized specifically for single-GPU inference (**RTX 4090 / 24GB VRAM**) and serving as the multimodal text/reference conditioning encoder for the **MiniMax H3 Video DiT pipeline**.

* **Base Model Source:** [MiniMaxAI/MiniMax-H3 (text_encoder)](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/text_encoder)

---

## Quickstart: How to Use in ComfyUI

This repository provides a single-file checkpoint, [`qwen3vl_32b_minimax_h3_w4a16_awq.safetensors`](qwen3vl_32b_minimax_h3_w4a16_awq.safetensors) (18.99 GB). Because it uses `compressed-tensors` Marlin W4A16 packing, load it using the dedicated [**ComfyUI-h3-explorations**](https://github.com/fblissjr/ComfyUI-h3-explorations) adapter ([`h3_awq_encoder.py`](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/h3_awq_encoder.py)) instead of stock `CLIPLoader`.

### 1. Install Custom Nodes
```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/fblissjr/ComfyUI-h3-explorations.git
```

### 2. Download the Encoder
```bash
cd /path/to/ComfyUI/models/text_encoders
uvx --from huggingface-hub hf download \
  fbjr/qwen3-vl-32b-W4A16-AWQ-H3 \
  qwen3vl_32b_minimax_h3_w4a16_awq.safetensors \
  --local-dir .
```

You may use a symlink instead if the checkpoint already lives elsewhere.

### 3. Add Loader to Workflow
* In ComfyUI, add the **`MiniMaxH3AWQEncoderLoader`** node (*"Load MiniMax H3 Compressed-Tensors AWQ Encoder"* under category `MiniMaxH3/loaders`).
* Select `qwen3vl_32b_minimax_h3_w4a16_awq.safetensors` and connect the `CLIP` output to `MiniMaxH3Conditioning` or `MiniMaxH3ReferenceConditioning`.
* The node is not hardcoded to this basename. It validates the selected file's embedded quantization/config metadata and full adapted H3 tensor inventory; a renamed compatible copy also loads, while a merely similar filename does not bypass validation.
* Start with [`h3_text_to_video.json`](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/workflows/h3_text_to_video.json), the simplest ready-to-run UI workflow using this loader.
* To exercise the checkpoint's BF16 vision tower and source-config image processor, use [`h3_image_ref_plus_text_to_video.json`](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/workflows/h3_image_ref_plus_text_to_video.json).
* API variants and the rest of the generated workflow family are in the [ComfyUI-h3-explorations `workflows/`](https://github.com/fblissjr/ComfyUI-h3-explorations/tree/main/workflows) directory; the current generated workflows use `MiniMaxH3AWQEncoderLoader` throughout.

---

## 1. Quantization Architecture & Component Breakdown

To maintain maximum visual fidelity on reference images while slashing the VRAM footprint for single-GPU serving, the model utilizes a **selective mixed-precision architecture**:

| Component | Target Layers | Precision | Technical Rationale & Pipeline Impact |
| :--- | :--- | :--- | :--- |
| **Vision Tower (ViT)** | `model.visual.blocks.*` (27 layers) | **100% BF16** | **0% feature loss:** Preserves fine spatial textures, facial geometry, and high-frequency reference details. |
| **DeepStack Mergers** | `model.visual.deepstack_merger_list.*` | **100% BF16** | Maintains continuous cross-scale feature adaptation between visual patch hierarchies and language backbone. |
| **Patch Projections** | `model.visual.patch_embed.*`, `merger.*` | **100% BF16** | Ensures linear patch embedding projections enter Layer 0 without numerical noise. |
| **Token Embeddings** | `model.language_model.embed_tokens` | **100% BF16** | Retains exact vector representations for critical H3 prompt delimiters (`<d>`, `</d>`, `<Subject N>`). |
| **Norms & LM Head** | `.*norm$`, `lm_head.weight` | **100% BF16** | Preserves numerical stability and prevents logit/representation drift during forward passes. |
| **Language Decoder** | `model.language_model.layers.*` (64 layers) | **W4A16 AWQ** | Compresses 30.6B language params with activation-informed channel scaling (Marlin-accelerated). |
| **Activations & KV** | Attention & MLP activations | **16-bit (A16)** | Full 16-bit dynamic precision to ensure zero drift in the Layer 50 hidden state tap. |

---

## 2. Calibration Strategy

Calibration was performed exclusively on **native MiniMax H3 multimodal prompt schemas** to ensure zero activation scale drift on H3 control syntax:

* **Data Mix & Formats:**
  * **Omni-Reference (`Ref2VA`):** Multi-reference conditioning with `<Subject N>`, `retention_analysis:`, and `summary: [reference generation]`.
  * **First/Last Keyframes (`I2VA` / `FL2VA`):** Temporal anchor conditioning (`At 0.00 seconds into the target video, <Picture 1>...`).
  * **Text-to-Video/Audio (`T2VA`):** Standard 3-field schemas (`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`).
  * **Portrait & Face Geometry:** High-resolution unblurred facial reference portraits with dialogue tags (`<d>...</d>`).
* **Tap Point Protection:** Optimized to preserve the fidelity of **unnormalized hidden states after Layer 50**, which directly condition the downstream 33B DiT video generator.

---

## 3. Hardware & Memory Footprint

| Metric | BF16 Base Model | W4A16 AWQ (This Model) | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Weight Footprint** | **66.7 GB** | **18.99 GB** | **3.5x Compression** |
| **Minimum Hardware** | 2× 80GB GPUs | **1× 24GB GPU (e.g. RTX 4090 / 3090)** | **Runs on Consumer Hardware** |
| **KV Cache Headroom (24GB)** | 0 GB (OOM) | **~5.5 GB** | Supports long multimodal context |

---

## 4. Multi-Shard Serving (vLLM & SGLang)

### Serving with vLLM
```bash
vllm serve fbjr/qwen3-vl-32b-W4A16-AWQ-H3 \
  --quantization compressed-tensors \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

### Serving with SGLang
```bash
python3 -m sglang.launch_server \
  --model-path fbjr/qwen3-vl-32b-W4A16-AWQ-H3 \
  --quantization compressed-tensors \
  --port 30000
```
