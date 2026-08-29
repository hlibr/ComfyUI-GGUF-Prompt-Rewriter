# ComfyUI GGUF Prompt Rewriter

A ComfyUI node that runs a local GGUF LLM (via `llama-cpp-python`) and returns its text output. It exposes the full chat-completion interface — system prompt, user prompt, and sampling parameters — so it can be used for prompt rewriting, booru-style tag generation, captioning, translation, or any other local text-generation task. By default it ships with a system prompt tuned for rewriting prompts into danbooru-style tags.

<img width="292" height="618" alt="Screenshot 2026-05-01 at 10 00 45 AM" src="https://github.com/user-attachments/assets/0393bfb2-e3ef-44dc-af4f-0b46d54d91b2" />

## Installation

Clone into your ComfyUI `custom_nodes` directory and install the dependency into ComfyUI's venv:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/hlibr/ComfyUI-GGUF-Prompt-Rewriter.git
/path/to/ComfyUI/.venv/bin/python -m pip install -r /path/to/ComfyUI/custom_nodes/ComfyUI-GGUF-Prompt-Rewriter/requirements.txt
```

Restart ComfyUI.

## Model location

The node scans `models/llm_gguf` for `.gguf` files (respecting `--models-directory` / `--base-directory` and any `extra_model_paths.yaml` entries). Drop your GGUF there and restart ComfyUI; it appears in the node's **model** dropdown.

## GPU support (optional)

`llama-cpp-python`'s prebuilt wheels are CPU-only. To use your GPU (Metal on macOS, CUDA on Windows/Linux), install a GPU-enabled build; the default `n_gpu_layers = -1` then offloads all layers to the GPU. The node runs on CPU without this — just slower.

```bash
# macOS (Metal)
CMAKE_ARGS="-DGGML_METAL=ON" /path/to/ComfyUI/.venv/bin/python -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
# Windows (CUDA) — in cmd.exe:
set CMAKE_ARGS=-DGGML_CUDA=ON
path\to\ComfyUI\.venv\Scripts\python -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
# Linux (CUDA)
CMAKE_ARGS="-DGGML_CUDA=ON" /path/to/ComfyUI/.venv/bin/python -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

Check ComfyUI's startup log for a `metal`/`CUDA` backend line to confirm.

## Usage

The node (category `prompt/LLM`) loads the selected `.gguf` and runs a chat completion with the given `system_prompt` and `user_prompt`.

**Outputs:** `rewritten_prompt` (cleaned result) and `raw_output` (model reply before cleanup).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | choice | — | GGUF file from `models/llm_gguf`. |
| `user_prompt` | text | empty | Prompt sent to the model. |
| `system_prompt` | text | booru tag template | Instructions/role for the model (editable). |
| `enable_thinking` | bool | `False` | Thinking mode for supporting models (e.g. Qwen). |
| `seed` | int | `0` | `-1` random; `0`/positive fixed. |
| `max_tokens` | int | `160` | Max tokens to generate. |
| `temperature` | float | `0.5` | Sampling temperature. |
| `top_p` | float | `0.9` | Nucleus cutoff. |
| `top_k` | int | `40` | Top-k cutoff. |
| `min_p` | float | `0.05` | Min probability threshold. |
| `presence_penalty` | float | `0.0` | Penalizes repeated topics. |
| `repeat_penalty` | float | `1.1` | Penalizes repeated tokens. |
| `n_ctx` | int | `4096` | Context window (tokens). |
| `n_batch` | int | `512` | Prompt-processing batch size. |
| `n_gpu_layers` | int | `-1` | Layers offloaded to GPU (`-1` all, `0` CPU). |
| `n_threads` | int | `0` | CPU threads (`0` auto). |
| `ignore_cache` | bool | `False` | Skip the result cache. |
| `keep_model_in_memory` | bool | `False` | Keep model loaded between runs (faster, uses VRAM). |

Tested with uncensored Gemma 4 and Qwen3.6 models.

## Dependencies

ComfyUI provides the node runtime and `folder_paths`.

## License

MIT
