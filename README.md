# ComfyUI GGUF Prompt Rewriter

Standalone ComfyUI custom nodes for rewriting plain-English prompts with a local GGUF LLM through `llama-cpp-python`.

This package was designed for prompt rewriting and booru-style tag generation without relying on remote APIs or merging into other prompt/tagger extensions.

<img width="292" height="618" alt="Screenshot 2026-05-01 at 10 00 45 AM" src="https://github.com/user-attachments/assets/0393bfb2-e3ef-44dc-af4f-0b46d54d91b2" />


## Nodes

### `GGUF Prompt Rewriter`

Loads a local `.gguf` model and rewrites a user prompt with a configurable system prompt and decoding settings.

Outputs:

- `rewritten_prompt`
- `raw_output`

Notes:

- Tested with uncensored Gemma 4 and Qwen3.6 models
- By default the node loads the model for each run and closes it immediately afterward, so it does not keep a cached model in memory. Enable `keep_model_in_memory` to keep the model resident between runs (faster, but uses VRAM).

## Model Locations

The node scans ComfyUI's `models/llm_gguf` folder for `.gguf` files. This follows the `--models-directory` / `--base-directory` launch settings, and any additional folders you configure for `llm_gguf` in `extra_model_paths.yaml`.

Put your local GGUF files in `ComfyUI/models/llm_gguf` and restart ComfyUI.

## GPU Support

`llama-cpp-python`'s prebuilt wheels are **CPU-only**. To actually use your GPU (Metal on macOS, CUDA on Windows/Linux) you must install a GPU-enabled build, then set `n_gpu_layers` to a value greater than 0 (e.g. `-1` for all layers).

Reinstall with the appropriate backend (use the ComfyUI venv's `pip`):

- **macOS (Metal / Apple Silicon):**
  ```bash
  CMAKE_ARGS="-DGGML_METAL=ON" /path/to/ComfyUI/.venv/bin/pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
  ```
- **Windows (CUDA):** requires the Visual Studio C++ build tools and the CUDA toolkit.
  ```bat
  set CMAKE_ARGS=-DGGML_CUDA=ON
  path\to\ComfyUI\.venv\Scripts\pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
  ```
- **Linux (CUDA):**
  ```bash
  CMAKE_ARGS="-DGGML_CUDA=ON" pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
  ```

Verify it worked by checking ComfyUI's startup log: it should report a `metal`/`CUDA` backend when the model loads, and inference will be dramatically faster.

## Installation

Clone into your ComfyUI custom nodes directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/hlibr/ComfyUI-GGUF-Prompt-Rewriter.git
```

Install dependencies into the same venv ComfyUI uses:

```bash
/path/to/ComfyUI/.venv/bin/python -m pip install -r /path/to/ComfyUI/custom_nodes/ComfyUI-GGUF-Prompt-Rewriter/requirements.txt
```

Restart ComfyUI.

## Dependencies

- `llama-cpp-python`

ComfyUI itself provides the node runtime and `folder_paths`.

## License

MIT
