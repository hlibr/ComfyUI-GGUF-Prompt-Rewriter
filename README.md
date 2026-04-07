# ComfyUI GGUF Prompt Rewriter

Standalone ComfyUI custom nodes for rewriting plain-English prompts with a local GGUF LLM through `llama-cpp-python`.

This package was designed for prompt rewriting and booru-style tag generation without relying on remote APIs or merging into other prompt/tagger extensions.

## Nodes

### `GGUF Prompt Rewriter`

Loads a local `.gguf` model and rewrites a user prompt with a configurable system prompt and decoding settings.

Outputs:

- `rewritten_prompt`
- `raw_output`
- `prompt_used`

Supported prompt modes:

- `gemma`
- `llama3`
- `chatml`
- `plain`
- `native_chat`

Notes:

- Gemma 4 models are automatically routed through native chat mode when `template=gemma`.
- The node caches one loaded model in memory for speed.

### `Unload GGUF Prompt Model`

Drops the cached model from memory so you can switch to another GGUF cleanly.

## Model Locations

The node scans these directories for `.gguf` files:

- `~/Documents/ComfyUI/models/llm_gguf`
- `~/AI`

Put your local GGUF files in either location and restart ComfyUI.

## Installation

Clone into your ComfyUI custom nodes directory:

```bash
cd /Users/gleb/Documents/ComfyUI/custom_nodes
git clone <your-repo-url> comfyui_gguf_prompt_rewriter
```

Install dependencies into the same venv ComfyUI uses:

```bash
/Users/gleb/Documents/ComfyUI/.venv/bin/python -m pip install -r /Users/gleb/Documents/ComfyUI/custom_nodes/comfyui_gguf_prompt_rewriter/requirements.txt
```

Restart ComfyUI.

## Dependencies

- `llama-cpp-python`

ComfyUI itself provides the node runtime and `folder_paths`.

## Recommended Defaults

For Gemma-style instruct models:

- `template = gemma`
- `temperature = 0.5`
- `top_p = 0.9`
- `top_k = 40`
- `repeat_penalty = 1.1`
- `n_gpu_layers = -1`

## Publishing

This folder is self-contained and can be published as its own repository.

Suggested repository name:

- `comfyui-gguf-prompt-rewriter`

## License

MIT
