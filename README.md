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
- The node caches one loaded model in memory for speed.

### `Unload GGUF Prompt Model`

Drops the cached model from memory so you can switch to another GGUF cleanly.

## Model Locations

The node scans these directories for `.gguf` files:

- `ComfyUI/models/llm_gguf`
- `~/AI`

Put your local GGUF files in either location and restart ComfyUI.

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
