import gc
import os
import re
import threading

import folder_paths
from llama_cpp import Llama


DEFAULT_SYSTEM_PROMPT = """You convert plain-English anime image descriptions into high-quality booru-style tags.

Rules:
- Output only a comma-separated tag list
- No prose
- No explanations
- No markdown
- Prefer common Danbooru-style or booru-style tags
- Keep character identity, action, clothing, environment, and strong visual details
- Do not invent extra subjects
- If the prompt implies one subject, keep it single-subject
- Avoid low-confidence tags
- Keep the result concise and useful
"""

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE = {"key": None, "model": None}


def _register_llm_gguf_paths():
    key = "llm_gguf"
    user_models_dir = os.path.join(
        os.path.dirname(folder_paths.get_user_directory()),
        "models",
        "llm_gguf",
    )
    home_ai_dir = os.path.expanduser("~/AI")
    try:
        os.makedirs(user_models_dir, exist_ok=True)
    except Exception:
        pass

    existing = folder_paths.folder_names_and_paths.get(key, ([], set()))
    existing_dirs = list(existing[0]) if isinstance(existing[0], (list, tuple, set)) else []
    dirs = []
    for path in [user_models_dir, home_ai_dir, *existing_dirs]:
        if path and path not in dirs and (path == user_models_dir or os.path.exists(path)):
            dirs.append(path)

    folder_paths.folder_names_and_paths[key] = (dirs, {".gguf"})


_register_llm_gguf_paths()


def _get_model_choices():
    try:
        files = folder_paths.get_filename_list("llm_gguf")
    except Exception:
        files = []
    return sorted(files) if files else ["No GGUF models found"]


def _resolve_model_path(model_name: str) -> str:
    return folder_paths.get_full_path("llm_gguf", model_name)


# def _build_prompt(template: str, system_prompt: str, user_prompt: str) -> tuple[str, list[str]]:
#     system_prompt = system_prompt.strip()
#     user_prompt = user_prompt.strip()

#     if template == "gemma":
#         prompt = (
#             "<|turn>system\n"
#             f"You are a helpful assistant.<turn|>\n"
#             f"<|turn>user\n"
#             "Hello.<turn|>"
#         )
#         return prompt, ["<end_of_turn>"]

#     if template == "llama3":
#         prompt = (
#             "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
#             f"{system_prompt}<|eot_id|>"
#             "<|start_header_id|>user<|end_header_id|>\n\n"
#             f"{user_prompt}<|eot_id|>"
#             "<|start_header_id|>assistant<|end_header_id|>\n\n"
#         )
#         return prompt, ["<|eot_id|>"]

#     if template == "chatml":
#         prompt = (
#             "<|im_start|>system\n"
#             f"{system_prompt}<|im_end|>\n"
#             "<|im_start|>user\n"
#             f"{user_prompt}<|im_end|>\n"
#             "<|im_start|>assistant\n"
#         )
#         return prompt, ["<|im_end|>"]

#     prompt = f"{system_prompt}\n\n{user_prompt}".strip()
#     return prompt, []


def _normalize_output(text: str) -> str:
    text = re.sub(r"^(?:<\|channel\>[^ \n]+\n<channel\|>\s*)+", "", text)
    text = re.sub(r"^(?:<\|channel\|>[^ \n]+\n<channel\|>\s*)+", "", text)
    return text.strip(" \n\r\t,")


def _should_use_native_chat(model_name: str, template: str) -> bool:
    name = model_name.lower()
    if template == "native_chat":
        return True
    return template == "gemma" and ("gemma-4" in name or "gemma 4" in name)


def _maybe_unload():
    model = _MODEL_CACHE.get("model")
    if model is not None:
        try:
            model.close()
        except Exception:
            pass
    _MODEL_CACHE["model"] = None
    _MODEL_CACHE["key"] = None
    gc.collect()


def _get_or_load_model(model_path: str, n_ctx: int, n_batch: int, n_gpu_layers: int, n_threads: int):
    cache_key = (model_path, n_ctx, n_batch, n_gpu_layers, n_threads)
    with _MODEL_LOCK:
        if _MODEL_CACHE["key"] == cache_key and _MODEL_CACHE["model"] is not None:
            return _MODEL_CACHE["model"]

        _maybe_unload()
        model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            n_threads=None if n_threads <= 0 else n_threads,
            verbose=False,
        )
        _MODEL_CACHE["key"] = cache_key
        _MODEL_CACHE["model"] = model
        return model


class GGUFPromptRewriter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_get_model_choices(),),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": DEFAULT_SYSTEM_PROMPT, "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "max_tokens": ("INT", {"default": 160, "min": 1, "max": 4096, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 500, "step": 1}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05}),
                "n_ctx": ("INT", {"default": 4096, "min": 256, "max": 32768, "step": 256}),
                "n_batch": ("INT", {"default": 512, "min": 32, "max": 4096, "step": 32}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999, "step": 1}),
                "n_threads": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("rewritten_prompt", "raw_output")
    FUNCTION = "rewrite"
    CATEGORY = "prompt/LLM"

    def rewrite(
        self,
        model,
        user_prompt,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        seed=0,
        max_tokens=160,
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        n_ctx=4096,
        n_batch=512,
        n_gpu_layers=-1,
        n_threads=0,
    ):
        if model == "No GGUF models found":
            raise ValueError("No GGUF models found. Put GGUF files in ComfyUI/models/llm_gguf or ~/AI.")

        model_path = _resolve_model_path(model)
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Could not resolve GGUF model path for: {model}")

        llm = _get_or_load_model(model_path, n_ctx, n_batch, n_gpu_layers, n_threads)
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            seed=seed,
            # stop=stops or None,
        )
        raw_text = response["choices"][0]["message"]["content"]
        return (_normalize_output(raw_text), raw_text)


class UnloadGGUFPromptModel:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "prompt/LLM"

    def unload(self):
        with _MODEL_LOCK:
            _maybe_unload()
        return ("Unloaded GGUF prompt model",)


NODE_CLASS_MAPPINGS = {
    "GGUFPromptRewriter": GGUFPromptRewriter,
    "UnloadGGUFPromptModel": UnloadGGUFPromptModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GGUFPromptRewriter": "GGUF Prompt Rewriter",
    "UnloadGGUFPromptModel": "Unload GGUF Prompt Model",
}
