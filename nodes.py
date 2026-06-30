import gc
import os
import re
import threading

import folder_paths
from llama_cpp import Llama
from collections import OrderedDict


DEFAULT_SYSTEM_PROMPT = """Convert plain-English image descriptions into high-quality danbooru tags that will be fed into an image generation model.

Rules:
- Output only a comma-separated tag list and nothing else
- Describe character identity, action, clothing, body, environment, and strong visual details
- Add professional scene details, describe lighting, shadows, camera angle
- Do not use _ (underscore) within the tags, instead use spaces (e.g. not black_hair - black hair)
"""

_MODEL_LOCK = threading.Lock()


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
    # Strip thinking blocks (common in DeepSeek-R1 and similar)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    # Strip other channel tags
    text = re.sub(r"^(?:<\|channel\>[^ \n]+\n<channel\|>\s*)+", "", text)
    text = re.sub(r"^(?:<\|channel\|>[^ \n]+\n<channel\|>\s*)+", "", text)
    return text.strip(" \n\r\t,")


def _close_model(model):
    if model is not None:
        try:
            model.close()
        except Exception:
            pass


def _load_model(model_path: str, n_ctx: int, n_batch: int, n_gpu_layers: int, n_threads: int, enable_thinking: bool):
    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        n_threads=None if n_threads <= 0 else n_threads,
        verbose=False,
    )

    import llama_cpp.llama_chat_format

    base_chat_handler = (
        model.chat_handler
        or model._chat_handlers.get(model.chat_format)
        or llama_cpp.llama_chat_format.get_chat_completion_handler(model.chat_format)
    )

    def chat_handler_with_kwargs(*args, **kwargs):
        return base_chat_handler(*args, **{"enable_thinking": enable_thinking, **kwargs})

    model.chat_handler = chat_handler_with_kwargs
    return model


class GGUFPromptRewriter:
    # Class-level shared cache (shared across all node instances)
    _shared_cache = OrderedDict()
    _shared_cache_lock = threading.Lock()
    _cache_size = 200  # Larger since it's shared across all instances

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_get_model_choices(),),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": DEFAULT_SYSTEM_PROMPT, "multiline": True}),
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "max_tokens": ("INT", {"default": 160, "min": 1, "max": 4096, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 500, "step": 1}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.05}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05}),
                "n_ctx": ("INT", {"default": 4096, "min": 256, "max": 32768, "step": 256}),
                "n_batch": ("INT", {"default": 512, "min": 32, "max": 4096, "step": 32}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999, "step": 1}),
                "n_threads": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "ignore_cache": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ["rewritten_prompt", "raw_output"]
    FUNCTION = "rewrite"
    CATEGORY = "prompt/LLM"

    def rewrite(
        self,
        model,
        user_prompt,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        enable_thinking=False,
        ignore_cache=False,
        seed=0,
        max_tokens=160,
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        min_p=0.05,
        presence_penalty=0.0,
        repeat_penalty=1.1,
        n_ctx=4096,
        n_batch=512,
        n_gpu_layers=-1,
        n_threads=0,
    ):
        
        cache_key = (
            model,
            user_prompt.strip(),
            system_prompt.strip(),
            enable_thinking,
            seed,
            max_tokens,
            round(temperature, 2),
            round(top_p, 2),
            top_k,
            round(min_p, 2),
            round(presence_penalty, 2),
            round(repeat_penalty, 2),
            n_ctx,
            n_batch,
            n_gpu_layers,
            n_threads,
        )

        # ------------------------------------------------------------
        # LRU CACHE CHECK
        # ------------------------------------------------------------
        if not ignore_cache:
            with self._shared_cache_lock:
                if cache_key in self._shared_cache:
                    print("[INFO] GGUF Rewriter -> Input prompt found, using cached")
                    self._shared_cache.move_to_end(cache_key)
                    normalized_cached, raw_cached = self._shared_cache[cache_key]
                    return (normalized_cached, raw_cached)
                else:
                    print(f"[INFO] GGUF Rewriter -> New prompt found: {user_prompt[:80]}")
        else:
            print(f"[INFO] GGUF Rewriter -> Cache bypassed, forcing rewrite: {user_prompt[:80]}")

        if model == "No GGUF models found":
            raise ValueError("No GGUF models found. Put GGUF files in ComfyUI/models/llm_gguf or ~/AI.")

        model_path = _resolve_model_path(model)
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Could not resolve GGUF model path for: {model}")

        with _MODEL_LOCK:
            llm = None
            try:
                llm = _load_model(model_path, n_ctx, n_batch, n_gpu_layers, n_threads, enable_thinking)
                response = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt.strip()},
                        {"role": "user", "content": user_prompt.strip()},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    presence_penalty=presence_penalty,
                    repeat_penalty=repeat_penalty,
                    seed=seed,
                )
            finally:
                _close_model(llm)
                llm = None
                gc.collect()
        raw_text = response["choices"][-1]["message"]["content"]
        normalized = _normalize_output(raw_text)

        # ------------------------------------------------------------
        # UPDATE LRU CACHE (only if not ignoring cache)
        # ------------------------------------------------------------
        if not ignore_cache:
            with self._shared_cache_lock:
                # Store both normalized and raw output
                self._shared_cache[cache_key] = (normalized, raw_text)
                self._shared_cache.move_to_end(cache_key)

                # Enforce max size
                if len(self._shared_cache) > self._cache_size:
                    self._shared_cache.popitem(last=False)  # remove oldest

        return (normalized, raw_text)


NODE_CLASS_MAPPINGS = {
    "GGUFPromptRewriter": GGUFPromptRewriter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GGUFPromptRewriter": "GGUF Prompt Rewriter",
}
