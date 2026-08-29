import gc
import os
import re
import threading
import time

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

# Optional resident model cache, used only when keep_model_in_memory=True.
# Keyed by load configuration (path + context params); enable_thinking is a
# per-generation parameter and therefore not part of the key.
_MODEL_CACHE: "dict[tuple, dict]" = {}
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE_SIZE = 1  # max models kept resident; raise if you juggle several


def _register_llm_gguf_paths():
    key = "llm_gguf"
    # Default location, correctly tracking --models-directory / --base-directory.
    # Any folders the user added for "llm_gguf" via extra_model_paths.yaml are
    # preserved (add_model_folder_path prepends rather than replacing).
    default_dir = os.path.join(folder_paths.models_dir, "llm_gguf")
    try:
        os.makedirs(default_dir, exist_ok=True)
    except Exception:
        pass

    folder_paths.add_model_folder_path(key, default_dir, is_default=True)

    # Restrict to .gguf files (add_model_folder_path registers with an empty extension set)
    folder_paths.folder_names_and_paths[key] = (
        folder_paths.folder_names_and_paths[key][0],
        {".gguf"},
    )


_register_llm_gguf_paths()


def _get_model_choices():
    try:
        files = folder_paths.get_filename_list("llm_gguf")
    except Exception:
        files = []
    return sorted(files) if files else ["No GGUF models found"]


def _resolve_model_path(model_name: str) -> str:
    return folder_paths.get_full_path_or_raise("llm_gguf", model_name)




def _normalize_output(text: str) -> str:
    # First, handle standard paired tags (if present)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    # Then strip everything before and including any remaining closing tags
    # (handles cases where opening tag is missing, like Qwen's format)
    text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</thinking>\s*", "", text, flags=re.DOTALL)
    # Strip Gemma-style channel tags: <|channel>NAME...content...<channel|>
    text = re.sub(r"<\|channel\>[^\n]*\n.*?<channel\|>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|channel\|>[^\n]*\n.*?<channel\|>\s*", "", text, flags=re.DOTALL)
    return text.strip(" \n\r\t,")


def _close_model(model):
    if model is not None:
        try:
            model.close()
        except Exception:
            pass


def _load_model(model_path: str, n_ctx: int, n_batch: int, n_gpu_layers: int, n_threads: int):
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        n_threads=None if n_threads <= 0 else n_threads,
        verbose=False,
    )


def _acquire_model(
    model_path: str,
    n_ctx: int,
    n_batch: int,
    n_gpu_layers: int,
    n_threads: int,
    keep_model_in_memory: bool,
):
    """Return (model, should_close).

    When keep_model_in_memory is False the model is loaded fresh and the caller
    must close it after use (original behaviour). When True a resident model is
    returned from the cache and must NOT be closed by the caller.

    Note: enable_thinking is passed per generation (create_chat_completion), not
    at load time, so it does not affect which model instance is cached.
    """
    if not keep_model_in_memory:
        return (
            _load_model(model_path, n_ctx, n_batch, n_gpu_layers, n_threads),
            True,
        )

    key = (model_path, n_ctx, n_batch, n_gpu_layers, n_threads)
    with _MODEL_CACHE_LOCK:
        entry = _MODEL_CACHE.get(key)
        if entry is not None:
            entry["last"] = time.time()
            return entry["model"], False

        model = _load_model(model_path, n_ctx, n_batch, n_gpu_layers, n_threads)
        _MODEL_CACHE[key] = {"model": model, "last": time.time()}
        # Evict least-recently-used beyond the cap. Safe because the caller holds
        # _MODEL_LOCK, so no other thread can be using a cached model right now.
        if len(_MODEL_CACHE) > _MODEL_CACHE_SIZE:
            old_key = min(_MODEL_CACHE, key=lambda k: _MODEL_CACHE[k]["last"])
            old = _MODEL_CACHE.pop(old_key)
            _close_model(old["model"])
    return model, False


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
                # -1 = random seed; 0 or positive = fixed/deterministic
                "seed": ("INT", {"default": 0, "min": -1, "max": 2**32 - 1}),
                "max_tokens": ("INT", {"default": 160, "min": 1, "max": 32768, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 500, "step": 1}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.05}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05}),
                "n_ctx": ("INT", {"default": 4096, "min": 256, "max": 262144, "step": 256}),
                "n_batch": ("INT", {"default": 512, "min": 32, "max": 4096, "step": 32}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999, "step": 1}),
                "n_threads": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "ignore_cache": ("BOOLEAN", {"default": False}),
                "keep_model_in_memory": ("BOOLEAN", {"default": False}),
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
        keep_model_in_memory=False,
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
            raise ValueError("No GGUF models found. Put GGUF files in the ComfyUI models/llm_gguf folder.")

        model_path = _resolve_model_path(model)

        with _MODEL_LOCK:
            llm, should_close = _acquire_model(
                model_path, n_ctx, n_batch, n_gpu_layers, n_threads, keep_model_in_memory
            )
            response = None
            try:
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
                    enable_thinking=enable_thinking,
                )
            finally:
                if should_close:
                    _close_model(llm)
                    llm = None
                    gc.collect()

        if response is None:
            raise RuntimeError("GGUF model failed to produce a response (see logs above).")
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
