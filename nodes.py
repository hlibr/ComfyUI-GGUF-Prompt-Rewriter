from __future__ import annotations

import os
import re
import threading
import time
from collections import OrderedDict

import folder_paths
from llama_cpp import Llama


DEFAULT_SYSTEM_PROMPT = """Convert plain-English image descriptions into high-quality danbooru tags that will be fed into an image generation model.

Rules:
- Output only a comma-separated tag list and nothing else
- Describe character identity, action, clothing, body, environment, and strong visual details
- Add professional scene details, describe lighting, shadows, camera angle
- Do not use _ (underscore) within the tags, instead use spaces (e.g. not black_hair - black hair)
"""

_MODEL_LOCK = threading.Lock()

# Optional resident model cache, used only when keep_model_in_memory=True.
# Keyed by load configuration (path + context params + enable_thinking, which is
# applied at load time via the chat handler). When keep_model_in_memory is False,
# _acquire_model clears this cache, so turning the flag off releases the model.
_MODEL_CACHE: "dict[tuple, dict]" = {}
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE_SIZE = 1  # max models kept resident; increase if you load several models at once


def _register_llm_gguf_paths():
    key = "llm_gguf"
    # Default location, correctly tracking --models-directory / --base-directory.
    # Any folders the user added for "llm_gguf" via extra_model_paths.yaml are
    # preserved (add_model_folder_path prepends rather than replacing).
    default_dir = os.path.join(folder_paths.models_dir, "llm_gguf")
    try:
        os.makedirs(default_dir, exist_ok=True)
    except OSError:
        pass

    try:
        folder_paths.add_model_folder_path(key, default_dir, is_default=True)
    except Exception:
        # folder_paths may not be fully initialised during early import
        pass

    # Restrict to .gguf files (add_model_folder_path registers with an empty extension set).
    # Preserve existing search paths; only fix the extension filter.
    try:
        paths, exts = folder_paths.folder_names_and_paths.get(key, ([], set()))
        if exts != {".gguf"}:
            folder_paths.folder_names_and_paths[key] = (paths, {".gguf"})
    except Exception:
        pass


_register_llm_gguf_paths()


def _get_model_choices():
    try:
        files = folder_paths.get_filename_list("llm_gguf")
    except (OSError, ValueError, KeyError):
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


def _load_model(model_path: str, n_ctx: int, n_batch: int, n_gpu_layers: int, n_threads: int, enable_thinking: bool):
    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        n_threads=None if n_threads <= 0 else n_threads,
        verbose=False,
    )

    # Forward enable_thinking to the chat handler (e.g. Qwen/Gemma). We inject it
    # here rather than passing it to create_chat_completion, because
    # llama-cpp-python (incl. 0.3.35 and current master) does not accept
    # enable_thinking in create_chat_completion. The handler is resolved via
    # the same fallback chain as Llama.create_chat_completion does, so this
    # works for Jinja (chat_template.default) models like Qwen3/3.6 and Gemma4
    # where model.chat_handler is None and the real handler lives in
    # model._chat_handlers[chat_format].
    import llama_cpp.llama_chat_format

    base_chat_handler = (
        model.chat_handler
        or model._chat_handlers.get(model.chat_format)
        or llama_cpp.llama_chat_format.get_chat_completion_handler(model.chat_format)
    )

    # base_chat_handler can be None if chat_format is unknown and no template
    # was registered; in that case there is nothing to wrap.
    if base_chat_handler is not None:
        def chat_handler_with_kwargs(*args, **kwargs):
            return base_chat_handler(*args, **{"enable_thinking": enable_thinking, **kwargs})

        model.chat_handler = chat_handler_with_kwargs
    return model


def _acquire_model(
    model_path: str,
    n_ctx: int,
    n_batch: int,
    n_gpu_layers: int,
    n_threads: int,
    enable_thinking: bool,
    keep_model_in_memory: bool,
):
    """Return (model, should_close).

    When keep_model_in_memory is False the model is loaded fresh and the caller
    must close it after use (original behaviour). When True a resident model is
    returned from the cache and must NOT be closed by the caller.

    Note: enable_thinking is applied at load time (wrapped into the chat handler),
    so it is part of the cache key to avoid reusing a model with the wrong setting.
    """
    if not keep_model_in_memory:
        # Not keeping the model resident: release any model held in the cache so
        # unchecking "keep_model_in_memory" actually frees VRAM. The fresh model
        # returned here is closed by the caller after use.
        with _MODEL_CACHE_LOCK:
            for old in _MODEL_CACHE.values():
                _close_model(old["model"])
            _MODEL_CACHE.clear()
        return (
            _load_model(model_path, n_ctx, n_batch, n_gpu_layers, n_threads, enable_thinking),
            True,
        )

    key = (model_path, n_ctx, n_batch, n_gpu_layers, n_threads, enable_thinking)
    with _MODEL_CACHE_LOCK:
        entry = _MODEL_CACHE.get(key)
        if entry is not None:
            entry["last"] = time.time()
            return entry["model"], False

        model = _load_model(model_path, n_ctx, n_batch, n_gpu_layers, n_threads, enable_thinking)
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
        
        # Resolve the model file up front so the cache key identifies the actual
        # file (path + mtime + size), not just its name. This invalidates cached
        # outputs when a GGUF is swapped for a different file with the same name.
        if model == "No GGUF models found":
            raise ValueError("No GGUF models found. Put GGUF files in the ComfyUI models/llm_gguf folder.")

        model_path = None
        try:
            model_path = _resolve_model_path(model)
            _st = os.stat(model_path)
            model_identity = (model_path, _st.st_mtime, _st.st_size)
        except (OSError, ValueError):
            # Fall back to the bare name if the file can't be resolved/stated
            # (e.g. missing); the lookup then simply won't match a later valid hit.
            model_identity = (model,)
            if model_path is None:
                model_path = model

        # seed=-1 means random: don't use the output cache (otherwise "random"
        # would return a cached deterministic result) and pass None to llama
        # so it picks a non-deterministic seed. All other seeds are fixed.
        effective_seed = None if seed == -1 else seed
        use_output_cache = not ignore_cache and seed != -1

        cache_key = None
        if use_output_cache:
            cache_key = (
                model_identity,
                user_prompt.strip(),
                system_prompt.strip(),
                enable_thinking,
                effective_seed,
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
        if use_output_cache:
            with self._shared_cache_lock:
                if cache_key in self._shared_cache:
                    print("[INFO] GGUF Rewriter -> Input prompt found, using cached")
                    self._shared_cache.move_to_end(cache_key)
                    normalized_cached, raw_cached = self._shared_cache[cache_key]
                    return (normalized_cached, raw_cached)
                else:
                    print(f"[INFO] GGUF Rewriter -> New prompt found: {user_prompt[:80]}")
        elif seed == -1:
            print(f"[INFO] GGUF Rewriter -> Random seed (-1), cache bypassed: {user_prompt[:80]}")
        else:
            print(f"[INFO] GGUF Rewriter -> Cache bypassed, forcing rewrite: {user_prompt[:80]}")

        with _MODEL_LOCK:
            llm, should_close = _acquire_model(
                model_path, n_ctx, n_batch, n_gpu_layers, n_threads, enable_thinking, keep_model_in_memory
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
                    seed=effective_seed,
                )
            finally:
                if should_close:
                    _close_model(llm)

        if response is None:
            raise RuntimeError("GGUF model failed to produce a response (see logs above).")
        raw_text = response["choices"][-1]["message"]["content"]
        normalized = _normalize_output(raw_text)

        # ------------------------------------------------------------
        # UPDATE LRU CACHE (only if not ignoring cache and not random)
        # ------------------------------------------------------------
        if use_output_cache:
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
