"""LLM providers. Default: template_only (no LLM). Optional local providers
only; no remote/paid endpoint is ever active by default. All outputs cached
by content hash."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..util import sha256_obj


class ProviderUnavailable(Exception):
    pass


class BaseProvider:
    kind = "base"

    def __init__(self, cfg: Dict[str, Any], cache_dir: Optional[Path] = None):
        self.cfg = cfg
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def available(self) -> bool:
        return False

    def complete(self, prompt: str, *, max_tokens: int = 512,
                 temperature: float = 0.0, n: int = 1,
                 seed: int = 0) -> List[str]:
        raise ProviderUnavailable(self.kind)

    def _cached(self, key_obj: Any) -> Optional[List[str]]:
        if not self.cache_dir:
            return None
        f = self.cache_dir / f"{sha256_obj(key_obj)}.json"
        if f.is_file():
            return json.loads(f.read_text(encoding="utf-8"))
        return None

    def _store(self, key_obj: Any, value: List[str]) -> None:
        if self.cache_dir:
            f = self.cache_dir / f"{sha256_obj(key_obj)}.json"
            f.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class TemplateOnlyProvider(BaseProvider):
    """No LLM. Query realization stays with deterministic templates."""
    kind = "template_only"

    def available(self) -> bool:
        return True


class OpenAICompatibleLocalProvider(BaseProvider):
    """LM Studio / Ollama / local vLLM via OpenAI-compatible HTTP API."""
    kind = "openai_compatible_local"

    def available(self) -> bool:
        base = self.cfg.get("base_url", "")
        if not base or not ("127.0.0.1" in base or "localhost" in base):
            return False
        try:
            req = urllib.request.Request(base.rstrip("/") + "/models")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            return False

    def complete(self, prompt: str, *, max_tokens: int = 512,
                 temperature: float = 0.0, n: int = 1,
                 seed: int = 0) -> List[str]:
        key = {"kind": self.kind, "model": self.cfg.get("model"),
               "prompt": prompt, "max_tokens": max_tokens,
               "temperature": temperature, "n": n, "seed": seed}
        cached = self._cached(key)
        if cached is not None:
            return cached
        body = json.dumps({
            "model": self.cfg.get("model", ""),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature,
            "n": n, "seed": seed,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.cfg["base_url"].rstrip("/") + "/chat/completions",
            data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        outs = [c["message"]["content"] for c in data.get("choices", [])]
        self._store(key, outs)
        return outs


class TransformersLocalProvider(BaseProvider):
    """Local transformers checkpoint (requires GPU for Qwen3-4B in practice)."""
    kind = "transformers_local"

    def __init__(self, cfg: Dict[str, Any], cache_dir: Optional[Path] = None):
        super().__init__(cfg, cache_dir)
        self._pipe = None

    def available(self) -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def complete(self, prompt: str, *, max_tokens: int = 512,
                 temperature: float = 0.0, n: int = 1,
                 seed: int = 0) -> List[str]:
        key = {"kind": self.kind, "model": self.cfg.get("model"),
               "prompt": prompt, "max_tokens": max_tokens,
               "temperature": temperature, "n": n, "seed": seed}
        cached = self._cached(key)
        if cached is not None:
            return cached
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = self.cfg.get("model") or "Qwen/Qwen3-4B-Instruct-2507"
        if self._pipe is None:
            tok = AutoTokenizer.from_pretrained(model_id)
            kwargs: Dict[str, Any] = {"device_map": "auto"}
            # the probe is only a difficulty proxy, so a 4-bit copy of the
            # exact student checkpoint is acceptable and fits a 6 GB GPU
            if self.cfg.get("load_in_4bit"):
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True)
            else:
                kwargs["torch_dtype"] = torch.bfloat16
            mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            mdl.eval()
            self._pipe = (tok, mdl)
        tok, mdl = self._pipe
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        outs = []
        for i in range(n):
            torch.manual_seed(seed + i)
            ids = tok(text, return_tensors="pt").to(mdl.device)
            with torch.inference_mode():
                gen = mdl.generate(**ids, max_new_tokens=max_tokens,
                                   do_sample=temperature > 0,
                                   temperature=max(temperature, 1e-5),
                                   pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(gen[0][ids["input_ids"].shape[1]:],
                                   skip_special_tokens=True))
        self._store(key, outs)
        return outs


def make_provider(cfg: Dict[str, Any], cache_dir: Optional[Path] = None,
                  no_llm: bool = False) -> BaseProvider:
    """`LOCAL_LLM_BASE_URL` / `LOCAL_LLM_MODEL` from the environment or the
    repository .env override the config (local probe, section 4)."""
    import os

    from ..paraphrase import load_env_file

    env = {**load_env_file(), **os.environ}
    cfg = dict(cfg)
    if env.get("LOCAL_LLM_BASE_URL"):
        cfg["base_url"] = env["LOCAL_LLM_BASE_URL"]
    if env.get("LOCAL_LLM_MODEL"):
        cfg["model"] = env["LOCAL_LLM_MODEL"]
    kind = "template_only" if no_llm else cfg.get("kind", "template_only")
    cls = {"template_only": TemplateOnlyProvider,
           "openai_compatible_local": OpenAICompatibleLocalProvider,
           "transformers_local": TransformersLocalProvider}.get(kind, TemplateOnlyProvider)
    return cls(cfg, cache_dir)
