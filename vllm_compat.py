"""
Compatibility shim for vLLM 0.8.4 + transformers >= 5.0.

vLLM 0.8.4 accesses `tokenizer.all_special_tokens_extended` which was
removed in transformers 5.x.  Import this module *before* importing vllm
to monkey-patch the missing attribute back in.
"""

import transformers.tokenization_utils_base as _tub

if not hasattr(_tub.PreTrainedTokenizerBase, "all_special_tokens_extended"):
    @property
    def _all_special_tokens_extended(self):
        return list(set(self.all_special_tokens))

    _tub.PreTrainedTokenizerBase.all_special_tokens_extended = (
        _all_special_tokens_extended
    )
