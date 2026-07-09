"""Autodata-style agentic synthetic data generation for NESTFUL-like tasks.

Implements the Agentic Self-Instruct loop (Autodata, arXiv:2606.25996) adapted
to executable tool-use: a data-scientist orchestrator drives a challenger LLM,
weak/strong solver LLMs and a verifier; a DETERMINISTIC executor (not the LLM)
is the source of truth for gold observations/answers; accepted examples must
be weak-fail / strong-pass.
"""
