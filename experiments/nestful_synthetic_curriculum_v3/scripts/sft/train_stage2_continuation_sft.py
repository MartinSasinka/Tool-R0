#!/usr/bin/env python3
"""Stage2 continuation SFT trainer.

Pure supervised fine-tuning on the Stage2 continuation SFT view
(outputs/sft/stage2_continuation/{train,val}.jsonl). This is INTENTIONALLY
separate from the GRPO curriculum pipeline:

  - no reward function, no rollout groups, no GRPO, no group-relative
    advantage — this is a standard causal-LM next-token loss.
  - no teacher forcing during evaluation (evaluation is handled by
    eval_stage2_sft.py / continuation_conditioned_eval.py, both of which use
    free generation only).
  - model loading (QLoRA config: r=16, alpha=32, NF4 4-bit) reuses
    experiments/nestful_mtgrpo_minimal/run.py::load_model_and_tokenizer so the
    adapter this script produces is a drop-in, directly comparable checkpoint
    for the SAME eval machinery (run.py --mode rollout_eval / final_eval)
    the GRPO pilot already uses.

Loss masking:
  Each training record's `messages` field is the FULL conversation (see
  scripts/sft/sft_common.py::build_continuation_messages) and
  `loss_target_message_indices` names which assistant messages are actual
  generation targets ([4, 6] for target_type="continuation": the second gold
  tool call and the terminal empty-finish). Loss is computed ONLY on the
  token span of those messages; every other token (system prompt, user
  turns, and the GIVEN first assistant turn) is masked with label=-100.
  Token spans are found by re-applying the tokenizer's chat template to each
  growing message prefix messages[:i] and diffing token-id lengths — this is
  a standard trick for chat-template models (Qwen3's template is a strict
  append per turn, so prefix boundaries are stable) but is NOT a fully
  general solution for chat templates with cross-turn side effects; verified
  against Qwen3-4B-Instruct-2507 in this script's --dry-run mode (see
  scripts/sft/README-ish docstring below; no GPU/model download required).

Usage (typical, on the GPU pod):
  python train_stage2_continuation_sft.py \
      --train-path outputs/sft/stage2_continuation/train.jsonl \
      --val-path   outputs/sft/stage2_continuation/val.jsonl \
      --output-dir outputs/sft/stage2_continuation/run_20260709_120000 \
      --epochs 1 --lr 1e-5 --batch-size 1 --grad-accum 16 --max-seq-len 3072

Dry run (no GPU / no bitsandbytes / no model download required — sanity
checks the masking logic and prints token-budget stats only):
  python train_stage2_continuation_sft.py --train-path ... --val-path ... \
      --output-dir /tmp/x --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sft_common import BASE_MODEL, MINIMAL, read_jsonl_raw  # noqa: E402

for _p in (str(MINIMAL),):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-path", required=True)
    ap.add_argument("--val-path", required=True)
    ap.add_argument("--output-dir", required=True, help="Run dir; adapter + sidecars written here.")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--resume-checkpoint", default=None,
                    help="Optional existing LoRA adapter dir to continue training from.")
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("SFT_EPOCHS", 1)))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("SFT_LR", 1e-5)))
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("SFT_BATCH_SIZE", 1)))
    ap.add_argument("--grad-accum", type=int, default=int(os.environ.get("SFT_GRAD_ACCUM", 16)))
    ap.add_argument("--max-seq-len", type=int, default=int(os.environ.get("SFT_MAX_SEQ_LEN", 3072)))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SFT_SEED", 42)))
    ap.add_argument("--lora-r", type=int, default=int(os.environ.get("SFT_LORA_R", 16)))
    ap.add_argument("--lora-alpha", type=int, default=int(os.environ.get("SFT_LORA_ALPHA", 32)))
    ap.add_argument("--lora-dropout", type=float, default=float(os.environ.get("SFT_LORA_DROPOUT", 0.05)))
    ap.add_argument("--no-4bit", action="store_true",
                    help="Disable QLoRA 4-bit quantization (plain LoRA in bf16).")
    ap.add_argument("--eval-every-epoch", action="store_true", default=True)
    ap.add_argument("--log-every", type=int, default=10, help="Log train_log.jsonl every N optimizer steps.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build + mask + token-budget-check the dataset only; no model load, no training.")
    ap.add_argument("--max-train-examples", type=int, default=None, help="Debug: cap train set size.")
    ap.add_argument(
        "--hf-device-map",
        default=os.environ.get("SFT_HF_DEVICE_MAP", "auto"),
        help="Passed to load_model_and_tokenizer as hardware.hf_device_map. "
             "Use '{\"\": 0}' (or SFT_HF_DEVICE_MAP='{\"\": 0}') to pin the "
             "whole 4B QLoRA model on a single GPU; 'auto' shards across all "
             "visible GPUs (model parallelism, not DDP).",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
#  Loss-masked tokenization
# ---------------------------------------------------------------------------

def _message_boundaries(tokenizer, messages: List[Dict[str, str]]) -> List[int]:
    """Cumulative token count after each messages[:i] prefix (i=0..len(messages))."""
    boundaries = [0]
    for i in range(1, len(messages) + 1):
        ids = tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=False)
        boundaries.append(len(ids))
    return boundaries


def tokenize_masked_example(
    tokenizer, messages: List[Dict[str, str]], loss_target_indices: List[int], max_seq_len: int,
) -> Optional[Dict[str, Any]]:
    """Returns {"input_ids": [...], "labels": [...]} or None if it would need
    truncation (skipped rather than silently corrupting the near-the-end
    target span — see train_stage2_continuation_sft.py module docstring)."""
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if len(full_ids) > max_seq_len:
        return None
    boundaries = _message_boundaries(tokenizer, messages)
    labels = [-100] * len(full_ids)
    n_label_tokens = 0
    for idx in loss_target_indices:
        start, end = boundaries[idx], boundaries[idx + 1]
        labels[start:end] = full_ids[start:end]
        n_label_tokens += max(0, end - start)
    if n_label_tokens == 0:
        return None
    return {"input_ids": full_ids, "labels": labels, "n_label_tokens": n_label_tokens,
            "n_total_tokens": len(full_ids)}


def build_tokenized_dataset(
    tokenizer, records: List[Dict[str, Any]], max_seq_len: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    out = []
    stats = {"kept": 0, "skipped_too_long": 0, "skipped_no_label_tokens": 0}
    for r in records:
        ex = tokenize_masked_example(
            tokenizer, r["messages"], r["loss_target_message_indices"], max_seq_len,
        )
        if ex is None:
            full_ids = tokenizer.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=False)
            if len(full_ids) > max_seq_len:
                stats["skipped_too_long"] += 1
            else:
                stats["skipped_no_label_tokens"] += 1
            continue
        ex["sample_id"] = r["sample_id"]
        out.append(ex)
        stats["kept"] += 1
    return out, stats


# ---------------------------------------------------------------------------
#  Collation
# ---------------------------------------------------------------------------

def _input_device(model) -> "torch.device":
    """Device for batch tensors when the model uses device_map (multi-GPU shard)."""
    import torch
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings().weight.device
    return next(p.device for p in model.parameters() if p.requires_grad)


def _parse_hf_device_map(raw: str):
    """Parse SFT_HF_DEVICE_MAP / --hf-device-map: auto | 0 | {\"\": 0}."""
    s = (raw or "auto").strip()
    if s == "auto":
        return "auto"
    if s.isdigit():
        return {"": int(s)}
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return s


def collate(batch: List[Dict[str, Any]], pad_token_id: int):
    import torch
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, attn, labels = [], [], []
    for ex in batch:
        n_pad = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_token_id] * n_pad)
        attn.append([1] * len(ex["input_ids"]) + [0] * n_pad)
        labels.append(ex["labels"] + [-100] * n_pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_records = read_jsonl_raw(args.train_path)
    val_records = read_jsonl_raw(args.val_path)
    if args.max_train_examples:
        train_records = train_records[: args.max_train_examples]
    print(f"[sft-train] train records = {len(train_records)}  val records = {len(val_records)}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_tok, train_stats = build_tokenized_dataset(tokenizer, train_records, args.max_seq_len)
    val_tok, val_stats = build_tokenized_dataset(tokenizer, val_records, args.max_seq_len)
    print(f"[sft-train] train tokenized: {train_stats}")
    print(f"[sft-train] val tokenized:   {val_stats}")
    if not train_tok:
        print("[sft-train] ERROR: zero trainable examples after tokenization/masking.", file=sys.stderr)
        return 1

    if args.dry_run:
        avg_total = sum(e["n_total_tokens"] for e in train_tok) / len(train_tok)
        avg_label = sum(e["n_label_tokens"] for e in train_tok) / len(train_tok)
        print(f"[sft-train] DRY RUN — no model loaded, no training performed.")
        print(f"[sft-train] avg total tokens/example = {avg_total:.1f}")
        print(f"[sft-train] avg LOSS-target tokens/example = {avg_label:.1f}")
        print(f"[sft-train] example[0] input_ids[:20] = {train_tok[0]['input_ids'][:20]}")
        n_label_span = sum(1 for t in train_tok[0]["labels"] if t != -100)
        print(f"[sft-train] example[0] labeled tokens = {n_label_span} / {len(train_tok[0]['labels'])}")
        with open(os.path.join(args.output_dir, "DRY_RUN_TOKENIZE_REPORT.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "train_stats": train_stats, "val_stats": val_stats,
                "avg_total_tokens": avg_total, "avg_label_tokens": avg_label,
                "max_seq_len": args.max_seq_len, "base_model": args.base_model,
            }, fh, indent=2)
        return 0

    # ------------------------------------------------------------------
    #  Real training path — requires torch + CUDA + peft (+ bitsandbytes
    #  if 4-bit). This branch is NOT exercised on the CPU/Windows dev
    #  machine used to author this script; see the report for the risk.
    # ------------------------------------------------------------------
    import torch
    from run import load_model_and_tokenizer  # nestful_mtgrpo_minimal/run.py

    if not torch.cuda.is_available():
        print("[sft-train] ERROR: CUDA is not available — this trainer requires a GPU.", file=sys.stderr)
        return 1
    if not args.no_4bit:
        try:
            import importlib.metadata as _imd
            _imd.version("bitsandbytes")
        except Exception:
            print(
                "[sft-train] ERROR: bitsandbytes is required for QLoRA (4-bit) but is not installed.\n"
                "  pip install 'bitsandbytes>=0.43' 'peft>=0.12' 'accelerate>=0.33'\n"
                "  or: bash experiments/nestful_mtgrpo_minimal/install_deps.sh",
                file=sys.stderr,
            )
            return 1
    try:
        import peft  # noqa: F401
    except ImportError:
        print(
            "[sft-train] ERROR: peft is not installed.\n"
            "  pip install 'peft>=0.12' 'bitsandbytes>=0.43' 'accelerate>=0.33'",
            file=sys.stderr,
        )
        return 1

    hf_device_map = _parse_hf_device_map(args.hf_device_map)
    n_gpus = torch.cuda.device_count()
    print(f"[sft-train] cuda devices visible = {n_gpus}  hf_device_map = {hf_device_map!r}")

    config = {
        "model": {"base_model": args.base_model, "lora_adapter": None},
        "hardware": {
            "bf16": True,
            "load_in_4bit": not args.no_4bit,
            "gradient_checkpointing": True,
            "hf_device_map": hf_device_map,
            "use_flash_attention": False,
        },
        "finetuning": {
            "method": "qlora" if not args.no_4bit else "lora",
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": "auto",
            "load_in_4bit": not args.no_4bit,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        },
    }
    model, tokenizer = load_model_and_tokenizer(config, checkpoint=args.resume_checkpoint, for_training=True)
    input_device = _input_device(model)
    print(f"[sft-train] input device (batch placement) = {input_device}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )

    train_log_path = os.path.join(args.output_dir, "train_log.jsonl")
    train_log_fh = open(train_log_path, "w", encoding="utf-8")

    def log(row: Dict[str, Any]) -> None:
        train_log_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        train_log_fh.flush()

    def run_eval(step: int, epoch: int) -> float:
        model.eval()
        losses = []
        rng = random.Random(args.seed)
        idxs = list(range(len(val_tok)))
        with torch.no_grad():
            for i in range(0, len(idxs), max(1, args.batch_size)):
                batch_idx = idxs[i:i + args.batch_size]
                batch = collate([val_tok[j] for j in batch_idx], tokenizer.pad_token_id)
                batch = {k: v.to(input_device) for k, v in batch.items()}
                out = model(**batch)
                losses.append(float(out.loss.item()))
        model.train()
        val_loss = sum(losses) / len(losses) if losses else float("nan")
        log({"event": "eval", "step": step, "epoch": epoch, "val_loss": val_loss})
        print(f"[sft-train] epoch {epoch} step {step} val_loss={val_loss:.4f}")
        return val_loss

    model.train()
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    running_count = 0
    train_losses_by_epoch: Dict[int, List[float]] = {}
    val_loss_by_epoch: Dict[int, Optional[float]] = {}
    rng = random.Random(args.seed)

    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_tok)))
        rng.shuffle(order)
        train_losses_by_epoch[epoch] = []
        optimizer.zero_grad()
        for i in range(0, len(order), max(1, args.batch_size)):
            batch_idx = order[i:i + args.batch_size]
            batch = collate([train_tok[j] for j in batch_idx], tokenizer.pad_token_id)
            batch = {k: v.to(input_device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running_loss += float(out.loss.item())
            running_count += 1
            micro_step += 1
            train_losses_by_epoch[epoch].append(float(out.loss.item()))

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0,
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.log_every == 0:
                    avg_loss = running_loss / max(1, running_count)
                    log({"event": "train_step", "epoch": epoch, "step": global_step,
                         "loss": avg_loss, "lr": args.lr})
                    print(f"[sft-train] epoch {epoch} step {global_step} loss={avg_loss:.4f}")
                    running_loss, running_count = 0.0, 0

        if args.eval_every_epoch:
            val_loss = run_eval(global_step, epoch)
        else:
            val_loss = None
        val_loss_by_epoch[epoch] = val_loss

        adapter_dir = os.path.join(args.output_dir, "adapter", f"epoch_{epoch}")
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        _write_sidecars(adapter_dir, args, config, epoch=epoch, global_step=global_step,
                        train_loss=sum(train_losses_by_epoch[epoch]) / max(1, len(train_losses_by_epoch[epoch])),
                        val_loss=val_loss)
        print(f"[sft-train] saved epoch {epoch} adapter -> {adapter_dir}")

    train_log_fh.close()

    final_adapter_dir = os.path.join(args.output_dir, "adapter", f"epoch_{args.epochs}")
    summary = _build_training_summary(
        args, config, train_records, val_records, train_stats, val_stats,
        train_losses_by_epoch, val_loss_by_epoch, global_step, final_adapter_dir,
    )
    with open(os.path.join(args.output_dir, "SFT_STAGE2_TRAINING_SUMMARY.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    _write_training_summary_md(args.output_dir, summary)
    print(f"[sft-train] done. Final adapter: {final_adapter_dir}")
    return 0


def _write_sidecars(adapter_dir: str, args, config, *, epoch: int, global_step: int,
                    train_loss: float, val_loss: Optional[float]) -> None:
    try:
        import yaml
        with open(os.path.join(adapter_dir, "config_used.yaml"), "w", encoding="utf-8") as fh:
            yaml.safe_dump({
                "sft_config": vars(args), "model_load_config": config,
                "epoch": epoch, "global_step": global_step,
            }, fh, sort_keys=False, allow_unicode=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[sft-train] WARNING: could not write config_used.yaml: {exc!r}")
    with open(os.path.join(adapter_dir, "trainer_state.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "epoch": epoch, "global_step": global_step, "train_loss": train_loss,
            "val_loss": val_loss, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "trained": True, "loss_masking": "assistant-target-only (see module docstring)",
        }, fh, indent=2)


def _build_training_summary(args, config, train_records, val_records, train_stats, val_stats,
                            train_losses_by_epoch, val_loss_by_epoch, global_step,
                            final_adapter_dir) -> Dict[str, Any]:
    per_epoch = {
        str(ep): {
            "mean_train_loss": sum(losses) / len(losses) if losses else None,
            "num_micro_batches": len(losses),
            "val_loss": val_loss_by_epoch.get(ep),
        }
        for ep, losses in train_losses_by_epoch.items()
    }
    last_epoch = max(val_loss_by_epoch) if val_loss_by_epoch else None
    return {
        "experiment": "stage2_continuation_sft (pure SFT — NO GRPO, NO reward, NO rollouts)",
        "model": args.base_model,
        "lora_config": {
            "r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
            "load_in_4bit": not args.no_4bit, "quant_type": "nf4" if not args.no_4bit else None,
        },
        "resume_checkpoint": args.resume_checkpoint,
        "train_examples_source": len(train_records),
        "train_examples_used_after_tokenize_filter": train_stats["kept"],
        "train_skipped": {k: v for k, v in train_stats.items() if k != "kept"},
        "val_examples_source": len(val_records),
        "val_examples_used_after_tokenize_filter": val_stats["kept"],
        "val_skipped": {k: v for k, v in val_stats.items() if k != "kept"},
        "epochs": args.epochs,
        "global_optimizer_steps": global_step,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "per_epoch": per_epoch,
        "final_train_loss": per_epoch.get(str(args.epochs), {}).get("mean_train_loss"),
        "final_val_loss": val_loss_by_epoch.get(last_epoch) if last_epoch is not None else None,
        "eval_every_epoch": args.eval_every_epoch,
        "loss_masking_used": True,
        "loss_masking_description": (
            "Loss computed ONLY on the assistant messages listed in each "
            "record's loss_target_message_indices (the second gold tool call "
            "+ terminal empty-finish for target_type=continuation). System, "
            "user, and the GIVEN first assistant turn are masked (-100)."
        ),
        "adapter_saved_at": final_adapter_dir,
        "full_model_save": False,
    }


def _write_training_summary_md(out_dir: str, s: Dict[str, Any]) -> None:
    lines = []
    w = lines.append
    w("# Stage2 Continuation SFT — Training Summary")
    w("")
    w(f"- experiment: {s['experiment']}")
    w(f"- model: `{s['model']}`")
    w(f"- LoRA: r={s['lora_config']['r']} alpha={s['lora_config']['alpha']} "
      f"dropout={s['lora_config']['dropout']} 4bit={s['lora_config']['load_in_4bit']} "
      f"quant={s['lora_config']['quant_type']}")
    w(f"- resume_checkpoint: {s['resume_checkpoint']}")
    w(f"- train examples (source / used after tokenize+mask filter): "
      f"{s['train_examples_source']} / {s['train_examples_used_after_tokenize_filter']}  "
      f"(skipped: {s['train_skipped']})")
    w(f"- val examples (source / used): {s['val_examples_source']} / "
      f"{s['val_examples_used_after_tokenize_filter']}  (skipped: {s['val_skipped']})")
    w(f"- epochs: {s['epochs']}, global optimizer steps: {s['global_optimizer_steps']}")
    w(f"- learning_rate: {s['learning_rate']}, batch_size: {s['batch_size']}, "
      f"grad_accum: {s['gradient_accumulation_steps']}, max_seq_len: {s['max_seq_len']}")
    w(f"- final train loss: {s['final_train_loss']}")
    w(f"- final val loss: {s['final_val_loss']} (eval_every_epoch={s['eval_every_epoch']})")
    w(f"- loss masking used: {s['loss_masking_used']} — {s['loss_masking_description']}")
    w(f"- adapter checkpoint (NOT full model): `{s['adapter_saved_at']}`")
    w("")
    w("## Per-epoch")
    w("")
    w("| epoch | mean train loss | val loss | micro-batches |")
    w("|---|---:|---:|---:|")
    for ep, d in s["per_epoch"].items():
        w(f"| {ep} | {d['mean_train_loss']} | {d['val_loss']} | {d['num_micro_batches']} |")
    w("")
    with open(os.path.join(out_dir, "SFT_STAGE2_TRAINING_SUMMARY.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
