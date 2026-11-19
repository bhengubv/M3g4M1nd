"""One command-line entry point with subcommands.

MiniMind ships seven training scripts, each ~170 lines, each a copy of the
same loop with small divergences: five of them build the learning rate one way
and two another, three save the checkpoint one way and four another, and the
agent script's masking diverged from the GRPO script's in a way that disabled
two safeguards. Seven copies is how that happens.

So there is one loop in `train.py` and one entry point here. Each subcommand
assembles a model, a dataset and a loss, and hands them over.

    python -m m3g4m1nd pretrain --data dataset/corpus.jsonl --epochs 2
    python -m m3g4m1nd sft      --data dataset/sft.jsonl --from-weights out/pretrain.pth
    python -m m3g4m1nd tokenizer --data dataset/corpus.jsonl --vocab-size 6400
    python -m m3g4m1nd export   --from-weights out/sft.pth --out exported/
    python -m m3g4m1nd serve    --weights out/sft.pth
    python -m m3g4m1nd info

Nothing here starts a run on import, so `--help` costs nothing and the tests
can exercise argument parsing without touching a GPU or a dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="m3g4m1nd",
        description="Train, serve and export a small language model.",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    subparsers = parser.add_subparsers(dest="command")

    def shared(sub, *, default_epochs: int = 1, default_lr: float = 5e-4):
        sub.add_argument("--data", type=Path, required=True, help="jsonl training data")
        sub.add_argument("--tokenizer", type=Path, default=Path("model"))
        sub.add_argument("--out", type=Path, default=Path("out"))
        sub.add_argument("--epochs", type=int, default=default_epochs)
        sub.add_argument("--batch-size", type=int, default=8)
        sub.add_argument("--max-length", type=int, default=512)
        sub.add_argument("--learning-rate", type=float, default=default_lr)
        sub.add_argument("--accumulation-steps", type=int, default=1)
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--hidden-size", type=int, default=768)
        sub.add_argument("--layers", type=int, default=8)
        sub.add_argument("--use-moe", action="store_true")
        sub.add_argument("--from-weights", type=Path, default=None)
        sub.add_argument("--resume", type=Path, default=None, help="checkpoint to continue")
        sub.add_argument("--device", default="cpu")
        sub.add_argument(
            "--dry-run", action="store_true",
            help="build everything, report the plan, and stop without training",
        )
        sub.add_argument(
            "--match-vocab", action="store_true",
            help="size the embedding to the tokenizer rather than the config default",
        )
        return sub

    shared(subparsers.add_parser("pretrain", help="next-token pretraining"), default_epochs=2)
    shared(subparsers.add_parser("sft", help="supervised fine-tuning"), default_lr=1e-5)

    dpo = shared(subparsers.add_parser("dpo", help="direct preference optimisation"),
                 default_lr=5e-8)
    dpo.add_argument("--beta", type=float, default=0.1)

    grpo = shared(subparsers.add_parser("grpo", help="group-relative policy optimisation"),
                  default_lr=3e-7)
    grpo.add_argument("--num-generations", type=int, default=6)
    grpo.add_argument("--inner-iters", type=int, default=2,
                      help="passes per rollout; 1 makes the clip inert and is refused")
    grpo.add_argument("--epsilon", type=float, default=0.2)
    grpo.add_argument("--loss-type", choices=["grpo", "cispo"], default="grpo")

    agent = shared(subparsers.add_parser("agent", help="multi-turn tool-use RL"),
                   default_lr=3e-7)
    agent.add_argument("--num-generations", type=int, default=4)
    agent.add_argument("--max-turns", type=int, default=3)
    agent.add_argument("--inner-iters", type=int, default=2)

    distil = shared(subparsers.add_parser("distill", help="white-box distillation"),
                    default_lr=5e-6)
    distil.add_argument("--teacher-weights", type=Path, required=True)
    distil.add_argument("--temperature", type=float, default=1.5)
    distil.add_argument("--alpha", type=float, default=0.5)

    tok = subparsers.add_parser("tokenizer", help="train a byte-level BPE tokenizer")
    tok.add_argument("--data", type=Path, required=True)
    tok.add_argument("--out", type=Path, default=Path("model"))
    tok.add_argument("--vocab-size", type=int, default=6400)

    export = subparsers.add_parser("export", help="write a Qwen3-shaped checkpoint")
    export.add_argument("--from-weights", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--tokenizer", type=Path, default=Path("model"))
    # No shape flags: the checkpoint is read for its own shape.

    serve_cmd = subparsers.add_parser("serve", help="run the OpenAI-compatible endpoint")
    serve_cmd.add_argument("--weights", type=Path, default=None)
    serve_cmd.add_argument("--tokenizer", type=Path, default=Path("model"))
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8998)
    serve_cmd.add_argument(
        "--allow-public", action="store_true",
        help="bind beyond loopback; there is no authentication, so say it out loud",
    )

    subparsers.add_parser("info", help="report the model shape and parameter count")
    return parser


def _config(args, tokenizer=None):
    """Build the model config from the flags.

    `--match-vocab` is honoured here rather than in one caller. It used to be
    applied only by the RL stages, so `pretrain --match-vocab` accepted the
    flag and ignored it: the embedding was sized 6,400 against a 733-token
    tokenizer, wasting 725k parameters and making every later strict load
    fail. A flag that is accepted and ignored is the defect this project was
    built to answer, and it keeps turning up in my own code.
    """
    from .config import M3Config

    settings = {
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.layers,
        "use_moe": getattr(args, "use_moe", False),
    }
    if getattr(args, "match_vocab", False):
        if tokenizer is None:
            raise ValueError("--match-vocab needs a tokenizer to match against")
        settings["vocab_size"] = len(tokenizer)
    return M3Config(**settings)


def _dataset(args, tokenizer, kind: str):
    from .data import PretrainDataset, SFTDataset

    if kind == "pretrain":
        return PretrainDataset(args.data, tokenizer, max_length=args.max_length)
    return SFTDataset(args.data, tokenizer, max_length=args.max_length)


def command_info(args) -> int:
    from .model import M3ForCausalLM, count_parameters

    for use_moe in (False, True):
        config = _config(argparse.Namespace(
            hidden_size=768, layers=8, use_moe=use_moe, match_vocab=False
        ))
        total, trainable = count_parameters(M3ForCausalLM(config))
        label = "moe (4 experts, top-1)" if use_moe else "dense"
        print(f"{label:24} {total:>12,} parameters  ({trainable:,} trainable)")
    return 0


def command_tokenizer(args) -> int:
    from .data import read_jsonl
    from .tokenizer import train_tokenizer

    rows = read_jsonl(args.data)
    texts = [str(row.get("text", "")) for row in rows]
    if not any(texts):
        texts = [
            "\n".join(m.get("content", "") for m in row.get("conversations", []))
            for row in rows
        ]
    out = train_tokenizer(texts, args.out, vocab_size=args.vocab_size)
    print(f"tokenizer written to {out}")
    return 0


def command_export(args) -> int:
    import torch

    from .checkpoints import read_state_dict
    from .export import config_from_state_dict, export_qwen3
    from .model import M3ForCausalLM

    state = read_state_dict(args.from_weights)
    # The checkpoint knows its own shape; do not ask the operator to retype it.
    config = config_from_state_dict(state)
    print(f"recovered shape: hidden={config.hidden_size} layers={config.num_hidden_layers} "
          f"heads={config.num_attention_heads}/{config.num_key_value_heads} "
          f"head_dim={config.head_dim} vocab={config.vocab_size}")
    model = M3ForCausalLM(config)
    model.load_state_dict(state, strict=True)
    out = export_qwen3(model, args.out, tokenizer_dir=args.tokenizer)
    print(f"exported to {out}")
    return 0


def command_serve(args) -> int:
    from . import serve as serving

    if args.weights is not None:
        import torch

        from .checkpoints import read_state_dict
        from .export import config_from_state_dict
        from .model import M3ForCausalLM

        state = read_state_dict(args.weights)
        model = M3ForCausalLM(config_from_state_dict(state))
        model.load_state_dict(state, strict=True)
        serving.model = model.eval()
    serving.tokenizer = __import__(
        "m3g4m1nd.tokenizer", fromlist=["load_tokenizer"]
    ).load_tokenizer(args.tokenizer)
    serving.serve(host=args.host, port=args.port, allow_public=args.allow_public)
    return 0


def command_train(args, kind: str) -> int:
    import torch

    from .checkpoints import load_weights
    from .losses import masked_cross_entropy
    from .model import M3ForCausalLM
    from .tokenizer import load_tokenizer
    from .train import StepPlan, Trainer

    tokenizer = load_tokenizer(args.tokenizer)
    model = M3ForCausalLM(_config(args, tokenizer))
    if args.from_weights is not None:
        load_weights(model, args.from_weights)
        print(f"loaded weights from {args.from_weights}")

    dataset = _dataset(args, tokenizer, "pretrain" if kind == "pretrain" else "sft")
    plan = StepPlan(
        num_rows=len(dataset),
        batch_size=args.batch_size,
        epochs=args.epochs,
        accumulation_steps=args.accumulation_steps,
    )

    def loss_fn(net, rows):
        input_ids = torch.stack([ids for ids, _ in rows]).to(args.device)
        labels = torch.stack([lab for _, lab in rows]).to(args.device)
        return masked_cross_entropy(net(input_ids).logits, labels)

    trainer = Trainer(
        model.to(args.device),
        torch.optim.AdamW(model.parameters(), lr=args.learning_rate),
        dataset,
        loss_fn,
        plan,
        seed=args.seed,
    )
    if args.resume is not None:
        trainer.load(args.resume)
        print(f"resumed from {args.resume} at epoch {trainer.state.epoch}")

    summary = {
        "command": kind,
        "rows": len(dataset),
        "batches_per_epoch": plan.batches_per_epoch,
        "planned_optimizer_steps": plan.planned_optimizer_steps,
        "device": args.device,
    }
    print(json.dumps(summary, indent=2))

    if args.dry_run:
        print("dry run: nothing trained")
        return 0

    trainer.run()
    trainer.verify_step_accounting()
    args.out.mkdir(parents=True, exist_ok=True)
    trainer.save(args.out / f"{kind}.pth")
    print(f"saved {args.out / f'{kind}.pth'}")
    return 0


def command_stage(args, kind: str) -> int:
    """Run one of the four post-SFT stages.

    Each assembles a model, a reference or teacher, a dataset and an optimiser,
    and hands them to `pipelines.py`. The loop itself lives there and is shared,
    so these four cannot drift apart the way MiniMind's seven scripts did.
    """
    import torch

    from .checkpoints import load_weights, read_state_dict
    from .data import PreferenceDataset, PromptDataset, SFTDataset
    from .model import M3ForCausalLM
    from .pipelines import run_agent, run_distillation, run_dpo, run_grpo
    from .tokenizer import load_tokenizer

    tokenizer = load_tokenizer(args.tokenizer)
    config = _config(args, tokenizer)

    def build() -> "M3ForCausalLM":
        model = M3ForCausalLM(config)
        if args.from_weights is not None:
            load_weights(model, args.from_weights)
        return model.to(args.device)

    model = build()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if kind == "dpo":
        dataset = PreferenceDataset(args.data, tokenizer, max_length=args.max_length)
        result = run_dpo(
            model, build().eval(), dataset, optimizer,
            epochs=args.epochs, batch_size=args.batch_size, beta=args.beta,
        )
    elif kind == "distill":
        # The teacher's checkpoint knows its own shape, and a teacher is
        # normally a *different* shape from the student — that is the point of
        # distillation. Building it from the student's flags was the same fault
        # the exporter had.
        from .export import config_from_state_dict

        teacher_state = read_state_dict(args.teacher_weights)
        teacher_config = config_from_state_dict(teacher_state)
        print(f"teacher shape: hidden={teacher_config.hidden_size} "
              f"layers={teacher_config.num_hidden_layers} "
              f"heads={teacher_config.num_attention_heads}/"
              f"{teacher_config.num_key_value_heads} "
              f"head_dim={teacher_config.head_dim}")
        teacher = M3ForCausalLM(teacher_config)
        teacher.load_state_dict(teacher_state, strict=True)
        dataset = SFTDataset(args.data, tokenizer, max_length=args.max_length)
        result = run_distillation(
            model, teacher.to(args.device).eval(), dataset, optimizer,
            epochs=args.epochs, batch_size=args.batch_size,
            temperature=args.temperature, alpha=args.alpha,
        )
    else:
        dataset = PromptDataset(args.data, tokenizer, max_length=args.max_length)
        runner = run_grpo if kind == "grpo" else run_agent
        extra = (
            {"epsilon": args.epsilon, "loss_type": args.loss_type}
            if kind == "grpo"
            else {"max_turns": args.max_turns}
        )
        result = runner(
            model, build().eval(), dataset, optimizer, tokenizer,
            epochs=args.epochs, batch_size=args.batch_size,
            num_generations=args.num_generations, inner_iters=args.inner_iters,
            **extra,
        )

    print(json.dumps({
        "stage": result.stage,
        "steps": result.steps,
        "final_loss": result.final_loss,
        **{k: round(v, 6) for k, v in result.metrics.items()},
    }, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {k: v.cpu() for k, v in model.state_dict().items()}, args.out / f"{kind}.pth"
    )
    print(f"saved {args.out / f'{kind}.pth'}")
    return 0


def command_not_implemented(kind: str) -> int:
    print(
        f"'{kind}' is not wired up yet. Its loss and updater exist and are tested "
        f"(see docs/ACCEPTANCE.md); what is missing is the rollout plumbing that "
        f"joins them. Saying so beats a subcommand that runs and trains nothing.",
        file=sys.stderr,
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__

        print(__version__)
        return 0
    if args.command is None:
        parser.print_help()
        return 1

    if args.command in {"pretrain", "sft"}:
        return command_train(args, args.command)
    if args.command in {"dpo", "grpo", "agent", "distill"}:
        if getattr(args, "dry_run", False):
            print(f"dry run: {args.command} would run; nothing trained")
            return 0
        return command_stage(args, args.command)
    return {
        "tokenizer": command_tokenizer,
        "export": command_export,
        "serve": command_serve,
        "info": command_info,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
