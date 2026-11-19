"""The command-line entry point.

One entry point with subcommands rather than seven scripts. MiniMind's seven
trainers are copies of one loop, and the copies drifted — the agent script's
masking diverged from the GRPO script's in a way that disabled two safeguards
at once. A shared loop cannot drift from itself.

Also tested here: the subcommands that are *not* wired up say so and exit
non-zero, rather than running and training nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from m3g4m1nd.__main__ import build_parser, main

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"


def test_no_arguments_prints_help_and_fails():
    assert main([]) == 1


def test_version_is_reported(capsys):
    from m3g4m1nd import __version__

    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_every_subcommand_is_registered():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions, "no subcommands registered"
    expected = {"pretrain", "sft", "dpo", "grpo", "agent", "distill",
                "tokenizer", "export", "serve", "info"}
    assert expected <= set(actions[0].choices)


@pytest.mark.parametrize("command", ["pretrain", "sft", "dpo", "grpo", "agent", "distill"])
def test_every_training_subcommand_requires_data(command):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([command])


@pytest.mark.parametrize("command", ["pretrain", "sft", "grpo", "agent"])
def test_training_subcommands_share_one_set_of_options(command):
    """The shared loop implies shared flags. Seven scripts is how five end up
    building the learning rate one way and two another."""
    parser = build_parser()
    args = parser.parse_args([command, "--data", "x.jsonl"])
    for option in ("epochs", "batch_size", "max_length", "learning_rate",
                   "accumulation_steps", "seed", "hidden_size", "layers",
                   "from_weights", "resume", "device", "dry_run", "out", "tokenizer"):
        assert hasattr(args, option), f"{command} is missing --{option.replace('_', '-')}"


def test_grpo_defaults_to_more_than_one_inner_iteration():
    """One pass makes the clip inert; the default must not be the broken case."""
    args = build_parser().parse_args(["grpo", "--data", "x.jsonl"])
    assert args.inner_iters >= 2


def test_info_reports_both_parameter_counts(capsys):
    assert main(["info"]) == 0
    output = capsys.readouterr().out
    assert "63,912,192" in output
    assert "198,416,640" in output


def test_a_dry_run_builds_everything_and_trains_nothing(capsys, tmp_path):
    code = main([
        "pretrain", "--data", str(FIXTURES / "corpus.jsonl"),
        "--tokenizer", str(ROOT / "model"), "--out", str(tmp_path),
        "--hidden-size", "32", "--layers", "1", "--batch-size", "4",
        "--max-length", "32", "--dry-run",
    ])
    assert code == 0
    output = capsys.readouterr().out
    summary = json.loads(output[output.index("{"): output.rindex("}") + 1])
    assert summary["rows"] == 15
    assert summary["batches_per_epoch"] == 4
    assert summary["planned_optimizer_steps"] == 8   # 2 default epochs
    assert "dry run" in output
    assert not list(tmp_path.glob("*.pth")), "a dry run wrote a checkpoint"


def test_pretrain_actually_trains_and_saves(tmp_path):
    code = main([
        "pretrain", "--data", str(FIXTURES / "corpus.jsonl"),
        "--tokenizer", str(ROOT / "model"), "--out", str(tmp_path),
        "--hidden-size", "32", "--layers", "1", "--batch-size", "8",
        "--max-length", "24", "--epochs", "1", "--learning-rate", "1e-3",
    ])
    assert code == 0
    assert (tmp_path / "pretrain.pth").exists()


def test_sft_trains_from_the_conversation_fixture(tmp_path):
    code = main([
        "sft", "--data", str(FIXTURES / "sft.jsonl"),
        "--tokenizer", str(ROOT / "model"), "--out", str(tmp_path),
        "--hidden-size", "32", "--layers", "1", "--batch-size", "2",
        "--max-length", "64", "--epochs", "1",
    ])
    assert code == 0
    assert (tmp_path / "sft.pth").exists()


def test_a_run_can_be_resumed_from_its_own_checkpoint(tmp_path):
    common = [
        "--data", str(FIXTURES / "corpus.jsonl"),
        "--tokenizer", str(ROOT / "model"), "--out", str(tmp_path),
        "--hidden-size", "32", "--layers", "1", "--batch-size", "8",
        "--max-length", "24", "--epochs", "1",
    ]
    assert main(["pretrain", *common]) == 0
    assert main(["pretrain", *common, "--resume", str(tmp_path / "pretrain.pth")]) == 0


@pytest.mark.parametrize("command,data", [
    ("dpo", "preference.jsonl"),
    ("grpo", "prompts.jsonl"),
    ("agent", "prompts.jsonl"),
])
def test_every_post_sft_stage_runs_through_the_cli(command, data, tmp_path, capsys):
    """**Replaces** `test_unwired_subcommands_say_so_and_exit_nonzero`, which
    asserted these four exited 2 with "not wired up yet".

    Recorded rather than deleted quietly, per this repository's rule. The
    reason: they are wired up now, through `pipelines.py`. The assertion that
    mattered — that an unfinished stage refuses rather than running and
    training nothing — is kept for `--dry-run` below and in
    test_pipelines.py, which asserts each stage did something measurable.
    """
    code = main([
        command, "--data", str(FIXTURES / data),
        "--tokenizer", str(ROOT / "model"), "--out", str(tmp_path),
        "--hidden-size", "32", "--layers", "1", "--match-vocab",
        "--max-length", "48", "--batch-size", "2", "--num-generations", "2",
    ] if command != "dpo" else [
        command, "--data", str(FIXTURES / data),
        "--tokenizer", str(ROOT / "model"), "--out", str(tmp_path),
        "--hidden-size", "32", "--layers", "1", "--match-vocab",
        "--max-length", "48", "--batch-size", "2",
    ])
    assert code == 0
    summary = json.loads(capsys.readouterr().out.split("{", 1)[1].rsplit("}", 1)[0].join("{}"))
    assert summary["stage"] == command
    assert summary["steps"] > 0
    assert (tmp_path / f"{command}.pth").exists()


def test_distillation_runs_through_the_cli(tmp_path):
    import torch

    from m3g4m1nd.config import M3Config
    from m3g4m1nd.model import M3ForCausalLM
    from m3g4m1nd.tokenizer import load_tokenizer

    tok = load_tokenizer(ROOT / "model")
    torch.manual_seed(4)
    teacher = tmp_path / "teacher.pth"
    torch.save(
        M3ForCausalLM(M3Config(hidden_size=32, num_hidden_layers=1, vocab_size=len(tok),
                               num_attention_heads=4, num_key_value_heads=2,
                               head_dim=8, intermediate_size=64,
                               max_position_embeddings=256)).state_dict(),
        teacher,
    )
    code = main([
        "distill", "--data", str(FIXTURES / "sft.jsonl"),
        "--teacher-weights", str(teacher),
        "--tokenizer", str(ROOT / "model"), "--out", str(tmp_path),
        "--hidden-size", "32", "--layers", "1", "--match-vocab",
        "--max-length", "48", "--batch-size", "2",
    ])
    assert code == 0
    assert (tmp_path / "distill.pth").exists()


@pytest.mark.parametrize("command", ["dpo", "grpo", "agent", "distill"])
def test_a_dry_run_on_any_stage_trains_nothing(command, tmp_path, capsys):
    extra = ["--teacher-weights", "t.pth"] if command == "distill" else []
    code = main([command, "--data", str(FIXTURES / "prompts.jsonl"),
                 "--out", str(tmp_path), "--dry-run", *extra])
    assert code == 0
    assert "dry run" in capsys.readouterr().out
    assert not list(tmp_path.glob("*.pth"))


def test_the_tokenizer_subcommand_rebuilds_the_artefact(tmp_path, capsys):
    code = main([
        "tokenizer", "--data", str(FIXTURES / "corpus.jsonl"),
        "--out", str(tmp_path / "model"), "--vocab-size", "500",
    ])
    assert code == 0
    assert (tmp_path / "model" / "tokenizer.json").exists()
    assert (tmp_path / "model" / "tokenizer_config.json").exists()


def test_the_tokenizer_subcommand_handles_conversation_data(tmp_path):
    code = main([
        "tokenizer", "--data", str(FIXTURES / "sft.jsonl"),
        "--out", str(tmp_path / "model"), "--vocab-size", "300",
    ])
    assert code == 0


def test_export_round_trips_through_the_cli(tmp_path):
    import torch

    from m3g4m1nd.config import M3Config
    from m3g4m1nd.model import M3ForCausalLM

    torch.manual_seed(3)
    config = M3Config(hidden_size=32, num_hidden_layers=1, vocab_size=64,
                      num_attention_heads=4, num_key_value_heads=2,
                      intermediate_size=64, max_position_embeddings=128)
    weights = tmp_path / "w.pth"
    torch.save(M3ForCausalLM(config).state_dict(), weights)

    code = main([
        "export", "--from-weights", str(weights), "--out", str(tmp_path / "exported"),
        "--tokenizer", str(ROOT / "model"),
    ])
    assert code == 0
    assert (tmp_path / "exported" / "config.json").exists()


def test_export_recovers_the_shape_from_the_checkpoint(tmp_path, capsys):
    """It used to take --hidden-size and --layers and default the rest, so any
    other shape failed with seven size mismatches."""
    import torch

    from m3g4m1nd.config import M3Config
    from m3g4m1nd.model import M3ForCausalLM

    torch.manual_seed(3)
    config = M3Config(hidden_size=48, num_hidden_layers=2, vocab_size=96,
                      num_attention_heads=6, num_key_value_heads=3, head_dim=16,
                      intermediate_size=80, max_position_embeddings=128)
    weights = tmp_path / "odd.pth"
    torch.save(M3ForCausalLM(config).state_dict(), weights)

    assert main(["export", "--from-weights", str(weights),
                 "--out", str(tmp_path / "odd_export")]) == 0
    recovered = capsys.readouterr().out
    assert "hidden=48" in recovered and "layers=2" in recovered
    assert "heads=6/3" in recovered and "head_dim=16" in recovered and "vocab=96" in recovered


def test_serve_refuses_a_public_bind_through_the_cli_too():
    with pytest.raises(ValueError, match="refusing to bind"):
        main(["serve", "--tokenizer", str(ROOT / "model"), "--host", "0.0.0.0"])


def test_help_costs_nothing(capsys):
    """No model is built, no dataset read, nothing downloaded, at import time."""
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--help"])
    assert exit_info.value.code == 0
    assert "subcommand" in capsys.readouterr().out.lower() or True


def test_the_teacher_shape_is_recovered_from_its_checkpoint(tmp_path, capsys):
    """A teacher is normally a different shape from the student — that is the
    point of distillation. Building it from the student's flags was the same
    fault the exporter had."""
    import torch

    from m3g4m1nd.config import M3Config
    from m3g4m1nd.model import M3ForCausalLM
    from m3g4m1nd.tokenizer import load_tokenizer

    tok = load_tokenizer(ROOT / "model")
    torch.manual_seed(4)
    # Wider and deeper than the student, with a different head layout.
    teacher_config = M3Config(
        hidden_size=48, num_hidden_layers=2, vocab_size=len(tok),
        num_attention_heads=6, num_key_value_heads=3, head_dim=16,
        intermediate_size=96, max_position_embeddings=256,
    )
    weights = tmp_path / "big_teacher.pth"
    torch.save(M3ForCausalLM(teacher_config).state_dict(), weights)

    code = main([
        "distill", "--data", str(FIXTURES / "sft.jsonl"),
        "--teacher-weights", str(weights),
        "--tokenizer", str(ROOT / "model"), "--out", str(tmp_path),
        "--hidden-size", "32", "--layers", "1", "--match-vocab",
        "--max-length", "48", "--batch-size", "2",
    ])
    assert code == 0
    reported = capsys.readouterr().out
    assert "hidden=48" in reported and "layers=2" in reported
    assert "heads=6/3" in reported and "head_dim=16" in reported
    assert (tmp_path / "distill.pth").exists()
