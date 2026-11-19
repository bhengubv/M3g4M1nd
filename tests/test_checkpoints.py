"""Loading weights strictly.

Both of the faults guarded here were found by *running* the pipeline end to
end, not by testing its parts — which is the argument for doing that at least
once before claiming a thing works.

1. `Trainer.save` writes a resume checkpoint (`model`, `optimizer`, `plan`…).
   The CLI passed it to `load_state_dict(strict=False)`, which matched none of
   the keys, reported nothing, and left the model randomly initialised. An SFT
   run trained thirty epochs from scratch and printed "saved".

2. `--match-vocab` was honoured by the RL stages and ignored by pretrain and
   SFT, so the embedding was sized 6,400 against a 733-token tokenizer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from m3g4m1nd.checkpoints import is_resume_checkpoint, load_weights, read_state_dict
from m3g4m1nd.config import M3Config
from m3g4m1nd.model import M3ForCausalLM

ROOT = Path(__file__).resolve().parent.parent

SMALL = dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
             num_key_value_heads=2, head_dim=8, intermediate_size=64,
             vocab_size=64, max_position_embeddings=128)


def model(**overrides) -> M3ForCausalLM:
    torch.manual_seed(3)
    return M3ForCausalLM(M3Config(**{**SMALL, **overrides}))


def test_a_bare_state_dict_loads():
    source = model()
    target = model()
    with torch.no_grad():
        target.lm_head.weight.add_(1.0)

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "weights.pth"
        torch.save(source.state_dict(), path)
        load_weights(target, path)

    torch.testing.assert_close(target.lm_head.weight, source.lm_head.weight)


def test_a_resume_checkpoint_is_unwrapped_rather_than_rejected(tmp_path):
    source = model()
    payload = {
        "model": source.state_dict(),
        "optimizer": {"state": {}, "param_groups": []},
        "epoch": 1,
        "plan": {"num_rows": 4},
    }
    path = tmp_path / "resume.pth"
    torch.save(payload, path)

    assert is_resume_checkpoint(torch.load(path, weights_only=True))
    target = model()
    load_weights(target, path)
    torch.testing.assert_close(target.lm_head.weight, source.lm_head.weight)


def test_a_checkpoint_supplying_nothing_is_an_error_at_any_strictness(tmp_path):
    """The exact silent failure: a file full of keys the model does not want.
    `strict=False` used to accept it and load nothing."""
    path = tmp_path / "junk.pth"
    torch.save({"something.else": torch.zeros(2)}, path)

    for strict in (True, False):
        with pytest.raises(ValueError, match="supplied none of the model's"):
            load_weights(model(), path, strict=strict)


def test_a_non_weights_file_is_named_as_such(tmp_path):
    path = tmp_path / "notweights.pth"
    torch.save({"epoch": 3, "seed": 42}, path)
    with pytest.raises(ValueError, match="is not weights"):
        read_state_dict(path)


def test_an_empty_file_is_refused(tmp_path):
    path = tmp_path / "empty.pth"
    torch.save({}, path)
    with pytest.raises(ValueError, match="does not contain a state dict"):
        read_state_dict(path)


def test_a_wrong_shape_is_reported_clearly_not_as_a_wall_of_sizes(tmp_path):
    path = tmp_path / "wide.pth"
    torch.save(model(hidden_size=64, head_dim=16).state_dict(), path)

    with pytest.raises(ValueError, match="parameters of the wrong shape") as caught:
        load_weights(model(), path)
    assert "config_from_state_dict" in str(caught.value), (
        "the error should point at the function that recovers the right shape"
    )


def test_a_partial_match_is_refused_when_strict(tmp_path):
    full = model().state_dict()
    del full["model.norm.weight"]
    path = tmp_path / "partial.pth"
    torch.save(full, path)

    with pytest.raises(ValueError, match="does not fit this model"):
        load_weights(model(), path, strict=True)

    # Non-strict is allowed to accept it, because something did load.
    load_weights(model(), path, strict=False)


def test_the_trainer_writes_weights_beside_the_resume_checkpoint(tmp_path):
    """A resume checkpoint and a set of weights are different things, and
    conflating them is what caused the silent failure."""
    from tests.test_train import Rows, squared_error, tiny_regressor
    from m3g4m1nd.train import StepPlan, Trainer

    net = tiny_regressor()
    trainer = Trainer(net, torch.optim.SGD(net.parameters(), lr=1e-3),
                      Rows(12), squared_error, StepPlan(12, 3))
    trainer.run(max_batches=1)
    trainer.save(tmp_path / "stage.pth")

    assert (tmp_path / "stage.pth").exists()
    assert (tmp_path / "stage_weights.pth").exists()
    assert is_resume_checkpoint(torch.load(tmp_path / "stage.pth", weights_only=True))
    assert not is_resume_checkpoint(
        torch.load(tmp_path / "stage_weights.pth", weights_only=True)
    )


def test_match_vocab_is_honoured_by_every_stage():
    """It used to apply only to the RL stages, so `pretrain --match-vocab`
    accepted the flag and sized the embedding 6,400 against a 733-token
    tokenizer — 725k wasted parameters and a later strict load that could not
    succeed."""
    import argparse

    from m3g4m1nd.__main__ import _config
    from m3g4m1nd.tokenizer import load_tokenizer

    tok = load_tokenizer(ROOT / "model")
    args = argparse.Namespace(hidden_size=32, layers=1, use_moe=False, match_vocab=True)
    assert _config(args, tok).vocab_size == len(tok)

    args.match_vocab = False
    assert _config(args, tok).vocab_size == 6400


def test_match_vocab_without_a_tokenizer_is_refused():
    import argparse

    from m3g4m1nd.__main__ import _config

    args = argparse.Namespace(hidden_size=32, layers=1, use_moe=False, match_vocab=True)
    with pytest.raises(ValueError, match="needs a tokenizer"):
        _config(args, None)


def test_pretrain_then_sft_actually_carries_the_weights_over(tmp_path):
    """The end-to-end run that found the bug, as a test.

    If the hand-off silently fails, SFT starts from random initialisation and
    its first loss is the untrained loss. That is the signal.
    """
    from m3g4m1nd.__main__ import main
    from m3g4m1nd.data import PretrainDataset
    from m3g4m1nd.losses import masked_cross_entropy
    from m3g4m1nd.tokenizer import load_tokenizer

    tok = load_tokenizer(ROOT / "model")
    fixtures = ROOT / "tests" / "fixtures"
    shape = ["--hidden-size", "32", "--layers", "1", "--match-vocab",
             "--tokenizer", str(ROOT / "model"), "--out", str(tmp_path)]

    assert main(["pretrain", "--data", str(fixtures / "corpus.jsonl"), *shape,
                 "--max-length", "32", "--batch-size", "5", "--epochs", "15",
                 "--learning-rate", "3e-3"]) == 0

    config = M3Config(hidden_size=32, num_hidden_layers=1, vocab_size=len(tok))
    dataset = PretrainDataset(fixtures / "corpus.jsonl", tok, max_length=32)
    ids = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.stack([dataset[i][1] for i in range(len(dataset))])

    def loss_of(path):
        net = M3ForCausalLM(config)
        load_weights(net, path)
        net.eval()
        with torch.no_grad():
            return masked_cross_entropy(net(ids).logits, labels).item()

    torch.manual_seed(0)
    untrained = M3ForCausalLM(config).eval()
    with torch.no_grad():
        baseline = masked_cross_entropy(untrained(ids).logits, labels).item()

    # The threshold detects a *silent total failure*, where nothing loaded and
    # the loss sits at the random baseline. It is not a quality benchmark: a
    # first pass at `baseline / 2` failed at 6.593 -> 3.412, which is training
    # working and an assertion miscalibrated.
    trained = loss_of(tmp_path / "pretrain_weights.pth")
    assert trained < baseline * 0.75, (
        f"pretraining barely moved: {baseline:.3f} -> {trained:.3f}"
    )

    assert main(["sft", "--data", str(fixtures / "sft.jsonl"), *shape,
                 "--from-weights", str(tmp_path / "pretrain_weights.pth"),
                 "--max-length", "48", "--batch-size", "3", "--epochs", "1",
                 "--learning-rate", "1e-5"]) == 0

    # One epoch at 1e-5 cannot undo pretraining, so if the hand-off worked the
    # SFT output is still far better than random on the pretraining data.
    after = loss_of(tmp_path / "sft_weights.pth")
    assert after < baseline * 0.75, (
        f"the pretrained weights did not reach SFT: {after:.3f} against a "
        f"random baseline of {baseline:.3f}. A hand-off that silently fails "
        f"lands at the baseline, which is what this number is watching for."
    )
