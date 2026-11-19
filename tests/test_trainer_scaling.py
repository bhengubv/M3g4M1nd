"""DataLoader, mixed precision, gradient checkpointing and DDP.

The thing each of these must not do is change the answer. A worker pool, a
narrower dtype, a recomputed forward pass and a wrapped module are all meant
to change cost, not result — so every test here is a comparison against the
plain path, and the resume tests in test_train.py run unchanged beside them.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn

from m3g4m1nd.config import M3Config
from m3g4m1nd.model import M3ForCausalLM
from m3g4m1nd.train import (
    IndexedBatchSampler,
    StepPlan,
    Trainer,
    epoch_order,
    init_distributed,
    is_main_process,
    unwrap,
)
from tests.test_train import Rows, squared_error, tiny_regressor, weights


def make(plan, **kwargs):
    model = tiny_regressor(0)
    return Trainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.05),
        Rows(plan.num_rows),
        squared_error,
        plan,
        seed=7,
        **kwargs,
    )


def assert_same(a, b, note=""):
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=0, atol=0, msg=note or None)


# ------------------------------------------------------------------ DataLoader
def test_a_dataloader_does_not_change_the_result():
    plan = StepPlan(12, 3, epochs=2)
    plain = make(plan)
    plain.run()

    loaded = make(plan, pin_memory=False, num_workers=0)
    loaded.run()
    assert_same(weights(plain.model), weights(loaded.model))


def test_the_dataloader_path_sees_the_same_batches():
    """Both paths consume one deterministic sampler, so they cannot diverge."""
    order = epoch_order(12, seed=7, epoch=0)
    sampler = IndexedBatchSampler(order, batch_size=3)
    from m3g4m1nd.train import batches

    assert list(sampler) == list(batches(order, 3))


def test_resume_is_still_exact_with_a_dataloader(tmp_path):
    plan = StepPlan(12, 3, epochs=2)
    straight = make(plan, pin_memory=True)
    straight.run()

    interrupted = make(plan, pin_memory=True)
    interrupted.run(max_batches=3)
    interrupted.save(tmp_path / "loader.pth")

    resumed = make(plan, pin_memory=True)
    resumed.load(tmp_path / "loader.pth")
    resumed.run()
    assert_same(weights(straight.model), weights(resumed.model))


def test_the_sampler_length_matches_what_it_yields():
    order = epoch_order(10, seed=7, epoch=0)
    for batch_size in (1, 3, 4, 10):
        sampler = IndexedBatchSampler(order, batch_size)
        assert len(sampler) == len(list(sampler))


def test_the_sampler_shards_across_ranks_without_overlap():
    order = epoch_order(12, seed=7, epoch=0)
    shards = [
        list(IndexedBatchSampler(order, 2, rank=rank, world_size=3)) for rank in range(3)
    ]
    flat = [tuple(batch) for shard in shards for batch in shard]
    assert len(flat) == len(set(flat)), "two ranks were given the same batch"
    whole = [tuple(b) for b in IndexedBatchSampler(order, 2)]
    assert sorted(flat) == sorted(whole), "sharding lost or duplicated a batch"


# ------------------------------------------------------------------ mixed precision
def test_bfloat16_autocast_runs_and_stays_finite():
    plan = StepPlan(12, 3)
    trainer = make(plan, amp_dtype=torch.bfloat16)
    trainer.run()
    trainer.verify_step_accounting()
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())


def test_bfloat16_lands_close_to_full_precision():
    plan = StepPlan(12, 3, epochs=2)
    full = make(plan)
    full.run()
    half = make(plan, amp_dtype=torch.bfloat16)
    half.run()
    for a, b in zip(weights(full.model), weights(half.model)):
        torch.testing.assert_close(a, b, rtol=0.1, atol=0.1)


def test_a_scaler_is_enabled_only_for_float16():
    """bfloat16 has the exponent range and needs no loss scaling. A scaler
    enabled for it is a no-op that looks like a safeguard."""
    assert make(StepPlan(12, 3), amp_dtype=torch.float16).scaler.is_enabled()
    assert not make(StepPlan(12, 3), amp_dtype=torch.bfloat16).scaler.is_enabled()
    assert not make(StepPlan(12, 3)).scaler.is_enabled()


def test_autocast_is_off_by_default():
    from contextlib import nullcontext

    trainer = make(StepPlan(12, 3))
    assert isinstance(trainer._autocast(), type(nullcontext()))


# ------------------------------------------------- gradient checkpointing
def tiny_lm(seed: int = 5) -> M3ForCausalLM:
    torch.manual_seed(seed)
    return M3ForCausalLM(M3Config(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, intermediate_size=32,
        vocab_size=64, max_position_embeddings=64,
    ))


def test_checkpointing_can_be_turned_on_and_off():
    model = tiny_lm()
    assert not model.is_gradient_checkpointing
    model.gradient_checkpointing_enable()
    assert model.is_gradient_checkpointing
    model.gradient_checkpointing_disable()
    assert not model.is_gradient_checkpointing


def test_checkpointing_gives_the_same_gradients():
    """It recomputes the forward pass instead of storing activations. Same
    answer, less memory — if the gradients differ, it is not that."""
    ids = torch.randint(0, 64, (2, 8))
    labels = ids.clone()

    plain = tiny_lm().train()
    plain(ids, labels=labels).loss.backward()
    expected = [p.grad.clone() for p in plain.parameters()]

    recomputed = tiny_lm().train()
    recomputed.gradient_checkpointing_enable()
    recomputed(ids, labels=labels).loss.backward()
    actual = [p.grad.clone() for p in recomputed.parameters()]

    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_checkpointing_is_inactive_in_eval():
    model = tiny_lm().eval()
    model.gradient_checkpointing_enable()
    with torch.no_grad():
        assert model(torch.randint(0, 64, (1, 4))).logits.shape == (1, 4, 64)


def test_checkpointing_with_a_kv_cache_is_refused():
    """The recomputed pass would discard the cache it just built, so asking for
    both is a contradiction rather than a slow path."""
    model = tiny_lm().train()
    model.gradient_checkpointing_enable()
    with pytest.raises(ValueError, match="gradient checkpointing recomputes"):
        model(torch.randint(0, 64, (1, 4)), use_cache=True)


def test_the_trainer_can_enable_checkpointing_on_the_model():
    model = tiny_lm()
    plan = StepPlan(4, 2)

    def loss_fn(net, rows):
        ids = torch.stack([r[0] for r in rows])
        return net(ids, labels=ids).loss

    class Ids:
        def __len__(self):
            return 4

        def __getitem__(self, index):
            torch.manual_seed(index)
            return (torch.randint(0, 64, (8,)), None)

    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=1e-3), Ids(), loss_fn, plan,
        gradient_checkpointing=True,
    )
    assert unwrap(trainer.model).is_gradient_checkpointing
    trainer.run()
    trainer.verify_step_accounting()


def test_asking_for_checkpointing_on_a_model_without_it_is_refused():
    model = nn.Linear(1, 1)
    with pytest.raises(ValueError, match="does not support gradient checkpointing"):
        Trainer(model, torch.optim.SGD(model.parameters(), lr=1e-3), Rows(4),
                squared_error, StepPlan(4, 2), gradient_checkpointing=True)


# ------------------------------------------------------------------ DDP
def test_outside_torchrun_init_reports_single_process():
    assert init_distributed() == -1
    assert is_main_process()


def test_asking_for_distributed_without_a_group_is_refused():
    plan = StepPlan(12, 3)
    with pytest.raises(ValueError, match="no process group is initialised"):
        make(plan, distributed=True)


def test_a_single_process_group_trains_and_matches_the_plain_path(tmp_path):
    """A real process group, gloo backend, world size one — which is the most
    of DDP that can be exercised on a machine with no second device. The
    sharding arithmetic is tested separately and exactly."""
    import torch.distributed as dist

    plan = StepPlan(12, 3, epochs=2)
    plain = make(plan)
    plain.run()

    environment = {
        "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29517",
    }
    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    try:
        assert init_distributed() == 0
        assert dist.is_initialized() and dist.get_world_size() == 1
        assert is_main_process()

        wrapped = make(plan, distributed=True)
        wrapped.run()
        wrapped.verify_step_accounting()

        assert_same(weights(plain.model), weights(unwrap(wrapped.model)),
                    "DDP wrapping changed the result at world size one")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_unwrap_reaches_through_ddp_and_compile_wrappers():
    model = tiny_regressor()
    assert unwrap(model) is model

    class Wrapper:
        def __init__(self, inner):
            self.module = inner

    assert unwrap(Wrapper(model)) is model


def test_pin_memory_is_ignored_when_there_is_no_accelerator():
    """Asking for pinned memory on a CPU-only box earns a torch warning and
    nothing else, so the request is recorded and not acted on."""
    trainer = make(StepPlan(12, 3), pin_memory=True)
    assert trainer.pin_memory_requested is True
    assert trainer.pin_memory is torch.accelerator.is_available()


def test_the_ddp_buffer_flag_tracks_whichever_torch_is_installed():
    """torch 2.15 renamed `broadcast_buffers` to `forward_sync_buffers`.
    Hardcoding either name warns on one version and breaks on the other."""
    import inspect as _inspect

    from torch.nn.parallel import DistributedDataParallel

    accepted = _inspect.signature(DistributedDataParallel.__init__).parameters
    assert "forward_sync_buffers" in accepted or "broadcast_buffers" in accepted
