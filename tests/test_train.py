"""Rows B8 and B9 — resume reproduces an uninterrupted run, and the step
count is planned and checked.

Neither was tested in MiniMind. Resume is the most intricate code there and
the property it exists to provide — that an interrupted run and an
uninterrupted one end in the same place — was never asserted once.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from m3g4m1nd.train import StepPlan, Trainer, batches, epoch_order


class Rows:
    """A tiny deterministic dataset: row i is the pair (i, i*2)."""

    def __init__(self, n: int = 12):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        return torch.tensor([float(index)]), torch.tensor([float(index) * 2.0])


def tiny_regressor(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Linear(1, 1)


def squared_error(model, rows):
    inputs = torch.stack([x for x, _ in rows])
    targets = torch.stack([y for _, y in rows])
    return ((model(inputs) - targets) ** 2).mean()


def make(plan: StepPlan, seed: int = 7, lr: float = 1e-3, model_seed: int = 0, scheduler=False):
    model = tiny_regressor(model_seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=plan.planned_optimizer_steps
        )
        if scheduler
        else None
    )
    return Trainer(model, optimizer, Rows(plan.num_rows), squared_error, plan, seed=seed,
                   scheduler=sched)


def weights(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def assert_same(a, b, note=""):
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=0, atol=0, msg=note or None)


# ---------------------------------------------------------------- the shuffle
def test_the_epoch_order_is_determined_only_by_seed_and_epoch():
    assert epoch_order(12, seed=7, epoch=0) == epoch_order(12, seed=7, epoch=0)
    assert epoch_order(12, seed=7, epoch=0) != epoch_order(12, seed=7, epoch=1)
    assert epoch_order(12, seed=7, epoch=0) != epoch_order(12, seed=8, epoch=0)


def test_the_epoch_order_survives_other_code_drawing_random_numbers():
    """MiniMind reseeded the global RNG per epoch and then consumed it elsewhere,
    which couples the data order to anything else that draws."""
    first = epoch_order(12, seed=7, epoch=0)
    torch.rand(1000)
    torch.manual_seed(999)
    assert epoch_order(12, seed=7, epoch=0) == first


def test_the_epoch_order_is_a_permutation():
    assert sorted(epoch_order(12, seed=7, epoch=3)) == list(range(12))


def test_skipping_batches_lands_on_the_batches_that_follow():
    order = epoch_order(12, seed=7, epoch=0)
    everything = list(batches(order, batch_size=3))
    resumed = list(batches(order, batch_size=3, skip=2))
    assert resumed == everything[2:]


def test_drop_last_discards_only_the_short_tail():
    order = list(range(10))
    assert len(list(batches(order, 3))) == 4
    assert len(list(batches(order, 3, drop_last=True))) == 3


# ---------------------------------------------------------------- B9
@pytest.mark.parametrize(
    "rows,batch,epochs,accum,inner,expected",
    [
        (12, 3, 1, 1, 1, 4),    # 4 batches, step each
        (12, 3, 2, 1, 1, 8),    # two epochs
        (12, 3, 1, 2, 1, 2),    # accumulate pairs
        (12, 3, 1, 4, 1, 1),    # one group
        (10, 3, 1, 1, 1, 4),    # ragged tail still steps
        (10, 3, 1, 2, 1, 2),    # 4 batches -> 2 groups
        (12, 3, 1, 1, 3, 12),   # inner iterations multiply
        (12, 3, 2, 2, 2, 8),    # everything at once
    ],
)
def test_the_plan_predicts_the_optimizer_step_count(rows, batch, epochs, accum, inner, expected):
    plan = StepPlan(rows, batch, epochs=epochs, accumulation_steps=accum, inner_iters=inner)
    assert plan.planned_optimizer_steps == expected


@pytest.mark.parametrize(
    "rows,batch,epochs,accum,inner",
    [(12, 3, 1, 1, 1), (12, 3, 2, 1, 1), (12, 3, 1, 2, 1), (10, 3, 1, 2, 1), (12, 3, 2, 2, 2)],
)
def test_the_run_takes_exactly_the_planned_number_of_steps(rows, batch, epochs, accum, inner):
    plan = StepPlan(rows, batch, epochs=epochs, accumulation_steps=accum, inner_iters=inner)
    trainer = make(plan)
    trainer.run()
    trainer.verify_step_accounting()
    assert trainer.state.optimizer_steps == plan.planned_optimizer_steps


def test_step_accounting_failure_is_an_error_not_a_shrug():
    plan = StepPlan(12, 3)
    trainer = make(plan)
    trainer.run()
    trainer.state.optimizer_steps += 1
    with pytest.raises(AssertionError, match="planned"):
        trainer.verify_step_accounting()


def test_a_scheduler_ends_where_the_plan_said_it_would():
    plan = StepPlan(12, 3, epochs=2)
    trainer = make(plan, lr=0.1, scheduler=True)
    trainer.run()
    trainer.verify_step_accounting()
    assert trainer.scheduler.last_epoch == plan.planned_optimizer_steps


def test_a_nonsense_plan_is_refused():
    for kwargs in ({"batch_size": 0}, {"epochs": 0}, {"accumulation_steps": 0}):
        with pytest.raises(ValueError, match="must be >= 1"):
            StepPlan(12, **{"batch_size": 3, **kwargs})


def test_a_plan_that_does_not_match_the_dataset_is_refused():
    plan = StepPlan(99, 3)
    model = tiny_regressor()
    with pytest.raises(ValueError, match="plan is for 99 rows"):
        Trainer(model, torch.optim.SGD(model.parameters(), lr=1e-3), Rows(12),
                squared_error, plan)


# ---------------------------------------------------------------- B8
def test_resume_reproduces_an_uninterrupted_run(tmp_path):
    """The property the whole feature exists for, asserted bit for bit."""
    plan = StepPlan(12, 3, epochs=2)

    straight = make(plan, lr=0.05)
    straight.run()
    expected = weights(straight.model)

    interrupted = make(plan, lr=0.05)
    interrupted.run(max_batches=3)
    interrupted.save(tmp_path / "ckpt.pth")

    resumed = make(plan, lr=0.05)
    resumed.load(tmp_path / "ckpt.pth")
    resumed.run()

    assert_same(expected, weights(resumed.model), "resumed run diverged from the straight one")
    assert resumed.state.optimizer_steps == straight.state.optimizer_steps
    resumed.verify_step_accounting()


@pytest.mark.parametrize("stop_after", [1, 2, 4, 5, 7])
def test_resume_is_exact_wherever_it_is_interrupted(tmp_path, stop_after):
    plan = StepPlan(12, 3, epochs=2)

    straight = make(plan, lr=0.05)
    straight.run()
    expected = weights(straight.model)

    interrupted = make(plan, lr=0.05)
    interrupted.run(max_batches=stop_after)
    interrupted.save(tmp_path / f"ckpt{stop_after}.pth")

    resumed = make(plan, lr=0.05)
    resumed.load(tmp_path / f"ckpt{stop_after}.pth")
    resumed.run()

    assert_same(expected, weights(resumed.model), f"diverged after stopping at {stop_after}")


def test_resume_is_exact_across_an_epoch_boundary(tmp_path):
    plan = StepPlan(12, 3, epochs=3)
    straight = make(plan, lr=0.05)
    straight.run()

    interrupted = make(plan, lr=0.05)
    interrupted.run(max_batches=4)           # exactly one epoch
    assert interrupted.state.epoch == 1 and interrupted.state.batch_in_epoch == 0
    interrupted.save(tmp_path / "boundary.pth")

    resumed = make(plan, lr=0.05)
    resumed.load(tmp_path / "boundary.pth")
    resumed.run()
    assert_same(weights(straight.model), weights(resumed.model))


def test_resume_carries_the_optimizer_momentum_not_just_the_weights(tmp_path):
    """Dropping optimizer state is the classic way resume silently diverges."""
    plan = StepPlan(12, 3, epochs=2)

    def momentum_trainer():
        model = tiny_regressor()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        return Trainer(model, optimizer, Rows(12), squared_error, plan, seed=7)

    straight = momentum_trainer()
    straight.run()

    interrupted = momentum_trainer()
    interrupted.run(max_batches=3)
    interrupted.save(tmp_path / "momentum.pth")

    resumed = momentum_trainer()
    resumed.load(tmp_path / "momentum.pth")
    resumed.run()

    assert_same(weights(straight.model), weights(resumed.model))


def test_resume_with_a_scheduler_lands_on_the_same_learning_rate(tmp_path):
    plan = StepPlan(12, 3, epochs=2)

    straight = make(plan, lr=0.1, scheduler=True)
    straight.run()

    interrupted = make(plan, lr=0.1, scheduler=True)
    interrupted.run(max_batches=3)
    interrupted.save(tmp_path / "sched.pth")

    resumed = make(plan, lr=0.1, scheduler=True)
    resumed.load(tmp_path / "sched.pth")
    resumed.run()

    assert_same(weights(straight.model), weights(resumed.model))
    assert resumed.scheduler.get_last_lr() == straight.scheduler.get_last_lr()


def test_a_checkpoint_from_a_different_plan_is_refused(tmp_path):
    original = make(StepPlan(12, 3, epochs=2), lr=0.05)
    original.run(max_batches=2)
    original.save(tmp_path / "other.pth")

    different = make(StepPlan(12, 4, epochs=2), lr=0.05)
    with pytest.raises(ValueError, match="different plan"):
        different.load(tmp_path / "other.pth")


def test_a_checkpoint_from_a_different_seed_is_refused(tmp_path):
    original = make(StepPlan(12, 3), seed=7)
    original.run(max_batches=1)
    original.save(tmp_path / "seed.pth")

    other = make(StepPlan(12, 3), seed=8)
    with pytest.raises(ValueError, match="seed"):
        other.load(tmp_path / "seed.pth")


def test_stopping_mid_accumulation_is_refused_rather_than_silently_lossy(tmp_path):
    plan = StepPlan(12, 3, accumulation_steps=2)
    trainer = make(plan)
    with pytest.raises(RuntimeError, match="mid-accumulation"):
        trainer.run(max_batches=1)


def test_the_checkpoint_is_written_atomically(tmp_path):
    trainer = make(StepPlan(12, 3))
    trainer.run(max_batches=1)
    target = tmp_path / "atomic.pth"
    trainer.save(target)
    assert target.exists()
    assert not list(tmp_path.glob("*.tmp")), "a temporary file was left behind"
