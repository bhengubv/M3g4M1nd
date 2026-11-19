"""Training loop, checkpointing, and resume that actually resumes.

MiniMind's resume path is the most intricate code in that repository — a
custom batch sampler that skips the first N batches, a checkpoint carrying
optimizer and scaler state, a step count rescaled when the GPU count changes —
and it has no test at all. Nothing anywhere establishes that a run interrupted
at step N and resumed produces the same weights as a run that was never
interrupted. That is the one property the whole feature exists to provide.

It is also easy to get wrong in a way nobody notices: if the shuffle is not
reproducible, resuming trains on a different slice of data and the loss curve
still looks fine.

So two things are enforced here.

**The shuffle is derived, not drawn.** `epoch_order` builds each epoch's
permutation from (seed, epoch) through a dedicated generator, so skipping the
first N batches lands on exactly the batches an uninterrupted run would have
reached. MiniMind reseeded the global RNG per epoch and then also consumed it
elsewhere, which couples the data order to everything else that draws a random
number.

**The step count is planned up front and checked against reality.** The
scheduler needs a total; if the plan and the run disagree, the learning rate
finishes somewhere other than where it was meant to. `StepPlan` computes it and
`Trainer.verify_step_accounting` asserts it.
"""

from __future__ import annotations

import inspect
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


def epoch_order(num_rows: int, seed: int, epoch: int) -> list[int]:
    """A permutation determined solely by (num_rows, seed, epoch).

    Nothing else in the process can disturb it, which is what makes skipping
    batches on resume land where it should.
    """
    generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
    return torch.randperm(num_rows, generator=generator).tolist()


def batches(order: list[int], batch_size: int, skip: int = 0, drop_last: bool = False):
    """Yield index batches, having skipped the first `skip` of them."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    produced = 0
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        if drop_last and len(chunk) < batch_size:
            break
        produced += 1
        if produced <= skip:
            continue
        yield chunk


@dataclass(frozen=True)
class StepPlan:
    """How many optimizer steps a run will take, worked out before it starts."""

    num_rows: int
    batch_size: int
    epochs: int = 1
    accumulation_steps: int = 1
    inner_iters: int = 1
    drop_last: bool = False

    def __post_init__(self):
        for name in ("num_rows", "batch_size", "epochs", "accumulation_steps", "inner_iters"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")

    @property
    def batches_per_epoch(self) -> int:
        if self.drop_last:
            return self.num_rows // self.batch_size
        return math.ceil(self.num_rows / self.batch_size)

    @property
    def batches_total(self) -> int:
        return self.batches_per_epoch * self.epochs

    @property
    def planned_optimizer_steps(self) -> int:
        """Accumulation groups per epoch, times epochs, times inner iterations.

        Counted per epoch rather than over the whole run, because a trailing
        partial group is flushed at each epoch boundary.
        """
        per_epoch = math.ceil(self.batches_per_epoch / self.accumulation_steps)
        return per_epoch * self.epochs * self.inner_iters


def init_distributed() -> int:
    """Join a torchrun-launched group, or report single-process.

    Returns the local rank, or -1 when not launched under torchrun. Uses nccl
    on CUDA and gloo elsewhere, so a single-process group is testable on a CPU.
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return -1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def unwrap(model: nn.Module) -> nn.Module:
    model = getattr(model, "module", model)
    return getattr(model, "_orig_mod", model)


class IndexedBatchSampler(torch.utils.data.Sampler):
    """A batch sampler over a deterministic order, with the first N skipped.

    This is what lets a DataLoader exist without costing exact resume: the
    order comes from `epoch_order`, which depends only on (seed, epoch), so
    the batches after a skip are the batches an uninterrupted run would have
    reached. A DataLoader with its own shuffling would not have that property.
    """

    def __init__(self, order, batch_size, skip=0, drop_last=False, rank=0, world_size=1):
        self.order = list(order)
        self.batch_size = batch_size
        self.skip = skip
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size

    def _all(self):
        return list(batches(self.order, self.batch_size, self.skip, self.drop_last))

    def __iter__(self):
        for position, batch in enumerate(self._all()):
            if self.world_size == 1 or position % self.world_size == self.rank:
                yield batch

    def __len__(self):
        return len(list(iter(self)))


@dataclass
class TrainerState:
    epoch: int = 0
    batch_in_epoch: int = 0
    global_batch: int = 0
    optimizer_steps: int = 0


class Trainer:
    """Minimal loop with exact resume.

    `loss_fn(model, rows) -> Tensor` keeps this independent of what is being
    trained, so the same loop and the same checkpoint logic serve pretraining,
    fine-tuning and distillation rather than being copied seven times with
    small divergences, which is how MiniMind's seven trainers drifted apart.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataset,
        loss_fn,
        plan: StepPlan,
        seed: int = 42,
        grad_clip: float = 1.0,
        scheduler=None,
        num_workers: int = 0,
        pin_memory: bool = False,
        amp_dtype: torch.dtype | None = None,
        device: str = "cpu",
        gradient_checkpointing: bool = False,
        distributed: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.plan = plan
        self.seed = seed
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.num_workers = num_workers
        # Pinned memory only helps a host-to-device copy. Asking for it on a
        # CPU-only box earns a warning and nothing else.
        self.pin_memory = pin_memory and torch.accelerator.is_available()
        self.pin_memory_requested = pin_memory
        self.amp_dtype = amp_dtype
        self.device = device
        self.state = TrainerState()
        self.losses: list[float] = []

        if gradient_checkpointing:
            target = unwrap(model)
            if not hasattr(target, "gradient_checkpointing_enable"):
                raise ValueError(
                    f"{type(target).__name__} does not support gradient checkpointing"
                )
            target.gradient_checkpointing_enable()

        # float16 needs loss scaling to stop small gradients flushing to zero;
        # bfloat16 has the exponent range and does not. A scaler enabled for
        # bf16 is a no-op that looks like a safeguard, so it is not enabled.
        self.scaler = torch.amp.GradScaler(
            device.split(":")[0], enabled=amp_dtype == torch.float16
        )

        self.rank, self.world_size = 0, 1
        if distributed:
            if not dist.is_initialized():
                raise ValueError(
                    "distributed=True but no process group is initialised; call "
                    "init_distributed() from a torchrun-launched process first"
                )
            self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
            # The rotary tables are derived identically on every rank, so
            # syncing them every step is pure waste. torch 2.15 renamed the
            # flag; read the signature rather than guess which torch is here.
            accepted = inspect.signature(DistributedDataParallel.__init__).parameters
            buffer_flag = (
                {"forward_sync_buffers": False}
                if "forward_sync_buffers" in accepted
                else {"broadcast_buffers": False}
            )
            self.model = DistributedDataParallel(
                model,
                device_ids=[self.rank] if torch.cuda.is_available() else None,
                **buffer_flag,
            )

        if plan.num_rows != len(dataset):
            raise ValueError(
                f"plan is for {plan.num_rows} rows but the dataset has {len(dataset)}"
            )

    # ------------------------------------------------------------ checkpoints
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.state.epoch,
            "batch_in_epoch": self.state.batch_in_epoch,
            "global_batch": self.state.global_batch,
            "optimizer_steps": self.state.optimizer_steps,
            "seed": self.seed,
            "plan": {
                "num_rows": self.plan.num_rows,
                "batch_size": self.plan.batch_size,
                "epochs": self.plan.epochs,
                "accumulation_steps": self.plan.accumulation_steps,
                "inner_iters": self.plan.inner_iters,
                "drop_last": self.plan.drop_last,
            },
        }
        if self.scheduler is not None:
            payload["scheduler"] = self.scheduler.state_dict()

        # Written beside the target then moved, so an interrupted save cannot
        # leave a half-written checkpoint where a good one used to be.
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

        # Also write the weights on their own. A resume checkpoint and a set of
        # weights are different things, and conflating them is how an SFT run
        # once started from random initialisation while reporting success.
        weights_path = path.with_name(path.stem + "_weights" + path.suffix)
        temporary = weights_path.with_suffix(weights_path.suffix + ".tmp")
        torch.save(payload["model"], temporary)
        temporary.replace(weights_path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)

        saved_plan = payload["plan"]
        live_plan = {
            "num_rows": self.plan.num_rows,
            "batch_size": self.plan.batch_size,
            "epochs": self.plan.epochs,
            "accumulation_steps": self.plan.accumulation_steps,
            "inner_iters": self.plan.inner_iters,
            "drop_last": self.plan.drop_last,
        }
        if saved_plan != live_plan:
            differences = {
                key: (saved_plan[key], live_plan[key])
                for key in live_plan
                if saved_plan.get(key) != live_plan[key]
            }
            raise ValueError(
                f"checkpoint was written under a different plan: {differences}. "
                f"Resuming would train on a different data order than the run it "
                f"claims to continue."
            )
        if payload["seed"] != self.seed:
            raise ValueError(
                f"checkpoint seed {payload['seed']} does not match this run's {self.seed}; "
                f"the data order would differ"
            )

        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None and "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.state = TrainerState(
            epoch=payload["epoch"],
            batch_in_epoch=payload["batch_in_epoch"],
            global_batch=payload["global_batch"],
            optimizer_steps=payload["optimizer_steps"],
        )

    # ------------------------------------------------------------ the loop
    def _autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.split(":")[0], dtype=self.amp_dtype)

    def _flush(self) -> None:
        if self.grad_clip > 0:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        for _ in range(self.plan.inner_iters):
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self.state.optimizer_steps += 1
        self.optimizer.zero_grad(set_to_none=True)

    def run(self, max_batches: int | None = None) -> TrainerState:
        """Train from the current state. `max_batches` stops early, to be resumed."""
        self.model.train()
        done = 0

        for epoch in range(self.state.epoch, self.plan.epochs):
            order = epoch_order(self.plan.num_rows, self.seed, epoch)
            skip = self.state.batch_in_epoch if epoch == self.state.epoch else 0
            position = skip

            for rows in self._iterate(order, skip):
                with self._autocast():
                    loss = self.loss_fn(self.model, rows) / self.plan.accumulation_steps
                self.scaler.scale(loss).backward()

                position += 1
                self.state.batch_in_epoch = position
                self.state.global_batch += 1
                self.losses.append(loss.item() * self.plan.accumulation_steps)

                at_group = position % self.plan.accumulation_steps == 0
                at_epoch_end = position == self.plan.batches_per_epoch
                if at_group or at_epoch_end:
                    self._flush()

                done += 1
                if max_batches is not None and done >= max_batches:
                    if not (at_group or at_epoch_end):
                        # Stopping mid-group would discard gradients that an
                        # uninterrupted run would have applied.
                        raise RuntimeError(
                            f"refusing to stop mid-accumulation at batch {position}; "
                            f"stop on a multiple of accumulation_steps"
                            f"={self.plan.accumulation_steps}"
                        )
                    if at_epoch_end:
                        # Normalise: "finished epoch N" and "about to start epoch
                        # N+1" are the same position, and only one of them should
                        # ever be written down. Leaving it as (N, last_batch) works
                        # by accident — the skip loop yields nothing — which is the
                        # kind of accident that stops being one after a refactor.
                        self.state.epoch = epoch + 1
                        self.state.batch_in_epoch = 0
                    return self.state

            self.state.epoch = epoch + 1
            self.state.batch_in_epoch = 0

        return self.state

    def _iterate(self, order, skip):
        """Yield batches of rows, through a DataLoader when one is wanted.

        Both paths consume the same deterministic batch sampler, so turning
        workers on cannot change which rows a step sees — the property every
        resume test depends on.
        """
        sampler = IndexedBatchSampler(
            order, self.plan.batch_size, skip, self.plan.drop_last,
            rank=self.rank, world_size=self.world_size,
        )
        if self.num_workers == 0 and not self.pin_memory:
            for indices in sampler:
                yield [self.dataset[i] for i in indices]
            return

        yield from DataLoader(
            self.dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=list,
        )

    def verify_step_accounting(self) -> None:
        planned = self.plan.planned_optimizer_steps
        if self.state.optimizer_steps != planned:
            raise AssertionError(
                f"took {self.state.optimizer_steps} optimizer steps but planned "
                f"{planned}. The scheduler was told the wrong total, so the learning "
                f"rate did not finish where it was meant to."
            )
