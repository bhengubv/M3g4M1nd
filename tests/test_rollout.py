"""Row C1 — the rollout engine never fabricates, and its snapshot is real."""

from __future__ import annotations

import pytest
import torch

from m3g4m1nd.rollout import HttpRolloutEngine, RolloutResult, TorchRolloutEngine
from tests.conftest import tiny_model


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(int(t)) for t in tokens)

    def batch_decode(self, rows, skip_special_tokens=True):
        return [self.decode(r) for r in rows]


# ---------------------------------------------------------------- C1
# MiniMind padded missing per-token scores with 0.0 — log p = 0, meaning total
# certainty — and fed them into the importance ratio.

def _reply(tokens, scores):
    return [{"meta_info": {"output_ids": tokens, "output_token_logprobs": [[s] for s in scores]}}]


def test_a_short_score_list_is_refused_not_padded():
    with pytest.raises(ValueError, match="2 log-probabilities for 4 tokens"):
        HttpRolloutEngine.assemble([[9, 9]], _reply([5, 6, 7, 8], [-0.1, -0.2]), FakeTokenizer())


def test_the_refusal_explains_what_padding_would_have_done():
    with pytest.raises(ValueError) as caught:
        HttpRolloutEngine.assemble([[9]], _reply([5, 6], [-0.1]), FakeTokenizer())
    assert "invented certainties" in str(caught.value)


def test_a_long_score_list_is_also_refused():
    with pytest.raises(ValueError, match="3 log-probabilities for 2 tokens"):
        HttpRolloutEngine.assemble([[9]], _reply([5, 6], [-0.1, -0.2, -0.3]), FakeTokenizer())


def test_an_empty_completion_is_refused():
    with pytest.raises(ValueError, match="returned no tokens"):
        HttpRolloutEngine.assemble([[9]], _reply([], []), FakeTokenizer())


def test_a_matching_reply_assembles_cleanly():
    result = HttpRolloutEngine.assemble(
        [[9, 9]], _reply([5, 6, 7], [-0.1, -0.2, -0.3]), FakeTokenizer()
    )
    assert result.completion_ids.tolist() == [[5, 6, 7]]
    assert result.per_token_logps.shape == (1, 3)
    assert result.completion_mask.tolist() == [[1.0, 1.0, 1.0]]
    assert result.prompt_lens.tolist() == [2]


def test_ragged_replies_pad_the_tokens_but_mask_the_padding():
    payload = [
        {"meta_info": {"output_ids": [5, 6, 7], "output_token_logprobs": [[-0.1], [-0.2], [-0.3]]}},
        {"meta_info": {"output_ids": [8], "output_token_logprobs": [[-0.4]]}},
    ]
    result = HttpRolloutEngine.assemble([[9], [9]], payload, FakeTokenizer())
    assert result.completion_mask.tolist() == [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]


def test_the_result_container_itself_refuses_a_score_shape_mismatch():
    """Belt and braces: even a hand-built result cannot carry a short score row."""
    with pytest.raises(ValueError, match="A score per token is not optional"):
        RolloutResult(
            output_ids=torch.zeros(1, 4, dtype=torch.long),
            completion_ids=torch.zeros(1, 3, dtype=torch.long),
            per_token_logps=torch.zeros(1, 2),
            completion_mask=torch.ones(1, 3),
            prompt_lens=torch.tensor([1]),
            completions=["x"],
        )


def test_result_refuses_a_mismatched_decode_count():
    with pytest.raises(ValueError, match="1 decoded strings for 2 rows"):
        RolloutResult(
            output_ids=torch.zeros(2, 4, dtype=torch.long),
            completion_ids=torch.zeros(2, 3, dtype=torch.long),
            per_token_logps=torch.zeros(2, 3),
            completion_mask=torch.ones(2, 3),
            prompt_lens=torch.tensor([1, 1]),
            completions=["x"],
        )


# The other half of C1: the snapshot must be a copy, which is what makes the
# importance ratio mean anything. MiniMind rebound the live object.

def test_the_snapshot_does_not_move_when_the_policy_is_trained():
    model = tiny_model().train()
    engine = TorchRolloutEngine(model, FakeTokenizer())

    before = engine._policy.lm_head.weight.detach().clone()
    with torch.no_grad():
        model.lm_head.weight.add_(1.0)

    torch.testing.assert_close(engine._policy.lm_head.weight, before, rtol=0, atol=0)
    assert not engine.is_live_policy


def test_opting_into_the_live_policy_is_possible_but_declares_itself():
    """MiniMind's behaviour is still reachable — it just has to be asked for,
    and `is_live_policy` tells any caller that the ratio will be 1."""
    model = tiny_model().train()
    engine = TorchRolloutEngine(model, FakeTokenizer(), snapshot=False)

    assert engine.is_live_policy
    assert engine._policy is model, "snapshot=False should hand back the live model"

    with torch.no_grad():
        model.lm_head.weight.add_(1.0)
    # Same object, so the sampler moved with the learner.
    torch.testing.assert_close(
        engine._policy.lm_head.weight, model.lm_head.weight, rtol=0, atol=0
    )


def test_staleness_is_counted_so_a_trainer_can_assert_what_it_believes():
    model = tiny_model().train()
    engine = TorchRolloutEngine(model, FakeTokenizer())
    assert engine.stale_by == 0
    engine.note_policy_update()
    engine.note_policy_update()
    assert engine.stale_by == 2
    engine.update_policy(model)
    assert engine.stale_by == 0


def test_rollout_returns_a_score_for_every_sampled_token():
    torch.manual_seed(31)
    model = tiny_model()
    engine = TorchRolloutEngine(model, FakeTokenizer())
    prompts = torch.randint(3, 60, (2, 4))

    result = engine.rollout(prompts, num_generations=2, max_new_tokens=5, temperature=0.9)

    rows = 2 * 2
    assert result.completion_ids.size(0) == rows
    assert result.per_token_logps.shape == result.completion_ids.shape
    assert bool(result.per_token_logps.isfinite().all())
    assert len(result.completions) == rows
    assert result.prompt_lens.tolist() == [4] * rows


# ------------------------------------------------- pushing weights to a server
class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"success": True}
        self.text = text or "ok"
        self.content = b"{}"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, response=None):
        self.response = response or FakeResponse()
        self.posts = []

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))
        return self.response

    def get(self, url, timeout=None):
        return self.response


def _engine(tmp_path, session):
    return HttpRolloutEngine(
        "http://inference:8998", FakeTokenizer(), session=session,
        shared_path=tmp_path / "weights",
    )


def test_pushing_weights_writes_them_and_tells_the_server_where(tmp_path):
    """Was a NotImplementedError."""
    session = FakeSession()
    engine = _engine(tmp_path, session)
    engine.update_policy(tiny_model())

    assert (tmp_path / "weights").is_dir()
    assert any((tmp_path / "weights").glob("*.safetensors")) or \
        any((tmp_path / "weights").glob("*.bin"))

    url, body = session.posts[-1]
    assert url.endswith("/update_weights_from_disk")
    assert body["model_path"] == str((tmp_path / "weights").resolve())
    assert engine.updates == 1


def test_a_refused_update_raises_instead_of_warning(tmp_path):
    """MiniMind printed a warning and set a flag. A silently failed push leaves
    the server sampling stale weights while the trainer believes otherwise,
    which makes the importance ratio meaningless."""
    session = FakeSession(FakeResponse(status_code=500, text="out of memory"))
    engine = _engine(tmp_path, session)
    with pytest.raises(RuntimeError, match="refused the weight update"):
        engine.update_policy(tiny_model())
    assert engine.updates == 0


def test_a_server_reporting_failure_in_the_body_also_raises(tmp_path):
    session = FakeSession(FakeResponse(payload={"success": False, "message": "bad shard"}))
    engine = _engine(tmp_path, session)
    with pytest.raises(RuntimeError, match="bad shard"):
        engine.update_policy(tiny_model())


def test_the_write_is_atomic_and_leaves_no_staging_directory(tmp_path):
    """A reload racing a half-written directory loads a mixture of two
    checkpoints, which is worse than either."""
    engine = _engine(tmp_path, FakeSession())
    engine.update_policy(tiny_model())
    assert not list(tmp_path.glob("*.incoming")), "a staging directory was left behind"


def test_a_second_push_replaces_the_first_cleanly(tmp_path):
    engine = _engine(tmp_path, FakeSession())
    engine.update_policy(tiny_model())
    engine.update_policy(tiny_model())
    assert engine.updates == 2
    assert not list(tmp_path.glob("*.incoming"))


def test_health_and_cache_flush_report_the_server_state(tmp_path):
    engine = _engine(tmp_path, FakeSession())
    assert engine.health() is True
    assert engine.flush_cache() is True

    unwell = _engine(tmp_path, FakeSession(FakeResponse(status_code=503)))
    assert unwell.health() is False
    assert unwell.flush_cache() is False
