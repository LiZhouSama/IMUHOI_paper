from types import SimpleNamespace

import torch

import eval_batch_online as batch_eval


class _FakeOnlineModel:
    def __init__(self):
        self.calls = []

    def inference(self, data_dict, **kwargs):
        human_imu = data_dict["human_imu"]
        self.calls.append((tuple(human_imu.shape), dict(kwargs)))
        batch_size, seq_len = human_imu.shape[:2]
        lookahead = int(kwargs["online_lookahead"])
        publish_len = max(seq_len - lookahead, 0)
        frame = human_imu[:, :publish_len, 0, 0:1]
        return {
            "p_pred": frame.expand(batch_size, publish_len, 6).clone(),
            "root_trans_pred": frame.expand(batch_size, publish_len, 3).clone(),
            "has_object": data_dict.get("has_object"),
        }


def _sample(seq_len):
    frame = torch.arange(seq_len, dtype=torch.float32).view(seq_len, 1, 1)
    return {
        "human_imu": frame.expand(seq_len, 6, 9).clone(),
        "has_object": True,
    }


def test_long_mixed_length_batch_is_padded_once_then_strictly_cropped(monkeypatch):
    monkeypatch.setattr(
        batch_eval,
        "build_model_input_dict",
        lambda raw_batch, _config, _device, add_noise=False: raw_batch,
    )
    model = _FakeOnlineModel()
    samples = [_sample(6), _sample(8), _sample(7)]

    outputs = batch_eval._run_batched_online_group(
        model,
        SimpleNamespace(),
        torch.device("cpu"),
        samples,
        online_window=3,
        online_lookahead=2,
        use_object_data=False,
        compute_fk=False,
        refine_human=False,
        desc="test",
    )

    assert len(model.calls) == 1
    assert model.calls[0][0] == (3, 8, 6, 9)
    assert [output["p_pred"].shape[1] for output in outputs] == [4, 6, 5]
    for seq_len, output in zip((6, 8, 7), outputs):
        expected = torch.arange(seq_len - 2, dtype=torch.float32)
        assert torch.equal(output["p_pred"][0, :, 0], expected)


def test_short_lookahead_warmups_are_grouped_by_exact_length(monkeypatch):
    monkeypatch.setattr(
        batch_eval,
        "build_model_input_dict",
        lambda raw_batch, _config, _device, add_noise=False: raw_batch,
    )
    model = _FakeOnlineModel()
    samples = [_sample(4), _sample(7), _sample(5)]

    outputs = batch_eval._run_batched_online_group(
        model,
        SimpleNamespace(),
        torch.device("cpu"),
        samples,
        online_window=3,
        online_lookahead=2,
        use_object_data=False,
        compute_fk=False,
        refine_human=False,
        desc="test-warmup",
    )

    assert sorted(call[0][1] for call in model.calls) == [4, 5, 7]
    assert [output["p_pred"].shape[1] for output in outputs] == [2, 5, 3]


def test_zero_lookahead_does_not_pad_short_legacy_warmups(monkeypatch):
    monkeypatch.setattr(
        batch_eval,
        "build_model_input_dict",
        lambda raw_batch, _config, _device, add_noise=False: raw_batch,
    )
    model = _FakeOnlineModel()
    samples = [_sample(2), _sample(5), _sample(3)]

    outputs = batch_eval._run_batched_online_group(
        model,
        SimpleNamespace(),
        torch.device("cpu"),
        samples,
        online_window=3,
        online_lookahead=0,
        use_object_data=False,
        compute_fk=False,
        refine_human=False,
        desc="test-zero",
    )

    assert sorted(call[0][1] for call in model.calls) == [2, 3, 5]
    assert [output["p_pred"].shape[1] for output in outputs] == [2, 5, 3]
