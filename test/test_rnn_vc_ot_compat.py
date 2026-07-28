from types import SimpleNamespace

import torch
import torch.nn as nn

from model.rnn.imuhoi_model import IMUHOIModel
from model.rnn.online import update_data_inits_from_history
from train.rnn.train_utils import BaseTrainer


class _EndpointAwareVC(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def _apply_segment_contact(self, output, contact_init=None):
        self.calls += 1
        result = dict(output)
        frame_contact = result["pred_hand_contact_prob"]
        result["pred_hand_contact_prob_frame"] = frame_contact
        result["pred_hand_contact_prob"] = torch.ones_like(frame_contact)
        result["seen_contact_init"] = contact_init
        return result


def _pipeline_shell(*, training, contact_from_segments=True):
    model = object.__new__(IMUHOIModel)
    nn.Module.__init__(model)
    model.cfg = SimpleNamespace(contact_from_segments=contact_from_segments)
    model.velocity_contact_module = _EndpointAwareVC()
    model.train(training)
    return model


def test_endpoint_contact_adapter_is_inference_only():
    contact = torch.zeros(1, 3, 3)
    contact_init = torch.tensor([[0.1, 0.2, 0.3]])

    inference_model = _pipeline_shell(training=False)
    inference_out = inference_model._apply_vc_segment_contact_if_enabled(
        {"pred_hand_contact_prob": contact},
        contact_init,
    )
    assert inference_model.velocity_contact_module.calls == 1
    assert torch.equal(inference_out["pred_hand_contact_prob"], torch.ones_like(contact))
    assert torch.equal(inference_out["pred_hand_contact_prob_frame"], contact)
    assert torch.equal(inference_out["seen_contact_init"], contact_init)

    training_model = _pipeline_shell(training=True)
    training_out = training_model._apply_vc_segment_contact_if_enabled(
        {"pred_hand_contact_prob": contact},
        contact_init,
    )
    assert training_model.velocity_contact_module.calls == 0
    assert training_out["pred_hand_contact_prob"] is contact


def test_online_contact_init_uses_window_frame_and_raw_contact():
    decoded = torch.zeros(1, 5, 3)
    raw = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3) / 15.0
    history = {
        "pred_obj_vel": torch.arange(15, dtype=torch.float32).reshape(1, 5, 3),
        "pred_hand_contact_prob": decoded,
        "pred_hand_contact_prob_frame": raw,
    }

    updated = update_data_inits_from_history(
        {"contact_init": torch.zeros(1, 3)},
        history,
        index=1,
        contact_index=3,
    )

    assert torch.equal(updated["obj_vel_init"], history["pred_obj_vel"][:, 1])
    assert torch.equal(updated["contact_init"], raw[:, 3])


class _OptimizerHookTrainer(BaseTrainer):
    def build_optimizer_and_scheduler(self):
        self.optimizer_hook_called = True
        return object(), object()


def test_base_trainer_uses_overridable_optimizer_hook():
    cfg = SimpleNamespace(
        device="cpu",
        resume_dir=None,
        pretrained_ckpt=None,
        use_multi_gpu=False,
        use_tensorboard=False,
        debug=True,
        save_dir=".",
    )
    trainer = _OptimizerHookTrainer(
        cfg,
        nn.Linear(1, 1),
        loss_fn=None,
        train_loader=[],
    )
    assert trainer.optimizer_hook_called
