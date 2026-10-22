"""Offline intervention: replace velocities while retaining original contact outputs."""
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_egoimu_vc_recipe import check_gpu, write_json, sha
from scripts.run_velocity_swap_ablation_20261019 import OUT, RUN, eval_command, state

check_gpu(3000)
import torch
from model.rnn.velocity_contact_v5 import V5VelocityContactModule
from scripts.probe_vc_joint_20261019 import worker

original_forward = V5VelocityContactModule.forward
original_state = state(torch.load(RUN / 'proposed/joint/best_velocity_contact.pt', map_location='cpu'))
shadows, calls = {}, [0]


def forward(self, data_dict, hp_out=None, velocity_only=False):
    if velocity_only:
        return original_forward(self, data_dict, hp_out=hp_out, velocity_only=True)
    assert not self.training
    if id(self) not in shadows:
        shadow = copy.deepcopy(self)
        shadow.load_state_dict(original_state, strict=True)
        shadow.eval()
        shadows[id(self)] = shadow
    # Original ++ velocity heads drive its contact computation. Only downstream
    # velocities are replaced, leaving all other VC outputs exactly as computed.
    prediction = original_forward(shadows[id(self)], data_dict, hp_out=hp_out)
    velocity = original_forward(self, data_dict, hp_out=hp_out, velocity_only=True)
    assert set(velocity) == {'pred_hand_glb_vel', 'pred_hand_glb_vel_flat', 'pred_obj_vel'}
    prediction.update(velocity)
    calls[0] += 1
    return prediction


V5VelocityContactModule.forward = forward
cmd, output = eval_command('swap_fixed_contact_outputs', OUT / 'baseline_velocity_in_proposed.pt', 'offline')
sys.argv = [__file__] + cmd[2:]
worker()
write_json(output.with_suffix('.intervention.json'), {
    'mode': 'offline', 'fixed_contact_outputs': 'Original ++ joint-Best forward on identical HP output and full-sequence input',
    'replaced': ['pred_hand_glb_vel', 'pred_hand_glb_vel_flat', 'pred_obj_vel'],
    'forward_calls': calls[0], 'source_sha256': sha(Path(__file__)),
})
