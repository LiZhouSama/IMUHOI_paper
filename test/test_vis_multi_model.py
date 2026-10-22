"""Repeated --model variant-directory parsing and manifest checkpoint selection."""
import json
from pathlib import Path

import pytest

from utils.vis_model_spec import resolve_vis_models


def _variant(tmp_path, label, arch, vc, hp_in_stage=False):
    root = tmp_path / label / 'experiment'
    variant = root / label
    stage = variant / 'run'
    stage.mkdir(parents=True)
    config = variant / 'config.yaml'
    config.write_text('model_arch: rnn\n')
    reused = root / 'hp.pt'
    reused.touch()
    vc_reused = root / 'vc.pt'
    vc_reused.touch()
    (stage / 'final_object_trans.pt').touch()
    (stage / 'final_velocity_contact.pt').touch()
    if hp_in_stage:
        (stage / 'final_human_pose.pt').touch()
    manifest = {
        'pipelines': {label: {'config': str(config), 'stage_dir': str(stage),
                              'joint_ot_ckpt': str(stage / 'final_object_trans.pt'),
                              'joint_vc_ckpt': str(stage / 'final_velocity_contact.pt'),
                              'object_trans_arch': arch, 'velocity_contact_arch': vc}},
        'reused_checkpoints': {'hp_ckpt': str(reused), 'vc_ckpt': str(vc_reused)},
    }
    (root / 'experiment_manifest.json').write_text(json.dumps(manifest))
    return variant


def test_two_variant_directories_select_independent_architectures(tmp_path):
    v2 = _variant(tmp_path, 'v2_combo', 'v2', 'v4')
    imuhoi = _variant(tmp_path, 'imuhoi', 'imuhoi_4heads', 'imuhoi', hp_in_stage=True)
    specs = resolve_vis_models([v2, imuhoi], tmp_path)
    assert [s.label for s in specs] == ['v2_combo', 'imuhoi']
    assert [s.velocity_contact_arch for s in specs] == ['v4', 'imuhoi']
    assert specs[0].checkpoints['human_pose'].name == 'hp.pt'
    assert specs[1].checkpoints['human_pose'].name == 'final_human_pose.pt'
    assert specs[0].checkpoints['object_trans'].name == 'final_object_trans.pt'


def test_duplicate_and_missing_checkpoints_fail_before_opening(tmp_path):
    v2 = _variant(tmp_path, 'v2_combo', 'v2', 'v4')
    with pytest.raises(ValueError, match='Duplicate model label'):
        resolve_vis_models([v2, v2], tmp_path)
    (v2 / 'run' / 'final_object_trans.pt').unlink()
    with pytest.raises(FileNotFoundError, match='object_trans checkpoint'):
        resolve_vis_models([v2], tmp_path)


def test_multi_lane_inference_uses_each_config_and_common_length(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import torch
    import vis_IMUHOI as vis

    configs = [SimpleNamespace(model_arch='rnn', test={'window': 60, 'online_lookahead': 10}),
               SimpleNamespace(model_arch='rnn', test={'window': 80, 'online_lookahead': 5})]
    bundles = [vis.ComparisonModel('v2', configs[0], object(), True),
               vis.ComparisonModel('imuhoi', configs[1], object(), False)]
    batch = {'human_imu': torch.zeros(1, 100, 6, 9), 'trans': torch.zeros(1, 100, 3),
             'obj_trans': torch.zeros(1, 100, 3)}
    viewer = SimpleNamespace(overlay_frames=[0, 89], online_window=None, online_lookahead=None,
                             online_chunk=None, use_fk=False, inference_mode='online',
                             show_objects=True, show_foot_contact=False, show_hand_traj=False,
                             show_obj_move=False, no_trans=False, obj_max_faces=20000,
                             pred_offset=[1.5, 0.0, 0.5])
    calls = []
    rendered = []
    monkeypatch.setattr(vis, '_move_comparison_model', lambda *args: None)
    monkeypatch.setattr(vis, 'build_model_input_dict', lambda batch, config, device, add_noise=False: {'config': config})

    def predict(model, data, targets, fk, arch, **kwargs):
        calls.append((data['config'], kwargs['online_window'], kwargs['online_lookahead'], kwargs['ot_refine']))
        length = 90 if len(calls) == 1 else 95
        return {'p_pred': torch.zeros(1, length, 60),
                'root_trans_pred': torch.zeros(1, length, 3),
                'pred_obj_trans': torch.zeros(1, length, 3)}

    monkeypatch.setattr(vis, '_run_model_prediction', predict)
    monkeypatch.setattr(vis, 'visualize_batch_data', lambda viewer, batch, model, smpl_model, device, obj_geo_root, **kw: rendered.append((viewer.config, viewer.scene._label, kw)))
    class Scene:
        def add(self, *args, **kwargs): pass
    viewer.scene = Scene()
    vis.visualize_comparison_data(viewer, batch, bundles, object(), torch.device('cpu'), None)
    assert calls == [(configs[0], 60, 10, True), (configs[1], 80, 5, False)]
    assert len(rendered) == 2
    assert [item[1] for item in rendered] == ['v2', 'imuhoi']
    assert [item[2]['render_gt'] for item in rendered] == [True, False]
    assert [item[2]['display_length'] for item in rendered] == [90, 90]
    # Lane i sits i steps of --pred_offset from GT, each in its own colours.
    assert np.allclose(rendered[0][2]['pred_offset_np'], [1.5, 0.0, 0.5])
    assert np.allclose(rendered[1][2]['pred_offset_np'], [3.0, 0.0, 1.0])
    assert rendered[0][2]['lane_colors'] is vis.PRED_LANE_COLORS[0]
    assert rendered[1][2]['lane_colors']['human'] != rendered[0][2]['lane_colors']['human']
    assert all(item[2]['clear_scene'] is False for item in rendered)


def test_overlay_outside_common_length_fails_before_any_render(monkeypatch):
    import torch
    from types import SimpleNamespace
    import vis_IMUHOI as vis
    bundle = vis.ComparisonModel('v2', SimpleNamespace(model_arch='rnn', test={'window': 60, 'online_lookahead': 10}), object(), True)
    batch = {'human_imu': torch.zeros(1, 20, 6, 9)}
    viewer = SimpleNamespace(overlay_frames=[18], online_window=None, online_lookahead=None, online_chunk=1,
                             use_fk=False, inference_mode='online', show_objects=False, show_foot_contact=False,
                             show_hand_traj=False, show_obj_move=False, no_trans=False, obj_max_faces=0)
    monkeypatch.setattr(vis, '_move_comparison_model', lambda *args: None)
    monkeypatch.setattr(vis, 'build_model_input_dict', lambda *args, **kwargs: {})
    monkeypatch.setattr(vis, '_run_model_prediction', lambda *args, **kwargs: {
        'p_pred': torch.zeros(1, 10, 60), 'root_trans_pred': torch.zeros(1, 10, 3), 'pred_obj_trans': torch.zeros(1, 10, 3)})
    monkeypatch.setattr(vis, 'visualize_batch_data', lambda *args, **kwargs: pytest.fail('rendered partial lane'))
    with pytest.raises(SystemExit, match='Invalid --overlay_frames'):
        vis.visualize_comparison_data(viewer, batch, [bundle], object(), torch.device('cpu'), None)


def test_parser_keeps_single_model_mode_and_repeats_variant_directories():
    import vis_IMUHOI as vis
    parser = vis._make_parser()
    assert parser.parse_args([]).model is None
    assert parser.parse_args(['--model', 'a/v2', '--model', 'b/imuhoi']).model == ['a/v2', 'b/imuhoi']


def test_three_model_lanes_share_gt_and_receive_distinct_names(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import torch
    import vis_IMUHOI as vis

    cfg = SimpleNamespace(model_arch='rnn', test={'window': 60, 'online_lookahead': 5})
    bundles = [vis.ComparisonModel(name, cfg, object(), False) for name in ('a', 'b', 'c')]
    batch = {'human_imu': torch.zeros(1, 40, 6, 9), 'trans': torch.zeros(1, 40, 3)}
    class Scene:
        def __init__(self): self.nodes = []
        def add(self, *nodes, **kwargs): self.nodes.extend(nodes)
    viewer = SimpleNamespace(scene=Scene(), overlay_frames=None, online_window=None, online_lookahead=None,
                             online_chunk=None, use_fk=False, inference_mode='online', show_objects=False,
                             show_foot_contact=False, show_hand_traj=False, show_obj_move=False,
                             no_trans=False, obj_max_faces=0, pred_offset=[2.0, 0.0, 0.0])
    monkeypatch.setattr(vis, '_move_comparison_model', lambda *args: None)
    monkeypatch.setattr(vis, 'build_model_input_dict', lambda *args, **kwargs: {})
    monkeypatch.setattr(vis, '_run_model_prediction', lambda *args, **kwargs: {
        'p_pred': torch.zeros(1, 35, 60), 'root_trans_pred': torch.zeros(1, 35, 3),
        'pred_obj_trans': torch.zeros(1, 35, 3)})
    offsets, colours = [], []
    def draw(view, *_args, **kw):
        view.scene.add(SimpleNamespace(name='Pred-Human'))
        offsets.append(kw['pred_offset_np'][0])
        colours.append(kw['lane_colors']['human'])
        if kw['render_gt']:
            view.scene.add(SimpleNamespace(name='GT-Human'))
    monkeypatch.setattr(vis, 'visualize_batch_data', draw)
    vis.visualize_comparison_data(viewer, batch, bundles, object(), torch.device('cpu'), None)
    assert [n.name for n in viewer.scene.nodes] == [
        'Pred-a::Pred-Human', 'GT-Human', 'Pred-b::Pred-Human', 'Pred-c::Pred-Human']
    assert np.allclose(offsets, [2., 4., 6.])
    assert len(set(colours)) == 3


def test_job_commit_replaces_exact_comparison_nodes():
    from types import SimpleNamespace
    import vis_IMUHOI as vis
    class Scene:
        def __init__(self): self.nodes = []; self.current_frame_id = 0
        def add(self, *nodes, **kw): self.nodes.extend(nodes)
        def remove(self, node): self.nodes.remove(node)
    scene = Scene()
    viewer = SimpleNamespace(scene=scene, comparison_models=[1], _comparison_nodes=[], current_index=0,
                             title='', _sequence_label=lambda index: str(index),
                             _frame_comparison_camera=lambda: None)
    first = SimpleNamespace(name='Pred-a::Pred-Human')
    j1 = SimpleNamespace(index=0, title='first', ops=[('add', (first,), {})],
                         started_at=0., finished_at=1.)
    vis.InteractiveViewer._apply_finished_job(viewer, j1)
    assert scene.nodes == [first] and viewer._comparison_nodes == [first]
    second = SimpleNamespace(name='Pred-b::Pred-Human')
    j2 = SimpleNamespace(index=1, title='second', ops=[('remove', (first,), {}), ('add', (second,), {})],
                         started_at=1., finished_at=2.)
    vis.InteractiveViewer._apply_finished_job(viewer, j2)
    assert scene.nodes == [second] and viewer._comparison_nodes == [second]


def test_comparison_camera_frames_lanes_clear_of_sidebar():
    from types import SimpleNamespace
    import numpy as np
    import vis_IMUHOI as vis

    camera = SimpleNamespace(forward=np.array([0., 0., -1.]), target=None, position=None)
    nodes = [SimpleNamespace(name=f'Pred-{i}', current_bounds=np.array([[x-1., x+1.], [0., 2.], [-1., 1.]]))
             for i, x in enumerate((0., 4., 8.))]
    viewer = SimpleNamespace(_comparison_nodes=nodes, scene=SimpleNamespace(camera=camera))
    vis.InteractiveViewer._frame_comparison_camera(viewer)
    assert camera.target[0] < 4.0  # leave room for the editor on the left
    assert camera.position[2] > camera.target[2] + 5


def test_lane_palette_and_offsets():
    import numpy as np
    import vis_IMUHOI as vis
    # Lane 1 keeps the single-model prediction colours; no lane reuses GT's.
    assert vis._lane_colors(1)['human'] == vis.PRED_HUMAN_COLOR
    assert vis._lane_colors(1)['object'] == vis.PRED_OBJECT_COLOR
    humans = [vis._lane_colors(lane)['human'] for lane in range(1, len(vis.PRED_LANE_COLORS) + 1)]
    assert len(set(humans)) == len(humans) and vis.GT_HUMAN_COLOR not in humans
    assert vis._lane_colors(len(vis.PRED_LANE_COLORS) + 1) is vis._lane_colors(1)
    assert np.allclose(vis._lane_offset([0.5, 0.0, 1.0], 3), [1.5, 0.0, 3.0])
    assert np.allclose(vis._lane_offset(None, 2), [2.0, 0.0, 0.0])

def test_job_commit_clears_removed_selection_before_releasing_mesh():
    from types import SimpleNamespace
    import vis_IMUHOI as vis
    child = SimpleNamespace(nodes=[])
    old = SimpleNamespace(nodes=[child], name='Pred-old')
    new = SimpleNamespace(nodes=[], name='Pred-new')
    class Scene:
        def __init__(self):
            self.nodes = [old]
            self.selected_object = child
            self.gui_selected_object = old
            self.current_frame_id = 3
        def select(self, node):
            self.selected_object = node
        def remove(self, node):
            assert self.selected_object is None
            assert self.gui_selected_object is None
            self.nodes.remove(node)
        def add(self, *nodes, **kwargs):
            self.nodes.extend(nodes)
    scene = Scene()
    viewer = SimpleNamespace(scene=scene, comparison_models=[1], _comparison_nodes=[old], current_index=0,
                             title='', _sequence_label=lambda index: str(index),
                             _frame_comparison_camera=lambda: None)
    job = SimpleNamespace(index=1, title='next', ops=[('remove', (old,), {}), ('add', (new,), {})],
                          started_at=0., finished_at=1.)
    vis.InteractiveViewer._apply_finished_job(viewer, job)
    assert scene.nodes == [new]
    assert scene.current_frame_id == 0 and viewer.current_index == 1


def test_removed_subtree_clears_property_selection_but_preserves_unrelated_selection():
    from types import SimpleNamespace
    import vis_IMUHOI as vis
    child = SimpleNamespace(nodes=[])
    old = SimpleNamespace(nodes=[child])
    camera = object()
    scene = SimpleNamespace(selected_object=camera, gui_selected_object=child)
    vis._clear_removed_node_selection(scene, [old])
    assert scene.selected_object is camera
    assert scene.gui_selected_object is None
