import errno

import pytest
import torch

from dataset.dataset_IMUHOI import IMUDataset, _compact_tensor_storage


def test_compaction_preserves_nested_values_and_repeated_tensor_identity():
    backing = torch.arange(400, dtype=torch.float32).reshape(100, 4)
    view = backing[20:25, 1:3]
    full = torch.arange(8)
    original = {'pose': view, 'nested': [view, (full, 'neutral')], 'version': 1}
    compact = _compact_tensor_storage(original)
    torch.testing.assert_close(compact['pose'], view, rtol=0, atol=0)
    assert compact['pose'].storage().size() == view.numel()
    assert view.storage().size() == backing.numel()
    assert compact['nested'][0] is compact['pose']
    assert compact['nested'][1][0] is full
    assert compact['nested'][1][1] == 'neutral'
    assert compact['version'] == 1


def test_preloading_compacts_without_changing_sample_metadata(tmp_path, monkeypatch):
    path = tmp_path / 'sequence.pt'
    pose = torch.arange(100 * 72, dtype=torch.float32).reshape(100, 72)[10:30]
    torch.save({'pose': pose, 'has_object': False, 'gender': 'neutral'}, path)
    monkeypatch.setenv('IMUHOI_COMPACT_PRELOAD', '0')
    monkeypatch.setenv('IMUHOI_STRICT_LOADING', '1')
    ordinary = IMUDataset(str(tmp_path), window_size=5)
    monkeypatch.setenv('IMUHOI_COMPACT_PRELOAD', '1')
    compact = IMUDataset(str(tmp_path), window_size=5)
    assert ordinary.sequence_info == compact.sequence_info
    stored = compact.loaded_data[str(path)]['pose']
    torch.testing.assert_close(stored, ordinary.loaded_data[str(path)]['pose'], rtol=0, atol=0)
    assert stored.storage().size() == stored.numel()


def test_strict_preload_aborts_on_allocation_failure(tmp_path, monkeypatch):
    path = tmp_path / 'failed.pt'
    path.touch()
    monkeypatch.setenv('IMUHOI_STRICT_LOADING', '1')

    def allocation_failure(*args, **kwargs):
        raise OSError(errno.ENOMEM, 'Cannot allocate memory')

    monkeypatch.setattr(torch, 'load', allocation_failure)
    with pytest.raises(RuntimeError, match='failed.pt') as caught:
        IMUDataset(str(tmp_path), window_size=5)
    assert isinstance(caught.value.__cause__, OSError)
    assert caught.value.__cause__.errno == errno.ENOMEM