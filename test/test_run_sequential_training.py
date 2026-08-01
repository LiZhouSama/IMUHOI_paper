from io import StringIO
from unittest.mock import patch

from train.rnn.run_sequential_training import CompactConsoleRenderer
from train.rnn.run_sequential_training import compact_tqdm_progress


def test_compact_tqdm_progress_keeps_only_total_loss():
    line = (
        "Epoch 12:  25%|██▌       | 10/40 "
        "[00:05<00:15,  2.00it/s, loss=0.123, obj_vel=0.456, "
        "contact_endpoint=0.789]"
    )

    compact = compact_tqdm_progress(line)

    assert compact == (
        "Epoch 12:  25%|██▌       | 10/40 "
        "[00:05<00:15, 2.00it/s] loss=0.123"
    )
    assert "obj_vel" not in compact
    assert "contact_endpoint" not in compact


def test_compact_tqdm_progress_supports_initial_and_test_progress():
    initial = "Epoch 0:   0%|          | 0/20 [00:00<?, ?it/s]"
    test = "Test 4: 100%|██████████| 5/5 [00:02<00:00, 2.50it/s]"

    assert compact_tqdm_progress(initial) == (
        "Epoch 0:   0%|          | 0/20 [00:00<?, ?it/s]"
    )
    assert compact_tqdm_progress(test) == (
        "Test 4: 100%|██████████| 5/5 [00:02<00:00, 2.50it/s]"
    )


def test_compact_tqdm_progress_ignores_normal_messages():
    assert compact_tqdm_progress("新的统一最佳测试损失: 0.1234") is None


def test_compact_console_renderer_updates_progress_in_place_across_chunks():
    output = StringIO()
    renderer = CompactConsoleRenderer()
    first = (
        "Epoch 3:  25%|██▌       | 1/4 [00:01<00:03, 1.00it/s, "
        "loss=0.500, obj_vel=0.200, endpoint=0.100]"
    )
    second = (
        "Epoch 3:  50%|█████     | 2/4 [00:02<00:02, 1.00it/s, "
        "loss=0.400, obj_vel=0.150, endpoint=0.080]"
    )

    with patch("train.rnn.run_sequential_training.sys.stdout", output):
        renderer.feed(first[:40])
        renderer.feed(first[40:] + "\r" + second + "\r")
        renderer.close()

    rendered = output.getvalue()
    assert rendered.count("\r\x1b[2K") == 2
    assert "loss=0.500" in rendered
    assert "loss=0.400" in rendered
    assert "obj_vel" not in rendered
    assert "endpoint" not in rendered
    assert rendered.endswith("\n")
