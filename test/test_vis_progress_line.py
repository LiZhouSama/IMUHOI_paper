"""The self-overwriting progress/ETA line of vis_IMUHOI's online inference."""
import io
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import vis_IMUHOI as vis


class _Tty(io.StringIO):
    def isatty(self):
        return True


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _progress(stream, label="seq | 1/2 v2"):
    clock = _Clock()
    line = vis._ProgressLine(stream)
    return vis._InferenceProgress(label, line=line, clock=clock), clock, line


def test_tty_redraws_one_line_and_ends_it_when_done():
    stream = _Tty()
    progress, clock, _ = _progress(stream)
    progress(60, 460)                 # warmup block, reported in one call
    clock.now += 10.0
    progress(160, 460)                # 100 frames in 10 s after the warmup
    clock.now += 0.1
    progress(161, 460)                # throttled: within TTY_INTERVAL
    clock.now += 30.0
    progress(460, 460)
    out = stream.getvalue()
    frames = out.split("\r")[1:]
    assert len(frames) == 3 and "\n" not in "".join(frames[:-1])
    # The rate excludes the warmup block: 10 fps, so 300 frames left = 30 s.
    assert "160/460 frames" in frames[1] and "10.0 fps" in frames[1] and "ETA 30s" in frames[1]
    assert frames[2].startswith("[seq | 1/2 v2] 460/460 frames in 40s") and out.endswith("\n")


def test_interruption_leaves_no_half_drawn_line():
    stream = _Tty()
    progress, clock, line = _progress(stream)
    progress(60, 460)
    line.break_line()
    assert stream.getvalue().endswith("\n")
    line.break_line()                  # nothing open: no blank line
    assert not stream.getvalue().endswith("\n\n")


def test_cancellation_is_checked_before_drawing(monkeypatch):
    stream = _Tty()
    progress, _, _ = _progress(stream)

    def cancelled():
        raise vis.SequenceCancelled()

    monkeypatch.setattr(vis, "_raise_if_cancelled", cancelled)
    with pytest.raises(vis.SequenceCancelled):
        progress(60, 460)
    assert stream.getvalue() == ""


def test_narrow_terminal_shortens_the_label_before_the_numbers():
    body = "160/460 frames  34.8% | 10.0 fps | elapsed 10s | ETA 30s"
    fitted = vis._ProgressLine._fit("Date01_Sub01_boxlarge_hand | 1/2 v2_armrot_biclose", body, 80)
    assert len(fitted) <= 80 and fitted.endswith(body) and "..." in fitted


def test_non_tty_prints_whole_lines_at_a_slower_rate():
    stream = io.StringIO()
    progress, clock, _ = _progress(stream)
    progress(60, 460)
    clock.now += 1.0
    progress(70, 460)                  # within PLAIN_INTERVAL: skipped
    clock.now += vis._ProgressLine.PLAIN_INTERVAL
    progress(200, 460)
    progress(460, 460)
    lines = stream.getvalue().splitlines()
    assert "\r" not in stream.getvalue() and len(lines) == 3
    assert lines[1].startswith("[seq | 1/2 v2] 200/460 frames")
