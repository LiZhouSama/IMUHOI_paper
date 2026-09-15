#!/usr/bin/env python3
"""Print the OT-phase 'Test Loss' (model-selection sum) per epoch of several ObjectTrans trainer logs side by side.

    python scripts/diag/ot_test_loss_series.py [step] name=path/to/log[,more.log] ...

Resumed / retried attempts append to the same log or to *_attemptN.log files; pass them comma-separated and
epochs are merged (later files override).  The joint phase restarts the epoch counter, so only the text before
the first joint-phase marker is parsed.
"""
import re
import sys


def series(paths):
    out = {}
    for path in paths:
        try:
            text = open(path, encoding="utf-8", errors="replace").read().replace("\r", "\n")
        except FileNotFoundError:
            continue
        cut = text.find("joint训练起点")
        ot_text = text if cut < 0 else text[:cut]
        # a "Test <epoch>:" progress line, an (optional) blank line, then ", Test Loss: x"
        epoch = None
        for line in ot_text.split("\n"):
            m = re.match(r"Test (\d+):", line)
            if m:
                epoch = int(m.group(1))
                continue
            m = re.search(r"Test Loss: ([0-9.]+)", line)
            if m and epoch is not None:
                out[epoch] = float(m.group(1))
                epoch = None
    return out


def main():
    args = sys.argv[1:]
    step = 1
    if args and args[0].isdigit():
        step, args = int(args[0]), args[1:]
    runs = {}
    for spec in args:
        name, _, paths = spec.partition("=")
        runs[name] = series(paths.split(","))
    epochs = sorted({e for s in runs.values() for e in s})
    print("| epoch | " + " | ".join(runs) + " |")
    print("|---|" + "---|" * len(runs))
    for e in epochs:
        if e % step:
            continue
        print(f"| {e} | " + " | ".join(f"{runs[n][e]:.3f}" if e in runs[n] else "—" for n in runs) + " |")
    for n, s in runs.items():
        if s:
            best = min(s, key=s.get)
            print(f"{n}: best {s[best]:.3f} @ epoch {best} (last epoch {max(s)})")


if __name__ == "__main__":
    main()
