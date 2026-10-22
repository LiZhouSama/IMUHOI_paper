#!/usr/bin/env python3
"""Use production evaluators but reject partial object/HOI coverage."""
import importlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_egoimu_vc_recipe import check_gpu, PYTHON


def main():
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    check_gpu(6000)
    mode = sys.argv.pop(1)
    config = Path(sys.argv[sys.argv.index('--config') + 1])
    protocol = json.loads((config.parents[2] / 'protocol.json').read_text())
    expected_count = len(protocol['data']['eval'])
    entry = importlib.import_module("eval_IMUHOI" if mode == "offline" else "eval_batch_online")
    original = entry.evaluate_model
    output = Path(sys.argv[sys.argv.index("--output_json") + 1])

    def checked(model, loader, *args, **kwargs):
        assert len(loader.dataset) == expected_count, f"Partial evaluation dataset: {len(loader.dataset)}"
        records = []
        kwargs["sequence_records"] = records
        result = original(model, loader, *args, **kwargs)
        assert len(records) == expected_count, f"Partial successful object evaluation: {len(records)}"
        rows = [{"seq_path": r["seq_path"], "obj_cm": r["obj_trans_err_fusion_published_cm"],
                 "hoi_cm": r["hoi_err_fusion_cm"]} for r in records]
        assert all(math.isfinite(r[k]) for r in rows for k in ("obj_cm", "hoi_cm"))
        assert len({r["seq_path"] for r in rows}) == expected_count, "Duplicate/missing sequence identities"
        receipt = {"status": "passed", "mode": mode, "successful_unique_sequences": expected_count, "sequences": rows}
        output.with_suffix(".coverage.json").write_text(json.dumps(receipt, indent=2, allow_nan=False))
        return result

    entry.evaluate_model = checked
    entry.main()


if __name__ == "__main__":
    main()


