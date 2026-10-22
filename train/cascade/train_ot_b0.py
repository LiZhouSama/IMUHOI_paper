#!/usr/bin/env python3
"""Cascade stage 3: ObjectTrans v2 without state feedback (the cascade's OT, baseline "B0").

Runs the stock ``train/rnn/train_object_trans.py`` trainer unchanged (same data,
schedule, losses, early stopping, checkpoints) with one substitution: the v2
module class is ``model.cascade.CascadeObjectTransModule``, whose fusion scan
hands the bimanual-closure gate each frame's own predictor hidden state when
there is no state feedback (see model/cascade/ot_nofb.py).  Pass the config with
``object_trans_state_feedback: false`` and omit ``--joint_train``: the cascade has
no VC+OT joint stage.

    python train/cascade/train_ot_b0.py --cfg <config.yaml> --object_trans_arch v2 \
        --velocity_contact_arch v5 --hp_ckpt ... --vc_ckpt ... [stock trainer args]
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import train.rnn.train_object_trans as stock_trainer  # noqa: E402
from model.cascade.ot_nofb import CascadeObjectTransModule  # noqa: E402


def main():
    if "--joint_train" in sys.argv:
        raise SystemExit("The cascade OT stage has no joint phase; drop --joint_train")
    # main() resolves the v2 class through this module-level name at call time.
    stock_trainer.V2ObjectTransModule = CascadeObjectTransModule
    stock_trainer.main()


if __name__ == "__main__":
    main()
