"""v2 defaults are filtered; inherited v3 remains unaffected."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_rnn_ot_known_prefix_fast_path import _cfg
from model.rnn.object_trans_v2 import V2ObjectTransModule
from model.rnn.object_trans_v3 import V3ObjectTransModule


def test_v2_filter_default_and_explicit_legacy_override():
    v2 = V2ObjectTransModule(_cfg())
    assert v2.fk_filter_enabled
    assert (v2.fk_filter_alpha, v2.fk_filter_beta) == (.35, .05)
    old = V2ObjectTransModule(_cfg(v2_fk_filter_alpha=1., v2_fk_filter_beta=0.))
    assert not old.fk_filter_enabled
    assert v2.state_dict().keys() == old.state_dict().keys()


def test_v3_does_not_inherit_v2_filter_even_with_shared_config():
    v3 = V3ObjectTransModule(_cfg(v2_fk_filter_alpha=.35, v2_fk_filter_beta=.05))
    assert not v3.fk_filter_enabled
