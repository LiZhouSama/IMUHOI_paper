"""Isolated cascade refine architecture (HP -> VC -> OT -> VC-refine -> OT-refine -> HP-refine).

Nothing in ``model/rnn`` imports this package; it subclasses the stock modules.
"""
from .ot_nofb import CascadeObjectTransModule
from .vc_refine import VCRefineModule, build_vc_refine_features
from .cascade_model import CascadeIMUHOIModel, build_cascade_model, publication_chain

__all__ = [
    "CascadeObjectTransModule",
    "VCRefineModule",
    "build_vc_refine_features",
    "CascadeIMUHOIModel",
    "build_cascade_model",
    "publication_chain",
]
