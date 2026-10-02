"""FOCUS3D-first observation and candidate primitives.

This package is intentionally independent of the Kaggle notebook runtime.  It
can be tested on synthetic arrays and later called from the GPU notebook.
"""

from .observations import (
    CenterObservation,
    FocusInstance,
    UnifiedNode,
    build_unified_nodes,
    extract_focus_instances,
    match_centers_to_instances,
)
from .candidates import (
    ContinuationCandidate,
    DivisionCandidate,
    enumerate_continuations,
    enumerate_divisions,
)
from .runtime import FocusFrameObservation, FocusObservationProvider
from .pipeline import FrameCandidateLayer, build_frame_candidate_layers

__all__ = [
    "CenterObservation",
    "FocusInstance",
    "UnifiedNode",
    "build_unified_nodes",
    "extract_focus_instances",
    "match_centers_to_instances",
    "ContinuationCandidate",
    "DivisionCandidate",
    "enumerate_continuations",
    "enumerate_divisions",
    "FocusFrameObservation",
    "FocusObservationProvider",
    "FrameCandidateLayer",
    "build_frame_candidate_layers",
]
