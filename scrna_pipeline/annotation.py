"""
Backward-compatible annotation API.
Prefer `scrna_pipeline.steps.annotation`.
"""

from scrna_pipeline.steps.annotation import score_markers_and_suggest_labels

__all__ = ["score_markers_and_suggest_labels"]
