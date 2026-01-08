"""
Backward-compatible clustering API.
Prefer `scrna_pipeline.steps.clustering`.
"""

from scrna_pipeline.steps.clustering import cluster_and_embed

__all__ = ["cluster_and_embed"]
