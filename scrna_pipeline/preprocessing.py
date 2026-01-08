"""
Backward-compatible preprocessing API.

NOTE:
This module is deprecated.
New code should use `scrna_pipeline.steps.preprocessing`.
"""

from scrna_pipeline.steps.preprocessing import preprocess_to_pca

__all__ = ["preprocess_to_pca"]
