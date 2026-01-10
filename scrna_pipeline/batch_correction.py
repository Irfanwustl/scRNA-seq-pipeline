"""
Backward-compatible API.

NOTE:
This module is deprecated.
New code should use `scrna_pipeline.steps.apply_batch_correction`.
"""

from scrna_pipeline.steps.batch_correction import apply_batch_correction

__all__ = ["apply_batch_correction"]