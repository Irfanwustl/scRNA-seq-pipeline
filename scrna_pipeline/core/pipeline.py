"""
Core pipeline framework (minimal, extensible).

This is the backbone for your pipeline design:

- Steps are small reusable units that mutate/return AnnData.
- StepSpec declares how to run a step (required/optional/off, error policy).
- run_steps executes a list of StepSpecs and stores a provenance log in adata.uns.

Design philosophy
-----------------
- Keep the core tiny and stable.
- Put biology/Scanpy logic in steps/* modules.
- Put opinions/presets in presets/* modules.
- Keep workflows/* as executors/drivers.

This module is intentionally dependency-light (standard library only).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional, Protocol, runtime_checkable

from anndata import AnnData


Mode = Literal["required", "optional", "off"]
OnError = Literal["raise", "skip"]


@runtime_checkable
class Step(Protocol):
    """
    Any object implementing this protocol can be used as a pipeline step.

    Minimal contract:
      - has a .run(adata, ctx) -> AnnData method
    """

    def run(self, adata: AnnData, ctx: "StepContext") -> AnnData:  # pragma: no cover
        ...


@dataclass(frozen=True)
class StepContext:
    """
    Execution context for steps.

    Parameters
    ----------
    verbose
        Print progress logs.

    strict
        If True, a failed required step raises immediately.
        If False, failures are logged and execution continues whenever possible.

    notes
        Free-form dict you can use to pass global info to steps (optional).
        Example: {"dataset_name": "...", "run_id": "..."}
    """
    verbose: bool = True
    strict: bool = True
    notes: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class StepSpec:
    """
    Declarative wrapper describing how to execute a Step.

    Parameters
    ----------
    step
        The Step instance.

    name
        Unique name for logging/provenance.

    mode
        "required" (default): must succeed
        "optional": may fail (depending on on_error)
        "off": not executed

    on_error
        "raise" (default): raise exception on failure
        "skip": log failure and continue (useful for optional steps)

    tags
        Optional list of tags for later filtering/reporting.
    """
    step: Step
    name: str
    mode: Mode = "required"
    on_error: OnError = "raise"
    tags: Optional[list[str]] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_history_container(adata: AnnData) -> list[dict[str, Any]]:
    adata.uns.setdefault("scrna_pipeline", {})
    adata.uns["scrna_pipeline"].setdefault("history", [])
    hist = adata.uns["scrna_pipeline"]["history"]
    # Ensure it's list-like
    if not isinstance(hist, list):
        raise TypeError("adata.uns['scrna_pipeline']['history'] must be a list")
    return hist


def run_steps(adata: AnnData, steps: list[StepSpec], *, ctx: StepContext) -> AnnData:
    """
    Execute StepSpecs in order and record provenance in adata.uns.

    Returns
    -------
    AnnData
        The processed AnnData (whatever the final step returns).

    Provenance
    ----------
    Writes a list of dicts to:
      adata.uns["scrna_pipeline"]["history"]

    Each entry includes:
      - name, mode, on_error
      - started_at, finished_at, duration_s
      - status: "success" | "skipped" | "failed"
      - error_type, error_message (if failed)
    """
    history = _ensure_history_container(adata)

    for i, spec in enumerate(steps, start=1):
        if spec.mode == "off":
            entry = {
                "index": i,
                "name": spec.name,
                "mode": spec.mode,
                "on_error": spec.on_error,
                "tags": spec.tags or [],
                "status": "skipped",
                "started_at": None,
                "finished_at": None,
                "duration_s": 0.0,
                "error_type": None,
                "error_message": None,
            }
            history.append(entry)
            if ctx.verbose:
                print(f"[steps] {i:02d}/{len(steps):02d} {spec.name}: OFF (skipped)")
            continue

        if ctx.verbose:
            print(f"[steps] {i:02d}/{len(steps):02d} {spec.name}: start")

        started_at = _utc_now_iso()
        t0 = datetime.now(timezone.utc)

        try:
            adata = spec.step.run(adata, ctx)
            finished_at = _utc_now_iso()
            dt = (datetime.now(timezone.utc) - t0).total_seconds()

            history.append(
                {
                    "index": i,
                    "name": spec.name,
                    "mode": spec.mode,
                    "on_error": spec.on_error,
                    "tags": spec.tags or [],
                    "status": "success",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_s": float(dt),
                    "error_type": None,
                    "error_message": None,
                }
            )

            if ctx.verbose:
                print(f"[steps] {i:02d}/{len(steps):02d} {spec.name}: done ({dt:.2f}s)")

        except Exception as e:
            finished_at = _utc_now_iso()
            dt = (datetime.now(timezone.utc) - t0).total_seconds()

            history.append(
                {
                    "index": i,
                    "name": spec.name,
                    "mode": spec.mode,
                    "on_error": spec.on_error,
                    "tags": spec.tags or [],
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_s": float(dt),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            )

            if ctx.verbose:
                print(f"[steps] {i:02d}/{len(steps):02d} {spec.name}: FAILED ({dt:.2f}s) -> {type(e).__name__}: {e}")

            # Decide behavior
            should_raise = (
                spec.on_error == "raise"
                and (ctx.strict or spec.mode == "required")
            )
            if should_raise:
                raise

            # otherwise skip/continue
            continue

    return adata
