"""Pre-execution validation of a generated Prolog KB.

Upgrades the AIEA-Lab baseline (which catches errors at execution time) by
running the KB through a fresh `swipl` subprocess and surfacing structured
errors before any query runs. A subprocess gives true isolation — pyswip
uses a process-wide shared SWI engine, so we cannot trust in-process probes
to distinguish "this KB consulted cleanly" from "predicates left over from a
previous consult."
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


# SWI emits errors to stderr prefixed with "ERROR:" and warnings with
# "Warning:". We accept "Warning:" lines without failing — only "ERROR:" is
# fatal. The verbose_load flag in the generated header suppresses banners.
_ERROR_MARKER = "ERROR:"
_WARNING_MARKER = "Warning:"


def validate(
    kb_source: str,
    *,
    swipl_path: str = "swipl",
    timeout_s: float = 5.0,
) -> ValidationResult:
    """Dry-consult `kb_source` in a fresh swipl subprocess."""
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".pl", delete=False, encoding="utf-8"
    )
    try:
        tmp.write(kb_source)
        tmp.flush()
        tmp.close()
        try:
            proc = subprocess.run(
                [
                    swipl_path,
                    "-q",
                    "-g",
                    f"consult('{tmp.name}')",
                    "-g",
                    "halt(0)",
                    "-t",
                    "halt(1)",
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except FileNotFoundError:
            return ValidationResult(
                ok=False,
                errors=(f"swipl not found at {swipl_path!r}",),
            )
        except subprocess.TimeoutExpired:
            return ValidationResult(
                ok=False,
                errors=(f"swipl validation exceeded {timeout_s}s",),
            )

        errors = _collect(proc.stderr, _ERROR_MARKER)
        warnings = _collect(proc.stderr, _WARNING_MARKER)
        if proc.returncode != 0 and not errors:
            errors = (f"swipl exited with code {proc.returncode}", *errors)
        return ValidationResult(
            ok=not errors,
            errors=errors,
            warnings=warnings,
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _collect(stderr: str, marker: str) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in stderr.splitlines()
        if line.lstrip().startswith(marker)
    )
