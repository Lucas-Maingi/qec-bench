"""Render a results JSON as human-readable tables.

The results file is the contract; this module is a convenience consumer for
READMEs, issues, and terminals (the dashboard renders the same file).
"""

from __future__ import annotations

import json
from pathlib import Path


def _fmt_rate(cell: dict) -> str:
    if cell["logical_errors"] == 0:
        return f"<{1 / cell['shots']:.1e}"
    return f"{cell['logical_error_rate']:.2e}"


def results_markdown(document: dict) -> str:
    """One markdown table per code distance: decoders x error rates."""
    results = document["results"]
    decoders = sorted({c["decoder"] for c in results})
    distances = sorted({c["distance"] for c in results})
    rates = sorted({c["error_rate"] for c in results})
    by_key = {(c["decoder"], c["distance"], c["error_rate"]): c for c in results}

    lines = [
        f"# {document['dataset']['name']} — logical error rate",
        "",
        f"{sum(c['shots'] for c in results):,} shots decoded; config "
        f"`{document['dataset']['config_hash']}`; `<x` means zero observed errors "
        "in a cell (rate below resolution).",
    ]
    for d in distances:
        lines += ["", f"## distance {d}", ""]
        lines.append("| decoder | " + " | ".join(f"p={p:g}" for p in rates) + " |")
        lines.append("|---" * (len(rates) + 1) + "|")
        for name in decoders:
            cells = [by_key.get((name, d, p)) for p in rates]
            row = [c and _fmt_rate(c) or "—" for c in cells]
            lines.append(f"| {name} | " + " | ".join(row) + " |")
        lines.append("")
        lines.append("| decoder (latency) | " + " | ".join(f"p={p:g}" for p in rates) + " |")
        lines.append("|---" * (len(rates) + 1) + "|")
        for name in decoders:
            cells = [by_key.get((name, d, p)) for p in rates]
            row = [c and f"{c['us_per_shot']:.1f} µs" or "—" for c in cells]
            lines.append(f"| {name} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def report(results_path: str | Path) -> str:
    return results_markdown(json.loads(Path(results_path).read_text()))
