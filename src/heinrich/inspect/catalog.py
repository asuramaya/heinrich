"""Provider module catalog -- ported from conker-detect modulecatalog.py."""
from __future__ import annotations

import re
from typing import Any


def catalog_provider_modules(
    provider: Any,
    *,
    pattern: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """List and group all modules exposed by a local provider.

    Raises ValueError if the provider does not implement list_modules().
    """
    from ..probe.trigger_core import describe_provider

    if not hasattr(provider, "list_modules") or not callable(provider.list_modules):
        raise ValueError("Provider does not expose list_modules(); use a provider with local module introspection support")
    modules = [str(name) for name in provider.list_modules()]
    if pattern:
        regex = re.compile(pattern)
        modules = [name for name in modules if regex.search(name)]
    groups = _group_modules(modules)
    if limit is not None:
        modules = modules[: int(limit)]
    return {
        "mode": "modulecatalog",
        "provider": describe_provider(provider),
        "pattern": pattern,
        "module_count": len(modules),
        "group_count": len(groups),
        "groups": groups[:20],
        "modules": modules,
    }


def _group_modules(modules: list[str]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for name in modules:
        parts = [part for part in name.split(".") if part]
        if not parts:
            continue
        if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
            key = ".".join(parts[:4]) if len(parts) >= 4 else ".".join(parts)
        else:
            key = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        counts[key] = counts.get(key, 0) + 1
    rows = [{"group": key, "count": int(value)} for key, value in counts.items()]
    rows.sort(key=lambda row: (-row["count"], row["group"]))
    return rows
