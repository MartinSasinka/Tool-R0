"""Workflow blueprint modules.

Every module exposes ``blueprints() -> list[Blueprint]``. Modules are discovered
by :func:`targeted_tool_data.pilot43.blueprints.all_blueprints`, so adding a
domain is adding a file -- there is no central list to forget to update.
"""
from __future__ import annotations
