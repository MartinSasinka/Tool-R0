"""Query contracts derived only from workflow instances and blueprints."""
from __future__ import annotations

from typing import Any, Dict

from ..repro import sha256_obj


def build_query_contract(instance: Dict[str, Any], blueprint: Any,
                         mode: str = "DOMAIN_GROUNDED_IMPLICIT") -> Dict[str, Any]:
    facts = instance["facts"]
    return {
        "workflow_id": blueprint.workflow_id,
        "domain": blueprint.domain,
        "user_goal": blueprint.user_goal,
        "entity": instance["entity"],
        "facts": [dict(facts[r]) for r in blueprint.input_roles],
        "input_roles": list(blueprint.input_roles),
        "target_role": blueprint.target_role,
        "target_semantic_type": blueprint.target_semantic_type,
        "query_mode": mode,
        "natural_language_assets": dict(blueprint.natural_language_assets),
    }


def contract_hash(contract: Dict[str, Any]) -> str:
    return sha256_obj(contract)
