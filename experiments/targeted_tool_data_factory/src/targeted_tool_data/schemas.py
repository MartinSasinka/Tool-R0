"""Pydantic data models — the canonical record contract (DESIGN.md §19)."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Track = Literal["A", "G"]  # adaptation / generalization


class ToolParam(BaseModel):
    name: str
    type: str                       # integer|number|string|array|boolean
    description: str = ""
    semantic: str = "value"
    required: bool = True
    enum: Optional[List[str]] = None
    items_type: Optional[str] = None  # for arrays


class ToolSpec(BaseModel):
    name: str
    description: str
    params: List[ToolParam]
    output_field: str = "output_0"
    output_type: str = "number"
    output_description: str = ""
    semantic_id: Optional[str] = None      # registry primitive; None never happens for executables
    surface_id: str = ""
    is_distractor: bool = False
    distractor_type: Optional[str] = None
    similarity_to_gold: Optional[Dict[str, float]] = None


class Call(BaseModel):
    name: str
    arguments: Dict[str, Any]
    label: str                      # "$var1" / "$var_1"


class GraphNode(BaseModel):
    node_id: str                    # "n1"
    semantic_id: str
    inputs: Dict[str, Any]          # param -> constant or {"ref": "n0"}
    output_type: str


class SemanticProgram(BaseModel):
    nodes: List[GraphNode]
    sink: str                       # node_id of final answer
    motif: str
    depth: int


class GenerationCell(BaseModel):
    generation_cell_id: str
    track: Track
    mode: str                       # "adaptation" | "generalization"
    call_count: int
    motif: str
    target_skill: str
    target_failure: str
    direct_argument_rate: float = 0.6
    numeric_string: bool = False
    reference_usage: bool = True
    offered_tools_bucket: str = "medium"
    hard_distractor_type: Optional[str] = None
    quota_weight: float = 1.0
    # pilot2 (engine v2)
    answer_kind: str = "float"      # float|int|bool|string|list|numeric_string
    hard_distractors: bool = True   # False -> ordinary offered set


class ValidationOutcome(BaseModel):
    passed: bool
    layer: str
    reasons: List[str] = Field(default_factory=list)


class ProbeResult(BaseModel):
    status: str = "NOT_RUN_LOCAL"   # NOT_RUN_LOCAL | P0 | P1 | P2 | P3
    structural_difficulty: Optional[float] = None
    rollouts: int = 0
    success_count: Optional[int] = None
    failure_classes: List[str] = Field(default_factory=list)
    first_tool_accuracy: Optional[float] = None
    continuation_accuracy: Optional[float] = None
    executability: Optional[float] = None
    reward_spread: Optional[float] = None
    completion_hashes: List[str] = Field(default_factory=list)


class TaskRecord(BaseModel):
    # identity + targeting
    task_id: str
    track: Track
    mode: str
    generation_cell_id: str
    target_skill: str
    target_failure_mode: str
    # language
    query: str
    template_id: str
    paraphrase_family: str
    # surface provenance (pilot2 paraphrasing)
    query_source: str = "template"          # template | openrouter_paraphrase
    query_template_original: Optional[str] = None
    paraphrase_meta: Dict[str, Any] = Field(default_factory=dict)
    # tools
    offered_tools: List[ToolSpec]
    offered_tool_count: int
    relevant_tool_count: int
    hard_distractor_count: int
    easy_distractor_count: int
    distractor_types: List[str]
    distractor_similarity: Dict[str, float] = Field(default_factory=dict)
    gold_tool_positions: List[int] = Field(default_factory=list)
    # program + oracle
    semantic_program: SemanticProgram
    graph_template_id: str
    semantic_program_family: str
    motif: str
    call_count: int
    minimal_valid_call_count: Optional[int] = None
    alternative_path_count: int = 0
    multi_path: bool = False
    shortcut_check: Dict[str, Any] = Field(default_factory=dict)
    dependency_depth: int
    canonical_calls: List[Call]
    alternative_valid_calls: List[List[Call]] = Field(default_factory=list)
    oracle_observations: List[Any]
    gold_answer: Any
    answer_type: str
    # semantic plausibility (pilot2)
    plausibility_class: str = "abstract_coherent"
    unit_trace: Dict[str, str] = Field(default_factory=dict)
    sink_unit: str = "abstract"
    # argument structure
    argument_type_pattern: List[str]
    reference_pattern: str          # e.g. "d,r|d,d" per call
    reference_arg_share: float
    numeric_string_args: int
    output_schema_pattern: List[str]
    # provenance / hygiene
    value_seed: int
    argument_skeleton_hash: str
    tool_combination_hash: str
    generator_version: str
    profile_version: str = ""
    registry_hash: str = ""
    executor_hash: str = ""
    config_hash: str = ""
    # downstream metadata
    validation: Dict[str, Any] = Field(default_factory=dict)
    student_probe_result: Optional[ProbeResult] = None
    split: Optional[str] = None
    split_group_ids: Dict[str, str] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)

    def split_groups(self) -> Dict[str, str]:
        return {
            "semantic_program_family": self.semantic_program_family,
            "graph_template_id": self.graph_template_id,
            "tool_combination": self.tool_combination_hash,
            "paraphrase_family": self.paraphrase_family,
            "argument_skeleton": self.argument_skeleton_hash,
            "value_seed": str(self.value_seed),
        }


class TargetProfile(BaseModel):
    target: str
    source: str
    n_rows: int
    profile_version: str
    profile_hash: str = ""
    call_count_dist: Dict[str, float]
    motif_dist: Dict[str, float] = Field(default_factory=dict)
    dependency_depth_dist: Dict[str, float] = Field(default_factory=dict)
    reference_task_rate: float
    reference_arg_share: float
    direct_arg_share: float
    arg_type_dist: Dict[str, float]
    numeric_string_rate: float
    answer_type_dist: Dict[str, float]
    output_field_names: Dict[str, float]
    tools_per_task: Dict[str, Any]           # {mean, p25, p50, p75, min, max, hist}
    relevant_ratio_mean: float
    tool_name_morphology: Dict[str, Any]
    tool_description_length: Dict[str, float]
    signature_similarity_mean: float
    question_length: Dict[str, float]
    hard_distractor_similarity_mean: float = 0.0
    student_failure_profile: Dict[str, Any] = Field(default_factory=dict)
