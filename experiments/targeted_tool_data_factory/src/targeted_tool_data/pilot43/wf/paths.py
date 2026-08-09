"""Path workflows: artefact location, file-type policy, manifest parts, nesting.

The four families differ in what they do to a path, not in what they are called:
one assembles and normalises a location, one rewrites and audits the file type,
one decomposes a manifest path into its components, and one grades how deeply an
artefact is buried. The long plans get their shape from the path algebra itself
-- the normalised path is needed by the basename, the parent and the depth
branch at once -- and the numeric tails always start from a real path property
(component count, name length, the sequence number inside the file name).
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── artefact location ───────────────────────────────────────────────
    root_dir = R("root_dir", "path_dir", "directory the archive is rooted at")
    relative_file = R("relative_file", "path_file",
                      "file path as it is recorded in the manifest")
    out.append(Blueprint(
        workflow_id="path.artifact_location",
        domain="path_processing",
        natural_user_goal=("work out where an artefact really sits once the "
                           "archive root and the recorded path are combined"),
        target_description="the resolved location or a property of it",
        value_generator_id="path.location",
        query_asset_family="artefact_location",
        hard_distractor_families=("string", "format"),
        entity_family="data_platform",
        plans=(
            Plan("location.v2", (root_dir, relative_file),
                 (S("n1", "path.join", ("root_dir", "relative_file"),
                    "root and recorded path as one location"),
                  S("n2", "path.normalize", ("@n1",),
                    "the location with . segments resolved")),
                 "n2", intent="resolved_location"),
            Plan("location.v4", (root_dir, relative_file),
                 (S("n1", "path.join", ("root_dir", "relative_file")),
                  S("n2", "path.normalize", ("@n1",)),
                  S("n3", "path.basename", ("@n2",)),
                  S("n4", "string.count_length", ("@n3",),
                    "how long the stored file name is")),
                 "n4", intent="file_name_length"),
            Plan("location.v6", (root_dir, relative_file),
                 (S("n1", "path.join", ("root_dir", "relative_file")),
                  S("n2", "path.normalize", ("@n1",),
                    "clean location, measured whole and by its leaf"),
                  S("n3", "string.count_length", ("@n2",)),
                  S("n4", "path.basename", ("@n2",)),
                  S("n5", "string.count_length", ("@n4",)),
                  S("n6", "rates.share_percent", ("@n5", "@n3"),
                    "share of the location taken up by the file name")),
                 "n6", intent="name_share_of_location"),
        )))

    # ── file-type policy ────────────────────────────────────────────────
    source_file = R("source_file", "path_file",
                    "file the loader picked up from the drop folder")
    archive_root = R("archive_root", "path_dir",
                     "root the file has to be filed under")
    required_extension = R("required_extension", "extension",
                           "extension the loader expects")
    min_depth = R("min_depth", "threshold_count",
                  "how many folders deep the file must sit")
    out.append(Blueprint(
        workflow_id="path.extension_policy",
        domain="path_processing",
        natural_user_goal=("check the file type of a pipeline input and work "
                           "out what it becomes once the policy is applied"),
        target_description="the retyped name, the depth verdict or the size gap",
        value_generator_id="path.policy",
        query_asset_family="pipeline_input",
        hard_distractor_families=("string", "comparison"),
        boolean_balancing_strategy="threshold_band",
        entity_family="data_platform",
        plans=(
            Plan("policy.v3", (source_file,),
                 (S("n1", "path.normalize", ("source_file",)),
                  S("n2", "path.extension", ("@n1",)),
                  S("n3", "string.normalize_upper", ("@n2",),
                    "file type as it appears in the policy table")),
                 "n3", intent="declared_file_type"),
            Plan("policy.v5", (archive_root, source_file, min_depth),
                 (S("n1", "path.join", ("archive_root", "source_file")),
                  S("n2", "path.normalize", ("@n1",)),
                  S("n3", "path.parent", ("@n2",)),
                  S("n4", "path.depth", ("@n3",)),
                  S("n5", "comparison.at_least", ("@n4", "min_depth"),
                    "is the file filed deeply enough?")),
                 "n5", intent="depth_verdict"),
            Plan("policy.v7", (source_file, required_extension),
                 (S("n1", "path.normalize", ("source_file",),
                    "clean path, read as a whole and by its parts"),
                  S("n2", "path.extension", ("@n1",)),
                  S("n3", "path.basename", ("@n1",)),
                  S("n4", "string.replace", ("@n3", "@n2", "required_extension"),
                    "the file name once the policy type is applied"),
                  S("n5", "string.count_length", ("@n1",)),
                  S("n6", "string.count_length", ("@n4",)),
                  S("n7", "rates.share_percent", ("@n6", "@n5"),
                    "how much of the path the retyped name occupies")),
                 "n7", intent="retyped_name_share"),
        )))

    # ── manifest segments ───────────────────────────────────────────────
    manifest_path = R("manifest_path", "path_file",
                      "path of the file the manifest lists")
    mount_dir = R("mount_dir", "path_dir", "mount the manifest is read from")
    part_separator = R("part_separator", "separator",
                       "character the path components are separated by")
    manifest_row = R("manifest_row", "text_record",
                     "delimited row the manifest holds for the artefact")
    out.append(Blueprint(
        workflow_id="path.manifest_segments",
        domain="path_processing",
        natural_user_goal=("break a manifest path into its components and "
                           "describe what those components add up to"),
        target_description="the component list, its size or a component report",
        value_generator_id="path.manifest",
        query_asset_family="path_manifest",
        hard_distractor_families=("list", "string"),
        entity_family="data_platform",
        plans=(
            Plan("segments.v4", (manifest_path, part_separator, manifest_row),
                 (S("n1", "path.basename", ("manifest_path",)),
                  S("n2", "string.concat", ("part_separator", "@n1"),
                    "the file name written as a new delimited field"),
                  S("n3", "string.concat", ("manifest_row", "@n2")),
                  S("n4", "string.split", ("@n3", "part_separator"),
                    "fields of the extended manifest row")),
                 "n4", intent="extended_row_fields"),
            Plan("segments.v6", (manifest_path, part_separator),
                 (S("n1", "path.normalize", ("manifest_path",),
                    "clean path, measured three independent ways"),
                  S("n2", "path.depth", ("@n1",)),
                  S("n3", "path.basename", ("@n1",)),
                  S("n4", "string.count_length", ("@n3",)),
                  S("n5", "string.split_count", ("@n1", "part_separator")),
                  S("n6", "arithmetic.sum_three", ("@n2", "@n4", "@n5"),
                    "combined size of the manifest entry")),
                 "n6", intent="entry_size_score"),
            Plan("segments.v9", (mount_dir, manifest_path),
                 (S("n1", "path.join", ("mount_dir", "manifest_path")),
                  S("n2", "path.normalize", ("@n1",),
                    "clean path, consumed by the name, depth and length branch"),
                  S("n3", "path.basename", ("@n2",)),
                  S("n4", "string.extract_digits", ("@n3",),
                    "sequence number carried by the file name"),
                  S("n5", "path.depth", ("@n2",)),
                  S("n6", "string.parse_number", ("@n4",)),
                  S("n7", "string.count_length", ("@n2",)),
                  S("n8", "arithmetic.sum_three", ("@n5", "@n6", "@n7")),
                  S("n9", "format.tag", ("@n3", "@n8"),
                    "manifest reference for the entry")),
                 "n9", intent="manifest_reference"),
        )))

    # ── nesting depth profile ───────────────────────────────────────────
    target_path = R("target_path", "path_file", "artefact the audit looks at")
    base_dir = R("base_dir", "path_dir", "directory the audit starts from")
    shallow_cut = R("shallow_cut", "cut_low", "depth that still counts as shallow")
    deep_cut = R("deep_cut", "cut_high", "depth from which nesting counts as deep")
    job_code = R("job_code", "identifier_code", "code of the audit job")
    out.append(Blueprint(
        workflow_id="path.depth_profile",
        domain="path_processing",
        natural_user_goal=("grade how deeply an artefact is buried and label "
                           "the audit result"),
        target_description="the nesting band, the audit tag or the size profile",
        value_generator_id="path.depth",
        query_asset_family="path_audit",
        hard_distractor_families=("classification", "arithmetic"),
        entity_family="data_platform",
        plans=(
            Plan("profile.v3", (target_path, shallow_cut, deep_cut),
                 (S("n1", "path.normalize", ("target_path",)),
                  S("n2", "path.depth", ("@n1",)),
                  S("n3", "classification.three_bands",
                    ("@n2", "shallow_cut", "deep_cut"), "nesting band")),
                 "n3", intent="nesting_band"),
            Plan("profile.v5", (target_path, job_code),
                 (S("n1", "path.normalize", ("target_path",)),
                  S("n2", "path.basename", ("@n1",)),
                  S("n3", "string.extract_digits", ("@n2",)),
                  S("n4", "string.parse_number", ("@n3",),
                    "sequence number read out of the file name"),
                  S("n5", "format.tag", ("job_code", "@n4"),
                    "audit tag for that sequence")),
                 "n5", intent="audit_tag"),
            Plan("profile.v8", (base_dir, target_path),
                 (S("n1", "path.join", ("base_dir", "target_path")),
                  S("n2", "path.normalize", ("@n1",),
                    "clean path, split into a folder branch and a name branch"),
                  S("n3", "path.parent", ("@n2",)),
                  S("n4", "path.basename", ("@n2",)),
                  S("n5", "path.depth", ("@n3",)),
                  S("n6", "string.count_length", ("@n2",)),
                  S("n7", "string.count_length", ("@n4",)),
                  S("n8", "statistics.mean_three", ("@n5", "@n6", "@n7"),
                    "average of the three nesting measures")),
                 "n8", intent="nesting_profile"),
        )))

    return out
