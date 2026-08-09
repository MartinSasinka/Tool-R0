"""File workflows: naming exports, auditing archives, sizing them, renaming batches.

These plans work on the file itself -- its stem, its extension, the folder it
sits in, how deep it is nested -- and only turn to arithmetic once the path has
produced a number. The recorded paths still carry the ``/./`` segments the
collector wrote, so anything structural has to normalise first.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── export naming ───────────────────────────────────────────────────
    source = R("source_file", "path_file", "where the collector wrote the file")
    target_dir = R("target_folder", "path_dir", "the folder the export has to land in")
    ext = R("export_format", "extension", "the format the export must be written in")
    depth_limit = R("depth_limit", "threshold_count",
                    "how deeply the archive is allowed to nest a file")

    out.append(Blueprint(
        workflow_id="files.export_naming",
        domain="files",
        natural_user_goal=("work out what the exported file should be called and "
                           "where it ends up once it is moved"),
        target_description="the export name or how well it follows the naming rules",
        value_generator_id="files.export_job",
        query_asset_family="export_job",
        hard_distractor_families=("path", "string"),
        entity_family="operations",
        plans=(
            Plan("exp.v2", (source,),
                 (S("n1", "path.stem", ("source_file",)),
                  S("n2", "string.normalize_slug", ("@n1",))),
                 "n2", intent="export_slug"),
            Plan("exp.v6", (source, target_dir, ext),
                 (S("n1", "path.normalize", ("source_file",),
                    "drop the ./ segments the collector left in"),
                  S("n2", "path.basename", ("@n1",)),
                  S("n3", "path.change_extension", ("@n1", "export_format")),
                  S("n4", "path.join", ("target_folder", "@n3")),
                  S("n5", "path.depth", ("@n4",)),
                  S("n6", "format.tag", ("@n2", "@n5"))),
                 "n6", intent="export_placement_tag"),
            Plan("exp.v8", (source, target_dir, ext, depth_limit),
                 (S("n1", "path.normalize", ("source_file",),
                    "cleaned path, read by three later calls"),
                  S("n2", "path.validate_extension", ("@n1", "export_format")),
                  S("n3", "path.change_extension", ("@n1", "export_format")),
                  S("n4", "path.join", ("target_folder", "@n3")),
                  S("n5", "path.validate_absolute", ("@n4",)),
                  S("n6", "path.depth", ("@n4",)),
                  S("n7", "comparison.at_least", ("@n6", "depth_limit")),
                  S("n8", "decision.count_true", ("@n2", "@n5", "@n7"))),
                 "n8", intent="naming_rules_passed"),
        )))

    # ── archive audit ───────────────────────────────────────────────────
    arc_file = R("archived_file", "path_file", "the path recorded in the archive index")
    arc_ext = R("expected_format", "extension", "the format the archive accepts")
    arc_depth = R("depth_limit", "threshold_count",
                  "how deep the archive lets a file sit")
    arc_start = R("expected_start", "prefix_text",
                  "how the file name of a compliant file begins")

    out.append(Blueprint(
        workflow_id="files.archive_audit",
        domain="files",
        natural_user_goal=("check whether a file that was archived actually "
                           "follows the archiving rules"),
        target_description="the audit verdict on the archived file",
        value_generator_id="files.archive_entry",
        query_asset_family="archive_index_entry",
        hard_distractor_families=("path", "validation"),
        boolean_balancing_strategy="calibrate_extension_and_prefix",
        entity_family="quality",
        plans=(
            Plan("arc.v3", (arc_file, arc_depth),
                 (S("n1", "path.normalize", ("archived_file",)),
                  S("n2", "path.depth", ("@n1",)),
                  S("n3", "comparison.at_least", ("@n2", "depth_limit"))),
                 "n3", intent="nested_too_deep"),
            Plan("arc.v5", (arc_file, arc_ext, arc_start),
                 (S("n1", "path.normalize", ("archived_file",)),
                  S("n2", "path.validate_extension", ("@n1", "expected_format")),
                  S("n3", "path.stem", ("@n1",)),
                  S("n4", "string.validate_prefix", ("@n3", "expected_start")),
                  S("n5", "boolean.and", ("@n2", "@n4"))),
                 "n5", intent="format_and_name_ok"),
            Plan("arc.v8", (arc_file, arc_ext, arc_depth, arc_start),
                 (S("n1", "path.normalize", ("archived_file",),
                    "cleaned path feeding three independent checks"),
                  S("n2", "path.parent", ("@n1",)),
                  S("n3", "path.depth", ("@n2",)),
                  S("n4", "comparison.at_least", ("@n3", "depth_limit")),
                  S("n5", "path.validate_extension", ("@n1", "expected_format")),
                  S("n6", "path.stem", ("@n1",)),
                  S("n7", "string.validate_prefix", ("@n6", "expected_start")),
                  S("n8", "decision.all_of", ("@n4", "@n5", "@n7"))),
                 "n8", intent="fully_compliant_archive"),
        )))

    # ── storage estimate ────────────────────────────────────────────────
    sto_file = R("sample_file", "path_file", "the file the sample was taken from")
    sto_bytes = R("file_bytes", "bytes_size", "how large that one file is in bytes")
    sto_count = R("file_count", "quantity_units",
                  "how many files of this kind the run produces")
    sto_places = R("places", "places", "how many decimals the estimate shows")

    out.append(Blueprint(
        workflow_id="files.storage_estimate",
        domain="files",
        natural_user_goal=("estimate how much space a nightly run of these files "
                           "will take and label the estimate"),
        target_description="the estimated size or the label carrying it",
        value_generator_id="files.storage_run",
        query_asset_family="storage_sample",
        hard_distractor_families=("unit_conversion", "path"),
        entity_family="operations",
        plans=(
            Plan("sto.v4", (sto_file, sto_bytes, sto_count),
                 (S("n1", "path.extension", ("sample_file",)),
                  S("n2", "unit_conversion.bytes_kib", ("file_bytes",)),
                  S("n3", "arithmetic.multiply", ("@n2", "file_count")),
                  S("n4", "format.tag", ("@n1", "@n3"))),
                 "n4", intent="size_by_format"),
            Plan("sto.v6", (sto_file, sto_bytes, sto_count, sto_places),
                 (S("n1", "path.normalize", ("sample_file",)),
                  S("n2", "path.stem", ("@n1",)),
                  S("n3", "unit_conversion.bytes_kib", ("file_bytes",)),
                  S("n4", "arithmetic.multiply", ("@n3", "file_count")),
                  S("n5", "format.fixed", ("@n4", "places")),
                  S("n6", "string.concat", ("@n2", "@n5"))),
                 "n6", intent="named_estimate"),
            Plan("sto.v9", (sto_file, sto_bytes, sto_count, sto_places),
                 (S("n1", "path.normalize", ("sample_file",),
                    "cleaned path, split into name and format"),
                  S("n2", "path.stem", ("@n1",)),
                  S("n3", "path.extension", ("@n1",)),
                  S("n4", "unit_conversion.bytes_kib", ("file_bytes",)),
                  S("n5", "arithmetic.multiply", ("@n4", "file_count")),
                  S("n6", "format.fixed", ("@n5", "places")),
                  S("n7", "string.concat", ("@n2", "@n6")),
                  S("n8", "string.concat", ("@n7", "@n3"),
                    "the format is only needed at the very end"),
                  S("n9", "string.normalize_slug", ("@n8",))),
                 "n9", intent="storage_ticket_slug"),
        )))

    # ── batch rename ────────────────────────────────────────────────────
    ren_file = R("source_file", "path_file", "one file from the batch to be renamed")
    ren_ext = R("new_format", "extension", "the format the batch is converted to")
    ren_dir = R("target_folder", "path_dir", "where the renamed batch is written")
    ren_width = R("name_width", "count_small",
                  "how many characters of the name the tool keeps")

    out.append(Blueprint(
        workflow_id="files.batch_rename",
        domain="files",
        natural_user_goal=("see what a file from the batch will be called after "
                           "the rename and conversion"),
        target_description="the new file name or its placement code",
        value_generator_id="files.rename_batch",
        query_asset_family="rename_batch",
        hard_distractor_families=("path", "string"),
        entity_family="operations",
        plans=(
            Plan("ren.v3", (ren_file, ren_width),
                 (S("n1", "path.stem", ("source_file",)),
                  S("n2", "string.normalize_slug", ("@n1",)),
                  S("n3", "string.truncate", ("@n2", "name_width"))),
                 "n3", intent="short_name"),
            Plan("ren.v5", (ren_file,),
                 (S("n1", "path.normalize", ("source_file",),
                    "cleaned path feeding both the name and the format"),
                  S("n2", "path.stem", ("@n1",)),
                  S("n3", "string.normalize_slug", ("@n2",)),
                  S("n4", "path.extension", ("@n1",)),
                  S("n5", "string.concat", ("@n3", "@n4"))),
                 "n5", intent="renamed_file"),
            Plan("ren.v9", (ren_file, ren_ext, ren_dir, ren_width),
                 (S("n1", "path.normalize", ("source_file",),
                    "cleaned path, still needed for the name five calls later"),
                  S("n2", "path.change_extension", ("@n1", "new_format")),
                  S("n3", "path.parent", ("@n2",)),
                  S("n4", "path.join", ("target_folder", "@n3")),
                  S("n5", "path.depth", ("@n4",)),
                  S("n6", "path.stem", ("@n1",)),
                  S("n7", "string.normalize_slug", ("@n6",)),
                  S("n8", "string.truncate", ("@n7", "name_width")),
                  S("n9", "format.tag", ("@n8", "@n5"))),
                 "n9", intent="rename_target_code"),
        )))

    return out
