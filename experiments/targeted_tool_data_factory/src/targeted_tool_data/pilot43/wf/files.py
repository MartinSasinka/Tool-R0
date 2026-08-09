"""File workflows: size reporting, archiving, retention checks, batch manifests.

Everything here is a path problem with a size or a count attached: the byte
figure only becomes an answer after the location it belongs to has been built,
normalised and taken apart. The families differ in what the file properties are
used for -- a printable report line, an archive entry name, a retention verdict
or a manifest summary -- and the longer plans reuse the resolved location for
the type branch and the depth branch at the same time.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── size reporting ──────────────────────────────────────────────────
    data_file = R("data_file", "path_file", "file the report is written for")
    payload_bytes = R("payload_bytes", "bytes_size", "size of the payload in bytes")
    index_bytes = R("index_bytes", "bytes_size", "size of the sidecar index in bytes")
    copy_count = R("copy_count", "quantity_units", "number of replicas kept")
    small_cut = R("small_cut", "cut_low", "footprint that still counts as small")
    large_cut = R("large_cut", "cut_high", "footprint from which a file counts as large")
    out.append(Blueprint(
        workflow_id="file.size_report",
        domain="file_processing",
        natural_user_goal=("report how much room a stored file takes up, named "
                           "the way the storage report names it"),
        target_description="the report line, the report entry or the size band",
        value_generator_id="file.size",
        query_asset_family="stored_file",
        hard_distractor_families=("unit_conversion", "format"),
        entity_family="storage",
        plans=(
            Plan("size.v3", (data_file, payload_bytes),
                 (S("n1", "path.basename", ("data_file",)),
                  S("n2", "unit_conversion.bytes_kib", ("payload_bytes",)),
                  S("n3", "format.tag", ("@n1", "@n2"),
                    "report line naming the file and its size")),
                 "n3", intent="size_report_line"),
            Plan("size.v5", (data_file, payload_bytes, copy_count),
                 (S("n1", "path.normalize", ("data_file",)),
                  S("n2", "path.basename", ("@n1",)),
                  S("n3", "unit_conversion.bytes_kib", ("payload_bytes",)),
                  S("n4", "arithmetic.multiply", ("@n3", "copy_count"),
                    "room taken by every replica together"),
                  S("n5", "record.build", ("@n2", "@n4"),
                    "the entry the storage report holds for the file")),
                 "n5", intent="size_report_entry"),
            Plan("size.v8", (data_file, payload_bytes, index_bytes, copy_count,
                             small_cut, large_cut),
                 (S("n1", "path.normalize", ("data_file",)),
                  S("n2", "path.depth", ("@n1",)),
                  S("n3", "unit_conversion.bytes_kib", ("payload_bytes",)),
                  S("n4", "unit_conversion.bytes_kib", ("index_bytes",)),
                  S("n5", "arithmetic.add", ("@n3", "@n4"),
                    "payload and index together"),
                  S("n6", "arithmetic.multiply", ("@n5", "copy_count")),
                  S("n7", "arithmetic.divide", ("@n6", "@n2"),
                    "room used per folder level"),
                  S("n8", "classification.three_bands",
                    ("@n7", "small_cut", "large_cut"), "footprint band")),
                 "n8", intent="footprint_band"),
        )))

    # ── archive entry ───────────────────────────────────────────────────
    incoming_file = R("incoming_file", "path_file", "file that arrived in the drop")
    archive_dir = R("archive_dir", "path_dir", "root of the archive")
    archive_extension = R("archive_extension", "extension",
                          "extension the archive stores files under")
    entry_copies = R("entry_copies", "quantity_units", "copies written per entry")
    out.append(Blueprint(
        workflow_id="file.archive_entry",
        domain="file_processing",
        natural_user_goal=("work out the entry an incoming file becomes once it "
                           "is filed into the archive"),
        target_description="the entry name, its type marker or its reference",
        value_generator_id="file.archive",
        query_asset_family="archive_entry",
        hard_distractor_families=("string", "path"),
        entity_family="storage",
        plans=(
            Plan("entry.v4", (archive_dir, incoming_file, archive_extension),
                 (S("n1", "path.join", ("archive_dir", "incoming_file")),
                  S("n2", "path.normalize", ("@n1",)),
                  S("n3", "path.change_extension", ("@n2", "archive_extension")),
                  S("n4", "string.normalize_slug", ("@n3",),
                    "archive entry name")),
                 "n4", intent="entry_name"),
            Plan("entry.v6", (archive_dir, incoming_file),
                 (S("n1", "path.join", ("archive_dir", "incoming_file")),
                  S("n2", "path.normalize", ("@n1",),
                    "filed location, read as a name and as a type"),
                  S("n3", "path.basename", ("@n2",)),
                  S("n4", "string.normalize_slug", ("@n3",)),
                  S("n5", "path.extension", ("@n2",)),
                  S("n6", "string.count_substring", ("@n4", "@n5"),
                    "how often the type marker survives in the entry name")),
                 "n6", intent="type_marker_count"),
            Plan("entry.v10", (archive_dir, incoming_file, archive_extension,
                               entry_copies),
                 (S("n1", "path.join", ("archive_dir", "incoming_file")),
                  S("n2", "path.normalize", ("@n1",),
                    "filed location, reused by the retype, name and depth branch"),
                  S("n3", "path.change_extension", ("@n2", "archive_extension")),
                  S("n4", "path.basename", ("@n2",)),
                  S("n5", "path.basename", ("@n3",)),
                  S("n6", "string.extract_digits", ("@n4",),
                    "sequence number the drop file carries"),
                  S("n7", "string.parse_number", ("@n6",)),
                  S("n8", "path.depth", ("@n2",)),
                  S("n9", "arithmetic.product_three",
                    ("@n7", "@n8", "entry_copies"),
                    "how many stored objects the entry accounts for"),
                  S("n10", "format.tag", ("@n5", "@n9"),
                    "archive reference for the entry")),
                 "n10", intent="entry_reference"),
        )))

    # ── retention ───────────────────────────────────────────────────────
    stored_file = R("stored_file", "path_file", "file already sitting in the archive")
    retention_dir = R("retention_dir", "path_dir", "archive the file was written to")
    keep_extension = R("keep_extension", "extension",
                       "extension the retention rule protects")
    stored_bytes = R("stored_bytes", "bytes_size", "size the file occupies in bytes")
    depth_floor = R("depth_floor", "threshold_count",
                    "how many folders deep a protected file must sit")
    out.append(Blueprint(
        workflow_id="file.retention_check",
        domain="file_processing",
        natural_user_goal=("decide whether an archived file is protected by the "
                           "retention rules and describe what is stored"),
        target_description="the retention verdict, the rules met or the entry line",
        value_generator_id="file.retention",
        query_asset_family="retained_file",
        hard_distractor_families=("validation", "decision"),
        boolean_balancing_strategy="threshold_band",
        entity_family="storage",
        plans=(
            Plan("retention.v3", (stored_file, depth_floor),
                 (S("n1", "path.normalize", ("stored_file",)),
                  S("n2", "path.depth", ("@n1",)),
                  S("n3", "comparison.at_least", ("@n2", "depth_floor"),
                    "is the file filed deeply enough to be protected?")),
                 "n3", intent="depth_protection_verdict"),
            Plan("retention.v6", (stored_file, keep_extension),
                 (S("n1", "path.normalize", ("stored_file",),
                    "clean path, checked by three independent rules"),
                  S("n2", "path.validate_absolute", ("@n1",)),
                  S("n3", "path.validate_extension", ("@n1", "keep_extension")),
                  S("n4", "path.stem", ("@n1",)),
                  S("n5", "string.validate_identifier", ("@n4",)),
                  S("n6", "decision.count_true", ("@n2", "@n3", "@n5"),
                    "how many retention rules the file satisfies")),
                 "n6", intent="rules_satisfied"),
            Plan("retention.v8", (retention_dir, stored_file, stored_bytes),
                 (S("n1", "path.join", ("retention_dir", "stored_file")),
                  S("n2", "path.normalize", ("@n1",),
                    "clean path, split into a folder branch and a type branch"),
                  S("n3", "path.parent", ("@n2",)),
                  S("n4", "path.basename", ("@n3",)),
                  S("n5", "path.extension", ("@n2",)),
                  S("n6", "string.concat", ("@n4", "@n5")),
                  S("n7", "unit_conversion.bytes_kib", ("stored_bytes",)),
                  S("n8", "format.tag", ("@n6", "@n7"),
                    "retention register line for the file")),
                 "n8", intent="retention_register_line"),
        )))

    # ── batch manifest ──────────────────────────────────────────────────
    manifest_file = R("manifest_file", "path_file", "manifest the batch was written to")
    batch_codes = R("batch_codes", "text_list_codes", "codes of the files in the batch")
    dataset_label = R("dataset_label", "text_label", "name the batch is filed under")
    field_separator = R("field_separator", "separator",
                        "character the manifest joins codes with")
    batch_copies = R("batch_copies", "quantity_units", "copies written for the batch")
    out.append(Blueprint(
        workflow_id="file.batch_manifest",
        domain="file_processing",
        natural_user_goal=("summarise the batch a manifest describes and label "
                           "it the way the batch register does"),
        target_description="the batch metrics, the register line or the nesting",
        value_generator_id="file.batch",
        query_asset_family="batch_manifest",
        hard_distractor_families=("list", "string"),
        entity_family="storage",
        plans=(
            Plan("batch.v2", (manifest_file,),
                 (S("n1", "path.normalize", ("manifest_file",)),
                  S("n2", "path.depth", ("@n1",),
                    "how deep the manifest itself is filed")),
                 "n2", intent="manifest_depth"),
            Plan("batch.v5", (manifest_file, batch_copies),
                 (S("n1", "path.normalize", ("manifest_file",),
                    "clean path, measured by depth and by name"),
                  S("n2", "path.depth", ("@n1",)),
                  S("n3", "path.basename", ("@n1",)),
                  S("n4", "string.count_length", ("@n3",)),
                  S("n5", "list.build", ("@n2", "@n4", "batch_copies"),
                    "the three figures the batch register records")),
                 "n5", intent="batch_metrics"),
            Plan("batch.v7", (dataset_label, manifest_file, batch_codes,
                              field_separator),
                 (S("n1", "string.normalize_title", ("dataset_label",)),
                  S("n2", "path.basename", ("manifest_file",)),
                  S("n3", "string.concat", ("@n1", "@n2")),
                  S("n4", "list.combine_join_text",
                    ("batch_codes", "field_separator")),
                  S("n5", "string.concat", ("@n3", "@n4")),
                  S("n6", "list.reduce_count_text", ("batch_codes",)),
                  S("n7", "format.tag", ("@n5", "@n6"),
                    "register line for the whole batch")),
                 "n7", intent="batch_register_line"),
        )))

    return out
