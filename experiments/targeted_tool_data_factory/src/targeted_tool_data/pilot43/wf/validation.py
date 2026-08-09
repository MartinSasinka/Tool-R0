"""Input validation over the things that actually get validated.

Asset codes, delimited scanner rows, file paths, published links, sampling
plans and retention dates -- each family binds the capability that really
inspects that kind of input (``string.validate_*``, ``path.validate_*``,
``validation.*``, ``date.compare``) instead of dressing arithmetic up as a
check. Plans that only count how many rules a record satisfies end in an
integer, so the family is not purely a stream of verdicts.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── asset code and scanner row ──────────────────────────────────────
    asset_roles = (
        R("asset_code", "identifier_code", "code stamped on the asset"),
        R("record_text", "text_record", "delimited row the scanner produced"),
        R("separator", "separator", "character the row is delimited with"),
        R("field_count_floor", "threshold_count",
          "fields the row layout is supposed to have"),
        R("code_length_floor", "threshold_count",
          "characters a well-formed code has"),
        R("serial_floor", "threshold_value",
          "serial number at which the current series starts"),
        R("reading_values", "list_readings", "readings attached to the row"),
        R("reading_ceiling", "threshold_value",
          "reading none of the attached samples may exceed"),
    )
    out.append(Blueprint(
        workflow_id="validate.asset_code",
        domain="validation",
        natural_user_goal="check whether a scanned asset row can be accepted",
        target_description="the acceptance verdict or the number of rules met",
        value_generator_id="validate.asset",
        query_asset_family="scanner_row",
        hard_distractor_families=("string", "validation"),
        boolean_balancing_strategy="threshold_band",
        entity_family="operations",
        plans=(
            Plan("asset.v3", asset_roles[1:4],
                 (S("n1", "string.split_count", ("record_text", "separator"),
                    "fields the row actually carries"),
                  S("n2", "comparison.at_least", ("@n1", "field_count_floor")),
                  S("n3", "boolean.not", ("@n2",), "the row is missing fields")),
                 "n3", intent="row_truncated"),
            Plan("asset.v7", asset_roles[:5],
                 (S("n1", "string.validate_identifier", ("asset_code",)),
                  S("n2", "string.split", ("record_text", "separator")),
                  S("n3", "list.reduce_count_text", ("@n2",)),
                  S("n4", "comparison.at_least", ("@n3", "field_count_floor")),
                  S("n5", "string.count_length", ("asset_code",)),
                  S("n6", "comparison.at_least", ("@n5", "code_length_floor")),
                  S("n7", "decision.count_true", ("@n1", "@n4", "@n6"),
                    "how many of the three intake rules the row satisfies")),
                 "n7", intent="intake_rule_count"),
            Plan("asset.v9",
                 (asset_roles[1], asset_roles[2], asset_roles[3],
                  asset_roles[0], asset_roles[5], asset_roles[6],
                  asset_roles[7]),
                 (S("n1", "string.split", ("record_text", "separator")),
                  S("n2", "list.reduce_count_text", ("@n1",)),
                  S("n3", "comparison.at_least", ("@n2", "field_count_floor")),
                  S("n4", "string.extract_digits", ("asset_code",)),
                  S("n5", "string.parse_number", ("@n4",),
                    "serial number hidden in the code"),
                  S("n6", "comparison.at_least", ("@n5", "serial_floor")),
                  S("n7", "validation.list_limit",
                    ("reading_values", "reading_ceiling")),
                  S("n8", "boolean.and", ("@n3", "@n6")),
                  S("n9", "boolean.and", ("@n8", "@n7"),
                    "layout, series and attached readings all have to hold")),
                 "n9", intent="full_row_acceptance"),
        )))

    # ── archive path and published link ─────────────────────────────────
    route_roles = (
        R("source_path", "path_file", "path the scanner wrote the file to"),
        R("archive_dir", "path_dir", "directory the archive keeps"),
        R("target_extension", "extension", "extension the archive stores"),
        R("endpoint", "url_link", "endpoint the archive is published under"),
        R("scheme", "scheme", "scheme the published link has to use"),
        R("host", "host", "host that serves the archive"),
        R("depth_floor", "threshold_count",
          "nesting level from which a file counts as buried"),
        R("archive_depth_limit", "threshold_count",
          "nesting the archive layout allows"),
    )
    out.append(Blueprint(
        workflow_id="validate.file_route",
        domain="validation",
        natural_user_goal=("check where a scanned file ends up once it is "
                           "archived and published"),
        target_description="the published link or the archiving verdict",
        value_generator_id="validate.route",
        query_asset_family="archive_route",
        hard_distractor_families=("path", "url"),
        boolean_balancing_strategy="threshold_band",
        entity_family="operations",
        plans=(
            Plan("route.v4", (route_roles[0], route_roles[6]),
                 (S("n1", "path.normalize", ("source_path",)),
                  S("n2", "path.depth", ("@n1",)),
                  S("n3", "comparison.at_least", ("@n2", "depth_floor")),
                  S("n4", "boolean.not", ("@n3",),
                    "the file is still reachable without digging")),
                 "n4", intent="not_buried"),
            Plan("route.v5",
                 (route_roles[0], route_roles[2], route_roles[3],
                  route_roles[4], route_roles[5]),
                 (S("n1", "path.normalize", ("source_path",)),
                  S("n2", "path.change_extension", ("@n1", "target_extension")),
                  S("n3", "url.path", ("endpoint",)),
                  S("n4", "path.join", ("@n3", "@n2")),
                  S("n5", "url.build", ("scheme", "host", "@n4"))),
                 "n5", intent="published_link"),
            Plan("route.v7",
                 (route_roles[0], route_roles[1], route_roles[2],
                  route_roles[7]),
                 (S("n1", "path.normalize", ("source_path",),
                    "clean source path, read again five steps later"),
                  S("n2", "path.parent", ("@n1",)),
                  S("n3", "path.join", ("archive_dir", "@n2")),
                  S("n4", "path.depth", ("@n3",)),
                  S("n5", "comparison.at_least", ("@n4", "archive_depth_limit")),
                  S("n6", "path.validate_extension",
                    ("@n1", "target_extension")),
                  S("n7", "boolean.and", ("@n5", "@n6"))),
                 "n7", intent="archive_layout_verdict"),
        )))

    # ── sampling plan and retention window ──────────────────────────────
    sample_roles = (
        R("readings", "list_readings", "readings logged for the batch"),
        R("sample_size", "count_small", "readings the sampling plan draws"),
        R("reading_ceiling", "threshold_value",
          "reading none of the drawn samples may exceed"),
        R("mean_floor", "threshold_value", "level the drawn sample must reach"),
        R("coverage_floor", "threshold_count", "samples the plan requires"),
        R("log_date", "date_iso", "date the batch was logged"),
        R("retention_days", "duration_days", "days the batch stays reviewable"),
        R("review_by", "date_deadline", "date the review has to happen by"),
    )
    out.append(Blueprint(
        workflow_id="validate.reading_batch",
        domain="validation",
        natural_user_goal=("check whether the sample drawn from a batch is "
                           "usable and still in date"),
        target_description="the sampling error or the sample verdict",
        value_generator_id="validate.sample",
        query_asset_family="sampling_plan",
        hard_distractor_families=("validation", "date"),
        boolean_balancing_strategy="threshold_band",
        entity_family="quality",
        plans=(
            Plan("sample.v4", sample_roles[:2],
                 (S("n1", "list.slice_first", ("readings", "sample_size")),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.mean", ("readings",)),
                  S("n4", "arithmetic.abs_difference", ("@n2", "@n3"),
                    "how far the sample misses the batch")),
                 "n4", intent="sampling_error"),
            Plan("sample.v6", sample_roles[:5],
                 (S("n1", "list.slice_first", ("readings", "sample_size"),
                    "the drawn sample, read again by the level check"),
                  S("n2", "validation.list_limit", ("@n1", "reading_ceiling")),
                  S("n3", "statistics.mean", ("@n1",)),
                  S("n4", "comparison.at_least", ("@n3", "mean_floor")),
                  S("n5", "comparison.at_least",
                    ("sample_size", "coverage_floor")),
                  S("n6", "decision.all_of", ("@n2", "@n4", "@n5"))),
                 "n6", intent="sample_usable"),
            Plan("sample.v8",
                 (sample_roles[0], sample_roles[1], sample_roles[2],
                  sample_roles[3], sample_roles[5], sample_roles[6],
                  sample_roles[7]),
                 (S("n1", "list.slice_first", ("readings", "sample_size"),
                    "the drawn sample, read by two later branches"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "validation.list_limit", ("@n1", "reading_ceiling")),
                  S("n4", "date.add_duration", ("log_date", "retention_days")),
                  S("n5", "date.compare", ("@n4", "review_by"),
                    "the retention window still closes before the review"),
                  S("n6", "comparison.at_least", ("@n2", "mean_floor")),
                  S("n7", "boolean.and", ("@n3", "@n6")),
                  S("n8", "boolean.and", ("@n7", "@n5"))),
                 "n8", intent="sample_usable_and_in_date"),
        )))

    return out
