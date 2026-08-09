"""URL workflows: endpoint assembly, host audits, link reports, transfer checks.

Two directions are covered on purpose. One family builds an address out of a
scheme, a host and a path and then reads the built address back, which forces
path and url capabilities into the same graph; the others take an address apart
-- scheme, host, port, path -- and turn the parts into a verdict, a report line
or a band. The wide plans branch on the parsed path and merge the branches
again, so the join is a property of the parse, not of a template.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── endpoint assembly ───────────────────────────────────────────────
    api_scheme = R("api_scheme", "scheme", "scheme the service is reached over")
    api_host = R("api_host", "host", "host the service runs on")
    resource_dir = R("resource_dir", "path_dir", "directory the resource lives in")
    resource_file = R("resource_file", "path_file", "resource the caller wants")
    out.append(Blueprint(
        workflow_id="url.endpoint_assembly",
        domain="url_processing",
        natural_user_goal=("assemble the address a service resource is reached "
                           "at and read back what that address resolves to"),
        target_description="the assembled address or a part of it",
        value_generator_id="url.endpoint",
        query_asset_family="service_endpoint",
        hard_distractor_families=("path", "string"),
        entity_family="integration",
        plans=(
            Plan("endpoint.v2", (resource_dir, resource_file, api_scheme, api_host),
                 (S("n1", "path.join", ("resource_dir", "resource_file")),
                  S("n2", "url.build", ("api_scheme", "api_host", "@n1"),
                    "address of the resource")),
                 "n2", intent="assembled_endpoint"),
            Plan("endpoint.v4", (resource_dir, resource_file, api_scheme, api_host),
                 (S("n1", "path.join", ("resource_dir", "resource_file")),
                  S("n2", "path.normalize", ("@n1",)),
                  S("n3", "url.build", ("api_scheme", "api_host", "@n2")),
                  S("n4", "url.domain_depth", ("@n3",),
                    "how many labels the host of that address has")),
                 "n4", intent="endpoint_host_labels"),
            Plan("endpoint.v6", (resource_dir, resource_file, api_scheme, api_host),
                 (S("n1", "path.join", ("resource_dir", "resource_file")),
                  S("n2", "path.normalize", ("@n1",)),
                  S("n3", "url.build", ("api_scheme", "api_host", "@n2"),
                    "address, read back as a host and as a path"),
                  S("n4", "url.domain", ("@n3",)),
                  S("n5", "url.path", ("@n3",)),
                  S("n6", "string.concat", ("@n4", "@n5"),
                    "how the endpoint is written in the routing table")),
                 "n6", intent="endpoint_routing_label"),
        )))

    # ── host audit ──────────────────────────────────────────────────────
    link = R("link", "url_link", "link the audit was given")
    host_floor = R("host_floor", "threshold_count",
                   "shortest host name the audit accepts")
    out.append(Blueprint(
        workflow_id="url.host_audit",
        domain="url_processing",
        natural_user_goal=("audit the host, port and path of a link that was "
                           "handed to the integration"),
        target_description="the host verdict, the address weight or its reference",
        value_generator_id="url.audit",
        query_asset_family="audited_link",
        hard_distractor_families=("string", "comparison"),
        boolean_balancing_strategy="threshold_band",
        entity_family="integration",
        plans=(
            Plan("audit.v3", (link, host_floor),
                 (S("n1", "url.domain", ("link",)),
                  S("n2", "string.count_length", ("@n1",)),
                  S("n3", "comparison.at_least", ("@n2", "host_floor"),
                    "is the host name long enough to be a real host?")),
                 "n3", intent="host_length_verdict"),
            Plan("audit.v5", (link,),
                 (S("n1", "url.parse_port", ("link",)),
                  S("n2", "url.domain_depth", ("link",)),
                  S("n3", "url.path", ("link",)),
                  S("n4", "path.depth", ("@n3",)),
                  S("n5", "arithmetic.sum_three", ("@n1", "@n2", "@n4"),
                    "combined weight of port, host and path")),
                 "n5", intent="address_weight"),
            Plan("audit.v10", (link,),
                 (S("n1", "url.scheme", ("link",)),
                  S("n2", "url.domain", ("link",)),
                  S("n3", "string.concat", ("@n1", "@n2"),
                    "origin of the link, needed again at the very end"),
                  S("n4", "url.path", ("link",)),
                  S("n5", "path.depth", ("@n4",)),
                  S("n6", "path.basename", ("@n4",)),
                  S("n7", "string.count_length", ("@n6",)),
                  S("n8", "url.parse_port", ("link",)),
                  S("n9", "arithmetic.sum_three", ("@n5", "@n7", "@n8")),
                  S("n10", "format.tag", ("@n3", "@n9"),
                    "audit reference for the link")),
                 "n10", intent="audit_reference"),
        )))

    # ── link report ─────────────────────────────────────────────────────
    report_places = R("report_places", "places", "decimals the report is printed with")
    terse_cut = R("terse_cut", "cut_low", "address length that still counts as terse")
    verbose_cut = R("verbose_cut", "cut_high",
                    "address length from which a link counts as verbose")
    out.append(Blueprint(
        workflow_id="url.link_report",
        domain="url_processing",
        natural_user_goal=("report where a link actually points and how much "
                           "address it spends doing so"),
        target_description="the target label, the printed weight or the length band",
        value_generator_id="url.report",
        query_asset_family="reported_link",
        hard_distractor_families=("format", "classification"),
        entity_family="integration",
        plans=(
            Plan("linkreport.v4", (link,),
                 (S("n1", "url.domain", ("link",)),
                  S("n2", "url.path", ("link",)),
                  S("n3", "path.basename", ("@n2",)),
                  S("n4", "string.concat", ("@n1", "@n3"),
                    "host and resource written as one label")),
                 "n4", intent="link_target_label"),
            Plan("linkreport.v7", (link, report_places),
                 (S("n1", "url.path", ("link",),
                    "path of the link, measured by name and by depth"),
                  S("n2", "path.basename", ("@n1",)),
                  S("n3", "string.count_length", ("@n2",)),
                  S("n4", "path.depth", ("@n1",)),
                  S("n5", "url.parse_port", ("link",)),
                  S("n6", "arithmetic.sum_three", ("@n3", "@n4", "@n5")),
                  S("n7", "format.fixed", ("@n6", "report_places"),
                    "the weight as the report prints it")),
                 "n7", intent="printed_link_weight"),
            Plan("linkreport.v8", (link, terse_cut, verbose_cut),
                 (S("n1", "url.domain", ("link",)),
                  S("n2", "string.count_length", ("@n1",)),
                  S("n3", "url.path", ("link",),
                    "path of the link, measured whole and by its leaf"),
                  S("n4", "string.count_length", ("@n3",)),
                  S("n5", "path.basename", ("@n3",)),
                  S("n6", "string.count_length", ("@n5",)),
                  S("n7", "arithmetic.sum_three", ("@n2", "@n4", "@n6")),
                  S("n8", "classification.three_bands",
                    ("@n7", "terse_cut", "verbose_cut"), "address length band")),
                 "n8", intent="address_length_band"),
        )))

    # ── transfer check ──────────────────────────────────────────────────
    mirror_scheme = R("mirror_scheme", "scheme", "scheme the mirror is served over")
    mirror_host = R("mirror_host", "host", "host of the download mirror")
    mirror_dir = R("mirror_dir", "path_dir", "directory the mirror exposes")
    path_floor = R("path_floor", "threshold_count",
                   "how many path segments a usable link must have")
    label_floor = R("label_floor", "count_small",
                    "how many host labels the mirror must have")
    out.append(Blueprint(
        workflow_id="url.transfer_check",
        domain="url_processing",
        natural_user_goal=("decide whether a download mirror is addressed well "
                           "enough to be used and describe the endpoint"),
        target_description="the link verdict, the checks met or the mirror line",
        value_generator_id="url.transfer",
        query_asset_family="download_mirror",
        hard_distractor_families=("validation", "decision"),
        boolean_balancing_strategy="threshold_band",
        entity_family="integration",
        plans=(
            Plan("transfer.v3", (link, path_floor),
                 (S("n1", "url.path", ("link",)),
                  S("n2", "path.depth", ("@n1",)),
                  S("n3", "comparison.at_least", ("@n2", "path_floor"),
                    "does the link address a deep enough resource?")),
                 "n3", intent="link_depth_verdict"),
            Plan("transfer.v7", (mirror_scheme, mirror_host, mirror_dir, label_floor),
                 (S("n1", "url.build", ("mirror_scheme", "mirror_host", "mirror_dir"),
                    "mirror address, checked by three independent rules"),
                  S("n2", "url.validate_secure", ("@n1",)),
                  S("n3", "url.path", ("@n1",)),
                  S("n4", "path.validate_absolute", ("@n3",)),
                  S("n5", "url.domain_depth", ("@n1",)),
                  S("n6", "comparison.at_least", ("@n5", "label_floor")),
                  S("n7", "decision.count_true", ("@n2", "@n4", "@n6"),
                    "how many mirror rules hold")),
                 "n7", intent="mirror_rules_met"),
            Plan("transfer.v8", (mirror_scheme, mirror_host, mirror_dir),
                 (S("n1", "path.normalize", ("mirror_dir",)),
                  S("n2", "url.build", ("mirror_scheme", "mirror_host", "@n1"),
                    "mirror address, read back as host, path and port"),
                  S("n3", "url.domain", ("@n2",)),
                  S("n4", "url.path", ("@n2",)),
                  S("n5", "path.basename", ("@n4",)),
                  S("n6", "url.parse_port", ("@n2",)),
                  S("n7", "string.concat", ("@n3", "@n5")),
                  S("n8", "format.tag", ("@n7", "@n6"),
                    "the line the mirror register keeps")),
                 "n8", intent="mirror_register_line"),
        )))

    return out
