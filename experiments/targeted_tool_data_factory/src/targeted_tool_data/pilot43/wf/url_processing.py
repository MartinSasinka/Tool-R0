"""URL workflows: endpoint inventories, security reviews, mirrors, host bands.

An address is taken apart into scheme, host, port and resource path, and those
parts are put back together again into a new address. Reading the path and
reading the host are separate roots, so the plans that need both really do merge
two independent branches.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── endpoint inventory ──────────────────────────────────────────────
    link = R("endpoint", "url_link", "the address the service is reachable on")
    prefix_dir = R("gateway_root", "path_dir",
                   "the folder the gateway puts in front of every resource")
    scheme = R("gateway_scheme", "scheme", "the protocol the gateway serves")
    host = R("gateway_host", "host", "the host the gateway runs on")
    min_port = R("port_floor", "threshold_count",
                 "the lowest port the network team allows")

    out.append(Blueprint(
        workflow_id="urls.endpoint_inventory",
        domain="urls",
        natural_user_goal=("write down what an endpoint really is before it goes "
                           "behind our gateway"),
        target_description="the gateway address or its inventory entry",
        value_generator_id="urls.endpoint",
        query_asset_family="service_endpoint",
        hard_distractor_families=("url", "path"),
        entity_family="operations",
        plans=(
            Plan("ep.v2", (link,),
                 (S("n1", "url.domain", ("endpoint",)),
                  S("n2", "string.normalize_upper", ("@n1",))),
                 "n2", intent="host_for_the_sheet"),
            Plan("ep.v3", (link, host),
                 (S("n1", "url.scheme", ("endpoint",)),
                  S("n2", "url.path", ("endpoint",)),
                  S("n3", "url.build", ("@n1", "gateway_host", "@n2"))),
                 "n3", intent="rehosted_endpoint"),
            Plan("ep.v5", (link, prefix_dir, host),
                 (S("n1", "url.scheme", ("endpoint",)),
                  S("n2", "url.path", ("endpoint",)),
                  S("n3", "path.join", ("gateway_root", "@n2")),
                  S("n4", "url.build", ("@n1", "gateway_host", "@n3")),
                  S("n5", "url.parse_port", ("@n4",))),
                 "n5", intent="gateway_port"),
            Plan("ep.v7", (link, prefix_dir),
                 (S("n1", "url.scheme", ("endpoint",)),
                  S("n2", "url.domain", ("endpoint",)),
                  S("n3", "url.path", ("endpoint",)),
                  S("n4", "path.join", ("gateway_root", "@n3")),
                  S("n5", "url.build", ("@n1", "@n2", "@n4"),
                    "the rewritten address, described by the next two calls"),
                  S("n6", "url.domain_depth", ("@n5",)),
                  S("n7", "format.tag", ("@n5", "@n6"))),
                 "n7", intent="inventory_entry"),
            Plan("ep.v8", (link, prefix_dir, scheme, host, min_port),
                 (S("n1", "url.path", ("endpoint",)),
                  S("n2", "path.join", ("gateway_root", "@n1")),
                  S("n3", "url.build", ("gateway_scheme", "gateway_host", "@n2")),
                  S("n4", "url.validate_secure", ("@n3",)),
                  S("n5", "path.validate_absolute", ("@n2",)),
                  S("n6", "url.parse_port", ("@n3",)),
                  S("n7", "comparison.at_least", ("@n6", "port_floor")),
                  S("n8", "decision.count_true", ("@n4", "@n5", "@n7"))),
                 "n8", intent="gateway_checks_passed"),
        )))

    # ── security review ─────────────────────────────────────────────────
    sec_link = R("endpoint", "url_link", "the address that has to be reviewed")
    sec_port = R("port_floor", "threshold_count",
                 "the lowest port the policy still accepts")
    sec_fragment = R("host_fragment", "needle_text",
                     "the fragment an internal host name has to contain")

    out.append(Blueprint(
        workflow_id="urls.security_review",
        domain="urls",
        natural_user_goal=("review whether an address someone added is safe to "
                           "call from our side"),
        target_description="the review verdict or how many checks the address passes",
        value_generator_id="urls.review_endpoint",
        query_asset_family="endpoint_review",
        hard_distractor_families=("url", "validation"),
        boolean_balancing_strategy="calibrate_host_fragment_and_port",
        entity_family="quality",
        plans=(
            Plan("sec.v2", (sec_link, sec_port),
                 (S("n1", "url.parse_port", ("endpoint",)),
                  S("n2", "comparison.at_least", ("@n1", "port_floor"))),
                 "n2", intent="port_high_enough"),
            Plan("sec.v5", (sec_link, sec_port, sec_fragment),
                 (S("n1", "url.domain", ("endpoint",)),
                  S("n2", "string.validate_contains", ("@n1", "host_fragment")),
                  S("n3", "url.parse_port", ("endpoint",),
                    "the port is read independently of the host"),
                  S("n4", "comparison.at_least", ("@n3", "port_floor")),
                  S("n5", "boolean.and", ("@n2", "@n4"))),
                 "n5", intent="internal_and_allowed"),
            Plan("sec.v6", (sec_link, sec_port),
                 (S("n1", "url.validate_secure", ("endpoint",)),
                  S("n2", "url.path", ("endpoint",)),
                  S("n3", "path.validate_absolute", ("@n2",)),
                  S("n4", "url.parse_port", ("endpoint",)),
                  S("n5", "comparison.at_least", ("@n4", "port_floor")),
                  S("n6", "decision.count_true", ("@n1", "@n3", "@n5"))),
                 "n6", intent="checks_passed"),
            Plan("sec.v9", (sec_link,),
                 (S("n1", "url.scheme", ("endpoint",)),
                  S("n2", "url.domain", ("endpoint",)),
                  S("n3", "url.path", ("endpoint",)),
                  S("n4", "url.parse_port", ("endpoint",)),
                  S("n5", "string.count_length", ("@n2",)),
                  S("n6", "path.depth", ("@n3",)),
                  S("n7", "arithmetic.sum_three", ("@n4", "@n5", "@n6")),
                  S("n8", "string.concat", ("@n1", "@n2"),
                    "the host is needed again five calls after it was read"),
                  S("n9", "format.tag", ("@n8", "@n7"))),
                 "n9", intent="endpoint_fingerprint"),
        )))

    # ── mirror rewrite ──────────────────────────────────────────────────
    mir_link = R("published_link", "url_link", "the address the file is published on")
    mir_host = R("mirror_host", "host", "the host that will serve the mirror")
    mir_scheme = R("mirror_scheme", "scheme", "the protocol the mirror uses")
    mir_root = R("mirror_root", "path_dir", "the folder the mirror serves from")
    mir_fill = R("fill_character", "separator",
                 "the character the report pads short addresses with")

    out.append(Blueprint(
        workflow_id="urls.mirror_rewrite",
        domain="urls",
        natural_user_goal=("point a published address at our mirror and write the "
                           "result into the report column"),
        target_description="the mirrored address as the report prints it",
        value_generator_id="urls.mirror_entry",
        query_asset_family="mirror_entry",
        hard_distractor_families=("url", "string"),
        entity_family="logistics",
        plans=(
            Plan("mir.v3", (mir_link, mir_host),
                 (S("n1", "url.domain", ("published_link",)),
                  S("n2", "string.replace", ("published_link", "@n1", "mirror_host"),
                    "swap the published host for the mirror host"),
                  S("n3", "string.normalize_lower", ("@n2",))),
                 "n3", intent="mirrored_link"),
            Plan("mir.v6", (mir_link, mir_host, mir_scheme, mir_root),
                 (S("n1", "url.path", ("published_link",)),
                  S("n2", "path.normalize", ("@n1",)),
                  S("n3", "path.join", ("mirror_root", "@n2")),
                  S("n4", "url.build", ("mirror_scheme", "mirror_host", "@n3")),
                  S("n5", "url.parse_port", ("@n4",)),
                  S("n6", "format.tag", ("@n4", "@n5"))),
                 "n6", intent="mirror_with_port"),
            Plan("mir.v8", (mir_link, mir_host, mir_scheme, mir_root, mir_fill),
                 (S("n1", "url.domain", ("published_link",)),
                  S("n2", "string.replace", ("published_link", "@n1", "mirror_host")),
                  S("n3", "string.normalize_lower", ("@n2",)),
                  S("n4", "url.path", ("published_link",),
                    "the resource path is read on its own branch"),
                  S("n5", "path.join", ("mirror_root", "@n4")),
                  S("n6", "url.build", ("mirror_scheme", "mirror_host", "@n5")),
                  S("n7", "string.count_length", ("@n3",)),
                  S("n8", "format.pad", ("@n6", "@n7", "fill_character"))),
                 "n8", intent="report_column_entry"),
        )))

    # ── host classification ─────────────────────────────────────────────
    hst_link = R("endpoint", "url_link", "the address being catalogued")
    hst_cut = R("spread_cut", "threshold_ratio",
                "the proportion at which a host counts as broadly named")
    hst_low = R("simple_cut", "cut_low", "still counts as a simple address")
    hst_high = R("complex_cut", "cut_high", "counts as a complicated address")

    out.append(Blueprint(
        workflow_id="urls.host_classification",
        domain="urls",
        natural_user_goal=("sort the addresses in our catalogue by how involved "
                           "they are"),
        target_description="the band the address falls into",
        value_generator_id="urls.catalogue_entry",
        query_asset_family="address_catalogue",
        hard_distractor_families=("url", "classification"),
        entity_family="operations",
        plans=(
            Plan("hst.v5", (hst_link, hst_cut),
                 (S("n1", "url.domain", ("endpoint",)),
                  S("n2", "string.count_length", ("@n1",)),
                  S("n3", "url.domain_depth", ("endpoint",),
                    "the label count is read independently of the host text"),
                  S("n4", "rates.ratio_of", ("@n3", "@n2")),
                  S("n5", "classification.ratio_band", ("@n4", "spread_cut"))),
                 "n5", intent="host_spread_band"),
            Plan("hst.v7", (hst_link, hst_low, hst_high),
                 (S("n1", "url.domain", ("endpoint",)),
                  S("n2", "url.path", ("endpoint",)),
                  S("n3", "string.count_length", ("@n1",)),
                  S("n4", "path.depth", ("@n2",)),
                  S("n5", "url.parse_port", ("endpoint",)),
                  S("n6", "arithmetic.sum_three", ("@n3", "@n4", "@n5")),
                  S("n7", "classification.three_bands",
                    ("@n6", "simple_cut", "complex_cut"))),
                 "n7", intent="address_complexity_band"),
        )))

    return out
