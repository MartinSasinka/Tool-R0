"""Path workflows: mount layouts, link-to-path rewrites, nesting policy.

Paths are treated as structure, not as text: they are normalised, joined,
climbed and measured, and the plans that mix a link with a mount point get two
independent roots because the path of the address and the host of the address
are read separately before they meet again.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── mount layout ────────────────────────────────────────────────────
    mount = R("mount_point", "path_dir", "where the share is mounted on our side")
    recorded = R("recorded_file", "path_file", "the file path the tool recorded")
    depth_cut = R("depth_cut", "threshold_ratio",
                  "how much deeper than recorded a mounted file may sit")

    out.append(Blueprint(
        workflow_id="paths.mount_layout",
        domain="paths",
        natural_user_goal=("work out where a recorded file ends up once the share "
                           "is mounted on our side"),
        target_description="the mounted location or how much deeper it sits",
        value_generator_id="paths.mount_plan",
        query_asset_family="mount_plan",
        hard_distractor_families=("path", "arithmetic"),
        entity_family="operations",
        plans=(
            Plan("mnt.v2", (mount, recorded),
                 (S("n1", "path.join", ("mount_point", "recorded_file")),
                  S("n2", "path.depth", ("@n1",))),
                 "n2", intent="mounted_depth"),
            Plan("mnt.v4", (mount, recorded),
                 (S("n1", "path.join", ("mount_point", "recorded_file")),
                  S("n2", "path.normalize", ("@n1",)),
                  S("n3", "path.parent", ("@n2",)),
                  S("n4", "path.basename", ("@n3",))),
                 "n4", intent="containing_folder"),
            Plan("mnt.v6", (mount, recorded),
                 (S("n1", "path.normalize", ("recorded_file",),
                    "cleaned path, measured again four calls later"),
                  S("n2", "path.parent", ("@n1",)),
                  S("n3", "path.join", ("mount_point", "@n2")),
                  S("n4", "path.depth", ("@n3",)),
                  S("n5", "path.depth", ("@n1",)),
                  S("n6", "rates.ratio_of", ("@n4", "@n5"),
                    "how the mounted depth compares with the recorded one")),
                 "n6", intent="extra_nesting"),
            Plan("mnt.v7", (mount, recorded, depth_cut),
                 (S("n1", "path.normalize", ("recorded_file",)),
                  S("n2", "path.parent", ("@n1",)),
                  S("n3", "path.join", ("mount_point", "@n2")),
                  S("n4", "path.depth", ("@n3",)),
                  S("n5", "path.depth", ("@n1",)),
                  S("n6", "rates.ratio_of", ("@n4", "@n5")),
                  S("n7", "classification.ratio_band", ("@n6", "depth_cut"))),
                 "n7", intent="nesting_band"),
        )))

    # ── link to local path ──────────────────────────────────────────────
    link = R("source_link", "url_link", "the address the file is published under")
    local = R("local_root", "path_dir", "the folder the mirror is written into")
    scheme = R("mirror_scheme", "scheme", "the protocol the mirror is served over")
    host = R("mirror_host", "host", "the host that serves the mirror")

    out.append(Blueprint(
        workflow_id="paths.link_to_local",
        domain="paths",
        natural_user_goal=("turn a published address into the local folder it "
                           "should be mirrored to, and back into an address"),
        target_description="the mirrored address or how it is laid out",
        value_generator_id="paths.mirror_plan",
        query_asset_family="mirror_plan",
        hard_distractor_families=("url", "path"),
        entity_family="logistics",
        plans=(
            Plan("lnk.v3", (link, local, scheme, host),
                 (S("n1", "url.path", ("source_link",)),
                  S("n2", "path.join", ("local_root", "@n1")),
                  S("n3", "url.build", ("mirror_scheme", "mirror_host", "@n2"))),
                 "n3", intent="mirror_address"),
            Plan("lnk.v5", (link, local, scheme, host),
                 (S("n1", "url.path", ("source_link",)),
                  S("n2", "path.normalize", ("@n1",)),
                  S("n3", "path.join", ("local_root", "@n2")),
                  S("n4", "url.build", ("mirror_scheme", "mirror_host", "@n3")),
                  S("n5", "url.domain_depth", ("@n4",))),
                 "n5", intent="mirror_host_labels"),
            Plan("lnk.v9", (link, local, scheme, host),
                 (S("n1", "url.path", ("source_link",)),
                  S("n2", "path.normalize", ("@n1",),
                    "cleaned resource path, reused five calls later"),
                  S("n3", "path.depth", ("@n2",)),
                  S("n4", "url.domain_depth", ("source_link",),
                    "independent reading of the same address"),
                  S("n5", "url.parse_port", ("source_link",)),
                  S("n6", "arithmetic.sum_three", ("@n3", "@n4", "@n5")),
                  S("n7", "path.join", ("local_root", "@n2")),
                  S("n8", "url.build", ("mirror_scheme", "mirror_host", "@n7")),
                  S("n9", "format.tag", ("@n8", "@n6"))),
                 "n9", intent="mirror_entry"),
        )))

    # ── nesting policy ──────────────────────────────────────────────────
    pol_file = R("stored_file", "path_file", "the path the file is stored under")
    pol_root = R("policy_root", "path_dir", "the root the policy applies to")
    pol_ext = R("allowed_format", "extension", "the only format the root accepts")
    pol_limit = R("depth_limit", "threshold_count",
                  "how many levels the policy allows")
    pol_low = R("shallow_cut", "cut_low", "still counts as a flat layout")
    pol_high = R("deep_cut", "cut_high", "counts as an over-nested layout")

    out.append(Blueprint(
        workflow_id="paths.nesting_policy",
        domain="paths",
        natural_user_goal=("check a stored file against the folder policy for the "
                           "shared drive"),
        target_description="the policy verdict or the layout band",
        value_generator_id="paths.policy_check",
        query_asset_family="folder_policy",
        hard_distractor_families=("path", "validation"),
        boolean_balancing_strategy="calibrate_depth_and_format",
        entity_family="facilities",
        plans=(
            Plan("pol.v3", (pol_file, pol_low, pol_high),
                 (S("n1", "path.normalize", ("stored_file",)),
                  S("n2", "path.depth", ("@n1",)),
                  S("n3", "classification.three_bands",
                    ("@n2", "shallow_cut", "deep_cut"))),
                 "n3", intent="layout_band"),
            Plan("pol.v6", (pol_file, pol_root, pol_ext, pol_limit),
                 (S("n1", "path.join", ("policy_root", "stored_file")),
                  S("n2", "path.normalize", ("@n1",),
                    "the path the policy is actually applied to"),
                  S("n3", "path.depth", ("@n2",)),
                  S("n4", "comparison.at_least", ("@n3", "depth_limit")),
                  S("n5", "path.validate_extension", ("@n2", "allowed_format")),
                  S("n6", "boolean.xor", ("@n4", "@n5"))),
                 "n6", intent="one_rule_broken"),
            Plan("pol.v9", (pol_file, pol_root),
                 (S("n1", "path.normalize", ("stored_file",),
                    "cleaned path, mounted again six calls later"),
                  S("n2", "path.parent", ("@n1",)),
                  S("n3", "path.parent", ("@n2",)),
                  S("n4", "path.depth", ("@n3",)),
                  S("n5", "path.basename", ("@n2",)),
                  S("n6", "string.count_length", ("@n5",)),
                  S("n7", "path.join", ("policy_root", "@n1")),
                  S("n8", "path.depth", ("@n7",)),
                  S("n9", "arithmetic.sum_three", ("@n4", "@n6", "@n8"))),
                 "n9", intent="path_budget"),
        )))

    return out
