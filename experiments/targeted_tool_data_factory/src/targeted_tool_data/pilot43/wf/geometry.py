"""Geometry workflows: floor covering, plot fencing, round tank shells.

Areas and lengths are both real quantities here, so the plans that price a job
have to build the area from the room and the area from the covering unit
separately and only then divide -- which is where the independent branches and
the repeated primitive come from.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── floor covering ──────────────────────────────────────────────────
    room_width_m = R("room_width_m", "length_m", "width of the room in metres")
    room_length_m = R("room_length_m", "length_m", "length of the room in metres")
    tile_width_m = R("tile_width_m", "length_m", "width of one tile in metres")
    tile_length_m = R("tile_length_m", "length_m", "length of one tile in metres")
    tile_price = R("tile_price", "money_price", "price the supplier charges per tile")
    skirting_price = R("skirting_price", "money_price",
                       "price charged for each metre of skirting")
    waste_percent = R("waste_percent", "percent_margin",
                      "extra material the fitter allows for cuts and breakage")
    tile_budget = R("tile_budget", "threshold_money",
                    "amount the owner has set aside for materials")
    unit_word = R("unit_word", "unit_word", "unit the quote is written in")

    out.append(Blueprint(
        workflow_id="geometry.floor_covering",
        domain="geometry",
        natural_user_goal=("work out how many tiles a room needs and what "
                           "covering the floor will cost"),
        target_description="the floor area, the tile count or the materials cost",
        value_generator_id="geometry.floor",
        query_asset_family="floor_plan",
        hard_distractor_families=("geometry", "rounding"),
        boolean_balancing_strategy="calibrate_material_budget",
        entity_family="property",
        plans=(
            Plan("floor.v2", (room_width_m, room_length_m, unit_word),
                 (S("n1", "geometry.rectangle_area", ("room_width_m",
                                                      "room_length_m"),
                    "area of the floor"),
                  S("n2", "format.with_unit", ("@n1", "unit_word"),
                    "the area as it appears on the plan")),
                 "n2", intent="floor_area_label"),
            Plan("floor.v4", (room_width_m, room_length_m, tile_width_m,
                              tile_length_m),
                 (S("n1", "geometry.rectangle_area", ("room_width_m",
                                                      "room_length_m")),
                  S("n2", "geometry.rectangle_area", ("tile_width_m",
                                                      "tile_length_m"),
                    "area one tile covers, worked out the same way"),
                  S("n3", "arithmetic.divide", ("@n1", "@n2")),
                  S("n4", "rounding.to_int", ("@n3",),
                    "whole tiles the room swallows")),
                 "n4", intent="tile_count"),
            Plan("floor.v6", (room_width_m, room_length_m, tile_width_m,
                              tile_length_m, tile_price, tile_budget),
                 (S("n1", "geometry.rectangle_area", ("room_width_m",
                                                      "room_length_m")),
                  S("n2", "geometry.rectangle_area", ("tile_width_m",
                                                      "tile_length_m")),
                  S("n3", "arithmetic.divide", ("@n1", "@n2")),
                  S("n4", "rounding.to_int", ("@n3",)),
                  S("n5", "arithmetic.multiply", ("tile_price", "@n4"),
                    "what the tiles cost"),
                  S("n6", "comparison.at_least", ("@n5", "tile_budget"))),
                 "n6", intent="materials_budget_verdict"),
            Plan("floor.v9", (room_width_m, room_length_m, tile_width_m,
                              tile_length_m, tile_price, waste_percent,
                              skirting_price),
                 (S("n1", "geometry.rectangle_area", ("room_width_m",
                                                      "room_length_m")),
                  S("n2", "geometry.rectangle_area", ("tile_width_m",
                                                      "tile_length_m")),
                  S("n3", "arithmetic.divide", ("@n1", "@n2")),
                  S("n4", "rounding.to_int", ("@n3",)),
                  S("n5", "arithmetic.multiply", ("tile_price", "@n4")),
                  S("n6", "rates.increase_by_percent", ("@n5",
                                                        "waste_percent"),
                    "tiles including the allowance for cuts"),
                  S("n7", "geometry.rectangle_perimeter", ("room_width_m",
                                                           "room_length_m"),
                    "skirting run around the room"),
                  S("n8", "arithmetic.multiply", ("skirting_price", "@n7")),
                  S("n9", "arithmetic.add", ("@n6", "@n8"),
                    "everything the job needs bought")),
                 "n9", intent="full_material_cost"),
        )))

    # ── plot fencing ────────────────────────────────────────────────────
    plot_width_m = R("plot_width_m", "length_m", "width of the plot in metres")
    plot_depth_m = R("plot_depth_m", "length_m", "depth of the plot in metres")
    fence_price = R("fence_price", "money_price",
                    "price charged for each metre of fencing")
    path_price = R("path_price", "money_price",
                   "price charged for each metre of the diagonal path")
    install_percent = R("install_percent", "percent_margin",
                        "installation charge added to the fencing")
    run_limit = R("run_limit", "threshold_value",
                  "kilometres of fencing the contractor can deliver in one visit")
    share_limit = R("share_limit", "threshold_percent",
                    "share of the boundary the shortcut may represent")
    cost_limit = R("cost_limit", "threshold_money",
                   "amount the owner is prepared to spend")

    out.append(Blueprint(
        workflow_id="geometry.plot_fencing",
        domain="geometry",
        natural_user_goal=("price fencing the boundary of a plot and laying a "
                           "path straight across it"),
        target_description="the boundary run, its cost or the affordability verdict",
        value_generator_id="geometry.fencing",
        query_asset_family="plot_survey",
        hard_distractor_families=("geometry", "unit_conversion"),
        boolean_balancing_strategy="calibrate_fencing_limits",
        entity_family="property",
        plans=(
            Plan("fence.v3", (plot_width_m, plot_depth_m),
                 (S("n1", "geometry.hypotenuse", ("plot_width_m",
                                                  "plot_depth_m"),
                    "the diagonal straight across the plot"),
                  S("n2", "geometry.rectangle_perimeter", ("plot_width_m",
                                                           "plot_depth_m"),
                    "the boundary, measured on its own"),
                  S("n3", "arithmetic.add", ("@n1", "@n2"),
                    "everything that has to be laid")),
                 "n3", intent="total_run_metres"),
            Plan("fence.v5", (plot_width_m, plot_depth_m, fence_price,
                              path_price),
                 (S("n1", "geometry.rectangle_perimeter", ("plot_width_m",
                                                           "plot_depth_m")),
                  S("n2", "arithmetic.multiply", ("fence_price", "@n1")),
                  S("n3", "geometry.hypotenuse", ("plot_width_m",
                                                  "plot_depth_m")),
                  S("n4", "arithmetic.multiply", ("path_price", "@n3")),
                  S("n5", "arithmetic.add", ("@n2", "@n4"),
                    "the whole groundworks quote")),
                 "n5", intent="groundworks_quote"),
            Plan("fence.v7", (plot_width_m, plot_depth_m, run_limit,
                              share_limit),
                 (S("n1", "geometry.rectangle_perimeter", ("plot_width_m",
                                                           "plot_depth_m"),
                    "the boundary, read twice"),
                  S("n2", "unit_conversion.length_m_km", ("@n1",)),
                  S("n3", "comparison.at_least", ("@n2", "run_limit")),
                  S("n4", "geometry.hypotenuse", ("plot_width_m",
                                                  "plot_depth_m")),
                  S("n5", "rates.share_percent", ("@n4", "@n1"),
                    "how much of a boundary walk the shortcut saves"),
                  S("n6", "comparison.at_least", ("@n5", "share_limit")),
                  S("n7", "boolean.and", ("@n3", "@n6"))),
                 "n7", intent="single_visit_verdict"),
            Plan("fence.v10", (plot_width_m, plot_depth_m, fence_price,
                               install_percent, path_price, cost_limit,
                               share_limit),
                 (S("n1", "geometry.rectangle_perimeter", ("plot_width_m",
                                                           "plot_depth_m")),
                  S("n2", "arithmetic.multiply", ("fence_price", "@n1")),
                  S("n3", "rates.increase_by_percent", ("@n2",
                                                        "install_percent")),
                  S("n4", "geometry.hypotenuse", ("plot_width_m",
                                                  "plot_depth_m")),
                  S("n5", "arithmetic.multiply", ("path_price", "@n4"),
                    "the path on its own, priced again below"),
                  S("n6", "arithmetic.add", ("@n3", "@n5")),
                  S("n7", "comparison.at_least", ("@n6", "cost_limit")),
                  S("n8", "rates.share_percent", ("@n5", "@n6"),
                    "how much of the job is the path"),
                  S("n9", "comparison.at_least", ("@n8", "share_limit")),
                  S("n10", "boolean.xor", ("@n7", "@n9"),
                    "the two objections the owner might raise rarely coincide")),
                 "n10", intent="groundworks_objection")
        )))

    # ── round tank shell ────────────────────────────────────────────────
    radius_m = R("radius_m", "length_m", "radius of the tank in metres")
    height_m = R("height_m", "length_m", "height of the tank wall in metres")
    ring_count = R("ring_count", "count_small",
                   "rings of plate the wall is built from")
    plate_area = R("plate_area", "generic_value",
                   "square metres one steel plate covers")
    plate_price = R("plate_price", "money_price", "price of one steel plate")
    band_low = R("band_low", "cut_low", "cost that still counts as a small job")
    band_high = R("band_high", "cut_high", "cost that counts as a major job")

    out.append(Blueprint(
        workflow_id="geometry.round_tank_shell",
        domain="geometry",
        natural_user_goal=("estimate how much steel a round tank's wall takes "
                           "and what that comes to"),
        target_description="the weld run, the wall-to-base ratio or the cost band",
        value_generator_id="geometry.tank_shell",
        query_asset_family="tank_drawing",
        hard_distractor_families=("geometry", "arithmetic"),
        entity_family="plant",
        plans=(
            Plan("shell.v3", (radius_m, ring_count),
                 (S("n1", "geometry.circumference", ("radius_m",),
                    "weld run around one ring"),
                  S("n2", "arithmetic.multiply", ("@n1", "ring_count"),
                    "weld run for the whole wall"),
                  S("n3", "unit_conversion.length_m_km", ("@n2",),
                    "the run in the unit the welder quotes in")),
                 "n3", intent="weld_run_km"),
            Plan("shell.v5", (radius_m, height_m),
                 (S("n1", "geometry.circle_area", ("radius_m",),
                    "the base of the tank"),
                  S("n2", "geometry.circumference", ("radius_m",)),
                  S("n3", "geometry.rectangle_area", ("@n2", "height_m"),
                    "the wall unrolled flat"),
                  S("n4", "arithmetic.divide", ("@n3", "@n1"),
                    "how much bigger the wall is than the base"),
                  S("n5", "rates.ratio_to_percent", ("@n4",),
                    "the wall as a percentage of the base")),
                 "n5", intent="wall_to_base_share"),
            Plan("shell.v8", (radius_m, height_m, plate_area, plate_price,
                              band_low, band_high),
                 (S("n1", "geometry.circumference", ("radius_m",)),
                  S("n2", "geometry.rectangle_area", ("@n1", "height_m")),
                  S("n3", "geometry.circle_area", ("radius_m",),
                    "the base, worked out independently"),
                  S("n4", "arithmetic.add", ("@n2", "@n3"),
                    "steel the tank needs altogether"),
                  S("n5", "arithmetic.divide", ("@n4", "plate_area")),
                  S("n6", "rounding.ceil", ("@n5",), "plates to order"),
                  S("n7", "arithmetic.multiply", ("plate_price", "@n6")),
                  S("n8", "classification.three_bands", ("@n7", "band_low",
                                                         "band_high"))),
                 "n8", intent="fabrication_cost_band"),
        )))

    return out
