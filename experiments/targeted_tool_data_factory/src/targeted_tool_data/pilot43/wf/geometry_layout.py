"""Geometry workflows: surveyed plots, panel layouts, coil fitting, footprints.

Dimensions stay in metres the whole way through, so a length only ever reaches
a length parameter and a route quoted in kilometres has to be converted before
a geometric step can touch it. Where a triangle is involved the two remaining
edges are derived from the reference edge by a stated percentage, which is what
keeps the triangle inequality satisfied instead of hoping three sampled sides
happen to close. The plans differ in how much of the shape the user needs: one
figure, the composition of the shape, or two independent limits combined.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── triangular plot survey ──────────────────────────────────────────
    plot_roles = (
        R("base_side", "length_m", "metres the surveyed base edge runs"),
        R("ridge_height", "length_m", "metres from the base edge up to the apex"),
        R("extension_percent", "percent_growth",
          "how much longer the second edge runs than the base"),
        R("taper_percent", "percent_margin",
          "how much shorter the third edge runs than the base"),
        R("plot_count", "quantity_units", "identical plots in the parcel"),
        R("fence_limit", "threshold_value", "metres of fencing available"),
        R("area_target", "threshold_value", "square metres the parcel has to yield"),
        R("cut_low", "cut_low",
          "edge share below which the plot counts as balanced"),
        R("cut_high", "cut_high",
          "edge share above which the plot counts as lopsided"),
    )
    out.append(Blueprint(
        workflow_id="geometry.triangular_plot_survey",
        domain="geometry",
        natural_user_goal=("work out what a triangular plot covers and how much "
                           "fencing its edges need"),
        target_description="the surveyed area, the edge balance or the survey verdict",
        value_generator_id="geometry.triangular_plot",
        query_asset_family="plot_survey",
        hard_distractor_families=("geometry", "rates"),
        boolean_balancing_strategy="threshold_band",
        entity_family="engineering",
        plans=(
            Plan("plot.v2", plot_roles[:2] + plot_roles[4:5],
                 (S("n1", "geometry.triangle_area", ("base_side", "ridge_height")),
                  S("n2", "arithmetic.multiply", ("@n1", "plot_count"),
                    "area of the whole parcel")),
                 "n2", intent="parcel_area"),
            Plan("plot.v5", plot_roles[:1] + plot_roles[2:4] + plot_roles[7:],
                 (S("n1", "rates.increase_by_percent",
                    ("base_side", "extension_percent"),
                    "the second edge, needed for the outline and for its share"),
                  S("n2", "rates.decrease_by_percent",
                    ("base_side", "taper_percent"), "the third edge"),
                  S("n3", "geometry.triangle_perimeter",
                    ("base_side", "@n1", "@n2")),
                  S("n4", "rates.share_percent", ("@n1", "@n3"),
                    "how much of the outline the longest edge takes"),
                  S("n5", "classification.three_bands",
                    ("@n4", "cut_low", "cut_high"))),
                 "n5", intent="edge_balance_band"),
            Plan("plot.v7", plot_roles[:4] + plot_roles[5:7],
                 (S("n1", "rates.increase_by_percent",
                    ("base_side", "extension_percent")),
                  S("n2", "rates.decrease_by_percent",
                    ("base_side", "taper_percent")),
                  S("n3", "geometry.triangle_perimeter",
                    ("base_side", "@n1", "@n2"), "fencing the plot needs"),
                  S("n4", "geometry.triangle_area", ("base_side", "ridge_height")),
                  S("n5", "comparison.at_least", ("@n3", "fence_limit")),
                  S("n6", "comparison.at_least", ("@n4", "area_target")),
                  S("n7", "boolean.and", ("@n5", "@n6"),
                    "the fencing and the yield both have to work out")),
                 "n7", intent="fence_and_yield_verdict"),
        )))

    # ── rectangular panel layout ────────────────────────────────────────
    panel_roles = (
        R("panel_width", "length_m", "metres across one panel"),
        R("panel_height", "length_m", "metres up one panel"),
        R("panel_count", "quantity_units", "panels the cladding job needs"),
        R("waste_percent", "percent_margin", "share of the sheet lost as offcuts"),
        R("places", "places", "decimals the cutting list is written to"),
        R("area_budget", "threshold_value", "square metres of sheet on order"),
        R("diagonal_limit", "threshold_value",
          "metres the hoist can take across the diagonal"),
    )
    out.append(Blueprint(
        workflow_id="geometry.rectangular_panel_layout",
        domain="geometry",
        natural_user_goal=("size up a cladding job from one panel: sheet needed, "
                           "trim length and whether the panels can be hoisted"),
        target_description="the sheet area, the hoist verdict or the cutting list",
        value_generator_id="geometry.panel_layout",
        query_asset_family="cladding_job",
        hard_distractor_families=("geometry", "arithmetic"),
        boolean_balancing_strategy="threshold_band",
        entity_family="engineering",
        plans=(
            Plan("panel.v3", panel_roles[:4],
                 (S("n1", "geometry.rectangle_area",
                    ("panel_width", "panel_height")),
                  S("n2", "arithmetic.multiply", ("@n1", "panel_count")),
                  S("n3", "rates.increase_by_percent", ("@n2", "waste_percent"),
                    "sheet to order once offcuts are allowed for")),
                 "n3", intent="sheet_area_to_order"),
            Plan("panel.v6", panel_roles[:3] + panel_roles[5:],
                 (S("n1", "geometry.rectangle_area",
                    ("panel_width", "panel_height")),
                  S("n2", "geometry.hypotenuse", ("panel_width", "panel_height"),
                    "diagonal the hoist has to clear"),
                  S("n3", "arithmetic.multiply", ("@n1", "panel_count")),
                  S("n4", "comparison.at_least", ("@n3", "area_budget")),
                  S("n5", "comparison.at_least", ("@n2", "diagonal_limit")),
                  S("n6", "boolean.and", ("@n4", "@n5"))),
                 "n6", intent="sheet_and_hoist_verdict"),
            Plan("panel.v8", panel_roles[:5],
                 (S("n1", "geometry.rectangle_area",
                    ("panel_width", "panel_height")),
                  S("n2", "geometry.rectangle_perimeter",
                    ("panel_width", "panel_height"), "trim around one panel"),
                  S("n3", "geometry.hypotenuse", ("panel_width", "panel_height"),
                    "diagonal, quoted again at the end of the list"),
                  S("n4", "arithmetic.multiply", ("@n1", "panel_count")),
                  S("n5", "rates.increase_by_percent", ("@n4", "waste_percent")),
                  S("n6", "arithmetic.multiply", ("@n2", "panel_count"),
                    "trim for the whole job"),
                  S("n7", "list.build", ("@n5", "@n6", "@n3")),
                  S("n8", "list.map_round", ("@n7", "places"))),
                 "n8", intent="cutting_list"),
        )))

    # ── circular coil fitting ───────────────────────────────────────────
    coil_roles = (
        R("coil_radius", "length_m", "metres of radius the drum winds at"),
        R("route_km", "length_km", "kilometres of route the cable has to cover"),
        R("circuits", "quantity_units", "circuits wound onto the drum"),
        R("gap_limit", "threshold_value",
          "metres by which the wound length may miss the route"),
        R("coverage_limit", "threshold_percent",
          "share of the route one drum has to cover"),
        R("coil_limit", "threshold_value",
          "metres one circuit of the drum may measure"),
    )
    out.append(Blueprint(
        workflow_id="geometry.circular_coil_fitting",
        domain="geometry",
        natural_user_goal=("check whether a drum wound at a given radius holds "
                           "enough cable for a route quoted in kilometres"),
        target_description="the shortfall in metres or the drum verdict",
        value_generator_id="geometry.coil_fitting",
        query_asset_family="cable_drum",
        hard_distractor_families=("geometry", "unit_conversion"),
        boolean_balancing_strategy="threshold_band",
        entity_family="logistics",
        plans=(
            Plan("coil.v4", coil_roles[:3],
                 (S("n1", "geometry.circumference", ("coil_radius",),
                    "metres in one circuit"),
                  S("n2", "arithmetic.multiply", ("@n1", "circuits"),
                    "metres wound onto the drum"),
                  S("n3", "unit_conversion.length_km_m", ("route_km",),
                    "the route in the unit the drum is measured in"),
                  S("n4", "arithmetic.abs_difference", ("@n3", "@n2"))),
                 "n4", intent="route_shortfall"),
            Plan("coil.v5", coil_roles[:4],
                 (S("n1", "geometry.circumference", ("coil_radius",)),
                  S("n2", "arithmetic.multiply", ("@n1", "circuits")),
                  S("n3", "unit_conversion.length_km_m", ("route_km",)),
                  S("n4", "arithmetic.abs_difference", ("@n3", "@n2")),
                  S("n5", "comparison.at_least", ("@n4", "gap_limit"))),
                 "n5", intent="shortfall_verdict"),
            Plan("coil.v9", coil_roles,
                 (S("n1", "unit_conversion.length_km_m", ("route_km",),
                    "the route in metres, read again for the coverage"),
                  S("n2", "geometry.circumference", ("coil_radius",),
                    "one circuit, checked again against the drum rating"),
                  S("n3", "arithmetic.multiply", ("@n2", "circuits")),
                  S("n4", "arithmetic.abs_difference", ("@n1", "@n3")),
                  S("n5", "rates.share_percent", ("@n3", "@n1")),
                  S("n6", "comparison.at_least", ("@n4", "gap_limit")),
                  S("n7", "comparison.at_least", ("@n5", "coverage_limit")),
                  S("n8", "comparison.at_least", ("@n2", "coil_limit")),
                  S("n9", "decision.all_of", ("@n6", "@n7", "@n8"),
                    "shortfall, coverage and drum rating all have to hold")),
                 "n9", intent="drum_fitting_compliance"),
        )))

    # ── composite footprint ─────────────────────────────────────────────
    bay_roles = (
        R("bay_width", "length_m", "metres across the bay"),
        R("bay_depth", "length_m", "metres deep the bay runs"),
        R("apex_height", "length_m", "metres from the bay up to the canopy apex"),
        R("corner_radius", "length_m", "metres of radius on the rounded apron"),
        R("bay_count", "quantity_units", "bays the site is laid out with"),
        R("share_limit", "threshold_percent",
          "share of the footprint the covered bay has to keep"),
        R("area_budget", "threshold_value", "square metres of hardstanding on site"),
    )
    out.append(Blueprint(
        workflow_id="geometry.composite_footprint",
        domain="geometry",
        natural_user_goal=("add up a footprint made of a rectangular bay, a gable "
                           "canopy and a rounded apron"),
        target_description=("the composite area, the covered share or the site "
                            "layout verdict"),
        value_generator_id="geometry.composite_bay",
        query_asset_family="site_layout",
        hard_distractor_families=("geometry", "arithmetic"),
        boolean_balancing_strategy="threshold_band",
        entity_family="warehouse",
        plans=(
            Plan("bay.v3", bay_roles[:3],
                 (S("n1", "geometry.rectangle_area", ("bay_width", "bay_depth")),
                  S("n2", "geometry.triangle_area", ("bay_width", "apex_height")),
                  S("n3", "arithmetic.add", ("@n1", "@n2"),
                    "bay and canopy together")),
                 "n3", intent="covered_area"),
            Plan("bay.v7", bay_roles[:4] + bay_roles[5:6],
                 (S("n1", "geometry.rectangle_area", ("bay_width", "bay_depth")),
                  S("n2", "geometry.triangle_area", ("bay_width", "apex_height")),
                  S("n3", "geometry.circle_area", ("corner_radius",),
                    "the rounded apron"),
                  S("n4", "arithmetic.add", ("@n1", "@n2"),
                    "covered area, compared against the whole footprint"),
                  S("n5", "arithmetic.add", ("@n4", "@n3"),
                    "footprint including the apron"),
                  S("n6", "rates.share_percent", ("@n4", "@n5")),
                  S("n7", "comparison.at_least", ("@n6", "share_limit"))),
                 "n7", intent="covered_share_verdict"),
            Plan("bay.v10", bay_roles,
                 (S("n1", "geometry.rectangle_area", ("bay_width", "bay_depth")),
                  S("n2", "geometry.triangle_area", ("bay_width", "apex_height")),
                  S("n3", "geometry.circle_area", ("corner_radius",),
                    "the apron, weighed up again much later"),
                  S("n4", "arithmetic.add", ("@n1", "@n2")),
                  S("n5", "arithmetic.add", ("@n4", "@n3")),
                  S("n6", "arithmetic.multiply", ("@n5", "bay_count"),
                    "hardstanding the whole site takes"),
                  S("n7", "comparison.at_least", ("@n6", "area_budget")),
                  S("n8", "rates.share_percent", ("@n3", "@n4"),
                    "apron against the covered area"),
                  S("n9", "comparison.at_least", ("@n8", "share_limit")),
                  S("n10", "boolean.and", ("@n7", "@n9"))),
                 "n10", intent="site_layout_verdict"),
        )))

    return out
