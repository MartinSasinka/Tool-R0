"""Arithmetic workflows: pooling, packing, dispersion and scaling of quantities.

The family covers the reasoning a planner does with plain counts and quantities
that carry no rate semantics: pooling two holdings before packing them,
reconciling meter readings against their own average, measuring how far three
depots sit from their mean, and scaling a mix up to a larger batch.

The plans of one blueprint differ in how much of that chain the user actually
asks for. A two-call plan stops at the packed quantity; the long plans re-use the
pooled total several calls later, merge two independently derived branches, and
end in a verdict whose threshold is calibrated after execution.
"""
from __future__ import annotations

from typing import List

from ..blueprints import Blueprint, Plan, Role, Step

R = Role
S = Step


def blueprints() -> List[Blueprint]:
    out: List[Blueprint] = []

    # ── pooled packing ──────────────────────────────────────────────────
    pack_roles = (
        R("units_a", "quantity_units", "units held at the first depot"),
        R("units_b", "quantity_units", "units held at the second depot"),
        R("pack_size", "count_small", "units that fit on one pallet"),
        R("loose_limit", "threshold_count",
          "loose units the dispatcher is willing to ship unpalletised"),
    )
    out.append(Blueprint(
        workflow_id="arithmetic.pooled_packing",
        domain="arithmetic",
        natural_user_goal=("work out how two depots' stock palletises once it is "
                           "pooled and how much stays loose"),
        target_description="the palletised or loose quantity, or the loose-stock verdict",
        value_generator_id="arithmetic.pooled_packing",
        query_asset_family="pooled_consignment",
        hard_distractor_families=("comparison", "statistics"),
        boolean_balancing_strategy="threshold_band",
        entity_family="warehouse",
        plans=(
            Plan("pool.v2", (pack_roles[0], pack_roles[2]),
                 (S("n1", "arithmetic.divide", ("units_a", "pack_size"),
                    "pallet slots the depot's stock needs"),
                  S("n2", "rounding.to_int", ("@n1",),
                    "pallet slots the dispatcher books")),
                 "n2", intent="pallet_slots_booked"),
            Plan("pool.v5", pack_roles[:3],
                 (S("n1", "arithmetic.add", ("units_a", "units_b"),
                    "pooled stock, read again by the last call"),
                  S("n2", "arithmetic.divide", ("@n1", "pack_size")),
                  S("n3", "rounding.floor", ("@n2",),
                    "pallets that fill completely"),
                  S("n4", "arithmetic.multiply", ("@n3", "pack_size")),
                  S("n5", "arithmetic.subtract", ("@n1", "@n4"),
                    "units left loose after pooling")),
                 "n5", intent="loose_after_pooling"),
            Plan("pool.v7", pack_roles,
                 (S("n1", "arithmetic.add", ("units_a", "units_b")),
                  S("n2", "arithmetic.divide", ("@n1", "pack_size")),
                  S("n3", "rounding.floor", ("@n2",)),
                  S("n4", "arithmetic.multiply", ("@n3", "pack_size")),
                  S("n5", "arithmetic.subtract", ("@n1", "@n4")),
                  S("n6", "comparison.at_least", ("@n5", "loose_limit")),
                  S("n7", "boolean.not", ("@n6",),
                    "the pooled consignment closes out cleanly")),
                 "n7", intent="clean_closeout_verdict"),
            Plan("pool.v9", pack_roles[:3],
                 (S("n1", "arithmetic.add", ("units_a", "units_b")),
                  S("n2", "arithmetic.divide", ("@n1", "pack_size")),
                  S("n3", "rounding.floor", ("@n2",)),
                  S("n4", "arithmetic.multiply", ("@n3", "pack_size")),
                  S("n5", "arithmetic.subtract", ("@n1", "@n4"),
                    "loose units once the depots pool their stock"),
                  S("n6", "arithmetic.modulo", ("units_a", "pack_size"),
                    "loose units if the first depot packs alone"),
                  S("n7", "arithmetic.modulo", ("units_b", "pack_size")),
                  S("n8", "arithmetic.add", ("@n6", "@n7"),
                    "loose units if both depots pack alone"),
                  S("n9", "arithmetic.subtract", ("@n8", "@n5"),
                    "units pooling rescues onto full pallets")),
                 "n9", intent="pooling_saving"),
        )))

    # ── meter reading reconciliation ────────────────────────────────────
    meter_roles = (
        R("readings", "list_readings", "consumption logged by each meter"),
        R("correction", "quantity_units",
          "correction the technician adds to every reading"),
        R("spread_limit", "threshold_value",
          "spread between the highest and lowest reading that is still accepted"),
        R("skew_limit", "threshold_value",
          "gap between mean and median that still counts as a symmetric set"),
    )
    out.append(Blueprint(
        workflow_id="arithmetic.meter_reconciliation",
        domain="arithmetic",
        natural_user_goal=("judge how consistent a set of meter readings is once "
                           "the technician's correction is applied"),
        target_description="the dispersion of the readings or the consistency verdict",
        value_generator_id="arithmetic.meter_readings",
        query_asset_family="corrected_meter_log",
        hard_distractor_families=("statistics", "list"),
        boolean_balancing_strategy="threshold_band",
        entity_family="field service",
        plans=(
            Plan("meter.v3", meter_roles[:1],
                 (S("n1", "statistics.mean", ("readings",)),
                  S("n2", "statistics.stdev", ("readings",)),
                  S("n3", "arithmetic.divide", ("@n2", "@n1"),
                    "dispersion relative to the average reading")),
                 "n3", intent="relative_dispersion"),
            Plan("meter.v5", meter_roles[:2],
                 (S("n1", "list.map_offset", ("readings", "correction"),
                    "corrected readings, consumed by both branches"),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.median", ("@n1",)),
                  S("n4", "arithmetic.abs_difference", ("@n2", "@n3"),
                    "how far the mean sits from the median"),
                  S("n5", "arithmetic.divide", ("@n4", "@n2"),
                    "that skew relative to the average")),
                 "n5", intent="relative_skew"),
            Plan("meter.v8", meter_roles,
                 (S("n1", "list.map_offset", ("readings", "correction")),
                  S("n2", "statistics.mean", ("@n1",)),
                  S("n3", "statistics.median", ("@n1",)),
                  S("n4", "statistics.range", ("@n1",)),
                  S("n5", "comparison.at_least", ("@n4", "spread_limit")),
                  S("n6", "arithmetic.abs_difference", ("@n2", "@n3"),
                    "mean-to-median gap, read four calls after the mean"),
                  S("n7", "comparison.at_least", ("@n6", "skew_limit")),
                  S("n8", "boolean.and", ("@n5", "@n7"),
                    "the log is both wide and skewed")),
                 "n8", intent="irregular_log_verdict"),
        )))

    # ── depot spread review ─────────────────────────────────────────────
    spread_roles = (
        R("units_a", "quantity_units", "units held at the northern depot"),
        R("units_b", "quantity_units", "units held at the central depot"),
        R("units_c", "quantity_units", "units held at the southern depot"),
        R("low_cut", "cut_low", "deviation share that still counts as balanced"),
        R("high_cut", "cut_high", "deviation share that counts as unbalanced"),
        R("gap_limit_a", "threshold_count",
          "deviation the northern depot may show"),
        R("gap_limit_b", "threshold_count",
          "deviation the central depot may show"),
        R("gap_limit_c", "threshold_count",
          "deviation the southern depot may show"),
    )
    out.append(Blueprint(
        workflow_id="arithmetic.depot_spread_review",
        domain="arithmetic",
        natural_user_goal=("see how evenly stock is spread over three depots and "
                           "whether any of them is out of line"),
        target_description="the deviation from the network average or its band",
        value_generator_id="arithmetic.depots",
        query_asset_family="depot_network",
        hard_distractor_families=("statistics", "classification"),
        boolean_balancing_strategy="threshold_band",
        entity_family="logistics",
        plans=(
            Plan("dspread.v4", spread_roles[:3],
                 (S("n1", "statistics.mean_three",
                    ("units_a", "units_b", "units_c"),
                    "average holding, consumed by both deviation branches"),
                  S("n2", "arithmetic.abs_difference", ("units_a", "@n1")),
                  S("n3", "arithmetic.abs_difference", ("units_b", "@n1")),
                  S("n4", "comparison.max", ("@n2", "@n3"),
                    "the larger of the two deviations")),
                 "n4", intent="worst_deviation"),
            Plan("dspread.v7", spread_roles[:5],
                 (S("n1", "statistics.mean_three",
                    ("units_a", "units_b", "units_c")),
                  S("n2", "arithmetic.abs_difference", ("units_a", "@n1")),
                  S("n3", "arithmetic.abs_difference", ("units_b", "@n1")),
                  S("n4", "arithmetic.abs_difference", ("units_c", "@n1")),
                  S("n5", "arithmetic.sum_three", ("@n2", "@n3", "@n4"),
                    "total deviation across the network"),
                  S("n6", "arithmetic.divide", ("@n5", "@n1"),
                    "total deviation measured against the average"),
                  S("n7", "classification.three_bands",
                    ("@n6", "low_cut", "high_cut"))),
                 "n7", intent="balance_band"),
            Plan("dspread.v9", spread_roles[:3] + spread_roles[5:],
                 (S("n1", "statistics.mean_three",
                    ("units_a", "units_b", "units_c")),
                  S("n2", "arithmetic.abs_difference", ("units_a", "@n1")),
                  S("n3", "arithmetic.abs_difference", ("units_b", "@n1")),
                  S("n4", "arithmetic.abs_difference", ("units_c", "@n1")),
                  S("n5", "comparison.at_least", ("@n2", "gap_limit_a")),
                  S("n6", "comparison.at_least", ("@n3", "gap_limit_b")),
                  S("n7", "comparison.at_least", ("@n4", "gap_limit_c")),
                  S("n8", "decision.majority", ("@n5", "@n6", "@n7"),
                    "most depots are out of line"),
                  S("n9", "boolean.not", ("@n8",),
                    "the network is balanced overall")),
                 "n9", intent="network_balanced_verdict"),
        )))

    # ── mix scaling ─────────────────────────────────────────────────────
    mix_roles = (
        R("base_units", "quantity_units", "units the recipe needs for one batch"),
        R("scale_factor", "count_small", "batches the workshop wants to run"),
        R("allowance_units", "quantity_units", "units kept back as an allowance"),
        R("unit_word", "unit_word", "unit the workshop measures the mix in"),
        R("stock_units", "quantity_stock", "units already in the store"),
        R("bag_size", "count_small", "units in one supplier bag"),
    )
    out.append(Blueprint(
        workflow_id="arithmetic.mix_scaling",
        domain="arithmetic",
        natural_user_goal=("scale a mix up to several batches and see what has to "
                           "be bought in whole bags"),
        target_description="the scaled requirement, the leftover, or its share",
        value_generator_id="arithmetic.mix",
        query_asset_family="mix_sheet",
        hard_distractor_families=("format", "comparison"),
        entity_family="fabrication",
        plans=(
            Plan("mix.v3", mix_roles[:4],
                 (S("n1", "arithmetic.multiply", ("base_units", "scale_factor")),
                  S("n2", "arithmetic.add", ("@n1", "allowance_units")),
                  S("n3", "format.with_unit", ("@n2", "unit_word"),
                    "requirement written the way the mix sheet reads")),
                 "n3", intent="requirement_label"),
            Plan("mix.v4", (mix_roles[0], mix_roles[1], mix_roles[4],
                            mix_roles[5]),
                 (S("n1", "arithmetic.multiply", ("base_units", "scale_factor")),
                  S("n2", "arithmetic.subtract", ("@n1", "stock_units"),
                    "units the store cannot cover"),
                  S("n3", "arithmetic.divide", ("@n2", "bag_size")),
                  S("n4", "rounding.ceil", ("@n3",),
                    "whole supplier bags to order")),
                 "n4", intent="bags_to_order"),
            Plan("mix.v6", (mix_roles[0], mix_roles[1], mix_roles[4],
                            mix_roles[5]),
                 (S("n1", "arithmetic.multiply", ("base_units", "scale_factor")),
                  S("n2", "arithmetic.subtract", ("@n1", "stock_units"),
                    "shortfall to buy in, read again by the last call"),
                  S("n3", "arithmetic.divide", ("@n2", "bag_size")),
                  S("n4", "rounding.ceil", ("@n3",)),
                  S("n5", "arithmetic.multiply", ("@n4", "bag_size"),
                    "units the whole-bag order delivers"),
                  S("n6", "arithmetic.subtract", ("@n5", "@n2"),
                    "units left over once whole bags arrive")),
                 "n6", intent="bag_leftover"),
        )))

    return out
