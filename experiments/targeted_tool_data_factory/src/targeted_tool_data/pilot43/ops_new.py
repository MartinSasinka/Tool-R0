"""Pilot4.3 generic/coding ops: the capability areas Pilot4.2 only had labels for.

Every op here is deterministic, total on its declared semantic types (it raises
on inadmissible input so the executor rejects the instance instead of inventing
a value), and carries three surfaces so the surface holdout can hold out
``G_GENERAL_2`` without ever showing it during training.

Ops whose output is identical to one of their inputs (``max_two``-style
selectors, value gates) are deliberately absent: they make a strictly shorter
program value-equivalent and would be rejected by the V4 gate anyway.
"""
from __future__ import annotations

import math
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Sequence, Tuple

from . import semtypes as st
from .ops import Op, Param, Surface

_OUT = "output_0"


def _s(track: str, name: str, pnames: Sequence[str], desc: str) -> Surface:
    return Surface(track=track, name=name, param_names=tuple(pnames),
                   description=desc, output_field=_OUT)


def _op(pid: str, capability: str, params: Sequence[Tuple[str, str, str]],
        out_sem: str, fn, a: Tuple[str, Sequence[str], str],
        g1: Tuple[str, Sequence[str], str], g2: Tuple[str, Sequence[str], str],
        notes: str = "") -> Op:
    return Op(
        pid=pid, capability=capability,
        params=tuple(Param(name=n, sem=s_, role=r) for n, s_, r in params),
        out_sem=out_sem, fn=fn,
        surfaces=(_s("A_NATIVE", a[0], a[1], a[2]),
                  _s("G_GENERAL_1", g1[0], g1[1], g1[2]),
                  _s("G_GENERAL_2", g2[0], g2[1], g2[2])),
        source="pilot43", notes=notes)


# ── shared guards ────────────────────────────────────────────────────────
def _nums(values: Any) -> List[float]:
    if not isinstance(values, list) or not values:
        raise ValueError("expected a non-empty numeric list")
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("non-numeric list element")
        out.append(float(v))
    return out


def _texts(items: Any) -> List[str]:
    if not isinstance(items, list) or not items:
        raise ValueError("expected a non-empty text list")
    if any(not isinstance(x, str) for x in items):
        raise ValueError("non-text list element")
    return list(items)


def _mapping(mapping: Any) -> Dict[str, Any]:
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("expected a non-empty mapping")
    if any(not isinstance(k, str) for k in mapping):
        raise ValueError("mapping keys must be text")
    return dict(mapping)


def _num_mapping(mapping: Any) -> Dict[str, float]:
    m = _mapping(mapping)
    out = {}
    for k, v in m.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"mapping value for {k} is not numeric")
        out[k] = float(v)
    return out


def _records(records: Any) -> List[Dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise ValueError("expected a non-empty record list")
    if any(not isinstance(r, dict) for r in records):
        raise ValueError("record list element is not a record")
    return [dict(r) for r in records]


def _text(value: Any) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError("expected non-empty text")
    return value


def _iso(value: Any) -> date:
    return date.fromisoformat(_text(value))


def _int_of(value: Any, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected an integer")
    iv = int(round(float(value)))
    if abs(float(value) - iv) > 1e-9:
        raise ValueError("expected a whole number")
    if minimum is not None and iv < minimum:
        raise ValueError(f"expected >= {minimum}")
    return iv


def _r6(x: float) -> float:
    return round(float(x) + 0.0, 6)


def _flag(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("expected a boolean flag")
    return value


def _clean_segments(path_text: str) -> Tuple[bool, List[str]]:
    absolute = path_text.startswith("/")
    parts: List[str] = []
    for raw in path_text.replace("\\", "/").split("/"):
        if raw in ("", "."):
            continue
        if raw == "..":
            if parts:
                parts.pop()
            continue
        parts.append(raw)
    return absolute, parts


_URL_RE = re.compile(
    r"^(?P<scheme>[a-z][a-z0-9+.\-]*)://(?P<host>[^/:?#]+)(?::(?P<port>\d+))?"
    r"(?P<path>/[^?#]*)?(?:\?(?P<query>[^#]*))?(?:#.*)?$")


def _url(url_text: str) -> Dict[str, str]:
    m = _URL_RE.match(_text(url_text))
    if not m:
        raise ValueError(f"not a well-formed url: {url_text!r}")
    return {k: (v or "") for k, v in m.groupdict().items()}


def new_ops() -> List[Op]:  # noqa: PLR0915 - one flat registration table
    ops: List[Op] = []
    add = ops.append

    N, T, C, I = st.GENERIC, st.TEXT, st.COUNT, st.INDEX
    NL, TL, MP, RC, RL = (st.NUMBER_LIST, st.TEXT_LIST, st.MAPPING,
                          st.RECORD, st.RECORD_LIST)
    FL, PCT, RAT, CAT = st.FLAG, st.PERCENTAGE, st.RATIO, st.CATEGORY

    # ══ list ════════════════════════════════════════════════════════════
    add(_op("list_filter_below", "list.filter",
            [("values", NL, "collection"), ("threshold", N, "threshold")], NL,
            lambda values, threshold: [v for v in _nums(values)
                                       if v < float(threshold)] or
            (_ for _ in ()).throw(ValueError("filter removed every element")),
            ("filter_below_threshold", ["values", "threshold"],
             "Returns only the entries strictly below the threshold."),
            ("entries_under_limit", ["series", "limit"],
             "Keeps the readings that stay under the given limit."),
            ("values_beneath_cutoff", ["measurements", "cutoff"],
             "Selects the measurements lying beneath the cutoff.")))

    add(_op("list_map_percent", "list.map_percent",
            [("values", NL, "collection"), ("rate_percent", PCT, "rate")], NL,
            lambda values, rate_percent: [_r6(v * float(rate_percent) / 100.0)
                                          for v in _nums(values)],
            ("apply_percent_to_each", ["values", "rate_percent"],
             "Applies the percentage rate to every entry of the list."),
            ("rate_applied_series", ["series", "rate"],
             "Returns the series with the rate applied entry by entry."),
            ("percentage_per_entry", ["items", "percentage"],
             "Computes the given percentage of each item.")))

    add(_op("list_map_round", "list.map_round",
            [("values", NL, "collection"), ("places", C, "option")], NL,
            lambda values, places: [round(v, _int_of(places, minimum=0))
                                    for v in _nums(values)],
            ("round_each_value", ["values", "places"],
             "Rounds every entry to the requested number of decimals."),
            ("rounded_series", ["series", "decimals"],
             "Returns the series rounded to the given number of decimals."),
            ("normalise_precision", ["numbers", "precision"],
             "Normalises every number to a fixed precision.")))

    add(_op("list_reduce_product", "list.reduce_product",
            [("values", NL, "collection")], N,
            lambda values: _r6(math.prod(_nums(values))),
            ("product_of_values", ["values"],
             "Multiplies every entry of the list together."),
            ("combined_multiplier", ["series"],
             "Returns the product of all entries in the series."),
            ("multiply_all_entries", ["numbers"],
             "Multiplies all supplied numbers into one result.")))

    add(_op("list_count_above", "list.reduce_count_above",
            [("values", NL, "collection"), ("threshold", N, "threshold")], C,
            lambda values, threshold: sum(1 for v in _nums(values)
                                          if v > float(threshold)),
            ("count_above_threshold", ["values", "threshold"],
             "Counts how many entries exceed the threshold."),
            ("entries_over_limit_count", ["series", "limit"],
             "Reports how many readings sit above the limit."),
            ("exceedance_count", ["measurements", "cutoff"],
             "Number of measurements above the cutoff.")))

    add(_op("list_index_of_value", "list.index_of_value",
            [("values", NL, "collection"), ("value", N, "value")], I,
            lambda values, value: (
                _nums(values).index(float(value)) + 1
                if float(value) in _nums(values)
                else (_ for _ in ()).throw(ValueError("value not present"))),
            ("position_of_value", ["values", "value"],
             "Returns the 1-based position of the value in the list."),
            ("rank_of_entry", ["series", "entry"],
             "Gives the 1-based index at which the entry occurs."),
            ("locate_reading", ["measurements", "reading"],
             "Finds where the reading occurs in the measurements.")))

    add(_op("list_take_first", "list.slice_first",
            [("values", NL, "collection"), ("count", C, "option")], NL,
            lambda values, count: _nums(values)[:_int_of(count, minimum=1)],
            ("first_n_values", ["values", "count"],
             "Returns the first n entries of the list."),
            ("front_entries", ["series", "n"],
             "Takes the leading n entries of the series."),
            ("opening_segment", ["items", "size"],
             "Returns the opening segment of the given size.")))

    add(_op("list_take_last", "list.slice_last",
            [("values", NL, "collection"), ("count", C, "option")], NL,
            lambda values, count: _nums(values)[-_int_of(count, minimum=1):],
            ("last_n_values", ["values", "count"],
             "Returns the final n entries of the list."),
            ("trailing_entries", ["series", "n"],
             "Takes the trailing n entries of the series."),
            ("closing_segment", ["items", "size"],
             "Returns the closing segment of the given size.")))

    add(_op("list_pairwise_sum", "list.combine_pairwise",
            [("values_a", NL, "collection"), ("values_b", NL, "collection")], NL,
            lambda values_a, values_b: (
                [_r6(x + y) for x, y in zip(_nums(values_a), _nums(values_b))]
                if len(values_a) == len(values_b)
                else (_ for _ in ()).throw(ValueError("length mismatch"))),
            ("add_lists_elementwise", ["values_a", "values_b"],
             "Adds two equally long lists position by position."),
            ("merge_series_totals", ["first_series", "second_series"],
             "Combines two series into per-position totals."),
            ("elementwise_totals", ["left_items", "right_items"],
             "Returns position-wise totals of two item lists.")))

    add(_op("list_distinct_count", "list.reduce_distinct",
            [("values", NL, "collection")], C,
            lambda values: len({v for v in _nums(values)}),
            ("distinct_value_count", ["values"],
             "Counts how many different values the list holds."),
            ("unique_entry_count", ["series"],
             "Number of distinct entries in the series."),
            ("variety_of_readings", ["measurements"],
             "How many different measurements occur.")))

    add(_op("list_sort_ascending", "list.map_sort_asc",
            [("values", NL, "collection")], NL,
            lambda values: sorted(_nums(values)),
            ("sort_values_ascending", ["values"],
             "Sorts the list from smallest to largest."),
            ("ordered_low_to_high", ["series"],
             "Orders the series from lowest to highest."),
            ("ascending_order", ["items"],
             "Returns the items in ascending order.")))

    add(_op("list_index_of_min", "list.index_of_min",
            [("values", NL, "collection")], I,
            lambda values: _nums(values).index(min(_nums(values))) + 1,
            ("position_of_minimum", ["values"],
             "Returns the 1-based position of the smallest entry."),
            ("lowest_entry_index", ["series"],
             "Index of the lowest entry in the series."),
            ("weakest_reading_position", ["measurements"],
             "Where the weakest measurement sits.")))

    add(_op("list_second_largest", "list.reduce_second_largest",
            [("values", NL, "collection")], N,
            lambda values: (
                sorted(set(_nums(values)))[-2]
                if len(set(_nums(values))) >= 2
                else (_ for _ in ()).throw(ValueError("needs two distinct values"))),
            ("second_largest_value", ["values"],
             "Returns the second largest distinct entry."),
            ("runner_up_entry", ["series"],
             "The runner-up value of the series."),
            ("next_best_reading", ["measurements"],
             "The next best measurement after the maximum.")))

    add(_op("list_running_max", "list.map_running_max",
            [("values", NL, "collection")], NL,
            lambda values: [max(_nums(values)[:i + 1])
                            for i in range(len(_nums(values)))],
            ("running_maximum", ["values"],
             "Returns the running maximum at every position."),
            ("peak_so_far_series", ["series"],
             "Peak value observed up to each position."),
            ("cumulative_high_watermark", ["items"],
             "Cumulative high-water mark of the items.")))

    add(_op("build_value_triple", "list.build",
            [("first", N, "value"), ("second", N, "value"), ("third", N, "value")],
            NL,
            lambda first, second, third: [_r6(float(first)), _r6(float(second)),
                                          _r6(float(third))],
            ("collect_three_values", ["first", "second", "third"],
             "Collects three separate values into one list."),
            ("assemble_series", ["value_one", "value_two", "value_three"],
             "Assembles three values into a series."),
            ("gather_readings", ["reading_a", "reading_b", "reading_c"],
             "Gathers three readings into a single list.")))

    # ══ text list ════════════════════════════════════════════════════════
    add(_op("text_list_join", "list.combine_join_text",
            [("items", TL, "collection"), ("separator", T, "option")], T,
            lambda items, separator: _text(separator).join(_texts(items)),
            ("join_text_items", ["items", "separator"],
             "Joins the text items with the given separator."),
            ("concatenate_labels", ["labels", "glue"],
             "Concatenates labels using the glue string."),
            ("stitch_names", ["names", "delimiter"],
             "Stitches the names together with a delimiter.")))

    add(_op("text_list_count", "list.reduce_count_text",
            [("items", TL, "collection")], C,
            lambda items: len(_texts(items)),
            ("count_text_items", ["items"],
             "Counts the entries of a text list."),
            ("label_count", ["labels"],
             "How many labels the list contains."),
            ("name_total", ["names"],
             "Total number of names supplied.")))

    add(_op("text_list_filter_prefix", "list.filter_prefix",
            [("items", TL, "collection"), ("prefix", T, "option")], TL,
            lambda items, prefix: [x for x in _texts(items)
                                   if x.startswith(_text(prefix))] or
            (_ for _ in ()).throw(ValueError("prefix matched nothing")),
            ("filter_by_prefix", ["items", "prefix"],
             "Keeps only text items starting with the prefix."),
            ("labels_with_prefix", ["labels", "leading_text"],
             "Selects labels that begin with the leading text."),
            ("names_starting_with", ["names", "start"],
             "Names beginning with the given start string.")))

    add(_op("text_list_sorted", "list.map_sort_text",
            [("items", TL, "collection")], TL,
            lambda items: sorted(_texts(items)),
            ("sort_text_items", ["items"],
             "Sorts the text items alphabetically."),
            ("alphabetical_labels", ["labels"],
             "Returns the labels in alphabetical order."),
            ("ordered_names", ["names"],
             "Returns the names in dictionary order.")))

    add(_op("text_list_at", "list.index_text",
            [("items", TL, "collection"), ("position", I, "option")], T,
            lambda items, position: _texts(items)[
                _int_of(position, minimum=1) - 1],
            ("text_item_at", ["items", "position"],
             "Returns the text item at the 1-based position."),
            ("label_at_index", ["labels", "index"],
             "Reads the label at the given 1-based index."),
            ("name_in_position", ["names", "slot"],
             "Returns the name occupying the given slot.")))

    # ══ dictionary ═══════════════════════════════════════════════════════
    add(_op("dict_lookup_number", "dictionary.lookup",
            [("mapping", MP, "collection"), ("key", T, "key")], N,
            lambda mapping, key: _r6(_num_mapping(mapping)[_text(key)]),
            ("lookup_numeric_entry", ["mapping", "key"],
             "Reads the numeric value stored under a key."),
            ("value_for_key", ["table", "entry_name"],
             "Returns the value the table holds for the entry."),
            ("rate_from_table", ["catalogue", "item_key"],
             "Looks the item up in the catalogue and returns its number.")))

    add(_op("dict_set_number", "dictionary.update",
            [("mapping", MP, "collection"), ("key", T, "key"),
             ("value", N, "value")], MP,
            lambda mapping, key, value: {**_num_mapping(mapping),
                                         _text(key): _r6(float(value))},
            ("set_numeric_entry", ["mapping", "key", "value"],
             "Stores the value under the key and returns the updated mapping."),
            ("update_table_entry", ["table", "entry_name", "new_value"],
             "Updates one table entry and returns the whole table."),
            ("override_catalogue_value", ["catalogue", "item_key", "amount"],
             "Overrides one catalogue value and returns the catalogue.")))

    add(_op("dict_remove_key", "dictionary.update_remove",
            [("mapping", MP, "collection"), ("key", T, "key")], MP,
            lambda mapping, key: (
                {k: v for k, v in _num_mapping(mapping).items()
                 if k != _text(key)}
                if _text(key) in _num_mapping(mapping) and len(mapping) > 1
                else (_ for _ in ()).throw(ValueError("cannot remove key"))),
            ("remove_entry", ["mapping", "key"],
             "Removes one entry and returns the remaining mapping."),
            ("drop_table_row", ["table", "entry_name"],
             "Drops the named row and returns the table."),
            ("exclude_catalogue_item", ["catalogue", "item_key"],
             "Excludes one catalogue item and returns the rest.")))

    add(_op("dict_keys_sorted", "dictionary.keys",
            [("mapping", MP, "collection")], TL,
            lambda mapping: sorted(_mapping(mapping)),
            ("sorted_keys", ["mapping"],
             "Returns the mapping keys in alphabetical order."),
            ("table_entry_names", ["table"],
             "Lists the names of the table entries, ordered."),
            ("catalogue_item_keys", ["catalogue"],
             "Ordered list of catalogue item keys.")))

    add(_op("dict_key_count", "dictionary.keys_count",
            [("mapping", MP, "collection")], C,
            lambda mapping: len(_mapping(mapping)),
            ("count_keys", ["mapping"],
             "Counts the entries of the mapping."),
            ("table_size", ["table"],
             "How many entries the table holds."),
            ("catalogue_item_count", ["catalogue"],
             "Number of items in the catalogue.")))

    add(_op("dict_values_sum", "dictionary.aggregate_sum",
            [("mapping", MP, "collection")], N,
            lambda mapping: _r6(sum(_num_mapping(mapping).values())),
            ("sum_mapping_values", ["mapping"],
             "Adds up every numeric value in the mapping."),
            ("table_value_total", ["table"],
             "Total of all values stored in the table."),
            ("catalogue_amount_total", ["catalogue"],
             "Sum of all catalogue amounts.")))

    add(_op("dict_values_max", "dictionary.aggregate_max",
            [("mapping", MP, "collection")], N,
            lambda mapping: _r6(max(_num_mapping(mapping).values())),
            ("max_mapping_value", ["mapping"],
             "Returns the largest numeric value in the mapping."),
            ("table_peak_value", ["table"],
             "Highest value held by the table."),
            ("catalogue_largest_amount", ["catalogue"],
             "Largest amount in the catalogue.")))

    add(_op("dict_key_of_max", "dictionary.aggregate_argmax",
            [("mapping", MP, "collection")], T,
            lambda mapping: max(sorted(_num_mapping(mapping)),
                                key=lambda k: _num_mapping(mapping)[k]),
            ("key_of_max_value", ["mapping"],
             "Returns the key whose value is the largest."),
            ("table_leading_entry", ["table"],
             "Name of the table entry with the highest value."),
            ("top_catalogue_item", ["catalogue"],
             "Key of the catalogue item with the largest amount.")))

    add(_op("dict_filter_above", "dictionary.aggregate_filter",
            [("mapping", MP, "collection"), ("threshold", N, "threshold")], MP,
            lambda mapping, threshold: (
                {k: v for k, v in _num_mapping(mapping).items()
                 if v > float(threshold)} or
                (_ for _ in ()).throw(ValueError("filter emptied the mapping"))),
            ("filter_mapping_above", ["mapping", "threshold"],
             "Keeps only entries whose value exceeds the threshold."),
            ("table_rows_over_limit", ["table", "limit"],
             "Returns the table rows above the limit."),
            ("catalogue_items_above", ["catalogue", "cutoff"],
             "Catalogue items whose amount is above the cutoff.")))

    add(_op("dict_values_list", "dictionary.values",
            [("mapping", MP, "collection")], NL,
            lambda mapping: [_r6(_num_mapping(mapping)[k])
                             for k in sorted(_num_mapping(mapping))],
            ("mapping_values", ["mapping"],
             "Returns the mapping values ordered by key."),
            ("table_value_series", ["table"],
             "Values of the table as a series ordered by entry name."),
            ("catalogue_amounts", ["catalogue"],
             "Catalogue amounts ordered by item key.")))

    # ══ record / record list ═════════════════════════════════════════════
    add(_op("record_field_number", "record.select",
            [("record", RC, "collection"), ("field", T, "key")], N,
            lambda record, field: (
                _r6(float(record[_text(field)]))
                if isinstance(record, dict) and _text(field) in record
                and isinstance(record[_text(field)], (int, float))
                and not isinstance(record[_text(field)], bool)
                else (_ for _ in ()).throw(ValueError("no numeric field"))),
            ("read_record_number", ["record", "field"],
             "Reads a numeric field from a single record."),
            ("row_numeric_field", ["row", "column"],
             "Returns the numeric column of the row."),
            ("entry_amount_field", ["entry", "attribute"],
             "Reads the numeric attribute of the entry.")))

    add(_op("record_field_text", "record.select_text",
            [("record", RC, "collection"), ("field", T, "key")], T,
            lambda record, field: (
                record[_text(field)]
                if isinstance(record, dict)
                and isinstance(record.get(_text(field)), str)
                else (_ for _ in ()).throw(ValueError("no text field"))),
            ("read_record_text", ["record", "field"],
             "Reads a text field from a single record."),
            ("row_text_field", ["row", "column"],
             "Returns the text column of the row."),
            ("entry_label_field", ["entry", "attribute"],
             "Reads the text attribute of the entry.")))

    add(_op("records_find", "record.lookup",
            [("records", RL, "collection"), ("field", T, "key"),
             ("value", T, "key")], RC,
            lambda records, field, value: next(
                (r for r in _records(records) if r.get(_text(field)) == _text(value)),
                None) or (_ for _ in ()).throw(ValueError("no matching record")),
            ("find_record_by_field", ["records", "field", "value"],
             "Returns the first record whose field equals the value."),
            ("row_matching_column", ["rows", "column", "target"],
             "Finds the row whose column matches the target."),
            ("entry_with_label", ["entries", "attribute", "label"],
             "Returns the entry carrying the given label.")))

    add(_op("records_sum_field", "record.aggregate_sum",
            [("records", RL, "collection"), ("field", T, "key")], N,
            lambda records, field: _r6(sum(
                float(r[_text(field)]) for r in _records(records)
                if isinstance(r.get(_text(field)), (int, float))
                and not isinstance(r.get(_text(field)), bool))) if all(
                isinstance(r.get(_text(field)), (int, float))
                and not isinstance(r.get(_text(field)), bool)
                for r in _records(records)) else (_ for _ in ()).throw(
                ValueError("field missing in some record")),
            ("sum_record_field", ["records", "field"],
             "Adds the given numeric field across all records."),
            ("column_total", ["rows", "column"],
             "Total of one column across all rows."),
            ("entry_attribute_total", ["entries", "attribute"],
             "Sums one attribute over all entries.")))

    add(_op("records_max_field", "record.aggregate_max",
            [("records", RL, "collection"), ("field", T, "key")], N,
            lambda records, field: _r6(max(
                float(r[_text(field)]) for r in _records(records)
                if isinstance(r.get(_text(field)), (int, float))
                and not isinstance(r.get(_text(field)), bool))),
            ("max_record_field", ["records", "field"],
             "Largest value of the given field across the records."),
            ("column_peak", ["rows", "column"],
             "Peak value of one column across the rows."),
            ("entry_attribute_peak", ["entries", "attribute"],
             "Highest value of one attribute over all entries.")))

    add(_op("records_mean_field", "record.aggregate_mean",
            [("records", RL, "collection"), ("field", T, "key")], N,
            lambda records, field: _r6(sum(
                float(r[_text(field)]) for r in _records(records)) /
                len(_records(records))) if all(
                isinstance(r.get(_text(field)), (int, float))
                and not isinstance(r.get(_text(field)), bool)
                for r in _records(records)) else (_ for _ in ()).throw(
                ValueError("field missing in some record")),
            ("mean_record_field", ["records", "field"],
             "Average of the given field across the records."),
            ("column_average", ["rows", "column"],
             "Average of one column across the rows."),
            ("entry_attribute_average", ["entries", "attribute"],
             "Mean of one attribute over all entries.")))

    add(_op("records_count_above", "record.aggregate_count",
            [("records", RL, "collection"), ("field", T, "key"),
             ("minimum", N, "threshold")], C,
            lambda records, field, minimum: sum(
                1 for r in _records(records)
                if isinstance(r.get(_text(field)), (int, float))
                and not isinstance(r.get(_text(field)), bool)
                and float(r[_text(field)]) > float(minimum)),
            ("count_records_above", ["records", "field", "minimum"],
             "Counts records whose field exceeds the minimum."),
            ("rows_over_column_limit", ["rows", "column", "limit"],
             "How many rows exceed the column limit."),
            ("entries_above_attribute", ["entries", "attribute", "cutoff"],
             "Number of entries above the attribute cutoff.")))

    add(_op("records_field_values", "record.project",
            [("records", RL, "collection"), ("field", T, "key")], NL,
            lambda records, field: [
                _r6(float(r[_text(field)])) for r in _records(records)] if all(
                isinstance(r.get(_text(field)), (int, float))
                and not isinstance(r.get(_text(field)), bool)
                for r in _records(records)) else (_ for _ in ()).throw(
                ValueError("field missing in some record")),
            ("record_field_series", ["records", "field"],
             "Projects one numeric field of every record into a list."),
            ("column_as_series", ["rows", "column"],
             "Returns one column of the rows as a series."),
            ("attribute_series", ["entries", "attribute"],
             "Collects one attribute of every entry.")))

    add(_op("records_field_texts", "record.project_text",
            [("records", RL, "collection"), ("field", T, "key")], TL,
            lambda records, field: [
                r[_text(field)] for r in _records(records)] if all(
                isinstance(r.get(_text(field)), str) for r in _records(records))
            else (_ for _ in ()).throw(ValueError("text field missing")),
            ("record_text_series", ["records", "field"],
             "Projects one text field of every record into a list."),
            ("column_labels", ["rows", "column"],
             "Returns the labels of one column of the rows."),
            ("attribute_labels", ["entries", "attribute"],
             "Collects one text attribute of every entry.")))

    add(_op("records_count", "record.aggregate_size",
            [("records", RL, "collection")], C,
            lambda records: len(_records(records)),
            ("count_records", ["records"],
             "Counts the records in the list."),
            ("row_count", ["rows"],
             "How many rows the table has."),
            ("entry_total", ["entries"],
             "Total number of entries supplied.")))

    add(_op("build_labelled_record", "record.build",
            [("label", T, "key"), ("amount", N, "value")], RC,
            lambda label, amount: {"label": _text(label),
                                   "amount": _r6(float(amount))},
            ("make_labelled_record", ["label", "amount"],
             "Builds a record holding a label and an amount."),
            ("compose_row", ["name", "value"],
             "Composes a row from a name and a value."),
            ("assemble_entry", ["entry_name", "entry_amount"],
             "Assembles an entry from a name and an amount.")))

    # ══ string ═══════════════════════════════════════════════════════════
    add(_op("text_trim_collapse", "string.normalize_whitespace",
            [("text", T, "text")], T,
            lambda text: re.sub(r"\s+", " ", _text(text)).strip(),
            ("normalise_whitespace", ["text"],
             "Collapses repeated whitespace and trims the text."),
            ("tidy_spacing", ["input_text"],
             "Removes redundant spacing from the text."),
            ("clean_spacing", ["raw_text"],
             "Returns the text with spacing cleaned up.")))

    add(_op("text_slug", "string.normalize_slug",
            [("text", T, "text")], st.IDENTIFIER,
            lambda text: re.sub(r"-+", "-", re.sub(
                r"[^a-z0-9]+", "-", _text(text).lower())).strip("-") or
            (_ for _ in ()).throw(ValueError("slug is empty")),
            ("to_slug", ["text"],
             "Converts the text into a lowercase hyphenated slug."),
            ("url_safe_label", ["input_text"],
             "Produces a url-safe label from the text."),
            ("handle_from_text", ["raw_text"],
             "Derives a lowercase handle from the text.")))

    add(_op("text_title_case", "string.normalize_title",
            [("text", T, "text")], T,
            lambda text: " ".join(w.capitalize() for w in _text(text).split()),
            ("to_title_case", ["text"],
             "Capitalises the first letter of every word."),
            ("display_name_case", ["input_text"],
             "Formats the text the way a display name is written."),
            ("headline_case", ["raw_text"],
             "Returns the text in headline capitalisation.")))

    add(_op("text_lower", "string.normalize_lower",
            [("text", T, "text")], T,
            lambda text: _text(text).lower(),
            ("to_lower_case", ["text"], "Converts the text to lower case."),
            ("lowercased_text", ["input_text"],
             "Returns a lowercased copy of the text."),
            ("normalise_case", ["raw_text"],
             "Normalises the text to lower case.")))

    add(_op("text_split_take", "string.split_take",
            [("text", T, "text"), ("separator", T, "option"),
             ("position", I, "option")], T,
            lambda text, separator, position: (
                _text(text).split(_text(separator))[
                    _int_of(position, minimum=1) - 1]),
            ("split_and_take", ["text", "separator", "position"],
             "Splits the text and returns the part at the 1-based position."),
            ("field_from_delimited", ["record_text", "delimiter", "field_index"],
             "Reads one field out of a delimited string."),
            ("segment_of_text", ["source_text", "split_on", "segment_index"],
             "Returns one segment of the split text.")))

    add(_op("text_split_count", "string.split_count",
            [("text", T, "text"), ("separator", T, "option")], C,
            lambda text, separator: len(_text(text).split(_text(separator))),
            ("count_split_parts", ["text", "separator"],
             "Counts the parts the text splits into."),
            ("delimited_field_count", ["record_text", "delimiter"],
             "How many fields the delimited string has."),
            ("segment_count", ["source_text", "split_on"],
             "Number of segments after splitting.")))

    add(_op("text_split_parts", "string.split",
            [("text", T, "text"), ("separator", T, "option")], TL,
            lambda text, separator: _text(text).split(_text(separator)),
            ("split_into_parts", ["text", "separator"],
             "Splits the text into a list of parts."),
            ("delimited_fields", ["record_text", "delimiter"],
             "Returns all fields of the delimited string."),
            ("segments_of_text", ["source_text", "split_on"],
             "All segments produced by splitting the text.")))

    add(_op("text_digits_only", "string.extract_digits",
            [("text", T, "text")], st.NUMERIC_TEXT,
            lambda text: re.sub(r"\D", "", _text(text)) or
            (_ for _ in ()).throw(ValueError("no digits in text")),
            ("extract_digits", ["text"],
             "Returns only the digit characters of the text."),
            ("numeric_part_of", ["input_text"],
             "Extracts the numeric part of the text."),
            ("digits_from_code", ["code_text"],
             "Keeps just the digits found in the code.")))

    add(_op("text_between", "string.extract_between",
            [("text", T, "text"), ("start_marker", T, "option"),
             ("end_marker", T, "option")], T,
            lambda text, start_marker, end_marker: (
                _text(text).split(_text(start_marker), 1)[1]
                .split(_text(end_marker), 1)[0]
                if _text(start_marker) in _text(text)
                and _text(end_marker) in _text(text).split(
                    _text(start_marker), 1)[1]
                else (_ for _ in ()).throw(ValueError("markers not found"))),
            ("extract_between_markers", ["text", "start_marker", "end_marker"],
             "Returns the text between the two markers."),
            ("inner_part", ["input_text", "opening", "closing"],
             "Reads the part enclosed by the opening and closing markers."),
            ("delimited_inner_text", ["source_text", "left_marker", "right_marker"],
             "Text found between the left and right markers.")))

    add(_op("text_replace_all", "string.replace",
            [("text", T, "text"), ("find", T, "option"),
             ("replacement", T, "option")], T,
            lambda text, find, replacement: (
                _text(text).replace(_text(find), replacement)
                if _text(find) in _text(text)
                else (_ for _ in ()).throw(ValueError("nothing to replace"))),
            ("replace_all_occurrences", ["text", "find", "replacement"],
             "Replaces every occurrence of one substring with another."),
            ("substitute_text", ["input_text", "search_for", "insert_instead"],
             "Substitutes one fragment of the text for another."),
            ("rewrite_fragment", ["source_text", "old_fragment", "new_fragment"],
             "Rewrites every old fragment as the new fragment.")))

    add(_op("text_count_substring", "string.count_substring",
            [("text", T, "text"), ("needle", T, "option")], C,
            lambda text, needle: _text(text).count(_text(needle)),
            ("count_occurrences", ["text", "needle"],
             "Counts how often the substring occurs in the text."),
            ("fragment_frequency", ["input_text", "fragment"],
             "How many times the fragment appears."),
            ("substring_hits", ["source_text", "pattern_text"],
             "Number of times the pattern text is present.")))

    add(_op("text_word_count", "string.count_words",
            [("text", T, "text")], C,
            lambda text: len(_text(text).split()),
            ("count_words", ["text"], "Counts the words in the text."),
            ("word_total", ["input_text"], "Total number of words."),
            ("token_count", ["source_text"],
             "How many whitespace-separated tokens the text has.")))

    add(_op("text_starts_with", "string.validate_prefix",
            [("text", T, "text"), ("prefix", T, "option")], FL,
            lambda text, prefix: _text(text).startswith(_text(prefix)),
            ("starts_with", ["text", "prefix"],
             "Checks whether the text starts with the prefix."),
            ("has_leading_text", ["input_text", "leading_text"],
             "Reports whether the text begins with the leading text."),
            ("prefix_present", ["source_text", "expected_start"],
             "True when the text begins with the expected start.")))

    add(_op("text_contains", "string.validate_contains",
            [("text", T, "text"), ("needle", T, "option")], FL,
            lambda text, needle: _text(needle) in _text(text),
            ("contains_substring", ["text", "needle"],
             "Checks whether the text contains the substring."),
            ("fragment_present", ["input_text", "fragment"],
             "Reports whether the fragment occurs in the text."),
            ("holds_pattern_text", ["source_text", "pattern_text"],
             "True when the pattern text occurs somewhere.")))

    add(_op("text_is_identifier", "string.validate_identifier",
            [("text", T, "text")], FL,
            lambda text: bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]{2,}",
                                           _text(text))),
            ("is_valid_identifier", ["text"],
             "Checks whether the text is a valid identifier."),
            ("identifier_well_formed", ["candidate_text"],
             "Reports whether the candidate is a well-formed identifier."),
            ("code_format_valid", ["code_text"],
             "True when the code follows the identifier format.")))

    add(_op("text_pad_left", "format.pad",
            [("text", T, "text"), ("width", C, "option"), ("fill", T, "option")], T,
            lambda text, width, fill: (
                _text(text).rjust(_int_of(width, minimum=1), _text(fill)[0])),
            ("pad_left", ["text", "width", "fill"],
             "Left-pads the text to the requested width."),
            ("fixed_width_label", ["input_text", "target_width", "fill_char"],
             "Returns the text padded to a fixed width."),
            ("aligned_code", ["code_text", "column_width", "padding"],
             "Pads the code so it fills the column width.")))

    add(_op("text_truncate", "string.truncate",
            [("text", T, "text"), ("width", C, "option")], T,
            lambda text, width: _text(text)[:_int_of(width, minimum=1)],
            ("truncate_text", ["text", "width"],
             "Cuts the text down to the requested length."),
            ("shortened_label", ["input_text", "max_length"],
             "Returns the label shortened to the maximum length."),
            ("clipped_text", ["source_text", "limit"],
             "Clips the text to the given limit.")))

    # ══ path ═════════════════════════════════════════════════════════════
    add(_op("path_normalize", "path.normalize",
            [("path", st.PATH, "path")], st.PATH,
            lambda path: (lambda ab, parts: ("/" if ab else "") + "/".join(parts)
                          if parts else
                          (_ for _ in ()).throw(ValueError("empty path")))(
                *_clean_segments(_text(path))),
            ("normalise_path", ["path"],
             "Resolves . and .. segments and returns a clean path."),
            ("canonical_location", ["file_path"],
             "Returns the canonical form of the location."),
            ("tidy_path", ["target_path"],
             "Cleans redundant segments out of the path.")))

    add(_op("path_parent", "path.parent",
            [("path", st.PATH, "path")], st.PATH,
            lambda path: (lambda ab, parts: ("/" if ab else "") +
                          "/".join(parts[:-1]) if len(parts) > 1 else
                          (_ for _ in ()).throw(ValueError("no parent")))(
                *_clean_segments(_text(path))),
            ("parent_directory", ["path"],
             "Returns the directory containing the path."),
            ("enclosing_folder", ["file_path"],
             "Gives the folder that encloses the location."),
            ("directory_above", ["target_path"],
             "Returns the directory one level above.")))

    add(_op("path_basename", "path.basename",
            [("path", st.PATH, "path")], T,
            lambda path: _clean_segments(_text(path))[1][-1],
            ("path_basename", ["path"],
             "Returns the final component of the path."),
            ("file_name_part", ["file_path"],
             "Reads the file name out of the location."),
            ("leaf_component", ["target_path"],
             "Returns the last component of the path.")))

    add(_op("path_stem", "path.stem",
            [("path", st.PATH, "path")], T,
            lambda path: _clean_segments(_text(path))[1][-1].rsplit(".", 1)[0],
            ("path_stem", ["path"],
             "Returns the file name without its extension."),
            ("name_without_suffix", ["file_path"],
             "File name with the suffix removed."),
            ("bare_file_name", ["target_path"],
             "The file name stripped of its extension.")))

    add(_op("path_depth", "path.depth",
            [("path", st.PATH, "path")], C,
            lambda path: len(_clean_segments(_text(path))[1]),
            ("path_depth", ["path"],
             "Counts the components of the path."),
            ("nesting_level", ["file_path"],
             "How deeply the location is nested."),
            ("component_count", ["target_path"],
             "Number of components in the path.")))

    add(_op("path_has_extension", "path.validate_extension",
            [("path", st.PATH, "path"), ("extension", T, "option")], FL,
            lambda path, extension: _clean_segments(_text(path))[1][-1].endswith(
                "." + _text(extension).lstrip(".")),
            ("has_extension", ["path", "extension"],
             "Checks whether the path ends in the given extension."),
            ("suffix_matches", ["file_path", "expected_suffix"],
             "Reports whether the location carries the expected suffix."),
            ("extension_is", ["target_path", "file_type"],
             "True when the path has the given file type.")))

    add(_op("path_change_extension", "path.change_extension",
            [("path", st.PATH, "path"), ("extension", T, "option")], st.PATH,
            lambda path, extension: (lambda ab, parts: (
                ("/" if ab else "") + "/".join(
                    parts[:-1] + [parts[-1].rsplit(".", 1)[0] + "." +
                                  _text(extension).lstrip(".")])))(
                *_clean_segments(_text(path))),
            ("change_extension", ["path", "extension"],
             "Replaces the extension of the path."),
            ("retype_file", ["file_path", "new_suffix"],
             "Returns the location with a different suffix."),
            ("swap_file_type", ["target_path", "file_type"],
             "Swaps the file type of the path.")))

    add(_op("path_is_absolute", "path.validate_absolute",
            [("path", st.PATH, "path")], FL,
            lambda path: _text(path).startswith("/"),
            ("is_absolute_path", ["path"],
             "Checks whether the path is absolute."),
            ("rooted_location", ["file_path"],
             "Reports whether the location is rooted."),
            ("absolute_form", ["target_path"],
             "True when the path is given in absolute form.")))

    # ══ url ══════════════════════════════════════════════════════════════
    add(_op("url_scheme", "url.scheme",
            [("url", st.URL, "url")], T,
            lambda url: _url(url)["scheme"],
            ("url_scheme", ["url"], "Returns the scheme of the url."),
            ("protocol_of_link", ["link"],
             "Reads the protocol out of the link."),
            ("address_scheme", ["address"],
             "Returns the scheme part of the address.")))

    add(_op("url_path_part", "url.path",
            [("url", st.URL, "url")], st.PATH,
            lambda url: _url(url)["path"] or "/",
            ("url_path", ["url"], "Returns the path component of the url."),
            ("link_path", ["link"], "Reads the path out of the link."),
            ("address_path", ["address"],
             "Returns the path portion of the address.")))

    add(_op("url_query_value", "url.parse_query",
            [("url", st.URL, "url"), ("name", T, "key")], T,
            lambda url, name: next(
                (kv.split("=", 1)[1] for kv in _url(url)["query"].split("&")
                 if "=" in kv and kv.split("=", 1)[0] == _text(name)),
                None) or (_ for _ in ()).throw(ValueError("query key absent")),
            ("url_query_parameter", ["url", "name"],
             "Returns the value of one query parameter."),
            ("link_parameter", ["link", "parameter_name"],
             "Reads one parameter out of the link's query string."),
            ("address_query_field", ["address", "field_name"],
             "Returns one query field of the address.")))

    add(_op("url_port", "url.parse_port",
            [("url", st.URL, "url")], C,
            lambda url: int(_url(url)["port"]) if _url(url)["port"] else
            (443 if _url(url)["scheme"] == "https" else 80),
            ("url_port", ["url"],
             "Returns the port of the url, defaulting by scheme."),
            ("link_port", ["link"],
             "Reads the port of the link, using the scheme default."),
            ("address_port", ["address"],
             "Port used by the address.")))

    add(_op("url_is_secure", "url.validate_secure",
            [("url", st.URL, "url")], FL,
            lambda url: _url(url)["scheme"] == "https",
            ("url_is_secure", ["url"],
             "Checks whether the url uses a secure scheme."),
            ("link_uses_tls", ["link"],
             "Reports whether the link is transport-encrypted."),
            ("secure_address", ["address"],
             "True when the address uses https.")))

    add(_op("url_build", "url.build",
            [("scheme", T, "option"), ("host", T, "key"),
             ("path", st.PATH, "path")], st.URL,
            lambda scheme, host, path: (
                f"{_text(scheme)}://{_text(host)}"
                f"{_text(path) if _text(path).startswith('/') else '/' + _text(path)}"),
            ("build_url", ["scheme", "host", "path"],
             "Assembles a url from scheme, host and path."),
            ("compose_link", ["protocol", "domain", "location"],
             "Composes a link out of protocol, domain and location."),
            ("assemble_address", ["scheme_name", "host_name", "resource_path"],
             "Assembles an address from its parts.")))

    add(_op("url_host_depth", "url.domain_depth",
            [("url", st.URL, "url")], C,
            lambda url: len(_url(url)["host"].split(".")),
            ("url_host_label_count", ["url"],
             "Counts the labels of the url host."),
            ("domain_label_count", ["link"],
             "How many dot-separated labels the domain has."),
            ("host_segment_count", ["address"],
             "Number of host segments in the address.")))

    # ══ date / duration ══════════════════════════════════════════════════
    add(_op("date_difference_days", "date.difference",
            [("date_a", st.DATE, "date"), ("date_b", st.DATE, "date")], st.DUR_D,
            lambda date_a, date_b: (_iso(date_a) - _iso(date_b)).days,
            ("days_between", ["date_a", "date_b"],
             "Returns the signed number of days from date_b to date_a."),
            ("day_gap", ["later_date", "earlier_date"],
             "Number of days separating the two dates."),
            ("calendar_distance", ["end_date", "start_date"],
             "Calendar distance between the dates in days.")))

    add(_op("date_add_days", "date.add_duration",
            [("start_date", st.DATE, "date"), ("days", st.DUR_D, "duration")],
            st.DATE,
            lambda start_date, days: (
                _iso(start_date) + timedelta(days=_int_of(days))).isoformat(),
            ("add_days_to_date", ["start_date", "days"],
             "Shifts a date by the given number of days."),
            ("date_after_days", ["from_date", "day_count"],
             "Returns the date reached after the day count."),
            ("shifted_calendar_date", ["base_date", "offset_days"],
             "Applies a day offset to the base date.")))

    add(_op("date_is_before", "date.compare",
            [("date_a", st.DATE, "date"), ("date_b", st.DATE, "date")], FL,
            lambda date_a, date_b: _iso(date_a) < _iso(date_b),
            ("date_is_before", ["date_a", "date_b"],
             "Checks whether the first date precedes the second."),
            ("earlier_than", ["candidate_date", "reference_date"],
             "Reports whether the candidate date is earlier."),
            ("precedes_deadline", ["event_date", "deadline_date"],
             "True when the event date precedes the deadline.")))

    add(_op("date_weekday_index", "date.weekday",
            [("target_date", st.DATE, "date")], I,
            lambda target_date: _iso(target_date).isoweekday(),
            ("weekday_index", ["target_date"],
             "Returns the ISO weekday index (Monday = 1)."),
            ("day_of_week_number", ["calendar_date"],
             "Day-of-week number with Monday as one."),
            ("weekday_position", ["date_value"],
             "Position of the day within the week.")))

    add(_op("date_month_index", "date.month",
            [("target_date", st.DATE, "date")], I,
            lambda target_date: _iso(target_date).month,
            ("month_index", ["target_date"],
             "Returns the month number of the date."),
            ("calendar_month_number", ["calendar_date"],
             "Month number of the calendar date."),
            ("month_position", ["date_value"],
             "Which month of the year the date falls in.")))

    add(_op("date_quarter", "date.quarter",
            [("target_date", st.DATE, "date")], I,
            lambda target_date: (_iso(target_date).month - 1) // 3 + 1,
            ("quarter_of_date", ["target_date"],
             "Returns the calendar quarter of the date."),
            ("fiscal_quarter_number", ["calendar_date"],
             "Calendar quarter the date belongs to."),
            ("quarter_position", ["date_value"],
             "Which quarter of the year the date falls in.")))

    add(_op("duration_days_to_hours", "duration.convert_days_hours",
            [("days", st.DUR_D, "duration")], st.DUR_H,
            lambda days: _r6(float(days) * 24.0),
            ("days_as_hours", ["days"], "Converts a number of days into hours."),
            ("hour_equivalent_of_days", ["day_count"],
             "Returns how many hours the day count represents."),
            ("day_span_in_hours", ["span_days"],
             "Length of the day span expressed in hours.")))

    add(_op("duration_hours_to_days", "duration.convert_hours_days",
            [("hours", st.DUR_H, "duration")], st.DUR_D,
            lambda hours: _r6(float(hours) / 24.0),
            ("hours_as_days", ["hours"], "Converts a number of hours into days."),
            ("days_in_hours", ["hour_count"],
             "Returns how many days the hour count represents."),
            ("hour_span_in_days", ["span_hours"],
             "Length of the hour span expressed in days.")))

    add(_op("duration_sum_hours", "duration.sum",
            [("hours_a", st.DUR_H, "duration"), ("hours_b", st.DUR_H, "duration")],
            st.DUR_H,
            lambda hours_a, hours_b: _r6(float(hours_a) + float(hours_b)),
            ("add_hour_spans", ["hours_a", "hours_b"],
             "Adds two durations given in hours."),
            ("combined_hours", ["first_span_hours", "second_span_hours"],
             "Total of the two hour spans."),
            ("total_hour_span", ["span_one_hours", "span_two_hours"],
             "Combined length of both hour spans.")))

    add(_op("duration_scale_hours", "duration.scale",
            [("hours", st.DUR_H, "duration"), ("factor", N, "rate")], st.DUR_H,
            lambda hours, factor: _r6(float(hours) * float(factor)),
            ("scale_hour_span", ["hours", "factor"],
             "Scales a duration in hours by a factor."),
            ("stretched_hours", ["span_hours", "multiplier"],
             "Returns the hour span multiplied by the factor."),
            ("adjusted_hour_span", ["base_hours", "scaling"],
             "Applies a scaling factor to the hour span.")))

    # ══ boolean / multi-condition decision ═══════════════════════════════
    add(_op("all_three_hold", "decision.all_of",
            [("condition_one", FL, "flag"), ("condition_two", FL, "flag"),
             ("condition_three", FL, "flag")], FL,
            lambda condition_one, condition_two, condition_three: (
                _flag(condition_one) and _flag(condition_two)
                and _flag(condition_three)),
            ("all_conditions_hold", ["condition_one", "condition_two",
                                     "condition_three"],
             "True only when all three conditions hold."),
            ("every_check_passes", ["check_one", "check_two", "check_three"],
             "Reports whether every one of the three checks passes."),
            ("fully_compliant", ["rule_one", "rule_two", "rule_three"],
             "True when all three rules are satisfied.")))

    add(_op("any_three_hold", "decision.any_of",
            [("condition_one", FL, "flag"), ("condition_two", FL, "flag"),
             ("condition_three", FL, "flag")], FL,
            lambda condition_one, condition_two, condition_three: (
                _flag(condition_one) or _flag(condition_two)
                or _flag(condition_three)),
            ("any_condition_holds", ["condition_one", "condition_two",
                                     "condition_three"],
             "True when at least one of the three conditions holds."),
            ("some_check_passes", ["check_one", "check_two", "check_three"],
             "Reports whether at least one check passes."),
            ("any_rule_triggered", ["rule_one", "rule_two", "rule_three"],
             "True when any of the three rules triggers.")))

    add(_op("exactly_one_holds", "boolean.xor",
            [("condition_one", FL, "flag"), ("condition_two", FL, "flag")], FL,
            lambda condition_one, condition_two: (
                _flag(condition_one) != _flag(condition_two)),
            ("exactly_one_condition", ["condition_one", "condition_two"],
             "True when exactly one of the two conditions holds."),
            ("checks_disagree", ["check_one", "check_two"],
             "Reports whether the two checks disagree."),
            ("single_rule_triggered", ["rule_one", "rule_two"],
             "True when precisely one rule triggers.")))

    add(_op("count_conditions_met", "decision.count_true",
            [("condition_one", FL, "flag"), ("condition_two", FL, "flag"),
             ("condition_three", FL, "flag")], C,
            lambda condition_one, condition_two, condition_three: (
                int(_flag(condition_one)) + int(_flag(condition_two))
                + int(_flag(condition_three))),
            ("count_conditions_met", ["condition_one", "condition_two",
                                      "condition_three"],
             "Counts how many of the three conditions hold."),
            ("passed_check_count", ["check_one", "check_two", "check_three"],
             "How many of the three checks pass."),
            ("satisfied_rule_count", ["rule_one", "rule_two", "rule_three"],
             "Number of satisfied rules.")))

    add(_op("majority_holds", "decision.majority",
            [("condition_one", FL, "flag"), ("condition_two", FL, "flag"),
             ("condition_three", FL, "flag")], FL,
            lambda condition_one, condition_two, condition_three: (
                int(_flag(condition_one)) + int(_flag(condition_two))
                + int(_flag(condition_three))) >= 2,
            ("majority_of_conditions", ["condition_one", "condition_two",
                                        "condition_three"],
             "True when at least two of the three conditions hold."),
            ("majority_checks_pass", ["check_one", "check_two", "check_three"],
             "Reports whether most of the checks pass."),
            ("majority_rule_met", ["rule_one", "rule_two", "rule_three"],
             "True when a majority of rules is met.")))

    add(_op("is_at_least", "comparison.at_least",
            [("value", N, "value"), ("minimum", N, "threshold")], FL,
            lambda value, minimum: float(value) >= float(minimum),
            ("meets_minimum", ["value", "minimum"],
             "Checks whether the value reaches the minimum."),
            ("at_or_above", ["measured_value", "required_value"],
             "Reports whether the measurement reaches the requirement."),
            ("requirement_met", ["observed", "required"],
             "True when the observed value is at least the required one.")))

    # ══ validation ═══════════════════════════════════════════════════════
    add(_op("within_tolerance", "validation.tolerance",
            [("value", N, "value"), ("target", N, "threshold"),
             ("tolerance", N, "threshold")], FL,
            lambda value, target, tolerance: (
                abs(float(value) - float(target)) <= abs(float(tolerance))),
            ("within_tolerance", ["value", "target", "tolerance"],
             "Checks whether the value is within tolerance of the target."),
            ("close_enough_to_target", ["measured", "nominal", "allowance"],
             "Reports whether the measurement is inside the allowance."),
            ("inside_spec_band", ["observed", "specification", "margin"],
             "True when the observation stays inside the specification.")))

    add(_op("list_all_positive", "validation.list_positive",
            [("values", NL, "collection")], FL,
            lambda values: all(v > 0 for v in _nums(values)),
            ("all_values_positive", ["values"],
             "Checks that every entry is strictly positive."),
            ("series_strictly_positive", ["series"],
             "Reports whether the whole series stays positive."),
            ("no_negative_readings", ["measurements"],
             "True when no measurement is zero or below.")))

    add(_op("list_within_limit", "validation.list_limit",
            [("values", NL, "collection"), ("limit", N, "threshold")], FL,
            lambda values, limit: all(v <= float(limit) for v in _nums(values)),
            ("all_values_within_limit", ["values", "limit"],
             "Checks that no entry exceeds the limit."),
            ("series_under_ceiling", ["series", "ceiling"],
             "Reports whether the series stays under the ceiling."),
            ("no_limit_breach", ["measurements", "maximum"],
             "True when no measurement breaches the maximum.")))

    # ══ formatting / classification ══════════════════════════════════════
    add(_op("format_percent_text", "format.percent",
            [("value", PCT, "value"), ("places", C, "option")], T,
            lambda value, places: (
                f"{float(value):.{_int_of(places, minimum=0)}f}%"),
            ("format_as_percent", ["value", "places"],
             "Renders a percentage with the requested decimals."),
            ("percent_label", ["rate_value", "decimals"],
             "Produces a percentage label."),
            ("rate_display_text", ["rate", "precision"],
             "Formats the rate for display.")))

    add(_op("format_currency_text", "format.currency",
            [("amount", st.MONEY, "value"), ("currency", T, "option")], T,
            lambda amount, currency: f"{_text(currency)} {float(amount):.2f}",
            ("format_as_currency", ["amount", "currency"],
             "Renders a monetary amount with its currency code."),
            ("money_label", ["value", "currency_code"],
             "Produces a monetary label."),
            ("amount_display_text", ["total", "currency_name"],
             "Formats the amount for display with its currency.")))

    add(_op("classify_three_bands", "classification.three_bands",
            [("value", N, "value"), ("low_cut", N, "threshold"),
             ("high_cut", N, "threshold")], CAT,
            lambda value, low_cut, high_cut: (
                "low" if float(value) < float(low_cut)
                else ("high" if float(value) >= float(high_cut) else "medium"))
            if float(low_cut) < float(high_cut)
            else (_ for _ in ()).throw(ValueError("cuts out of order")),
            ("classify_into_bands", ["value", "low_cut", "high_cut"],
             "Classifies the value as low, medium or high."),
            ("band_of_value", ["measured_value", "lower_bound", "upper_bound"],
             "Returns which band the measurement falls into."),
            ("severity_band", ["observed", "first_cut", "second_cut"],
             "Assigns the observation to a severity band.")))

    add(_op("classify_ratio_band", "classification.ratio_band",
            [("ratio", RAT, "value"), ("cut", RAT, "threshold")], CAT,
            lambda ratio, cut: ("above_target" if float(ratio) >= float(cut)
                                else "below_target"),
            ("classify_ratio", ["ratio", "cut"],
             "Reports whether the ratio is above or below the cut."),
            ("proportion_band", ["proportion", "target_proportion"],
             "Band of the proportion relative to its target."),
            ("share_verdict", ["share", "target_share"],
             "Verdict on the share against its target.")))

    # ══ statistics / rates ═══════════════════════════════════════════════
    add(_op("weighted_average_two", "statistics.weighted_average",
            [("value_a", N, "value"), ("weight_a", N, "rate"),
             ("value_b", N, "value"), ("weight_b", N, "rate")], N,
            lambda value_a, weight_a, value_b, weight_b: (
                _r6((float(value_a) * float(weight_a)
                     + float(value_b) * float(weight_b))
                    / (float(weight_a) + float(weight_b)))
                if float(weight_a) + float(weight_b) != 0
                else (_ for _ in ()).throw(ValueError("zero total weight"))),
            ("weighted_average", ["value_a", "weight_a", "value_b", "weight_b"],
             "Computes the weighted average of two values."),
            ("blended_value", ["first_value", "first_weight", "second_value",
                               "second_weight"],
             "Blends two values according to their weights."),
            ("weighted_blend", ["value_one", "weight_one", "value_two",
                                "weight_two"],
             "Returns the weight-adjusted blend of two values.")))

    add(_op("stdev_values", "statistics.stdev",
            [("values", NL, "collection")], N,
            lambda values: (
                _r6(math.sqrt(sum(
                    (v - sum(_nums(values)) / len(_nums(values))) ** 2
                    for v in _nums(values)) / (len(_nums(values)) - 1)))
                if len(_nums(values)) >= 2
                else (_ for _ in ()).throw(ValueError("needs two values"))),
            ("sample_standard_deviation", ["values"],
             "Sample standard deviation of the list."),
            ("series_dispersion", ["series"],
             "Statistical spread of the series."),
            ("reading_dispersion", ["measurements"],
             "Dispersion of the measurements.")))

    add(_op("percent_change", "rates.percent_change",
            [("old_value", N, "value"), ("new_value", N, "value")], PCT,
            lambda old_value, new_value: (
                _r6((float(new_value) - float(old_value))
                    / float(old_value) * 100.0)
                if float(old_value) != 0
                else (_ for _ in ()).throw(ValueError("zero baseline"))),
            ("percent_change", ["old_value", "new_value"],
             "Percentage change from the old value to the new one."),
            ("relative_movement", ["baseline", "current"],
             "Relative movement between baseline and current value."),
            ("growth_rate_percent", ["previous", "latest"],
             "Growth rate in percent between the two values.")))

    add(_op("share_percent", "rates.share_percent",
            [("part", N, "value"), ("total", N, "value")], PCT,
            lambda part, total: (
                _r6(float(part) / float(total) * 100.0) if float(total) != 0
                else (_ for _ in ()).throw(ValueError("zero total"))),
            ("share_as_percent", ["part", "total"],
             "Expresses the part as a percentage of the total."),
            ("portion_percentage", ["portion", "whole"],
             "Percentage the portion represents of the whole."),
            ("contribution_percent", ["component", "aggregate"],
             "Percentage contribution of the component.")))

    add(_op("compound_growth", "rates.compound_growth",
            [("base_value", N, "value"), ("rate_percent", PCT, "rate"),
             ("periods", C, "option")], N,
            lambda base_value, rate_percent, periods: _r6(
                float(base_value) * (1.0 + float(rate_percent) / 100.0)
                ** _int_of(periods, minimum=1)),
            ("compound_growth", ["base_value", "rate_percent", "periods"],
             "Applies compound growth over a number of periods."),
            ("grown_over_periods", ["starting_value", "rate", "period_count"],
             "Value after compounding the rate over the periods."),
            ("multi_period_growth", ["initial", "growth_rate", "cycles"],
             "Compounds the growth rate over the given cycles.")))

    add(_op("apply_tax", "rates.apply_tax",
            [("amount", st.MONEY, "value"), ("tax_percent", PCT, "rate")],
            st.MONEY,
            lambda amount, tax_percent: _r6(
                float(amount) * (1.0 + float(tax_percent) / 100.0)),
            ("apply_tax_rate", ["amount", "tax_percent"],
             "Adds the tax rate to a monetary amount."),
            ("amount_with_tax", ["net_amount", "tax_rate"],
             "Returns the amount including tax."),
            ("taxed_total", ["base_amount", "levy_rate"],
             "Total after applying the levy rate.")))

    # ══ measurement / unit conversion ════════════════════════════════════
    add(_op("kg_to_grams", "unit_conversion.mass_kg_g",
            [("mass_kg", st.MASS_KG, "quantity")], st.MASS_G,
            lambda mass_kg: _r6(float(mass_kg) * 1000.0),
            ("kilograms_to_grams", ["mass_kg"],
             "Converts kilograms into grams."),
            ("grams_from_kilograms", ["weight_kg"],
             "Returns the weight expressed in grams."),
            ("mass_in_grams", ["kilogram_value"],
             "Mass converted to grams.")))

    add(_op("grams_to_kg", "unit_conversion.mass_g_kg",
            [("mass_g", st.MASS_G, "quantity")], st.MASS_KG,
            lambda mass_g: _r6(float(mass_g) / 1000.0),
            ("grams_to_kilograms", ["mass_g"],
             "Converts grams into kilograms."),
            ("kilograms_from_grams", ["weight_g"],
             "Returns the weight expressed in kilograms."),
            ("mass_in_kilograms", ["gram_value"],
             "Mass converted to kilograms.")))

    add(_op("litres_to_millilitres", "unit_conversion.volume_l_ml",
            [("volume_l", st.VOL_L, "quantity")], st.VOL_ML,
            lambda volume_l: _r6(float(volume_l) * 1000.0),
            ("litres_to_millilitres", ["volume_l"],
             "Converts litres into millilitres."),
            ("millilitres_from_litres", ["capacity_l"],
             "Returns the capacity in millilitres."),
            ("volume_in_millilitres", ["litre_value"],
             "Volume converted to millilitres.")))

    add(_op("millilitres_to_litres", "unit_conversion.volume_ml_l",
            [("volume_ml", st.VOL_ML, "quantity")], st.VOL_L,
            lambda volume_ml: _r6(float(volume_ml) / 1000.0),
            ("millilitres_to_litres", ["volume_ml"],
             "Converts millilitres into litres."),
            ("litres_from_millilitres", ["capacity_ml"],
             "Returns the capacity in litres."),
            ("volume_in_litres", ["millilitre_value"],
             "Volume converted to litres.")))

    add(_op("bytes_to_kibibytes", "unit_conversion.bytes_kib",
            [("size_bytes", st.BYTES, "quantity")], N,
            lambda size_bytes: _r6(_int_of(size_bytes, minimum=0) / 1024.0),
            ("bytes_to_kibibytes", ["size_bytes"],
             "Converts a byte count into kibibytes."),
            ("kibibytes_from_bytes", ["byte_count"],
             "Returns the size in kibibytes."),
            ("size_in_kibibytes", ["raw_bytes"],
             "Size converted to kibibytes.")))

    # ══ geometry ═════════════════════════════════════════════════════════
    add(_op("triangle_area", "geometry.triangle_area",
            [("base_length", st.LEN_M, "quantity"),
             ("height", st.LEN_M, "quantity")], st.AREA,
            lambda base_length, height: (
                _r6(float(base_length) * float(height) / 2.0)
                if float(base_length) > 0 and float(height) > 0
                else (_ for _ in ()).throw(ValueError("non-positive dimension"))),
            ("triangle_area", ["base_length", "height"],
             "Area of a triangle from its base and height."),
            ("triangular_surface", ["base_side", "vertical_height"],
             "Surface of a triangular shape."),
            ("wedge_area", ["base_measure", "height_measure"],
             "Area covered by a wedge-shaped section.")))

    add(_op("triangle_perimeter", "geometry.triangle_perimeter",
            [("side_a", st.LEN_M, "quantity"), ("side_b", st.LEN_M, "quantity"),
             ("side_c", st.LEN_M, "quantity")], st.LEN_M,
            lambda side_a, side_b, side_c: (
                _r6(float(side_a) + float(side_b) + float(side_c))
                if min(float(side_a), float(side_b), float(side_c)) > 0
                and float(side_a) + float(side_b) > float(side_c)
                and float(side_a) + float(side_c) > float(side_b)
                and float(side_b) + float(side_c) > float(side_a)
                else (_ for _ in ()).throw(
                    ValueError("sides violate the triangle inequality"))),
            ("triangle_perimeter", ["side_a", "side_b", "side_c"],
             "Perimeter of a triangle, validating the triangle inequality."),
            ("triangular_outline", ["first_side", "second_side", "third_side"],
             "Outline length of a triangular shape."),
            ("wedge_edge_total", ["edge_a", "edge_b", "edge_c"],
             "Total edge length of a wedge-shaped section.")))

    add(_op("circle_circumference", "geometry.circumference",
            [("radius", st.LEN_M, "quantity")], st.LEN_M,
            lambda radius: (_r6(2.0 * math.pi * float(radius))
                            if float(radius) > 0 else
                            (_ for _ in ()).throw(ValueError("non-positive radius"))),
            ("circle_circumference", ["radius"],
             "Circumference of a circle from its radius."),
            ("round_outline_length", ["circle_radius"],
             "Outline length of a circular shape."),
            ("ring_length", ["radius_measure"],
             "Length around a circular ring.")))

    return ops
