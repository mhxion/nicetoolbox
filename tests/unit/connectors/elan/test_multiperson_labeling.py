"""Tests for multiperson labeling data model, conversion, and NPZ schema."""

import math

import pytest

from nicetoolbox.connectors.elan.elan_data import ElanData, Interval, Tier
from nicetoolbox.connectors.elan.labeling_data import LabeledInterval, apply_category_defaults
from nicetoolbox.connectors.elan.labeling_from_elan import elan_data_to_hierarchical, parse_tier_name
from nicetoolbox.connectors.elan.npz_schema import schema_from_config, schema_from_data
from nicetoolbox.connectors.elan.toolbox_writer import hierarchical_to_npz_dict


def _make_elan(tiers_spec):
    tiers = [Tier(tier_name=name, intervals=[Interval(s, e, a) for s, e, a in ivs]) for name, ivs in tiers_spec]
    return ElanData(header=None, tiers=tiers)


def _npz(hier, schema, end_sec, serialize="text"):
    return hierarchical_to_npz_dict(
        hier,
        schema,
        fps=10.0,
        start_sec=0.0,
        end_sec=end_sec,
        serialize=serialize,
        category_gap_fills={},
        reset_frames=False,
    )


# --- parse_tier_name ---


def test_parse_tier_name_basic():
    assert parse_tier_name("th head") == ("th", "head")


def test_parse_tier_name_raises_on_no_space():
    with pytest.raises(ValueError, match="does not follow the expected"):
        parse_tier_name("headonly")


def test_parse_tier_name_raises_on_empty_parts():
    with pytest.raises(ValueError, match="does not follow the expected"):
        parse_tier_name(" head")


# --- elan_data_to_hierarchical ---


def test_subjects_and_categories_extracted():
    elan = _make_elan(
        [
            ("th head", [(0.0, 2.0, "htrn")]),
            ("cl head", [(0.0, 2.0, "hnod")]),
            ("th eyes", [(0.0, 2.0, "eymov")]),
            ("cl eyes", [(0.0, 2.0, "eyga")]),
        ]
    )
    result = elan_data_to_hierarchical(elan)

    assert result.subjects == {"cl", "th"}
    assert result.categories == {"eyes", "head"}


def test_data_contains_labeled_interval_objects():
    elan = _make_elan([("th head", [(0.0, 2.0, "htrn"), (2.0, 4.0, "")])])
    intervals = elan_data_to_hierarchical(elan).data["th"]["head"]

    assert len(intervals) == 2
    assert isinstance(intervals[0], LabeledInterval)
    assert intervals[0].labels == frozenset({"htrn"})
    assert intervals[1].labels == frozenset()


def test_data_multi_label_stored_as_set():
    elan = _make_elan([("th head", [(0.0, 2.0, "hnod, htrn")])])
    assert elan_data_to_hierarchical(elan).data["th"]["head"][0].labels == frozenset({"hnod", "htrn"})


def test_raises_on_tier_name_without_space():
    with pytest.raises(ValueError, match="does not follow the expected"):
        elan_data_to_hierarchical(_make_elan([("headonly", [(0.0, 2.0, "x")])]))


def test_raises_on_incomplete_tiers():
    elan = _make_elan([("th head", [(0.0, 2.0, "htrn")]), ("cl eyes", [(0.0, 2.0, "eyga")])])
    with pytest.raises(ValueError, match="missing categories"):
        elan_data_to_hierarchical(elan)


# --- hierarchical_to_npz_dict: text mode ---


def test_npz_array_shape_single_category():
    elan = _make_elan([("th head", [(0.0, 2.0, "htrn")]), ("cl head", [(0.0, 2.0, "hnod")])])
    hier = elan_data_to_hierarchical(elan)
    arr = _npz(hier, schema_from_data(hier), 2.0)["labels"]

    assert arr.shape == (2, 1, 20, 1)


def test_npz_array_shape_two_categories():
    elan = _make_elan(
        [
            ("th head", [(0.0, 1.0, "htrn")]),
            ("th eyes", [(0.0, 1.0, "eymov")]),
            ("cl head", [(0.0, 1.0, "hnod")]),
            ("cl eyes", [(0.0, 1.0, "eyga")]),
        ]
    )
    hier = elan_data_to_hierarchical(elan)
    arr = _npz(hier, schema_from_data(hier), 1.0)["labels"]

    assert arr.shape == (2, 1, 10, 2)


def test_npz_single_interval_fills_all_frames():
    elan = _make_elan([("th head", [(0.0, 2.0, "htrn")])])
    hier = elan_data_to_hierarchical(elan)
    arr = _npz(hier, schema_from_data(hier), 2.0)["labels"]

    assert all(arr[0, 0, f, 0] == "htrn" for f in range(20))


def test_npz_interval_boundary_is_half_open():
    elan = _make_elan([("th head", [(0.0, 1.0, "x"), (1.0, 2.0, "y")])])
    hier = elan_data_to_hierarchical(elan)
    arr = _npz(hier, schema_from_data(hier), 2.0)["labels"]

    assert all(arr[0, 0, f, 0] == "x" for f in range(10))
    assert all(arr[0, 0, f, 0] == "y" for f in range(10, 20))


def test_npz_gap_between_intervals_stays_empty():
    elan = _make_elan([("th head", [(0.0, 1.0, "a"), (1.5, 2.0, "b")])])
    hier = elan_data_to_hierarchical(elan)
    arr = _npz(hier, schema_from_data(hier), 2.0)["labels"]

    assert all(arr[0, 0, f, 0] == "a" for f in range(10))
    assert all(arr[0, 0, f, 0] == "" for f in range(10, 15))
    assert all(arr[0, 0, f, 0] == "b" for f in range(15, 20))


def test_npz_subjects_are_sorted():
    elan = _make_elan([("th head", [(0.0, 2.0, "htrn")]), ("cl head", [(0.0, 2.0, "hnod")])])
    hier = elan_data_to_hierarchical(elan)
    result = _npz(hier, schema_from_data(hier), 2.0)
    subjects = result["data_description"].item()["labels"]["axis0"]

    assert subjects == ["cl", "th"]


def test_npz_data_description_axes():
    elan = _make_elan([("th head", [(0.0, 1.0, "htrn")]), ("cl head", [(0.0, 1.0, "hnod")])])
    hier = elan_data_to_hierarchical(elan)
    desc = _npz(hier, schema_from_data(hier), 1.0)["data_description"].item()["labels"]

    assert desc["axis0"] == ["cl", "th"]
    assert desc["axis1"] == ["3d"]
    assert len(desc["axis2"]) == 10
    assert desc["axis3"] == ["head"]


def test_npz_frame_indices_non_zero_start():
    elan = _make_elan([("th head", [(10.0, 12.0, "x")])])
    hier = elan_data_to_hierarchical(elan)
    result = hierarchical_to_npz_dict(
        hier,
        schema_from_data(hier),
        fps=10.0,
        start_sec=10.0,
        end_sec=12.0,
        serialize="text",
        category_gap_fills={},
        reset_frames=False,
    )
    desc = result["data_description"].item()["labels"]

    assert desc["axis2"][0] == "000000100"
    assert desc["axis2"][-1] == "000000119"


# --- boolean mode ---


def test_boolean_mode_returns_per_category_arrays():
    elan = _make_elan([("th head", [(0.0, 1.0, "htrn")]), ("th eyes", [(0.0, 1.0, "eymov")])])
    hier = elan_data_to_hierarchical(elan)
    result = _npz(hier, schema_from_data(hier), 1.0, serialize="boolean")

    assert "head" in result
    assert "eyes" in result
    assert "labels" not in result


def test_boolean_mode_array_shape():
    elan = _make_elan([("th head", [(0.0, 1.0, "htrn")]), ("cl head", [(0.0, 1.0, "hnod")])])
    hier = elan_data_to_hierarchical(elan)
    result = _npz(hier, schema_from_data(hier), 1.0, serialize="boolean")

    assert result["head"].shape == (2, 1, 10, 2)


def test_boolean_mode_active_label_is_one():
    elan = _make_elan([("th head", [(0.0, 1.0, "htrn")])])
    hier = elan_data_to_hierarchical(elan)
    arr = _npz(hier, schema_from_data(hier), 1.0, serialize="boolean")["head"]

    assert all(arr[0, 0, f, 0] == 1.0 for f in range(10))


def test_boolean_mode_inactive_label_is_zero():
    elan = _make_elan([("th head", [(0.0, 1.0, "htrn"), (1.0, 2.0, "hnod")])])
    hier = elan_data_to_hierarchical(elan)
    result = _npz(hier, schema_from_data(hier), 2.0, serialize="boolean")
    labels = result["data_description"].item()["head"]["axis3"]
    arr = result["head"]
    hnod_idx, htrn_idx = labels.index("hnod"), labels.index("htrn")

    assert arr[0, 0, 0, htrn_idx] == 1.0
    assert arr[0, 0, 0, hnod_idx] == 0.0
    assert arr[0, 0, 10, hnod_idx] == 1.0
    assert arr[0, 0, 10, htrn_idx] == 0.0


def test_boolean_mode_gap_is_nan():
    elan = _make_elan([("th head", [(0.0, 1.0, "a"), (1.5, 2.0, "a")])])
    hier = elan_data_to_hierarchical(elan)
    arr = _npz(hier, schema_from_data(hier), 2.0, serialize="boolean")["head"]

    assert math.isnan(arr[0, 0, 12, 0])
    assert arr[0, 0, 0, 0] == 1.0


def test_boolean_mode_multi_label():
    elan = _make_elan([("th head", [(0.0, 1.0, "hnod, htrn")])])
    hier = elan_data_to_hierarchical(elan)
    arr = _npz(hier, schema_from_data(hier), 1.0, serialize="boolean")["head"]

    assert arr[0, 0, 0, 0] == 1.0
    assert arr[0, 0, 0, 1] == 1.0


def test_boolean_mode_raises_on_invalid_serialize():
    elan = _make_elan([("th head", [(0.0, 1.0, "x")])])
    hier = elan_data_to_hierarchical(elan)
    with pytest.raises(ValueError, match="serialize must be"):
        _npz(hier, schema_from_data(hier), 1.0, serialize="invalid")


# --- apply_category_defaults ---


def test_apply_defaults_replaces_empty_annotations():
    elan = _make_elan([("th head", [(0.0, 1.0, "htrn"), (1.0, 2.0, "")]), ("th eyes", [(0.0, 2.0, "")])])
    hier = elan_data_to_hierarchical(elan)
    result = apply_category_defaults(hier, {"head": "neutral", "eyes": "open"})

    assert result.data["th"]["head"][0].labels == frozenset({"htrn"})
    assert result.data["th"]["head"][1].labels == frozenset({"neutral"})
    assert result.data["th"]["eyes"][0].labels == frozenset({"open"})


def test_apply_defaults_leaves_non_empty_unchanged():
    elan = _make_elan([("th head", [(0.0, 1.0, "htrn")])])
    hier = elan_data_to_hierarchical(elan)
    assert apply_category_defaults(hier, {"head": "neutral"}).data["th"]["head"][0].labels == frozenset({"htrn"})


def test_apply_defaults_raises_on_unknown_category():
    elan = _make_elan([("th head", [(0.0, 1.0, "")])])
    hier = elan_data_to_hierarchical(elan)
    with pytest.raises(ValueError, match="unknown categories"):
        apply_category_defaults(hier, {"head": "neutral", "bogus": "x"})


def test_apply_defaults_returns_copy_not_original():
    elan = _make_elan([("th head", [(0.0, 1.0, "")])])
    hier = elan_data_to_hierarchical(elan)
    result = apply_category_defaults(hier, {"head": "neutral"})

    assert hier.data["th"]["head"][0].labels == frozenset()
    assert result.data["th"]["head"][0].labels == frozenset({"neutral"})


# --- schema_from_config ---


def test_schema_from_config_exact_match_passes():
    elan = _make_elan([("th head", [(0.0, 1.0, "htrn"), (1.0, 2.0, "hnod")])])
    hier = elan_data_to_hierarchical(elan)
    schema = schema_from_config(hier, {"head": {"hnod", "htrn"}})

    assert schema.categories == ["head"]
    assert schema.labels_per_category == {"head": ["hnod", "htrn"]}


def test_schema_from_config_raises_on_unexpected_label():
    elan = _make_elan([("th head", [(0.0, 1.0, "htrn"), (1.0, 2.0, "typo")])])
    hier = elan_data_to_hierarchical(elan)
    with pytest.raises(ValueError, match="Unexpected.*typo"):
        schema_from_config(hier, {"head": {"htrn"}})


def test_schema_from_config_warns_on_missing_label(caplog):
    elan = _make_elan([("th head", [(0.0, 1.0, "htrn")])])
    hier = elan_data_to_hierarchical(elan)
    schema = schema_from_config(hier, {"head": {"hnod", "htrn"}})

    assert "hnod" in caplog.text
    assert "hnod" in schema.labels_per_category["head"]


def test_schema_from_config_raises_on_unknown_category():
    elan = _make_elan([("th head", [(0.0, 1.0, "htrn")])])
    hier = elan_data_to_hierarchical(elan)
    with pytest.raises(ValueError, match="not defined in config"):
        schema_from_config(hier, {"bogus": {"x"}})


# --- schema_from_data ---


def test_schema_from_data_sorted():
    elan = _make_elan([("th head", [(0.0, 1.0, "htrn")]), ("cl head", [(0.0, 1.0, "hnod")])])
    hier = elan_data_to_hierarchical(elan)
    schema = schema_from_data(hier)

    assert schema.subjects == ["cl", "th"]
    assert schema.categories == ["head"]
    assert schema.labels_per_category == {"head": ["hnod", "htrn"]}
