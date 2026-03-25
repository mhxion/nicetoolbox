import pytest
from pydantic import BaseModel, ValidationError

from nicetoolbox.configs.models.video_timestamp import VideoTimestamp, timestamp_to_frame_index, timestamp_to_ms


class _TimestampModel(BaseModel):
    value: VideoTimestamp


# --- Parsing tests ---


def test_parse_dash_format():
    """
    Given: A string in "HH-MM-SS" dash format.
    When:  The model is validated.
    Then:  It stores the dash-separated string.
    """
    model = _TimestampModel.model_validate({"value": "01-23-45"})
    assert model.value == "01-23-45"


def test_parse_dash_format_with_milliseconds():
    """
    Given: A string in "HH-MM-SS.mmm" dash format.
    When:  The model is validated.
    Then:  It stores the dash-separated string with milliseconds.
    """
    model = _TimestampModel.model_validate({"value": "01-23-45.678"})
    assert model.value == "01-23-45.678"


def test_parse_colon_format_normalized_to_dashes():
    """
    Given: A string in "HH:MM:SS" colon format.
    When:  The model is validated.
    Then:  It normalizes colons to dashes.
    """
    model = _TimestampModel.model_validate({"value": "01:23:45"})
    assert model.value == "01-23-45"


def test_parse_colon_format_with_milliseconds_normalized():
    """
    Given: A string in "HH:MM:SS.mmm" colon format.
    When:  The model is validated.
    Then:  It normalizes colons to dashes, preserving milliseconds.
    """
    model = _TimestampModel.model_validate({"value": "01:23:45.678"})
    assert model.value == "01-23-45.678"


def test_parse_zero_timestamp():
    """
    Given: A zero timestamp "00-00-00".
    When:  The model is validated.
    Then:  It stores "00-00-00".
    """
    model = _TimestampModel.model_validate({"value": "00-00-00"})
    assert model.value == "00-00-00"


def test_parse_large_hours():
    """
    Given: A timestamp with large hours value "99-00-00".
    When:  The model is validated.
    Then:  It parses correctly (hours are not bounded).
    """
    model = _TimestampModel.model_validate({"value": "99-00-00"})
    assert model.value == "99-00-00"


# --- Rejection tests ---


def test_reject_invalid_minutes():
    """
    Given: A timestamp with minutes > 59.
    When:  The model is validated.
    Then:  A ValidationError is raised.
    """
    with pytest.raises(ValidationError, match="Minutes must be 0-59"):
        _TimestampModel.model_validate({"value": "00-60-00"})


def test_reject_invalid_seconds():
    """
    Given: A timestamp with seconds > 59.
    When:  The model is validated.
    Then:  A ValidationError is raised.
    """
    with pytest.raises(ValidationError, match="Seconds must be 0-59"):
        _TimestampModel.model_validate({"value": "00-00-60"})


def test_reject_non_string():
    """
    Given: A non-string value (float).
    When:  The model is validated.
    Then:  A ValidationError is raised.
    """
    with pytest.raises(ValidationError):
        _TimestampModel.model_validate({"value": 12.5})


def test_reject_malformed_string():
    """
    Given: A string that does not match the timestamp format.
    When:  The model is validated.
    Then:  A ValidationError is raised.
    """
    with pytest.raises(ValidationError, match="Invalid timestamp format"):
        _TimestampModel.model_validate({"value": "abc"})


def test_reject_single_digit_fields():
    """
    Given: A timestamp with single-digit fields like "1-2-3".
    When:  The model is validated.
    Then:  A ValidationError is raised (format requires 2-digit fields).
    """
    with pytest.raises(ValidationError, match="Invalid timestamp format"):
        _TimestampModel.model_validate({"value": "1-2-3"})


def test_reject_mixed_separators():
    """
    Given: A timestamp mixing colons and dashes like "00:01-30".
    When:  The model is validated.
    Then:  A ValidationError is raised.
    """
    with pytest.raises(ValidationError, match="Invalid timestamp format"):
        _TimestampModel.model_validate({"value": "00:01-30"})


# --- Serialization roundtrip ---


def test_roundtrip_json():
    """
    Given: A timestamp string.
    When:  Parsed and then serialized to JSON.
    Then:  The output matches the normalized (dash) form.
    """
    model = _TimestampModel.model_validate({"value": "02:30:15.100"})
    serialized = model.model_dump(mode="json")["value"]
    assert serialized == "02-30-15.100"


# --- timestamp_to_frame_index tests ---


def test_frame_index_from_timestamp_string():
    """
    Given: A timestamp string "00-00-03" and fps=30.
    When:  timestamp_to_frame_index is called.
    Then:  It returns 90 (3 * 30).
    """
    assert timestamp_to_frame_index("00-00-03", fps=30) == 90


def test_frame_index_passthrough_int():
    """
    Given: An int frame index.
    When:  timestamp_to_frame_index is called.
    Then:  It returns the same int unchanged.
    """
    assert timestamp_to_frame_index(100, fps=30) == 100


def test_frame_index_from_zero_timestamp():
    """
    Given: A zero timestamp string.
    When:  timestamp_to_frame_index is called.
    Then:  It returns 0.
    """
    assert timestamp_to_frame_index("00-00-00", fps=30) == 0


def test_frame_index_with_milliseconds():
    """
    Given: A timestamp with milliseconds "00-00-01.500" and fps=30.
    When:  timestamp_to_frame_index is called.
    Then:  It returns 45 (1.5 * 30).
    """
    assert timestamp_to_frame_index("00-00-01.500", fps=30) == 45


def test_frame_index_passthrough_negative():
    """
    Given: A negative int (-1, meaning full length).
    When:  timestamp_to_frame_index is called.
    Then:  It returns -1 unchanged (special value preserved).
    """
    assert timestamp_to_frame_index(-1, fps=30) == -1


# --- timestamp_to_milliseconds tests ---


def test_milliseconds_from_timestamp_string():
    """
    Given: A timestamp string "00-00-03" and fps=30.
    When:  timestamp_to_milliseconds is called.
    Then:  It returns 3000 (3 seconds).
    """
    assert timestamp_to_ms("00-00-03", fps=30) == 3000


def test_milliseconds_from_timestamp_with_milliseconds():
    """
    Given: A timestamp string "00-00-01.500" and fps=30.
    When:  timestamp_to_milliseconds is called.
    Then:  It returns 1500 (1.5 seconds).
    """
    assert timestamp_to_ms("00-00-01.500", fps=30) == 1500


def test_milliseconds_from_timestamp_hours_minutes():
    """
    Given: A timestamp string "01-30-00" and fps=30.
    When:  timestamp_to_milliseconds is called.
    Then:  It returns 5400000 (1h30m in ms).
    """
    assert timestamp_to_ms("01-30-00", fps=30) == 5_400_000


def test_milliseconds_from_zero_timestamp():
    """
    Given: A zero timestamp string "00-00-00" and fps=30.
    When:  timestamp_to_milliseconds is called.
    Then:  It returns 0.
    """
    assert timestamp_to_ms("00-00-00", fps=30) == 0


def test_milliseconds_from_frame_index():
    """
    Given: A frame index 90 and fps=30.
    When:  timestamp_to_milliseconds is called.
    Then:  It returns 3000 (90 / 30 * 1000).
    """
    assert timestamp_to_ms(90, fps=30) == 3000


def test_milliseconds_from_frame_index_with_remainder():
    """
    Given: A frame index 45 and fps=30.
    When:  timestamp_to_milliseconds is called.
    Then:  It returns 1500 (45 / 30 * 1000).
    """
    assert timestamp_to_ms(45, fps=30) == 1500


def test_milliseconds_from_zero_frame_index():
    """
    Given: A frame index 0 and fps=30.
    When:  timestamp_to_milliseconds is called.
    Then:  It returns 0.
    """
    assert timestamp_to_ms(0, fps=30) == 0
