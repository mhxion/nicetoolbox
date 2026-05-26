"""Tests for ELAN validation and trimming logic."""

import pytest

from nicetoolbox.connectors.elan.elan_data import ElanData, ElanHeader, Interval, Tier
from nicetoolbox.connectors.elan.elan_processing import VideoMeta, trim_tiers, validate_video_alignment


def _make_header(ms_per_sample=33.33, offset=0, duration_ms=1827300):
    return ElanHeader(
        ms_per_sample=ms_per_sample, offset=offset, duration_ms=duration_ms, media_files=["cam1.mp4"], data_start_line=1
    )


def _make_elan(header, tiers=None):
    return ElanData(header=header, tiers=tiers or [])


def _make_tier(tier_name, intervals):
    return Tier(tier_name=tier_name, intervals=[Interval(s, e, a) for s, e, a in intervals])


# --- validate_video_alignment ---


def test_no_header_returns_without_error():
    validate_video_alignment(_make_elan(None), VideoMeta(fps=30.0, duration_sec=1827.3))


def test_no_header_logs_warning(caplog):
    with caplog.at_level("WARNING"):
        validate_video_alignment(_make_elan(None), VideoMeta(fps=30.0, duration_sec=1827.3))
    assert "No ELAN header found" in caplog.text


def test_raises_on_nonzero_offset():
    with pytest.raises(NotImplementedError, match="non-zero offset"):
        validate_video_alignment(_make_elan(_make_header(offset=500)), VideoMeta(fps=30.0, duration_sec=1827.3))


def test_passes_matching_fps():
    validate_video_alignment(_make_elan(_make_header(ms_per_sample=33.33)), VideoMeta(fps=30.003, duration_sec=1827.3))


def test_tolerates_small_fps_difference():
    validate_video_alignment(
        _make_elan(_make_header(ms_per_sample=33.33333206176758)), VideoMeta(fps=30.0, duration_sec=1827.3)
    )


def test_raises_on_fps_mismatch():
    with pytest.raises(ValueError, match="FPS mismatch"):
        validate_video_alignment(
            _make_elan(_make_header(ms_per_sample=33.33)), VideoMeta(fps=25.0, duration_sec=1827.3)
        )


def test_tolerates_small_duration_difference():
    validate_video_alignment(_make_elan(_make_header(duration_ms=1827300)), VideoMeta(fps=30.0, duration_sec=1827.8))


def test_raises_on_duration_mismatch():
    with pytest.raises(ValueError, match="Duration mismatch"):
        validate_video_alignment(
            _make_elan(_make_header(duration_ms=1827300)), VideoMeta(fps=30.003, duration_sec=1900.0)
        )


# --- trim_tiers ---


def test_trim_keeps_intervals_within_window():
    elan = _make_elan(None, [_make_tier("th head", [(2.0, 4.0, "a"), (4.0, 6.0, "b")])])
    result = trim_tiers(elan, 0.0, 10.0)

    assert len(result.tiers[0].intervals) == 2


def test_trim_clips_interval_spanning_end():
    elan = _make_elan(None, [_make_tier("th head", [(8.0, 12.0, "a")])])
    iv = trim_tiers(elan, 0.0, 10.0).tiers[0].intervals[0]

    assert iv.start_sec == 8.0
    assert iv.end_sec == 10.0


def test_trim_clips_interval_spanning_start():
    elan = _make_elan(None, [_make_tier("th head", [(2.0, 7.0, "a")])])
    iv = trim_tiers(elan, 5.0, 10.0).tiers[0].intervals[0]

    assert iv.start_sec == 5.0
    assert iv.end_sec == 7.0


def test_trim_drops_interval_entirely_past_end():
    elan = _make_elan(None, [_make_tier("th head", [(0.0, 2.0, "a"), (12.0, 14.0, "b")])])
    result = trim_tiers(elan, 0.0, 10.0)

    assert len(result.tiers[0].intervals) == 1
    assert result.tiers[0].intervals[0].annotation == "a"


def test_trim_drops_interval_entirely_before_start():
    elan = _make_elan(None, [_make_tier("th head", [(0.0, 2.0, "a"), (6.0, 8.0, "b")])])
    result = trim_tiers(elan, 5.0, 10.0)

    assert len(result.tiers[0].intervals) == 1
    assert result.tiers[0].intervals[0].annotation == "b"


def test_trim_drops_interval_starting_exactly_at_end():
    elan = _make_elan(None, [_make_tier("th head", [(0.0, 2.0, "a"), (10.0, 12.0, "b")])])
    assert len(trim_tiers(elan, 0.0, 10.0).tiers[0].intervals) == 1


def test_trim_drops_interval_ending_exactly_at_start():
    elan = _make_elan(None, [_make_tier("th head", [(0.0, 5.0, "a"), (6.0, 8.0, "b")])])
    result = trim_tiers(elan, 5.0, 10.0)

    assert len(result.tiers[0].intervals) == 1
    assert result.tiers[0].intervals[0].annotation == "b"


def test_trim_preserves_multiple_tiers():
    elan = _make_elan(
        None,
        [
            _make_tier("th head", [(0.0, 2.0, "a"), (11.0, 13.0, "b")]),
            _make_tier("cl eyes", [(9.0, 12.0, "c")]),
        ],
    )
    result = trim_tiers(elan, 0.0, 10.0)

    assert len(result.tiers) == 2
    assert len(result.tiers[0].intervals) == 1
    assert result.tiers[1].intervals[0].end_sec == 10.0


def test_trim_preserves_header():
    header = _make_header()
    elan = _make_elan(header, [_make_tier("th head", [(0.0, 2.0, "a")])])
    assert trim_tiers(elan, 0.0, 10.0).header is header
