"""Tests for elan_parser module."""

from pathlib import Path

import pytest

from nicetoolbox.connectors.elan.elan_parser import parse_elan_file, parse_header, parse_tiers

_META = "offset: {offset}, duration: {dur}, ms per sample: {ms}"
_DATA_LINE = "th head\t\t00:00:00.000\t0.0\t00:00:02.000\t2.0\t00:00:02.000\t2.0\t\n"


def _header_line(cam, offset=0, dur="00:30:27.300 / 1827.300 / 1827300", ms="33.33"):
    meta = _META.format(offset=offset, dur=dur, ms=ms)
    return f'"#file:///path/{cam} -- {meta}"\n'


# --- parse_header ---


def test_parse_header_extracts_metadata():
    lines = [
        '"#file:///path/to/Cam1.mp4 -- offset: 0, duration: 00:30:27.300 / 1827.300 / 1827300,'
        ' ms per sample: 33.33333206176758"\n',
        "\n",
        "th head\t\t00:00:00.000\t0.0\t00:00:02.000\t2.0\t00:00:02.000\t2.0\thtrn\n",
    ]
    header = parse_header(lines)

    assert header is not None
    assert header.ms_per_sample == pytest.approx(33.333332, abs=1e-4)
    assert header.offset == 0
    assert header.duration_ms == 1827300
    assert header.data_start_line == 2


def test_parse_header_extracts_media_files():
    lines = [
        '"#file:///Volumes/data/Cam4.mp4 -- offset: 0, duration: 00:30:27.300 / 1827.300 / 1827300,'
        ' ms per sample: 33.0"\n',
        "\n",
        _DATA_LINE,
    ]
    header = parse_header(lines)

    assert header is not None
    assert len(header.media_files) == 1
    assert "Volumes/data/Cam4.mp4" in header.media_files[0]


def test_parse_header_allows_multiple_consistent_media_files():
    lines = [_header_line("Cam1.mp4"), _header_line("Cam2.mp4"), _header_line("Cam4.mp4"), "\n", _DATA_LINE]
    header = parse_header(lines)

    assert header is not None
    assert len(header.media_files) == 3


def test_parse_header_raises_on_inconsistent_ms_per_sample():
    lines = [_header_line("Cam1.mp4"), _header_line("Cam2.mp4", ms="40.0"), "\n", _DATA_LINE]
    with pytest.raises(ValueError, match="Inconsistent ms_per_sample"):
        parse_header(lines)


def test_parse_header_raises_on_inconsistent_offset():
    lines = [_header_line("Cam1.mp4"), _header_line("Cam2.mp4", offset=100), "\n", _DATA_LINE]
    with pytest.raises(ValueError, match="Inconsistent offset"):
        parse_header(lines)


def test_parse_header_raises_on_inconsistent_duration():
    lines = [
        _header_line("Cam1.mp4"),
        _header_line("Cam2.mp4", dur="00:10:00.000 / 600.000 / 600000"),
        "\n",
        _DATA_LINE,
    ]
    with pytest.raises(ValueError, match="Inconsistent duration"):
        parse_header(lines)


def test_parse_header_raises_on_malformed_line():
    lines = ['"#file:///path/to/Cam1.mp4 -- some garbage"\n', "\n", _DATA_LINE]
    with pytest.raises(ValueError, match="does not match expected format"):
        parse_header(lines)


def test_parse_header_missing():
    lines = [_DATA_LINE, "th head\t\t00:00:02.000\t2.0\t00:00:04.000\t4.0\t00:00:02.000\t2.0\t\n"]
    assert parse_header(lines) is None


def test_parse_header_skips_blank_lines():
    lines = ["\n", "\n", _DATA_LINE]
    assert parse_header(lines) is None


# --- parse_tiers ---


def test_parse_tiers_basic():
    lines = [
        "th head\t\t00:00:00.000\t0.0\t00:00:02.000\t2.0\t00:00:02.000\t2.0\thtrn\n",
        "th head\t\t00:00:02.000\t2.0\t00:00:04.000\t4.0\t00:00:02.000\t2.0\t\n",
        "th head\t\t00:00:04.000\t4.0\t00:00:06.000\t6.0\t00:00:02.000\t2.0\thnod, htrn\n",
    ]
    tiers = parse_tiers(lines, 0)

    assert len(tiers) == 1
    assert tiers[0].tier_name == "th head"
    assert len(tiers[0].intervals) == 3


def test_parse_tiers_interval_values():
    lines = ["cl eyes\t\t00:01:00.000\t60.0\t00:01:02.000\t62.0\t00:00:02.000\t2.0\teymov\n"]
    iv = parse_tiers(lines, 0)[0].intervals[0]

    assert iv.start_sec == 60.0
    assert iv.end_sec == 62.0
    assert iv.annotation == "eymov"


def test_parse_tiers_empty_labels():
    lines = ["th head\t\t00:00:00.000\t0.0\t00:00:02.000\t2.0\t00:00:02.000\t2.0\t\n"]
    assert parse_tiers(lines, 0)[0].intervals[0].annotation == ""


def test_parse_tiers_multiple_tiers():
    lines = [
        "th head\t\t00:00:00.000\t0.0\t00:00:02.000\t2.0\t00:00:02.000\t2.0\t\n",
        "th eyes\t\t00:00:00.000\t0.0\t00:00:02.000\t2.0\t00:00:02.000\t2.0\teymov\n",
        "cl head\t\t00:00:00.000\t0.0\t00:00:02.000\t2.0\t00:00:02.000\t2.0\thnod\n",
    ]
    tiers = parse_tiers(lines, 0)

    assert len(tiers) == 3
    assert {t.tier_name for t in tiers} == {"th head", "th eyes", "cl head"}


def test_parse_tiers_skips_blank_lines():
    lines = [
        "th head\t\t00:00:00.000\t0.0\t00:00:02.000\t2.0\t00:00:02.000\t2.0\t\n",
        "\n",
        "th head\t\t00:00:02.000\t2.0\t00:00:04.000\t4.0\t00:00:02.000\t2.0\t\n",
    ]
    assert len(parse_tiers(lines, 0)[0].intervals) == 2


# --- parse_elan_file validation ---


def test_parse_elan_file_rejects_non_txt():
    with pytest.raises(ValueError, match="Expected a .txt file"):
        parse_elan_file(Path("data.csv"))


def test_parse_elan_file_rejects_eaf():
    with pytest.raises(NotImplementedError, match=".eaf files aren't supported"):
        parse_elan_file(Path("annotation.eaf"))
