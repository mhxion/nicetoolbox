"""Parsing ELAN tab-delimited txt export files."""

import logging
import re
from pathlib import Path

from .elan_data import ElanData, ElanHeader, Interval, Tier

_HEADER_LINE_RE = re.compile(
    r'"#file:///(?P<path>.+?)\s+--\s+'
    r"offset:\s*(?P<offset>\d+),\s+"
    r"duration:\s*\S+\s*/\s*\S+\s*/\s*(?P<duration_ms>\d+),\s+"
    r"ms per sample:\s*(?P<ms_per_sample>[\d.]+)"
)


def parse_elan_file(file_path: Path) -> ElanData:
    if file_path.suffix == ".eaf":
        raise NotImplementedError(".eaf files aren't supported, please use ELAN .txt export")
    if file_path.suffix != ".txt":
        raise ValueError(f"Expected a .txt file, got: {file_path}")

    logging.info(f"Reading ELAN file: {file_path}")
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    logging.info("Trying to read ELAN header...")
    header = parse_header(lines)
    if header is not None:
        logging.info(f"Found {header}")
        data_start_line = header.data_start_line
    else:
        logging.warning("No ELAN header found!")
        data_start_line = 0

    logging.info("Parsing ELAN data...")
    tiers = parse_tiers(lines, data_start_line)
    data = ElanData(header, tiers)
    logging.info(f"Parsed {data}")
    return data


def parse_header(lines: list[str]) -> ElanHeader | None:
    media_files: list[str] = []
    ms_per_samples: list[float] = []
    offsets: list[int] = []
    durations_ms: list[int] = []
    data_start_line = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('"#'):
            match = _HEADER_LINE_RE.search(stripped)
            if not match:
                raise ValueError(f"Header line {i + 1} does not match expected format: {stripped[:120]}")
            media_files.append(match.group("path"))
            ms_per_samples.append(float(match.group("ms_per_sample")))
            offsets.append(int(match.group("offset")))
            durations_ms.append(int(match.group("duration_ms")))
        elif stripped == "":
            continue
        else:
            data_start_line = i
            break

    if not media_files:
        return None

    if len(set(ms_per_samples)) > 1:
        raise ValueError(f"Inconsistent ms_per_sample across header lines: {ms_per_samples}")
    if len(set(offsets)) > 1:
        raise ValueError(f"Inconsistent offset across header lines: {offsets}")
    if len(set(durations_ms)) > 1:
        raise ValueError(f"Inconsistent duration across header lines: {durations_ms}")

    return ElanHeader(ms_per_samples[0], offsets[0], durations_ms[0], media_files, data_start_line)


def parse_tiers(lines: list[str], start_idx: int) -> list[Tier]:
    tier_ivs: dict[str, list[Interval]] = {}

    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            continue

        fields = stripped.split("\t")
        if len(fields) < 8:
            raise ValueError(f"Expected interval record, got {line}")

        tier_name = fields[0]
        start_sec = float(fields[3])
        end_sec = float(fields[5])
        annotation = fields[8].strip() if len(fields) > 8 else ""

        if tier_name not in tier_ivs:
            tier_ivs[tier_name] = []
        tier_ivs[tier_name].append(Interval(start_sec, end_sec, annotation))

    return [Tier(tier_name, intervals) for tier_name, intervals in tier_ivs.items()]
