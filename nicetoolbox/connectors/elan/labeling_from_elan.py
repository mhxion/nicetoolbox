"""Convert flat ELAN tier data into hierarchical multiperson labeling structure."""

from collections import defaultdict
from typing import Optional

from .elan_data import ElanData, Interval
from .labeling_data import LabeledInterval, MultipersonHierarchicalData


def parse_tier_name(tier_name: str) -> tuple[str, str]:
    """Split '<subject> <category>' tier name into (subject, category)."""
    parts = tier_name.split(" ", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Tier name '{tier_name}' does not follow the expected '<subject> <category>' convention")
    return parts[0], parts[1]


def parse_labels(annotation: str) -> frozenset[str]:
    annotations = annotation.split(",")
    return frozenset(p.strip() for p in annotations if p.strip())


def elan_data_to_hierarchical(elan_data: ElanData) -> MultipersonHierarchicalData:
    categories_per_subject: defaultdict[str, set[str]] = defaultdict(set)
    raw_intervals: dict[tuple[str, str], list[Interval]] = {}

    for tier in elan_data.tiers:
        subject, category = parse_tier_name(tier.tier_name)
        categories_per_subject[subject].add(category)
        raw_intervals[(subject, category)] = tier.intervals

    categories: Optional[set[str]] = None
    for subject, subject_categories in categories_per_subject.items():
        if categories is None:
            categories = subject_categories
            continue
        missing = categories.difference(subject_categories)
        if missing:
            raise ValueError(f"Subject '{subject}' has missing categories: {missing}")

    if not categories:
        raise ValueError("Data doesn't contain any categories!")

    subjects = set(categories_per_subject)
    data: dict[str, dict[str, list[LabeledInterval]]] = {}
    for subject in subjects:
        data[subject] = {}
        for category in categories:
            labeled: list[LabeledInterval] = []
            for iv in raw_intervals[(subject, category)]:
                labels = parse_labels(iv.annotation)
                labeled.append(LabeledInterval(iv.start_sec, iv.end_sec, labels))
            data[subject][category] = labeled

    return MultipersonHierarchicalData(data=data)
