"""NPZ output schema: defines array axes independently of the annotation data."""

import logging
from dataclasses import dataclass

from .labeling_data import MultipersonHierarchicalData


@dataclass(frozen=True)
class NpzSchema:
    subjects: list[str]
    categories: list[str]
    labels_per_category: dict[str, list[str]]

    def __str__(self) -> str:
        return f"NpzSchema: subjects={self.subjects}, labels_per_category={self.labels_per_category}"


def schema_from_data(data: MultipersonHierarchicalData) -> NpzSchema:
    return NpzSchema(
        subjects=sorted(data.subjects),
        categories=sorted(data.categories),
        labels_per_category={cat: sorted(labels) for cat, labels in data.labels_per_category.items()},
    )


def schema_from_config(data: MultipersonHierarchicalData, expected_labels: dict[str, set[str]]) -> NpzSchema:
    unknown_cats = data.categories - set(expected_labels)
    if unknown_cats:
        raise ValueError(
            f"Data contains categories not defined in config: {sorted(unknown_cats)} "
            f"(config defines: {sorted(expected_labels)})"
        )

    actual_labels = data.labels_per_category
    for category, expected in expected_labels.items():
        actual = actual_labels.get(category, set())
        unexpected = sorted(actual - expected)
        if unexpected:
            raise ValueError(f"Unexpected labels in category '{category}': {unexpected}. Expected: {sorted(expected)}")
        missing = sorted(expected - actual)
        if missing:
            logging.warning(f"Category '{category}': expected labels not found in data: {missing}")

    logging.info(f"Label validation passed for categories: {sorted(expected_labels.keys())}")

    return NpzSchema(
        subjects=sorted(data.subjects),
        categories=sorted(expected_labels),
        labels_per_category={cat: sorted(labels) for cat, labels in expected_labels.items()},
    )
