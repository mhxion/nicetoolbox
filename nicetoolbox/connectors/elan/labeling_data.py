"""Multiperson annotations grouped by subject, category, and labeled intervals."""

import copy
import logging
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class LabeledInterval:
    start_sec: float
    end_sec: float
    labels: frozenset[str]


@dataclass
class MultipersonHierarchicalData:
    data: dict[str, dict[str, list[LabeledInterval]]]

    @property
    def subjects(self) -> set[str]:
        return set(self.data.keys())

    @property
    def categories(self) -> set[str]:
        if not self.data:
            return set()
        return set(next(iter(self.data.values())).keys())

    @property
    def labels_per_category(self) -> dict[str, set[str]]:
        result: defaultdict[str, set[str]] = defaultdict(set)
        for subject_data in self.data.values():
            for cat, intervals in subject_data.items():
                for iv in intervals:
                    result[cat].update(iv.labels)
        return dict(result)

    def __str__(self) -> str:
        return (
            f"MultipersonHierarchicalData: "
            f"subjects=[{self.subjects}], "
            f"label_per_category={self.labels_per_category}"
        )


def apply_category_defaults(
    data: MultipersonHierarchicalData, category_defaults: dict[str, str]
) -> MultipersonHierarchicalData:
    unknown = set(category_defaults) - data.categories
    if unknown:
        raise ValueError(f"Category defaults reference unknown categories: {unknown} (known: {data.categories})")

    new_data = copy.deepcopy(data.data)
    counts: dict[str, int] = {}
    for subject in new_data:
        for category, intervals in new_data[subject].items():
            if category not in category_defaults:
                continue
            default = frozenset({category_defaults[category]})
            for idx, iv in enumerate(intervals):
                if not iv.labels:
                    intervals[idx] = LabeledInterval(iv.start_sec, iv.end_sec, default)
                    counts[category] = counts.get(category, 0) + 1

    for category, n in sorted(counts.items()):
        logging.info(f"Replaced {n} empty annotations in category '{category}' with '{category_defaults[category]}'")

    return MultipersonHierarchicalData(data=new_data)


def rename_subjects(data: MultipersonHierarchicalData, subject_mapping: dict[str, str]) -> MultipersonHierarchicalData:
    unknown = set(subject_mapping) - data.subjects
    if unknown:
        raise ValueError(f"Subject mapping references unknown subjects: {unknown} (known: {data.subjects})")

    new_data = {subject_mapping.get(subj, subj): intervals for subj, intervals in data.data.items()}
    logging.info(f"Renamed subjects: {subject_mapping}")
    return MultipersonHierarchicalData(data=new_data)
