import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from ..configs.schemas.detectors_config import DetectorsConfig


def topological_sort(
    graph: Dict[str, List[str]],
    missing: Optional[List[Tuple[str, str]]] = None,
) -> List[str]:
    """
    Topological sort using Kahn's algorithm.
    Deterministic: alphabetical tie-breaking for nodes at the same level.

    Args:
        graph: Mapping of node -> list of nodes it depends on.
        missing: If provided, (node, dep) pairs for missing dependencies are
            appended here.

    Raises:
        ValueError: On circular dependencies.
    """
    if missing is None:
        missing = []
    in_degree = {name: 0 for name in graph}
    dependents = defaultdict(list)

    # check all outcomming connections
    for node, deps in graph.items():
        for dep in deps:
            if dep not in graph:  # we ignore missing dependencies
                missing.append((node, dep))
                continue
            dependents[dep].append(node)
            in_degree[node] += 1

    # start from the graph top
    # we use name sorting for determenism
    queue = deque(sorted(name for name, deg in in_degree.items() if deg == 0))
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        newly_ready = []
        for dep in dependents[node]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                newly_ready.append(dep)
        queue.extend(sorted(newly_ready))

    # loop detection
    if len(order) != len(graph):
        remaining = [n for n in graph if n not in set(order)]
        raise ValueError(remaining)

    return order


def sort_detectors_order(
    detectors_config: DetectorsConfig,
    selected_algorithms: List[str],
    strict: bool = True,
) -> List[str]:
    """
    Sort detectors algorithms in topological order.

    Args:
        detectors_config: All detectors configurations containing input_detector_names.
        selected_algorithms: Algorithm names to order.
        strict: If True, raise on missing dependencies. If False, log a warning and skip them.

    Raises:
        KeyError: On missing dependencies (only when strict mode).
        ValueError: On circular dependencies.
    """
    # get the dependencies for each selected detector
    graph = {}
    for algo_name in selected_algorithms:
        config = detectors_config.algorithms[algo_name]
        input_deps = getattr(config, "input_detector_names", None) or []
        graph[algo_name] = [pair[1] for pair in input_deps]

    missing = []
    try:
        order = topological_sort(graph, missing)
    except ValueError as e:
        cycle = e.args[0]
        raise ValueError(
            f"Circular dependency detected among detectors: {cycle}. "
            f"Check 'input_detector_names' in your detectors config. "
            f"Selected algorithms: {selected_algorithms}"
        ) from None

    # raise if we have any missing deps
    if missing and strict:
        raise KeyError(
            f"Missing dependencies in detector configuration: "
            f"{'; '.join(f'{dep} -> {node}' for node, dep in missing)}. "
            f"Ensure all required detectors are included in the run config, "
            f"or set check_missing_detectors_dependnencies = false in run_config.toml."
        )

    # or we are in non-strict mode - just write a warning
    for node, dep in missing:
        logging.warning(
            f"Detector '{node}' depends on '{dep}', which is not in the selected algorithms. "
            f"Assuming output from a previous run."
        )

    return order
