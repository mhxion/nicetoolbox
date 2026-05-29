from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple


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

    for node, deps in graph.items():
        for dep in deps:
            if dep not in graph:
                missing.append((node, dep))
                continue
            dependents[dep].append(node)
            in_degree[node] += 1

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

    if len(order) != len(graph):
        remaining = [n for n in graph if n not in set(order)]
        raise ValueError(remaining)

    return order
