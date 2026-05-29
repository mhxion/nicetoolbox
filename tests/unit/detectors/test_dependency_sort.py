import pytest

from nicetoolbox.utils.dependency_sort import topological_sort


def test_no_dependencies():
    """All independent nodes => alphabetical order."""
    graph = {"c": [], "a": [], "b": []}
    assert topological_sort(graph) == ["a", "b", "c"]


def test_simple_chain():
    """A -> B -> C."""
    graph = {"c": ["b"], "b": ["a"], "a": []}
    assert topological_sort(graph) == ["a", "b", "c"]


def test_diamond_dependency():
    """A depends on B and C, both depend on D."""
    graph = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
    result = topological_sort(graph)
    assert result.index("d") < result.index("b")
    assert result.index("d") < result.index("c")
    assert result.index("b") < result.index("a")
    assert result.index("c") < result.index("a")


def test_current_bipartite_structure():
    """Methods before features, matching current behavior."""
    graph = {
        "velocity_body": ["hrnetw48"],
        "gaze_distance": ["eth_xgaze", "hrnetw48"],
        "hrnetw48": [],
        "eth_xgaze": [],
    }
    result = topological_sort(graph)
    assert result.index("hrnetw48") < result.index("velocity_body")
    assert result.index("hrnetw48") < result.index("gaze_distance")
    assert result.index("eth_xgaze") < result.index("gaze_distance")


def test_deterministic_order():
    """Same graph always produces same output, regardless of insertion order."""
    graph_a = {"velocity_body": ["hrnetw48"], "hrnetw48": [], "spiga": []}
    graph_b = {"spiga": [], "hrnetw48": [], "velocity_body": ["hrnetw48"]}
    assert topological_sort(graph_a) == topological_sort(graph_b)


def test_circular_dependency_raises():
    graph = {"a": ["b"], "b": ["a"]}
    with pytest.raises(ValueError) as exc_info:
        topological_sort(graph)
    cycle = exc_info.value.args[0]
    assert "a" in cycle
    assert "b" in cycle


def test_missing_dependency_raises():
    graph = {"a": ["b"]}
    topological_sort(graph)


def test_missing_dependency_collected():
    """When missing list is provided, skip missing deps and collect them."""
    graph = {"a": ["b"], "c": ["d", "a"]}
    missing = []
    result = topological_sort(graph, missing=missing)
    assert result == ["a", "c"]
    assert ("a", "b") in missing
    assert ("c", "d") in missing


def test_missing_dependency_collected_empty():
    """When missing list is provided but no deps are missing, list stays empty."""
    graph = {"a": [], "b": ["a"]}
    missing = []
    result = topological_sort(graph, missing=missing)
    assert result == ["a", "b"]
    assert missing == []


def test_three_level_chain():
    """Arbitrary depth: feature -> feature -> method."""
    graph = {"summary": ["gaze_distance"], "gaze_distance": ["hrnetw48"], "hrnetw48": []}
    assert topological_sort(graph) == ["hrnetw48", "gaze_distance", "summary"]


def test_empty_graph():
    assert topological_sort({}) == []


def test_single_node():
    assert topological_sort({"a": []}) == ["a"]
