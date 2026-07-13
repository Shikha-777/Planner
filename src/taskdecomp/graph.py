from __future__ import annotations

from collections import defaultdict, deque


def transitive_reduction_like(nodes: list[str], edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop duplicate, self-loop, cyclic, and obviously transitively redundant edges."""
    node_set = set(nodes)
    clean: list[tuple[str, str]] = []
    seen = set()
    for a, b in edges:
        if a == b or a not in node_set or b not in node_set or (a, b) in seen:
            continue
        seen.add((a, b))
        clean.append((a, b))

    dag: list[tuple[str, str]] = []
    for edge in clean:
        candidate = dag + [edge]
        if is_acyclic(nodes, candidate):
            dag.append(edge)

    reduced: list[tuple[str, str]] = []
    for edge in dag:
        without = [e for e in dag if e != edge]
        if not has_path(edge[0], edge[1], without):
            reduced.append(edge)
    return reduced


def is_acyclic(nodes: list[str], edges: list[tuple[str, str]]) -> bool:
    indeg = {n: 0 for n in nodes}
    out = defaultdict(list)
    for a, b in edges:
        out[a].append(b)
        indeg[b] = indeg.get(b, 0) + 1
    queue = deque([n for n in nodes if indeg.get(n, 0) == 0])
    seen = 0
    while queue:
        node = queue.popleft()
        seen += 1
        for child in out[node]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    return seen == len(nodes)


def has_path(src: str, dst: str, edges: list[tuple[str, str]]) -> bool:
    out = defaultdict(list)
    for a, b in edges:
        out[a].append(b)
    queue = deque([src])
    visited = set()
    while queue:
        node = queue.popleft()
        if node == dst:
            return True
        if node in visited:
            continue
        visited.add(node)
        queue.extend(out[node])
    return False

