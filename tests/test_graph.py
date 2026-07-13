from taskdecomp.graph import is_acyclic, transitive_reduction_like


def test_is_acyclic_rejects_cycle():
    assert is_acyclic(["s1", "s2"], [("s1", "s2")])
    assert not is_acyclic(["s1", "s2"], [("s1", "s2"), ("s2", "s1")])


def test_transitive_reduction_drops_redundant_edge():
    edges = transitive_reduction_like(["s1", "s2", "s3"], [("s1", "s2"), ("s2", "s3"), ("s1", "s3")])
    assert edges == [("s1", "s2"), ("s2", "s3")]

