import math

from decide.scorer import log_softmax, permutations


def test_permutations_start_with_identity_then_reverse():
    assert permutations(3, 1) == [[0, 1, 2]]
    assert permutations(3, 2) == [[0, 1, 2], [2, 1, 0]]


def test_permutations_are_unique_and_capped():
    perms = permutations(3, 10)
    assert len(perms) == len({tuple(p) for p in perms}) == 6  # 2n distinct cyclic/reversed orders
    assert all(sorted(p) == [0, 1, 2] for p in perms)
    assert permutations(2, 5) == [[0, 1], [1, 0]]


def test_log_softmax_normalises_and_is_stable():
    lp = log_softmax([1000.0, 1000.0, 999.0])
    assert math.isclose(sum(math.exp(v) for v in lp), 1.0)
    assert lp[0] == lp[1] > lp[2]
