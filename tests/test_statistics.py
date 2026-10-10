import pytest

from phi3_pooling.statistics import exact_sign_flip_pvalue, paired_bootstrap_interval


def test_exact_sign_flip_for_two_positive_differences() -> None:
    assert exact_sign_flip_pvalue([0.0, 0.0], [1.0, 1.0]) == pytest.approx(0.5)


def test_exact_sign_flip_rejects_unpaired_input() -> None:
    with pytest.raises(ValueError, match="equally sized"):
        exact_sign_flip_pvalue([1.0], [1.0, 2.0])


def test_bootstrap_is_reproducible() -> None:
    first = [0.50, 0.55, 0.60]
    second = [0.60, 0.65, 0.70]
    interval_a = paired_bootstrap_interval(first, second, replicates=1000, seed=17)
    interval_b = paired_bootstrap_interval(first, second, replicates=1000, seed=17)
    assert interval_a == interval_b
    assert interval_a[0] == pytest.approx(0.1)
    assert interval_a[1] == pytest.approx(0.1)

