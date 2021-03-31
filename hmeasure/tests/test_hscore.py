import numpy
import pytest

from ..hsource import h_score


def test_h_score_binary_validation_smaller():
    # Given
    prand = numpy.random.default_rng(42)
    # When
    # y_true randomly sampled integers from interval [0, 2)
    y_true = prand.integers(low=0, high=2, size=10)
    # y_pred random sampled in interval [-1, 0)
    y_pred = (0 - (-1)) * prand.random(10) + (-1)
    # Then
    with pytest.raises(TypeError, match=r".*smaller.*"):
        h_score(y_true, y_pred)


def test_h_score_binary_validation_larger():
    # Given
    prand = numpy.random.default_rng(42)
    # When
    # y_true randomly sampled integers from interval [0, 2)
    y_true = prand.integers(low=0, high=2, size=10)
    # y_pred random sampled in interval [1, 2)
    y_pred = (2 - 1) * prand.random(10) + 1
    # Then
    with pytest.raises(TypeError, match=r".*larger.*"):
        h_score(y_true, y_pred)


def test_h_score_validation_nonbinary():
    # Given
    prand = numpy.random.default_rng(42)
    # When
    # y_true randomly sampled integers from interval [0, 4)
    y_true = prand.integers(low=0, high=4, size=10)
    # y_pred random sampled in interval [1, 2)
    y_pred = (2 - 1) * prand.random(10) + 1
    # Then
    with pytest.raises(TypeError, match=r".*binary.*"):
        h_score(y_true, y_pred)


def test_h_score_validation_nonbinary_smaller():
    # Given
    prand = numpy.random.default_rng(42)

    # When
    # y_true randomly sampled integers from interval [0, 4)
    y_true = prand.integers(low=0, high=4, size=10)
    # y_pred random sampled in interval [-1, 2)
    y_pred = (2 - (-1)) * prand.random(10) + (-1)

    # Then
    with pytest.raises(TypeError, match=r".*binary.*"):
        h_score(y_true, y_pred)


def test_h_score_validation_nonbinary_larger():
    # Given
    prand = numpy.random.default_rng(42)

    # When
    # y_true randomly sampled integers from interval [0, 4)
    y_true = prand.integers(low=0, high=4, size=10)
    # y_pred random sampled in interval [2, 7)
    y_pred = (7 - (2)) * prand.random(10) + (2)

    # Then
    with pytest.raises(TypeError, match=r".*binary.*"):
        h_score(y_true, y_pred)
