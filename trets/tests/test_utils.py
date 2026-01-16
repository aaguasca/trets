import astropy.units as u
import numpy as np
import pytest
from trets.utils import (
    get_intervals,
    fraction_outside_interval,
    weighted_average_error_calculation,
    get_intervals_sum,
    split_data_from_intervals,
)


@pytest.fixture
def data():
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.mark.parametrize(
    "n, expected_output",
    [
        (2, np.array([1, 3, 5, 7, 9, 10])),
        (3, np.array([1, 4, 7, 10])),
        (4, np.array([1, 5, 9, 10])),
    ],
)
def test_get_intervals(data, n, expected_output):
    selected_data = get_intervals(data, n)
    assert np.array_equal(selected_data, expected_output)


def test_get_intervals_output_type(data):
    selected_data = get_intervals(data, 2)
    assert isinstance(selected_data, np.ndarray)


def test_get_intervals_last_element(data):
    selected_data = get_intervals(data, 2)
    assert selected_data[-1] == data[-1]


def test_fraction_outside_interval():
    x = [0, 10]
    xmin = 3
    xmax = 7
    expected_output = 0.6
    output = fraction_outside_interval(x, xmin, xmax)
    assert np.isclose(output, expected_output)

    x = [0, 10]
    xmin = -3
    xmax = 13
    expected_output = 0.0
    output = fraction_outside_interval(x, xmin, xmax)
    assert np.isclose(output, expected_output)

    x = [0, 10]
    xmin = 5
    xmax = 15
    expected_output = 0.5
    output = fraction_outside_interval(x, xmin, xmax)
    assert np.isclose(output, expected_output)

    x = [0, 10]
    xmin = -5
    xmax = 5
    expected_output = 0.5
    output = fraction_outside_interval(x, xmin, xmax)
    assert np.isclose(output, expected_output)


def test_weighted_average_error_calculation():
    errors = [0.1, 0.2, 0.3]
    weights = [0.2, 0.3, 0.5]
    expected_output = 0.0265
    output = weighted_average_error_calculation(errors, weights)
    assert np.isclose(output, expected_output)

    errors = [1, 2, 3, 4]
    weights = [0.25, 0.25, 0.25, 0.25]
    expected_output = 1.875
    output = weighted_average_error_calculation(errors, weights)
    assert np.isclose(output, expected_output)

    errors = [0.1, 0.2, 0.3]
    weights = [0.2, 0.3, 0.4]
    expected_output = 0.02271604
    output = weighted_average_error_calculation(errors, weights)
    assert np.isclose(output, expected_output)

    errors = [0.1, 0.2, 0.3]
    weights = [0.2, 0.3, 0.6]
    expected_output = 0.0300826
    output = weighted_average_error_calculation(errors, weights)
    assert np.isclose(output, expected_output)


def test_get_intervals_sum():
    start = np.array([1, 8, 12, 21, 31, 40, 52, 60]) * u.s
    stop = np.array([6, 9, 17, 26, 36, 51, 53, 70]) * u.s
    thd_sum = 10 * u.s

    intervals = get_intervals_sum(start, stop, thd_sum)

    expected_intervals = [
        [1 * u.s, 9 * u.s],
        [12 * u.s, 26 * u.s],
        [31 * u.s, 36 * u.s],
        [40 * u.s, 51 * u.s],
        [52 * u.s, 53 * u.s],
        [60 * u.s, 70 * u.s],
    ]

    assert intervals == expected_intervals


def test_get_intervals_sum_with_different_units():
    start = np.array([1, 8, 12, 21, 31, 40, 52, 60]) * u.s
    stop = np.array([6, 9, 17, 26, 36, 51, 53, 70]) * u.s
    thd_sum = 600 * u.s

    intervals = get_intervals_sum(start, stop, thd_sum)

    expected_intervals = [[1 * u.s, 70 * u.s]]

    assert intervals == expected_intervals


def test_get_intervals_sum_with_rounding():
    start = np.array([1, 8]) * u.s
    stop = np.array([6, 10.0001]) * u.s
    thd_sum = 7 * u.s

    intervals_1 = get_intervals_sum(start, stop, thd_sum, digit_res=5)
    expected_intervals_1 = [[1 * u.s, 6 * u.s], [8 * u.s, 10.0001 * u.s]]

    intervals_2 = get_intervals_sum(start, stop, thd_sum, digit_res=3)
    expected_intervals_2 = [[1 * u.s, 10.0001 * u.s]]

    assert intervals_1 == expected_intervals_1


def test_split_data_from_intervals():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    intervals = [[2, 5] * u.s, [7, 11] * u.s]
    start = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * u.s
    stop = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) * u.s

    split_data = split_data_from_intervals(data, intervals, start, stop)
    print(split_data)
    expected_split_data = [np.array([2, 3, 4]), np.array([7, 8, 9, 10])]

    assert len(split_data) == len(expected_split_data)
    for i in range(len(split_data)):
        assert np.array_equal(split_data[i], expected_split_data[i])
