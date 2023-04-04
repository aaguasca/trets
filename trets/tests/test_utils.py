import numpy as np
import pytest
from trets.utils import get_intervals, fraction_outside_interval, weighted_average_error_calculation

@pytest.fixture
def data():
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

@pytest.mark.parametrize("n, expected_output", [(2, np.array([1, 3, 5, 7, 9, 10])), (3, np.array([1, 4, 7, 10])), (4, np.array([1, 5, 9, 10]))])
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