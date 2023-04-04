import numpy as np
import pytest
from trets.utils import get_intervals,fraction_outside_interval

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