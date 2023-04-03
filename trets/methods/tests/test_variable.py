import numpy as np
import pytest
from trets.utils import get_intervals

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