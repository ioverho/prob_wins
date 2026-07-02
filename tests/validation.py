import numpy as np
import pytest

from prob_wins.utils.validation import validate_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate(results, baseline):
    return validate_results(results, baseline)


# ---------------------------------------------------------------------------
# Happy path: valid inputs are accepted and returned as arrays
# ---------------------------------------------------------------------------

class TestValidInputs:
    def test_numpy_float_arrays(self):
        r, b = validate(np.array([1.0, 2.0]), np.array([0.5, 1.5]))
        assert isinstance(r, np.ndarray)
        assert isinstance(b, np.ndarray)

    def test_numpy_int_arrays(self):
        r, b = validate(np.array([1, 2, 3]), np.array([3, 2, 1]))
        assert isinstance(r, np.ndarray)
        assert isinstance(b, np.ndarray)

    def test_python_lists_coerced(self):
        r, b = validate([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        assert isinstance(r, np.ndarray)
        assert isinstance(b, np.ndarray)

    def test_python_tuples_coerced(self):
        r, b = validate((1.0, 2.0), (2.0, 1.0))
        assert isinstance(r, np.ndarray)
        assert isinstance(b, np.ndarray)

    def test_single_element_arrays(self):
        r, b = validate([1.0], [0.0])
        assert r.shape == (1,)
        assert b.shape == (1,)

    def test_values_unchanged(self):
        left  = [1.0, 2.0, 3.0]
        right = [3.0, 2.0, 1.0]
        r, b = validate(left, right)
        np.testing.assert_array_equal(r, left)
        np.testing.assert_array_equal(b, right)

    def test_negative_values_allowed(self):
        validate([-1.0, -2.0], [-2.0, -1.0])

    def test_zero_values_allowed(self):
        validate([0.0, 0.0], [0.0, 0.0])


# ---------------------------------------------------------------------------
# Type errors
# ---------------------------------------------------------------------------

class TestTypeErrors:
    def test_string_results_raises_type_error(self):
        with pytest.raises(TypeError):
            validate(["a", "b"], [1.0, 2.0])

    def test_string_baseline_raises_type_error(self):
        with pytest.raises(TypeError):
            validate([1.0, 2.0], ["a", "b"])

    def test_none_results_raises_type_error(self):
        with pytest.raises((TypeError, ValueError)):
            validate(None, [1.0, 2.0])

    def test_none_baseline_raises_type_error(self):
        with pytest.raises((TypeError, ValueError)):
            validate([1.0, 2.0], None)

    def test_mixed_str_numeric_results_raises(self):
        with pytest.raises((TypeError, ValueError)):
            validate([1.0, "b"], [1.0, 2.0])


# ---------------------------------------------------------------------------
# Shape errors
# ---------------------------------------------------------------------------

class TestShapeErrors:
    def test_empty_results_raises_value_error(self):
        with pytest.raises(ValueError):
            validate([], [])

    def test_empty_results_nonempty_baseline_raises(self):
        with pytest.raises(ValueError):
            validate([], [1.0])

    def test_mismatched_lengths_raises_value_error(self):
        with pytest.raises(ValueError):
            validate([1.0, 2.0], [1.0])

    def test_mismatched_lengths_raises_value_error_reversed(self):
        with pytest.raises(ValueError):
            validate([1.0], [1.0, 2.0])

    def test_2d_results_raises_value_error(self):
        with pytest.raises(ValueError):
            validate([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0])

    def test_2d_baseline_raises_value_error(self):
        with pytest.raises(ValueError):
            validate([1.0, 2.0], [[1.0, 2.0], [3.0, 4.0]])

    def test_0d_results_raises_value_error(self):
        with pytest.raises(ValueError):
            validate(np.array(1.0), np.array([1.0]))

    def test_0d_baseline_raises_value_error(self):
        with pytest.raises(ValueError):
            validate(np.array([1.0]), np.array(1.0))


# ---------------------------------------------------------------------------
# Non-finite value errors
# ---------------------------------------------------------------------------

class TestNonFiniteErrors:
    def test_nan_in_results_raises_value_error(self):
        with pytest.raises(ValueError):
            validate([1.0, float("nan")], [1.0, 2.0])

    def test_nan_in_baseline_raises_value_error(self):
        with pytest.raises(ValueError):
            validate([1.0, 2.0], [1.0, float("nan")])

    def test_inf_in_results_raises_value_error(self):
        with pytest.raises(ValueError):
            validate([1.0, float("inf")], [1.0, 2.0])

    def test_neg_inf_in_results_raises_value_error(self):
        with pytest.raises(ValueError):
            validate([1.0, float("-inf")], [1.0, 2.0])

    def test_inf_in_baseline_raises_value_error(self):
        with pytest.raises(ValueError):
            validate([1.0, 2.0], [1.0, float("inf")])

    def test_all_nan_raises_value_error(self):
        with pytest.raises(ValueError):
            validate([float("nan")] * 3, [float("nan")] * 3)


# ---------------------------------------------------------------------------
# Integer arrays are exempt from the NaN/inf check (no NaN in integer dtype)
# ---------------------------------------------------------------------------

class TestIntegerArrays:
    def test_integer_arrays_bypass_finite_check(self):
        # np.isfinite is not checked for integer dtypes (they can't be NaN/inf)
        r, b = validate(np.array([1, 2, 3]), np.array([3, 2, 1]))
        assert r.dtype == np.array([1, 2, 3]).dtype

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint8])
    def test_various_integer_dtypes_accepted(self, dtype):
        validate(np.array([1, 2], dtype=dtype), np.array([2, 1], dtype=dtype))
