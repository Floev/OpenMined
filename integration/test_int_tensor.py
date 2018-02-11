import numpy as np
import pytest
from syft import IntTensor
import syft.controller

# ------ Test settings ------
decimal_accuracy = 4  # tests will verify  abs(desired-actual) < 1.5 * 10**(-decimal)
verbosity = False


#
# IntTensor tests
#

def test_int_abs():
    data = np.array([-1, -2, 3, 4, 5, -6])
    expected = np.array([1, 2, 3, 4, 5, 6])
    a = IntTensor(data)
    b = a.abs()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change (non-inline)
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)

def test_int_abs_():
    data = np.array([-1, -2, 3, 4, 5, -6])
    expected = np.array([1, 2, 3, 4, 5, 6])
    a = IntTensor(data)
    a.abs_()

    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)

def test_int_cos():
    data = np.array([30, 60, 90, 180])
    expected = np.array([.1542515, -0.952413, -0.4480736, -0.5984601])
    a = IntTensor(data)
    b = a.cos()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change (non-inline)
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)
    
def test_int_acos():
    data = np.array([-1, 0, 1, 1, 2])
    expected = np.array([3.14159265, 1.57079633, 0, 0, np.nan])
    a = IntTensor(data)
    b = a.acos()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)

def test_int_lt():
    data = np.array([1,2,3,4])
    compare_data = np.array([2,2,5,1])
    tensor = IntTensor(data)
    compare_to = IntTensor(compare_data)
    expected = np.array([1,0,1,0])

    res = tensor.lt(compare_to)

    np.testing.assert_almost_equal(res.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    np.testing.assert_almost_equal(tensor.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)

def test_int_lt_():
    data = np.array([1,2,3,4])
    compare_data = np.array([2,2,5,1])
    tensor = IntTensor(data)
    compare_to = IntTensor(compare_data)
    expected = np.array([1,0,1,0])

    tensor.lt_(compare_to)

    np.testing.assert_almost_equal(tensor.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_int_equal():
    data = np.array([1,2,3,4])
    compare_not_eq = np.array([2,2,5,1])
    compare_eq = np.array([1,2,3,4])

    tensor = IntTensor(data)
    not_eq_tensor = IntTensor(compare_not_eq)
    eq_tensor = IntTensor(compare_eq)

    assert(tensor.equal(not_eq_tensor) is False)
    assert(tensor.equal(eq_tensor))


def test_int_neg():
    data = np.array([-1, -2, 3, 4, 5, -6])
    expected = np.array([1, 2, -3, -4, -5, 6])
    a = IntTensor(data)
    b = a.neg()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change (non-inline)
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)

def test_int_neg_():
    data = np.array([-1, -2, 3, 4, 5, -6])
    expected = np.array([1, 2, -3, -4, -5, 6])
    a = IntTensor(data)
    a.neg_()

    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)\


def test_int_shape():
    a_data = np.array([1,2,3,4])
    a_expected = [4]
    a = IntTensor(a_data)

    b_data = np.random.rand(3,2)
    b_expected = [3,2]
    b = IntTensor(b_data)

    assert(a.shape() == a_expected)
    assert(b.shape() == b_expected)

def test_int_sin():
    data = np.array([15, 60, 90, 180])
    expected = np.array([0.65028784, -0.30481062, 0.89399666, -0.80115264])
    a = IntTensor(data)
    b = a.sin()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)

def test_int_sqrt():
    data = np.random.rand(3,2)
    expected = np.sqrt(data)
    a = IntTensor(data)
    b = a.sqrt()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change (non-inline)
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_int_reciprocal():
    data = np.random.rand(3,2)
    expected = np.reciprocal(data)
    a = IntTensor(data)
    b = a.reciprocal()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change (non-inline)
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)

def test_int_trace():
    data = np.random.rand(5,5)
    expected = data.trace()
    a = IntTensor(data)
    b = a.trace()

    np.testing.assert_almost_equal(b, expected,
                                decimal=decimal_accuracy, verbose=verbosity)

    # a doesn't change (non-inline)
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)

def test_int_max_():
    data = np.array([4,0,6,-3,8,-2])
    compare_data = np.array([1,-2,2,-3,0,-1])
    tensor = IntTensor(data)
    compare_to = IntTensor(compare_data)
    expected = np.array([4,0,6,-3,8,-1])

    tensor.max_(compare_to)

    np.testing.assert_almost_equal(tensor.to_numpy(), expected,

def test_int_topk():
    data = np.array([[[3, 2], [12, 4]], [[6, 44], [43, 2]]])
    a = IntTensor(data)
    result = a.topk(1)
    expected = np.array([[[3], [12]], [[43], [44]]])

    np.testing.assert_almost_equal(result.to_numpy(), expected,
                                decimal=decimal_accuracy, verbose=verbosity)

    # a doesn't change (non-inline)
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)
