import numpy as np
import pytest
from syft.syft import FloatTensor, SyftController


# ------ Test settings ------
decimal_accuracy = 4  # tests will verify  abs(desired-actual) < 1.5 * 10**(-decimal)
verbosity = False


def pytest_namespace():
    return {
        'sc': None
    }


@pytest.yield_fixture(autouse=True)
def setup_controller():
    pytest.sc = SyftController()


def test_abs():
    data = np.array([-1., -2., 3., 4., 5., -6.])
    expected = np.array([1., 2., 3., 4., 5., 6.])
    a = pytest.sc.FloatTensor(data)
    b = a.abs()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_abs_():
    data = np.array([-1., -2., 3., 4., 5., -6.])
    expected = np.array([1., 2., 3., 4., 5., 6.])
    a = pytest.sc.FloatTensor(data)
    a.abs_()

    # a does change when inlined
    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_acos():
    data = np.array([-0.6366, 0.2718, 0.4469, 1.3122])
    expected = np.array([2.26087785, 1.2955333, 1.10749924, np.nan])
    a = pytest.sc.FloatTensor(data)
    b = a.acos()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_acos_():
    data = np.array([-1., -2., 3., 4., 5., -6.])
    expected = np.array([1., 2., 3., 4., 5., 6.])
    a = pytest.sc.FloatTensor(data)
    a.abs_()

    # a does change when inlined
    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_addcdiv():
    data = np.array([[-0.39069918,  0.18299954,  0.31636572],
                     [1.13772225, -0.3253836, -0.88367993]]).astype('float')
    t1 = np.array([[-0.59233409,  0.05522861, -2.57116127,
                    1.35875595, -0.87830114, 0.53922689]]).astype('float')
    t2 = np.array([[0.30240816], [0.48581797], [-0.61623448],
                   [-1.08655083], [-0.45116752], [-0.72556847]]).astype('float')
    value = 0.1
    expected = np.array([[-0.58657157,  0.19436771,  0.73360324],
                         [1.01267004, -0.13071065, -0.9579978]]).astype('float')
    a = pytest.sc.FloatTensor(data)
    numerator = pytest.sc.FloatTensor(t1)
    denominator = pytest.sc.FloatTensor(t2)
    b = a.addcdiv(value, numerator, denominator)

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_addcdiv_():
    data = np.array([[-0.39069918,  0.18299954,  0.31636572],
                     [1.13772225, -0.3253836, -0.88367993]]).astype('float')
    t1 = np.array([[-0.59233409,  0.05522861, -2.57116127,
                    1.35875595, -0.87830114, 0.53922689]]).astype('float')
    t2 = np.array([[0.30240816], [0.48581797], [-0.61623448],
                   [-1.08655083], [-0.45116752], [-0.72556847]]).astype('float')
    value = 0.1
    expected = np.array([[-0.58657157,  0.19436771,  0.73360324],
                         [1.01267004, -0.13071065, -0.9579978]]).astype('float')
    a = pytest.sc.FloatTensor(data)
    numerator = pytest.sc.FloatTensor(t1)
    denominator = pytest.sc.FloatTensor(t2)
    a.addcdiv_(value, numerator, denominator)

    # a does change when inlined
    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_addcmul():
    data = np.array([[-0.39069918,  0.18299954,  0.31636572],
                     [1.13772225, -0.3253836, -0.88367993]]).astype('float')
    t1 = np.array([[-0.59233409,  0.05522861, -2.57116127,
                    1.35875595, -0.87830114, 0.53922689]]).astype('float')
    t2 = np.array([[0.30240816], [0.48581797], [-0.61623448],
                   [-1.08655083], [-0.45116752], [-0.72556847]]).astype('float')
    value = 0.1
    expected = np.array([[-0.40861183,  0.18568264,  0.47480953],
                         [0.9900865, -0.28575751, -0.92280453]]).astype('float')
    a = pytest.sc.FloatTensor(data)
    tensor1 = pytest.sc.FloatTensor(t1)
    tensor2 = pytest.sc.FloatTensor(t2)
    b = a.addcmul(value, tensor1, tensor2)

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_addcmul_():
    data = np.array([[-0.39069918,  0.18299954,  0.31636572],
                     [1.13772225, -0.3253836, -0.88367993]]).astype('float')
    t1 = np.array([[-0.59233409,  0.05522861, -2.57116127,
                    1.35875595, -0.87830114, 0.53922689]]).astype('float')
    t2 = np.array([[0.30240816], [0.48581797], [-0.61623448],
                   [-1.08655083], [-0.45116752], [-0.72556847]]).astype('float')
    value = 0.1
    expected = np.array([[-0.40861183,  0.18568264,  0.47480953],
                         [0.9900865, -0.28575751, -0.92280453]]).astype('float')
    a = pytest.sc.FloatTensor(data)
    tensor1 = pytest.sc.FloatTensor(t1)
    tensor2 = pytest.sc.FloatTensor(t2)
    a.addcmul_(value, tensor1, tensor2)

    # a does change when inlined
    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_asin():
    data = np.array([-0.6366, 0.2718, 0.4469, 1.3122])
    expected = np.array([-0.69008148, 0.27526295, 0.46329704, np.nan])
    a = pytest.sc.FloatTensor(data)
    b = a.asin()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_asin_():
    data = np.array([-0.6366, 0.2718, 0.4469, 1.3122])
    expected = np.array([-0.69008148, 0.27526295, 0.46329704, np.nan])
    a = pytest.sc.FloatTensor(data)
    a.asin_()

    # a does change when inlined
    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_atan():
    data = np.array([-0.6366, 0.2718, 0.4469, 1.3122])
    expected = np.array([-0.56689745, 0.26538879, 0.42027298, 0.91960937])
    a = pytest.sc.FloatTensor(data)
    b = a.atan()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_atan_():
    data = np.array([-0.6366, 0.2718, 0.4469, 1.3122])
    expected = np.array([-0.56689745, 0.26538879, 0.42027298, 0.91960937])
    a = pytest.sc.FloatTensor(data)
    a.atan_()

    # a does change when inlined
    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_ceil():
    data = np.array([1.3869, 0.3912, -0.8634, -0.5468])
    expected = np.array([2., 1., -0., -0.])
    a = pytest.sc.FloatTensor(data)
    b = a.ceil()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_ceil_():
    data = np.array([1.3869, 0.3912, -0.8634, -0.5468])
    expected = np.array([2., 1., -0., -0.])
    a = pytest.sc.FloatTensor(data)
    a.ceil_()

    # a does change when inlined
    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_clamp():
    data = np.array([1.3869, 0.3912, -0.8634, -0.5468])
    a = pytest.sc.FloatTensor(data)

    b = a.clamp(min=-0.5, max=0.5)
    expected_b = np.array([0.5000, 0.3912, -0.5000, -0.5000])

    c = a.clamp(min=0.5)
    expected_c = np.array([1.3869, 0.5000, 0.5000, 0.5000])

    d = a.clamp(max=0.5)
    expected_d = np.array([0.5000, 0.3912, -0.8634, -0.5468])

    np.testing.assert_almost_equal(b.to_numpy(), expected_b,
                                   decimal=decimal_accuracy, verbose=verbosity)
    np.testing.assert_almost_equal(c.to_numpy(), expected_c,
                                   decimal=decimal_accuracy, verbose=verbosity)
    np.testing.assert_almost_equal(d.to_numpy(), expected_d,
                                   decimal=decimal_accuracy, verbose=verbosity)

    # a doesn't change
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_clamp_():
    data = np.array([1.3869, 0.3912, -0.8634, -0.5468])

    # inline case for min and max
    expected = np.array([0.5000, 0.3912, -0.5000, -0.5000])
    a = pytest.sc.FloatTensor(data)
    a.clamp_(min=-0.5, max=0.5)

    # a does change when inlined
    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)

    # inline case for min only
    expected = np.array([1.3869, 0.5000, 0.5000, 0.5000])
    a = pytest.sc.FloatTensor(data)
    a.clamp_(min=0.5)

    # a does change when inlined
    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)

    # inline case for max only
    expected = np.array([0.5000, 0.3912, -0.8634, -0.5468])
    a = pytest.sc.FloatTensor(data)
    a.clamp_(max=0.5)

    # a does change when inlined
    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_cos():
    data = np.array([-0.6366, 0.2718, 0.4469, 1.3122])
    expected = np.array([0.80412155, 0.9632892, 0.90179116, 0.25572386])
    a = pytest.sc.FloatTensor(data)
    b = a.cos()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_cos_():
    data = np.array([-0.6366, 0.2718, 0.4469, 1.3122])
    expected = np.array([0.80412155, 0.9632892, 0.90179116, 0.25572386])
    a = pytest.sc.FloatTensor(data)
    a.cos_()

    # a does change when inlined
    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_cosh():
    data = np.array([-0.6366, 0.2718, 0.4469, 1.3122])
    expected = np.array([1.209566, 1.03716552, 1.10153294, 1.99178159])
    a = pytest.sc.FloatTensor(data)
    b = a.cosh()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_cosh_():
    data = np.array([-0.6366, 0.2718, 0.4469, 1.3122])
    expected = np.array([1.209566, 1.03716552, 1.10153294, 1.99178159])
    a = pytest.sc.FloatTensor(data)
    a.cosh_()

    # a does change when inlined
    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_erf():
    data = np.array([0, -1., 10.])
    expected = np.array([0., -0.84270078, 1.])
    a = pytest.sc.FloatTensor(data)
    b = a.erf()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_erf_():
    data = np.array([0, -1., 10.])
    expected = np.array([0., -0.84270078, 1.])
    a = pytest.sc.FloatTensor(data)
    a.erf_()

    # a does change when inlined
    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_erfinv():
    data = np.array([0, 0.5, -1.])
    expected = np.array([0., 0.47693628, -np.inf])
    a = pytest.sc.FloatTensor(data)
    b = a.erfinv()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_erfinv_():
    data = np.array([0, 0.5, -1.])
    expected = np.array([0., 0.47693628, -np.inf])
    a = pytest.sc.FloatTensor(data)
    a.erfinv_()

    # a does change when inlined
    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_exp():
    data = np.array([0, np.log(2)])
    expected = np.array([1., 2.])
    a = pytest.sc.FloatTensor(data)
    b = a.exp()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_exp_():
    data = np.array([0, np.log(2)])
    expected = np.array([1., 2.])
    a = pytest.sc.FloatTensor(data)
    a.exp_()

    # a does change when inlined
    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_floor():
    data = np.array([1.3869, 0.3912, -0.8634, -0.5468])
    expected = np.array([1., 0., -1., -1.])
    a = pytest.sc.FloatTensor(data)
    b = a.floor()

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_floor_():
    data = np.array([1.3869, 0.3912, -0.8634, -0.5468])
    expected = np.array([1., 0., -1., -1.])
    a = pytest.sc.FloatTensor(data)
    a.floor_()

    # a does change when inlined
    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_fmod():
    data = np.array([-3, -2, -1, 1, 2, 3])
    expected = np.array([-1., -0., -1., 1., 0., 1.])
    a = pytest.sc.FloatTensor(data)
    b = a.fmod(2)

    np.testing.assert_almost_equal(b.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)
    # a doesn't change
    np.testing.assert_almost_equal(a.to_numpy(), data,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_fmod_():
    data = np.array([-3, -2, -1, 1, 2, 3])
    expected = np.array([-1., -0., -1., 1., 0., 1.])
    a = pytest.sc.FloatTensor(data)
    a.fmod_(2)

    # a does change when inlined
    np.testing.assert_almost_equal(a.to_numpy(), expected,
                                   decimal=decimal_accuracy, verbose=verbosity)


def test_exp():
    data = np.array([0., 1., 2., 5.])
    expected = np.array([1., 2.71828183, 7.3890561, 148.4131591])
    a = pytest.sc.FloatTensor(data)
    b = a.exp()

    np.testing.assert_almost_equal(expected, b.to_numpy(), decimal=4)
    # a doesn't change
    np.testing.assert_array_equal(data, a.to_numpy())


def test_exp_():
    data = np.array([0., 1., 2., 5.])
    expected = np.array([1., 2.71828183, 7.3890561, 148.4131591])
    a = pytest.sc.FloatTensor(data)
    a.exp_()

    # a does change when inlined
    np.testing.assert_almost_equal(expected, a.to_numpy(), decimal=4)


def test_log1p():
    data = np.array([1.2, -0.9, 9.9, 0.1, -0.455])
    expected = np.array([0.78845736, -2.30258509,  2.38876279,  0.09531018, -0.60696948])
    a = pytest.sc.FloatTensor(data)
    b = a.log1p()

    np.testing.assert_almost_equal(expected, b.to_numpy(), decimal=5)
    # a doesn't change
    np.testing.assert_array_equal(data, a.to_numpy())


def test_log1p_():
    data = np.array([1.2, -0.9, 9.9, 0.1, -0.455])
    expected = np.array([0.78845736, -2.30258509,  2.38876279,  0.09531018, -0.60696948])
    a = pytest.sc.FloatTensor(data)
    a.log1p_()

    # a does change when inlined
    np.testing.assert_almost_equal(expected, a.to_numpy(), decimal=5)
    

def test_reciprocal():
    data = np.array([1., 2., 3., 4.])
    expected = np.array([1., 0.5, 0.33333333, 0.25])
    a = pytest.sc.FloatTensor(data)
    b = a.reciprocal()

    np.testing.assert_almost_equal(expected, b.to_numpy(), decimal=4)
    # a doesn't change
    np.testing.assert_array_equal(data, a.to_numpy())

def test_reciprocal_():
    data = np.array([1., 2., 3., 4.])
    expected = np.array([1., 0.5, 0.33333333, 0.25])
    a = pytest.sc.FloatTensor(data)
    a.reciprocal_()

    # a does change when inlined
    np.testing.assert_almost_equal(expected, a.to_numpy(), decimal=4)


def test_round():
    data = np.array([12.7292, -3.11, 9.00, 20.4999, 20.5001])
    expected = np.array([13, -3, 9, 20, 21])
    a = pytest.sc.FloatTensor(data)
    b = a.round()

    np.testing.assert_array_equal(expected, b.to_numpy())
    # a doesn't change
    np.testing.assert_array_equal(data, a.to_numpy())


def test_round_():
    data = np.array([12.7292, -3.11, 9.00, 20.4999, 20.5001])
    expected = np.array([13, -3, 9, 20, 21])
    a = pytest.sc.FloatTensor(data)
    a.round_()

    # a does change when inlined
    np.testing.assert_array_equal(expected, a.to_numpy())

def test_rsqrt():
    data = np.array([1., 2., 3., 4.])
    expected = np.array([1., 0.7071068, 0.5773503, 0.5])
    a = pytest.sc.FloatTensor(data)
    b = a.rsqrt()

    np.testing.assert_almost_equal(expected, b.to_numpy(), decimal=4)
    # a doesn't change
    np.testing.assert_array_equal(data, a.to_numpy())


def test_rsqrt_():
    data = np.array([1., 2., 3., 4.])
    expected = np.array([1., 0.7071068, 0.5773503, 0.5])
    a = pytest.sc.FloatTensor(data)
    a.rsqrt_()

    # a does change when inlined
    np.testing.assert_almost_equal(expected, a.to_numpy(), decimal=4)

def test_sqrt():
    data = np.array([1., 2., 3., 4.])
    expected = np.array([1., 1.41421356, 1.73205081, 2.])
    a = pytest.sc.FloatTensor(data)
    b = a.sqrt()

    np.testing.assert_almost_equal(expected, b.to_numpy(), decimal=4)
    # a doesn't change
    np.testing.assert_array_equal(data, a.to_numpy())


def test_sqrt_():
    data = np.array([1., 2., 3., 4.])
    expected = np.array([1., 1.41421356, 1.73205081, 2.])
    a = pytest.sc.FloatTensor(data)
    a.sqrt_()

    # a does change when inlined
    np.testing.assert_almost_equal(expected, a.to_numpy(), decimal=4)

def test_trace():
    data = np.random.randn(17, 17).astype('float')
    expected = data.trace()

    a = pytest.sc.FloatTensor(data)
    actual = a.trace()
    np.testing.assert_almost_equal(actual, expected,
                                   decimal=decimal_accuracy, verbose=verbosity)

    a = a.gpu()
    actual = a.trace()
    np.testing.assert_almost_equal(actual, expected,
                                   decimal=decimal_accuracy, verbose=verbosity)

