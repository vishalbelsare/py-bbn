import numpy as np
from numpy.random import normal
from nose import with_setup
from nose.tools import assert_almost_equal

from pybbn.lg.inference import MvnGaussian


def setup():
    """
    Setup.
    :return: None.
    """
    np.random.seed(37)


def teardown():
    """
    Teardown.
    :return: None.
    """
    pass


@with_setup(setup, teardown)
def test_serial_inference():
    """
    Tests inference on a serial structure X -> Y -> Z.
    :return: None.
    """
    N = 10000
    x0 = normal(0, 1, N)
    x1 = normal(1 + 2 * x0, 1, N)
    x2 = normal(1 + 2 * x1, 1, N)

    X = np.hstack([x0.reshape(-1, 1), x1.reshape(-1, 1), x2.reshape(-1, 1)])
    M = np.mean(X, axis=0)
    S = np.cov(X.T)

    mvn = MvnGaussian(M, S, N)
    mvn.update_mean_cov(np.array([2.0]), [1])
    M_u, S_u = mvn.get_params()

    M_e = np.array([0.40296894, 2.0, 4.98759174])
    assert(M.shape[0] == M_e.shape[0])
    for m, m_u in zip(M_e, M_u):
        assert_almost_equal(m, m_u, delta=0.1)

    S_e = np.array([[1.98683996e-01, 1.96882311e+00, -7.51882725e-03],
                    [1.96882311e+00, 1.00000000e-02, 9.73351352e+00],
                    [-7.51882725e-03, 9.73351352e+00, 1.01390146e+00]])
    assert (S_e.shape[0] == S_u.shape[0])
    assert (S_e.shape[1] == S_u.shape[1])

    rows, cols = S_e.shape
    for r in range(rows):
        for c in range(cols):
            assert_almost_equal(S_e[r, c], S_u[r, c], delta=0.5)

    C_o = mvn.get_corr()
    C_e = np.array([[1., -0.02477834, 0.77462271],
                    [-0.02477834, 1., 0.05241713],
                    [0.77462271, 0.05241713, 1.]])

    assert (C_e.shape[0] == C_o.shape[0])
    assert (C_e.shape[1] == C_o.shape[1])

    rows, cols = C_e.shape
    for r in range(rows):
        for c in range(cols):
            assert_almost_equal(C_e[r, c], C_o[r, c], delta=0.5)

    samples = mvn.get_samples()
    assert(samples.shape[0] == N)
    assert(samples.shape[1] == 3)

    mvn.clear()

    M_u, S_u = mvn.get_params()
    assert (M.shape[0] == M_u.shape[0])
    for m, m_u in zip(M, M_u):
        assert_almost_equal(m, m_u, delta=0.0001)

    assert (S.shape[0] == S_u.shape[0])
    assert (S.shape[1] == S_u.shape[1])

    rows, cols = S.shape
    for r in range(rows):
        for c in range(cols):
            assert_almost_equal(S[r, c], S_u[r, c], delta=0.0001)

    mvn.update_mean_cov(None, None)
    M_u, S_u = mvn.get_params()
    assert (M.shape[0] == M_u.shape[0])
    for m, m_u in zip(M, M_u):
        assert_almost_equal(m, m_u, delta=0.0001)

    assert (S.shape[0] == S_u.shape[0])
    assert (S.shape[1] == S_u.shape[1])

    rows, cols = S.shape
    for r in range(rows):
        for c in range(cols):
            assert_almost_equal(S[r, c], S_u[r, c], delta=0.0001)