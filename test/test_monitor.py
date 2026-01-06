from risk_system.monitor import psi

import numpy as np


def test_psi_zero_when_identical():
    p = np.array([0.2, 0.5, 0.3])
    q = np.array([0.2, 0.5, 0.3])
    assert psi(p, q) < 1e-6

def test_psi_shifted():
    p = np.array([0.2, 0.5, 0.3])
    q = np.array([0.4, 0.4, 0.2])
    assert psi(p, q) > 1e-6
