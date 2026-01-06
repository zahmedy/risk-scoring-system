from risk_system.monitor import (psi, histogram_proportions, 
                                 categorical_psi)

import numpy as np


def test_psi_zero_when_identical():
    p = np.array([0.2, 0.5, 0.3])
    q = np.array([0.2, 0.5, 0.3])
    assert psi(p, q) < 1e-6

def test_psi_shifted():
    p = np.array([0.2, 0.5, 0.3])
    q = np.array([0.4, 0.4, 0.2])
    assert psi(p, q) > 1e-6

def test_histogram_proportions_sum_to_one():
    x = np.array([10, 20, 40, 50 , 70])
    edges = np.array([0, 30, 60, 90])
    p = histogram_proportions(x, edges)
    assert abs(p.sum() - 1.0) < 1e-6

def test_categorical_psi_zero_when_same_distribution():
    baseline = {"a": 0.5, "b": 0.5}
    current = {"a": 0.5, "b": 0.5}
    assert categorical_psi(baseline, current) < 1e-6

def test_categorical_psi_positive_when_distribution_changes():
    baseline = {"a": 0.9, "b": 0.1}
    current = {"a": 0.1, "b": 0.9}
    assert categorical_psi(baseline, current) > 0.1
