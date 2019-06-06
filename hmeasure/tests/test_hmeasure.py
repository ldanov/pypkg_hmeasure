import numpy
import pytest

from sklearn.metrics import roc_auc_score, roc_curve
from hmeasure.hmeasure import generate_convex_hull_points
from hmeasure.hmeasure import generate_beta_params
from hmeasure.hmeasure import generate_cost
from hmeasure.hmeasure import generate_b_vecs
from hmeasure.hmeasure import generate_LH_coef
from hmeasure.hmeasure import generate_B_coefs
from hmeasure.hmeasure import generate_h_measure
from hmeasure.hmeasure import transform_roc_to_invF
from hmeasure.hmeasure import h_score
from .testing_utils import *

def test_generate_beta_params():
    n0, n1 = get_n_case1() 
    exp_t1_a, exp_t1_b = get_shapes_case1()
    
    t1_a, t1_b = generate_beta_params(n0, n1, sev_ratio=1)
    assert t1_a == exp_t1_a
    assert t1_b == exp_t1_b

    t2_a, t2_b = generate_beta_params(n0, n1, sev_ratio=None)
    assert t2_a == 2
    assert t2_b == 1 + (n0/n1)

    t3_a, t3_b = generate_beta_params(n0, n1, sev_ratio=-1)
    assert t3_a == (n1/(n1+n0)) + 1
    assert t3_b == (n0/(n1+n0)) + 1

    with pytest.raises(ValueError):
        _, _ = generate_beta_params(n0, n1, sev_ratio=0)

def test_generate_cost():
    n0, n1 = get_n_case1() 
    G0, G1 = get_G_case1()
    exp_cost = get_cost_case1()

    cost = generate_cost(n0, n1, G0, G1)
    assert numpy.allclose(cost, exp_cost)

def test_generate_b_vecs():
    cost = get_cost_case1()
    a, b = get_shapes_case1()
    exp_b0, exp_b1 = get_b_vecs_case1()

    b0, b1 = generate_b_vecs(cost, a, b)

    assert numpy.allclose(exp_b0, b0)
    assert numpy.allclose(exp_b1, b1)

def test_generate_LH_coef():
    n0, n1 = get_n_case1()
    G0, G1 = get_G_case1()
    b0, b1 = get_b_vecs_case1()
    exp_LH = get_LH_case1()

    LH = generate_LH_coef(n0, n1, G0, G1, b0, b1)
    assert numpy.isclose(LH, exp_LH)

def test_generate_B_coefs():
    a, b = get_shapes_case1()
    n0, n1 = get_n_case1()
    exp_B0, exp_B1 = get_B_coefs_case1()

    B0, B1 = generate_B_coefs(a, b, n0, n1)
    assert numpy.isclose(B0, exp_B0)
    assert numpy.isclose(B1, exp_B1)

def test_generate_h_measure():
    n0, n1 = get_n_case1()
    B0, B1 = get_B_coefs_case1()
    LH = get_LH_case1()
    exp_H = get_H_case1()

    H = generate_h_measure(n0=n0, n1=n1, B0=B0, B1=B1, LH=LH)
    assert numpy.isclose(exp_H, H)

def test_generate_convex_hull_points():

    exp_G0, exp_G1 = get_G_case1()
    invF0, invF1 = get_invF_values_case1()
    assert isinstance(invF0, numpy.ndarray)
    assert isinstance(invF1, numpy.ndarray)
    
    G0, G1 = generate_convex_hull_points(invF0, invF1)
    assert numpy.allclose(G0, exp_G0)
    assert numpy.allclose(G1, exp_G1)

def test_transform_roc_to_invF():
    exp_invF0, exp_invF1 = get_invF_values_case1()
    y_true, y_pred = get_model_data_case1()
    fpr_untr, tpr_untr, _ = roc_curve(y_true, y_pred, drop_intermediate=False)
    fpr, tpr = transform_roc_to_invF(fpr_untr, tpr_untr)

    assert numpy.allclose(exp_invF0, fpr)
    assert numpy.allclose(exp_invF1, tpr)

def test_h_score():
    y_true, y_pred = get_model_data_case1()
    sev_r = 1
    exp_h = get_H_case1()
    h = h_score(y_true=y_true, y_score=y_pred, severity_ratio=sev_r)
    assert numpy.isclose(h, exp_h)

def test_case1():
    n0, n1 = get_n_case1() 

    y_true, y_pred = get_model_data_case1()
    fpr_untr, tpr_untr, _ = roc_curve(y_true, y_pred, drop_intermediate=False)

    exp_invF0, exp_invF1 = get_invF_values_case1()
    invF0, invF1 = transform_roc_to_invF(fpr_untr, tpr_untr)
    assert numpy.allclose(exp_invF0, exp_invF0)
    assert numpy.allclose(exp_invF1, exp_invF1)

    exp_G0, exp_G1 = get_G_case1()        
    G0, G1 = generate_convex_hull_points(invF0, invF1)
    assert numpy.allclose(G0, exp_G0)
    assert numpy.allclose(G1, exp_G1)
    
    exp_a, exp_b = get_shapes_case1()
    a, b = generate_beta_params(n0, n1, sev_ratio=1)
    assert a == exp_a
    assert b == exp_b

    exp_cost = get_cost_case1()
    cost = generate_cost(n0, n1, G0, G1)
    assert numpy.allclose(cost, exp_cost)

    exp_b0, exp_b1 = get_b_vecs_case1()
    b0, b1 = generate_b_vecs(cost, a, b)
    assert numpy.allclose(exp_b0, b0)
    assert numpy.allclose(exp_b1, b1)

    exp_LH = get_LH_case1()
    LH = generate_LH_coef(n0, n1, G0, G1, b0, b1)
    assert numpy.isclose(LH, exp_LH)

    exp_B0, exp_B1 = get_B_coefs_case1()
    B0, B1 = generate_B_coefs(a, b, n0, n1)
    assert numpy.isclose(B0, exp_B0)
    assert numpy.isclose(B1, exp_B1)

    exp_H = get_H_case1()
    H = generate_h_measure(n0=n0, n1=n1, B0=B0, B1=B1, LH=LH)
    assert numpy.isclose(exp_H, H)

    hsc = h_score(y_true, y_pred, severity_ratio=1)
    assert numpy.isclose(exp_H, hsc)