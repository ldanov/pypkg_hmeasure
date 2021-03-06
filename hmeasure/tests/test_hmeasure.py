import numpy
import os
import pytest

from sklearn.metrics import roc_curve
from ..hsource import _generate_convex_hull_points
from ..hsource import _generate_beta_params
from ..hsource import _generate_cost
from ..hsource import _generate_b_vecs
from ..hsource import _generate_LH_coef
from ..hsource import _generate_B_coefs
from ..hsource import _generate_h_measure
from ..hsource import _transform_roc_to_invF
from ..hsource import h_score
from .utils import get_case_data_dict, drop_rep_zeros

_cases_to_run = [
    "hmeasure_r_case_0.json",
    "hmeasure_r_case_1.json",
    "hmeasure_r_case_2.json",
    "hmeasure_r_case_3.json",
    "hmeasure_r_case_4.json",
    "hmeasure_r_case_5.json",
    "hmeasure_r_case_6.json"
]


@pytest.fixture(params=_cases_to_run)
def case_data(request):
    loc = os.path.dirname(os.path.abspath(__file__))
    path_full = os.path.join(loc, 'resources', request.param)

    case_data = get_case_data_dict(path_full)

    return case_data


def test__generate_beta_params(case_data):
    data = case_data
    n0 = data['n0']
    n1 = data['n1']
    sev_ratio = data['sev_ratio']

    exp_a = data['a']
    exp_b = data['b']
    a, b = _generate_beta_params(n0, n1, sev_ratio=sev_ratio)

    assert numpy.isclose(a, exp_a)
    assert numpy.isclose(b, exp_b)


def test__generate_beta_params_fail():
    with pytest.raises(ValueError):
        _, _ = _generate_beta_params(1, 1, sev_ratio=0)


def test__generate_cost(case_data):
    n0 = case_data['n0']
    n1 = case_data['n1']
    G0 = case_data['G0']
    G1 = case_data['G1']

    exp_cost = case_data['cost']

    cost = _generate_cost(n0, n1, G0, G1)
    assert numpy.allclose(cost, exp_cost)


def test__generate_b_vecs(case_data):
    cost = case_data['cost']
    a = case_data['a']
    b = case_data['b']

    exp_b0 = case_data['b0']
    exp_b1 = case_data['b1']
    b0, b1 = _generate_b_vecs(cost, a, b)

    assert numpy.allclose(exp_b0, b0)
    assert numpy.allclose(exp_b1, b1)


def test__generate_LH_coef(case_data):
    n0 = case_data['n0']
    n1 = case_data['n1']
    G0 = case_data['G0']
    G1 = case_data['G1']
    b0 = case_data['b0']
    b1 = case_data['b1']

    exp_LH = case_data['LH']
    LH = _generate_LH_coef(n0, n1, G0, G1, b0, b1)

    assert numpy.isclose(LH, exp_LH)


def test__generate_B_coefs(case_data):
    a = case_data['a']
    b = case_data['b']
    n0 = case_data['n0']
    n1 = case_data['n1']

    exp_B0 = case_data['B0']
    exp_B1 = case_data['B1']
    B0, B1 = _generate_B_coefs(a, b, n0, n1)

    assert numpy.isclose(B0, exp_B0)
    assert numpy.isclose(B1, exp_B1)


def test__generate_h_measure(case_data):
    n0 = case_data['n0']
    n1 = case_data['n1']
    B0 = case_data['B0']
    B1 = case_data['B1']
    LH = case_data['LH']

    exp_H = case_data['H']
    H = _generate_h_measure(n0=n0, n1=n1, B0=B0, B1=B1, LH=LH)

    assert numpy.isclose(exp_H, H)

def test__generate_h_measure_negative():
    n0 = 1
    n1 = 1
    B0 = 0.1
    B1 = 0.1
    LH = 0.1000000001

    exp_H = 0
    H0 = _generate_h_measure(n0=n0, n1=n1, B0=B0, B1=B1, LH=LH, fix_prec=False)
    H1 = _generate_h_measure(n0=n0, n1=n1, B0=B0, B1=B1, LH=LH, fix_prec=True)

    # test that H0 is a very small negative value
    assert exp_H > H0 and numpy.isclose(exp_H, H0)
    assert exp_H == H1


def test__generate_convex_hull_points(case_data):
    invF0 = case_data['invF0']
    invF1 = case_data['invF1']

    exp_G0 = case_data['G0']
    exp_G1 = case_data['G1']
    G0, G1 = _generate_convex_hull_points(invF0, invF1)

    assert numpy.allclose(G0, exp_G0)
    assert numpy.allclose(G1, exp_G1)

def test__generate_convex_hull_points_exception():
    invF0 = numpy.array([0,0.1,0.2,0.3,0.4,1])
    invF1 = numpy.array([0,0.01,0.02,0.03,0.04,1])

    exp_G0 = numpy.array([0, 1])
    exp_G1 = numpy.array([0, 1])
    G0, G1 = _generate_convex_hull_points(invF0, invF1)

    assert numpy.allclose(G0, exp_G0)
    assert numpy.allclose(G1, exp_G1)

def test__transform_roc_to_invF(case_data):
    y_true = case_data['y_true']
    y_pred = case_data['y_pred']
    exp_invF0 = case_data['invF0']
    exp_invF1 = case_data['invF1']
    fpr_untr, tpr_untr, _ = roc_curve(y_true, y_pred, drop_intermediate=False)
    fpr, tpr = _transform_roc_to_invF(fpr_untr, tpr_untr)

    # drop repeated 0s
    fpr = drop_rep_zeros(fpr)
    tpr = drop_rep_zeros(tpr)
    exp_invF0 = drop_rep_zeros(exp_invF0)
    exp_invF1 = drop_rep_zeros(exp_invF1)

    assert numpy.allclose(exp_invF0, fpr)
    assert numpy.allclose(exp_invF1, tpr)


def test_h_score(case_data):
    y_true = case_data['y_true']
    y_pred = case_data['y_pred']
    sev_r = case_data['sev_ratio']

    exp_h_sc = case_data['H']
    h_sc = h_score(y_true=y_true, y_score=y_pred, severity_ratio=sev_r)

    assert numpy.isclose(h_sc, exp_h_sc)
