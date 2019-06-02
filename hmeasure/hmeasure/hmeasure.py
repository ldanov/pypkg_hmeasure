#!/usr/bin/env python3

"""Implementation of the h-measure for classification evaluation

"""

# Authors: Lyubomir Danov <->
# License: -


import numpy

from scipy.stats import beta as beta_dist
from scipy.special import beta as beta_func
from scipy.spatial import ConvexHull

from sklearn.metrics import roc_curve

def generate_B_coefs(a, b, n0, n1):
    pi1 = n1 / (n1 + n0)
    b10 = beta_func((1+a), b)
    b01 = beta_func(a, (1+b))
    b00 = beta_func(a, b)
    B0 = beta_dist.cdf(x=pi1, a=(1+a), b=b) * b10/b00
    B1 = ( beta_dist.cdf(x=1, a=a, b=(1+b)) - beta_dist.cdf(x=pi1, a=a, b=(1+b)) ) * b01/b00
    return B0, B1

def generate_LH_coef(n0, n1, G0, G1, b0, b1, debug:bool = False):
    pi1 = n1 / (n1 + n0)
    pi0 = n0 / (n1 + n0)
    b0_head = b0[1:]
    b0_norm = b0[:-1]
    b1_head = b1[1:]
    b1_norm = b1[:-1]
    
    LH_array = ( pi0 * (1 - G0) * (b0_head - b0_norm) ) + ( pi1 * G1 * (b1_head - b1_norm))
    if debug:
        return LH_array
    return sum(LH_array)

def generate_b_vecs(cost, a, b):
    b10 = beta_func((1+a), b)
    b01 = beta_func(a, (1+b))
    b00 = beta_func(a, b)
    b0 = beta_dist.cdf(x=cost, a = (1 + a), b = b) * b10 / b00
    b1 = beta_dist.cdf(x=cost, a = a, b = (1 + b)) * b01 / b00
    return b0, b1

def generate_cost(n0, n1, G0: numpy.ndarray, G1: numpy.ndarray):
    
    G1_head = G1[1:]
    G1_norm = G1[:-1]
    G0_head = G0[1:]
    G0_norm = G0[:-1]
    
    c1 = n1 / (n1 + n0) * (G1_head - G1_norm)
    c0 = n0 / (n1 + n0) * (G0_head - G0_norm)
    
    c_more = c1 / (c1 + c0)
    
    cost = numpy.concatenate([[0], c_more, [1]])
    
    return cost

def generate_beta_params(n0, n1, sev_ratio):
    if sev_ratio is None:
        sr = n1 / n0
    else:
        sr = sev_ratio
    if sr > 0:
        a = 2
        b = 1 + (a - 1) * 1/sr
    elif sr < 0:
        a = n1/(n0+n1) + 1
        b = n0/(n0+n1) + 1
    else:
        raise ValueError
    return a, b

def generate_convex_hull_points(invF0:numpy.ndarray, invF1:numpy.ndarray):
    pair_max = numpy.maximum(invF0, invF1)
    chull_cand = numpy.array(list(zip(invF0, pair_max)))
    
    hull = ConvexHull(chull_cand)
    G0 = numpy.sort(invF0[hull.vertices])
    G1 = numpy.sort(invF1[hull.vertices])
    return G0, G1

def transform_roc_to_invF(fpr:numpy.ndarray, tpr:numpy.ndarray):
    fpr = -numpy.sort(-fpr)
    tpr = -numpy.sort(-tpr)
    return fpr, tpr

def h_score(y_true:numpy.ndarray, y_score:numpy.ndarray, severity_ratio: float = None, pos_label = None) -> float:
    
    assert isinstance(y_true, numpy.ndarray)
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label)

    fpr, tpr = transform_roc_to_invF(fpr, tpr)
    
    G0, G1 = generate_convex_hull_points(fpr, tpr)
    
    if pos_label is None:
        pos_label = 1.
    # make y_true a boolean vector
    y_true = (y_true == pos_label)
    
    n1 = sum(y_true)
    n0 = len(y_true) - n1

    a, b = generate_beta_params(n0=n0, n1=n1, sev_ratio=severity_ratio)
    cost = generate_cost(n0=n0, n1=n1, G0=G0, G1=G1)
    b0, b1 = generate_b_vecs(cost=cost, a=a, b=b)
    
    LH = generate_LH_coef(n0=n0, n1=n1, G0=G0, G1=G1, b0=b0, b1=b1)
    B0, B1 = generate_B_coefs(a=a, b=b, n0=n0, n1=n1)
    
    hm_score = generate_h_measure(n0=n0, n1=n1, B0=B0, B1=B1, LH=LH)
    return hm_score

def generate_h_measure(n0, n1, B0, B1, LH):
    return 1 - (LH / ((n0*B0 + n1*B1)/(n0+n1)))