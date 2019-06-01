"""Implementation of the h-measure for classification evaluation

"""

# Authors: Lyubomir Danov <->
# License: -


import numpy

from scipy.stats import beta as beta_dist
from scipy.special import beta as beta_func
from scipy.spatial import ConvexHull

from sklearn.metrics import roc_curve

def h_measure_fin(LH, pi0, pi1, B0, B1):
    return 1 - (LH/(pi0*B0 + pi1*B1))

def generate_B_coefs(a, b, n0, n1):
    pi1 = n1 / (n1 + n0)
    b10 = beta_func((1+a), b)
    b01 = beta_func(a, (1+b))
    b00 = beta_func(a, b)
    B0 = beta_dist.cdf(x=pi1, a=(1+a), b=b) * b10/b00
    B1 = ( beta_dist.cdf(x=1, a=a, b=(1+b)) - beta_dist.cdf(x=pi1, a=a, b=(1+b)) ) * b01/b00
    return B0, B1

def generate_LH(n0, n1, G0, G1, b0, b1, debug:bool = False):
    pi1 = n1 / (n1 + n0)
    pi0 = n0 / (n1 + n0)
    b0_head = b0[1:]
    b0_norm = b0[:-1]
    b1_head = b0[1:]
    b1_norm = b0[:-1]
    
    LH_array = ( pi0 * (1 - G0) * (b0_head - b0_norm) ) + ( pi1 * G1 * (b1_head - b1_norm))
    if debug:
        return LH_array
    return sum(LH_array)

def generate_cost(n0, n1, G0, G1):
    
    G1_head = G1[1:]
    G1_norm = G1[:-1]
    G0_head = G0[1:]
    G0_norm = G0[:-1]
    
    c1 = n1 / (n1 + n0) * (G1_head - G1_norm)
    c0 = n0 / (n1 + n0) * (G0_head - G0_norm)
    
    c_more = c1 / (c1 + c0)
    
    cost = numpy.concatenate([[0], c_more, [1]])
    
    return cost

def generate_b_vecs(cost, a, b):
    b10 = beta_func((1+a), b)
    b01 = beta_func(a, (1+b))
    b00 = beta_func(a, b)
    b0 = beta_dist.cdf(x=cost, a = (1 + a), b = b) * b10 / b00
    b1 = beta_dist.cdf(x=cost, a = a, b = (1 + b)) * b01 / b00
    return b0, b1

def generate_beta_params(n0, n1, sev_ratio):
    if sev_ratio is None:
        sev_ratio = n1 / n0
    if sev_ratio > 0:
        a = 2
        b = 1 + (a - 1) * 1/sev_ratio
    elif sev_ratio < 0:
        a = n0/(n0+n1)
        b = n1/(n0+n1)
    else:
        raise ValueError
    return a, b

def generate_chull_points(y_true, y_score, pos_label):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label)
    fnr = 1 - tpr
    
    pairwise_max = numpy.maximum(fnr, fpr)
    chull_candidates = numpy.array([fpr, pairwise_max]).transpose()
    
    assert chull_candidates.shape == (len(fnr), 2)
    hull = ConvexHull(chull_candidates)
    G0 = chull_candidates[hull.vertices, 0]
    G1 = chull_candidates[hull.vertices, 1]
    return G0, G1

def hmeasure_score(y_true:numpy.ndarray, y_score:numpy.ndarray, severity_ratio: float = None, pos_label = None) -> float:
    
    assert isinstance(y_true, numpy.ndarray)

    # TODO: rewrite as in R?
    G0, G1 = generate_chull_points(y_true=y_true, y_score=y_score, pos_label=pos_label)
    # G0 = numpy.array(G0_r)
    # G1 = numpy.array(G1_r)
    print(G0, G1)

    if pos_label is None:
        pos_label = 1.
    # make y_true a boolean vector
    y_true = (y_true == pos_label)
    
    n1 = sum(y_true)
    n0 = len(y_true) - n1

    a, b = generate_beta_params(n0=n0, n1=n1, sev_ratio=severity_ratio)
    cost = generate_cost(n0=n0, n1=n1, G0=G0, G1=G1)
    b0, b1 = generate_b_vecs(cost=cost, a=a, b=b)
    
    LH = generate_LH(n0=n0, n1=n1, G0=G0, G1=G1, b0=b0, b1=b1)
    B0, B1 = generate_B_coefs(a=a, b=b, n0=n0, n1=n1)
    
    hm_score = 1 - (LH / ((n0*B0 + n1*B1)/(n0+n1)))
    return hm_score