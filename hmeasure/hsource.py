"""Implementation of the h-measure for classification evaluation

"""

# Authors: Lyubomir Danov <->
# License: -


import numpy

from scipy.stats import beta as beta_dist
from scipy.special import beta as beta_func
from scipy.spatial import ConvexHull

from sklearn.metrics import roc_curve


def _generate_B_coefs(a, b, n0, n1):

    pi1 = n1 / (n1 + n0)
    b10 = beta_func((1+a), b)
    b01 = beta_func(a, (1+b))
    b00 = beta_func(a, b)

    B0 = beta_dist.cdf(x=pi1, a=(1+a), b=b) * b10/b00
    B1 = (beta_dist.cdf(x=1, a=a, b=(1+b)) -
          beta_dist.cdf(x=pi1, a=a, b=(1+b))) * b01/b00

    return B0, B1


def _generate_LH_coef(n0, n1, G0, G1, b0, b1):

    pi1 = n1 / (n1 + n0)
    pi0 = n0 / (n1 + n0)
    b0_head = b0[1:]
    b0_norm = b0[:-1]
    b1_head = b1[1:]
    b1_norm = b1[:-1]

    LH_array = (pi0 * (1 - G0) * (b0_head - b0_norm)) + \
        (pi1 * G1 * (b1_head - b1_norm))

    return sum(LH_array)


def _generate_b_vecs(cost, a, b):

    b10 = beta_func((1+a), b)
    b01 = beta_func(a, (1+b))
    b00 = beta_func(a, b)

    b0 = beta_dist.cdf(x=cost, a=(1 + a), b=b) * (b10 / b00)
    b1 = beta_dist.cdf(x=cost, a=a, b=(1 + b)) * (b01 / b00)

    return b0, b1


def _generate_cost(n0, n1, G0: numpy.ndarray, G1: numpy.ndarray):

    G1_head = G1[1:]
    G1_norm = G1[:-1]
    G0_head = G0[1:]
    G0_norm = G0[:-1]

    c1 = (n1 / (n1 + n0)) * (G1_head - G1_norm)
    c0 = (n0 / (n1 + n0)) * (G0_head - G0_norm)

    c_more = c1 / (c1 + c0)

    cost = numpy.concatenate([[0], c_more, [1]])

    return cost


def _generate_beta_params(n0, n1, sev_ratio):

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


def _generate_convex_hull_points(invF0: numpy.ndarray, invF1: numpy.ndarray):

    pair_max = numpy.maximum(invF0, invF1)

    # if all invF0>=invF1, pair_max = invF0
    # chull_cand is a line and
    # ConvexHull results in QHullError
    if not numpy.array_equal(invF0, pair_max):
        chull_cand = numpy.array(list(zip(invF0, pair_max)))
        hull = ConvexHull(chull_cand)
        vert = hull.vertices

        G0 = numpy.sort(invF0[vert])
        G1 = numpy.sort(invF1[vert])

    else:
        G0 = numpy.array([0, 1])
        G1 = numpy.array([0, 1])

    return G0, G1


def _generate_h_measure(n0, n1, B0, B1, LH, fix_prec=True):

    j = (LH * (n0+n1))
    k = (n0*B0 + n1*B1)

    # due to floating point imprecision
    # set lower bound at 0
    if fix_prec and numpy.isclose(j, k) and j > k:
        j = k

    return 1 - numpy.divide(j, k)


def _transform_roc_to_invF(fpr_untr: numpy.ndarray, tpr_untr: numpy.ndarray):
    fpr = -numpy.sort(-fpr_untr)
    tpr = -numpy.sort(-tpr_untr)

    # extend vector with explicit (0, 0)
    fpr = numpy.concatenate([fpr, [0]])
    tpr = numpy.concatenate([tpr, [0]])

    return fpr, tpr


def h_score(y_true: numpy.ndarray, y_score: numpy.ndarray,
            severity_ratio: float = None, pos_label=None) -> float:
    """Compute the h-measure as sklearn-compatible metric score

    Note: this implementation is restricted to the binary classification task.

    Read more in the original implementation: 
    https://github.com/canagnos/hmeasure

    Parameters
    ----------

    y_true : numpy.ndarray, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score : numpy.ndarray, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    severity_ratio: float, default = None
        The relative cost of misclassification of the positive class to the 
        other class(es). Value of 0 raises error. By default None, which is
        translated into number of positive / number of other class(es). See
        [3]_ or [4]_ for more detail.

    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.

    Returns
    -------
    h_score : float

    Notes
    -----
    The H-measure is a measure of classification performance proposed by 
    D.J.Hand. It successfully overcomes the problem of capturing performance 
    across multiple potential scenaria. Moreover, it is important in that it 
    proposes a sensible criterion for coherence of performance metrics, which 
    the H-measure satisfies but surprisingly several popular alternatives do 
    not, notably including the Area Under the Curve (AUC) and its variants, 
    such as the Gini coefficient [1]_ [2]_ [3]_ [4]_. 

    References
    ----------
    .. [1] Hand, D.J. 2009. Measuring classifier performance: a coherent
     alternative to the area under the ROC curve. Machine Learning, 77, 103–123.

    .. [2] Hand, D.J. 2010. Evaluating diagnostic tests: the area under the 
     ROC curve and the balance of errors. Statistics in Medicine, 29, 1502–1510.

    .. [3] Hand, D.J. and Anagnostopoulos, C. 2014. A better Beta for the H 
     measure of classification performance. Pattern Recognition Letters, 40, 41-46.

    .. [4] Hmeasure CRAN Reference for original R package
     https://cran.r-project.org/package=hmeasure

    Examples
    --------
    >>> import numpy
    >>> from hmeasure import h_score
    >>> rng = numpy.random.default_rng(66)
    >>> y_true = rng.integers(low=0, high=2, size=10)
    >>> y_true
    array([1, 1, 0, 1, 1, 0, 1, 1, 1, 0])
    >>> # y_pred random sampled in interval [0, 1)
    >>> y_pred = (1 - 0) * rng.random(10) + 0
    >>> y_pred
    array([0.84901876, 0.10282827, 0.43752488, 0.46004468, 0.90878931,
    ...    0.79177719, 0.5297229 , 0.13803906, 0.73166264, 0.22959056])
    >>> h_score(y_true, y_pred)
    0.18889596344769588
    >>> n1, n0 = y_true.sum(), y_true.shape[0]-y_true.sum()
    >>> h_score(y_true, y_pred, severity_ratio=(n1/n0))
    0.18889596344769588
    >>> h_score(y_true, y_pred, severity_ratio=0.7)
    0.13502616807120948
    >>> h_score(y_true, y_pred, severity_ratio=-0.7)
    0.18310946512079307
    >>> h_score(y_true, y_pred, severity_ratio=0.1)
    0.001212529211507385
    >>> h_score(y_true, y_pred, severity_ratio=0.5)
    0.10750123502531805

    """
    if not isinstance(y_true, numpy.ndarray):
        raise TypeError("y_true must be of type numpy.ndarray!")
    if not isinstance(y_score, numpy.ndarray):
        raise TypeError("y_score must be of type numpy.ndarray!")
    if len(numpy.unique(y_true)) != 2:
        raise TypeError("y_true must represent a binary classification!")
    true_max, true_min = y_true.max(), y_true.min()
    if not (y_score <= true_max).all():
        raise TypeError(
            "y_score must contain values not larger than max true label {}!".format(true_max))
    if not (y_score >= true_min).all():
        raise TypeError(
            "y_score must contain values not smaller than min true label {}!".format(true_min))

    if pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    fpr_untr, tpr_untr, _ = roc_curve(y_true=y_true,
                                      y_score=y_score,
                                      pos_label=True,
                                      drop_intermediate=False)

    fpr, tpr = _transform_roc_to_invF(fpr_untr, tpr_untr)

    G0, G1 = _generate_convex_hull_points(fpr, tpr)

    n1 = y_true.sum()
    n0 = y_true.shape[0] - n1

    a, b = _generate_beta_params(n0=n0, n1=n1, sev_ratio=severity_ratio)
    cost = _generate_cost(n0=n0, n1=n1, G0=G0, G1=G1)
    b0, b1 = _generate_b_vecs(cost=cost, a=a, b=b)

    LH = _generate_LH_coef(n0=n0, n1=n1, G0=G0, G1=G1, b0=b0, b1=b1)
    B0, B1 = _generate_B_coefs(a=a, b=b, n0=n0, n1=n1)

    h_score = _generate_h_measure(n0=n0, n1=n1,
                                  B0=B0, B1=B1,
                                  LH=LH, fix_prec=True)

    return h_score
