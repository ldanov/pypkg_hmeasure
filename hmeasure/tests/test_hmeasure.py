from hmeasure import *
def test_generate_beta_params():
    
    generate_beta_params()
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