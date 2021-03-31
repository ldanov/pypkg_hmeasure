# README

## Description 

A Python translation of the R package `hmeasure` ([GitHub](https://github.com/canagnos/hmeasure)) ([CRAN](https://cran.r-project.org/package=hmeasure)).

## Installation 

To install the hmeasure library use pip:

```
pip install hmeasure
```
or install directly from source:

```
python setup.py install
```

## Usage

``` 
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
```
