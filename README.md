# hmeasure

## Description 

A Python translation of the R package `hmeasure` ([Link](https://github.com/canagnos/hmeasure)).

## Installation 
```
python path/to/hmeasure's/setup.py install
```

## Usage

``` 
>>> import numpy
>>> from hmeasure import h_score
>>> y = numpy.array([1, 1, 2, 2])
>>> scores = numpy.array([0.1, 0.4, 0.35, 0.8])
>>> h_score(y, preds, pos_label = 2)
0.6484375
>>> h_score(y, preds, severity_ratio=0.1, pos_label = 2)
0.742006883572737
```