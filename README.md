# hscore

## Description 

A Python translation of the R package `hmeasure` ([Link](https://github.com/canagnos/hmeasure)).

## Usage

``` 
from hscore.hscore import h_score 
import numpy

y_true = numpy.array([False, False, False, True, False, True])
y_pred = numpy.array([0.2, 0.3, 0.3, 0.6, 0.51, 0.51])

h_score(y_true, y_pred)
```