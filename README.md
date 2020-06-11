# WI4475: Data Assimilation - Tidal Waves Project

This repository contains the script used to address the questions of the project. 
In order to work properly it requires the following libraries:

```
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import errno
from scipy.sparse.linalg import inv
from scipy.sparse import hstack, vstack, csc_matrix
from numpy import loadtxt
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates  
``` 

The repository contains the following files:

## tide_STATIONNAME.txt
Those files contains the observed tide at five different stations. Those files are used to load the observations and computed a first analysis of the performances of the model.
Particularly, tide_cadnaz.txt is used asboundary condition.

## waterlevel_STATIONNAME.txt
Those files contain the real observation with the storm effects included and are used for questions 9 and 10. 

## timeseries.py
This file is needed to properly read the observations data from the tide and waterlel files

## wave1d_ex1
This file addresses question 1 of the project

## wave1d_ex3
This file addresses question 3 of the project


## wave1d_ex4
This file addresses question 4 of the project. An AR is implemented to the left boundary. 

## wave1d_ex6_ex7_ex8
This file can be used to address questions 6 to 8 of the project. It requires ex6_observationdata_wave1d.py as input. For question 7 it is needed to change the ensemble size and run the code.

##ex6_observationdata_wave1d.py
This file is used to produce the observations data used as real data input in exercise 6. The initial condition needs to be modified to perform task of question 8.

## wave1d_ex9
This file addresses question 9 of the project. It requires as input wave_1d.py

## wave1d_ex9
This file addresses question 10 of the project. It requires as input wave_1d.py

## wave_1d.py
This file is used as input for wave_ex9 and wave_ex10.

@author: Anna Spinosa, Xiao Li
