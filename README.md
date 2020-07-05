# SportPredictingWinner

This repo demonstrates that a ML model can predict accurately


## Steps to follow

  1. We will clean our dataset
  2. Split it into training and testing data (12 features & 1 target (winning team           (Home/Away/Draw))
  3. Train the data with XGBoost 
  4. Use the best version of the classifier to predict which team will win
  
  
  
## Dataset
  
The graph shows the total number of games **TNM**, the number of home winnings **HW**, the number of Away winnings **AW** and finally the number of **Draws**.
  
<p align="center"> 
<img src="https://github.com/BardisRenos/SportPredictingWinner/blob/master/myplot.png" width="450" height="450" style=centerme>
</p>
  
## Using the necessary dependencies

```python
# Import Dependencies

# Import pandas library for data preprocessing
import pandas as pd
# Import the matplotlib 
import matplotlib.pyplot as plt
import numpy as np
# Import the classifier
import xgboost as xgb
# Import the sklearn library to standardising the data.
from sklearn.preprocessing import scale

```
 
 
## Preparing the Data




## Training and Evaluating the Model
