# SportPredictingWinner

This repo demonstrates that a ML model can predict accurately


## Steps to follow

  1. We will clean our dataset
  2. Split it into training and testing data (12 features & 1 target (winning team           (Home/Away/Draw))
  3. Train the data with XGBoost 
  4. Use the best version of the classifier to predict which team will win
  
  
## Dataset 

The dataset is downloaded from [link](http://football-data.co.uk/data.php) and I choose the English championship.  

## Data exploration
  
The graph shows the total number of games **TNM**, the number of home winnings **HW**, the number of Away winnings **AW** and finally the number of **Draws**.
  
<p align="center"> 
<img src="https://github.com/BardisRenos/SportPredictingWinner/blob/master/myplot.png" width="450" height="450" style=centerme>
</p>

```python
  df = pd.read_csv('/home/renos/Desktop/E1.csv')
  print(df.head())
  print("The shape of the dataframe : ", df.shape)
```
The dataset has 552 rows and 62 columns.

```python
  # The number of games are as number of rows
  num_matches = df.shape[0]
  
  # The number of features - 1 (the number of the labels). The number of features are the 
  # number of columns - 1 which is the label column
  num_features = df.shape[1] - 1
  
  print("Total number of matches: {}".format(num_matches))
  print("Number of features: {}".format(num_features))
  
  print("Number of matches won by home team: {}".format(num_home_team_wins))
  print("Number of matches won by away team: {}".format(num_away_team_wins))
  print("Number of matches draw: {}".format(num_home_drew))
```

```text
  Total number of matches: 552
  Number of features: 61
  
  Number of matches won by home team: 240
  Number of matches won by away team: 150
  Number of matches draw: 162

```

  
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
