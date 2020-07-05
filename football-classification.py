# Import Dependencies

# Import pandas library for data preprocessing
import pandas as pd
# Import the matplotlib
import matplotlib.pyplot as plt
import numpy as np
# Import the classifier
import xgboost as xgb
# Import the sklearn library to standardising the data.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, f1_score
import re

df = pd.read_csv('/home/renos/Desktop/E1.csv', index_col=False)
# print("The shape of the dataframe : ", df.shape)
# The number of games are as number of rows
num_matches = df.shape[0]
# The number of features - 1 (the number of the labels)
num_features = df.shape[1] - 1
num_home_team_wins = len(df[df.FTR == 'H'])
num_away_team_wins = len(df[df.FTR == 'A'])
num_home_drew = len(df[df.FTR == 'D'])
# Calculate win rate for home team.
win_home_rate = (float(num_home_team_wins) / num_matches) * 100

objects = ('TNM', 'HW', 'AW', 'Draw')
y_pos = np.arange(len(objects))
performance = [num_matches, num_home_team_wins, num_away_team_wins, num_home_drew]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.title('Total Matches/Winning/Loses/Draw')
# plt.show()

# print("Total number of matches: {}".format(num_matches))
# print("Number of features: {}".format(num_features))
# print("Number of matches won by home team: {}".format(num_home_team_wins))
# print("Number of matches won by away team: {}".format(num_away_team_wins))
# print("Number of matches draw: {}".format(num_home_drew))
# print("Win rate of home team is: {:.2f}%".format(win_home_rate))

X_data = df.drop(['FTR'], axis=1)
y_data = df['FTR']


def preprocess_data(X):
    for col, col_data in X.iteritems():
        if col_data.dtype == object:
            X = X.drop([col], axis=1)
    return X


def scalling_data(X):
    X_scaled = preprocess_data(X)
    for col in X_scaled.columns:
        X_scaled[col] = scale(X[col])
    return X_scaled


X_data = scalling_data(X_data)

regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X_data.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                  X_data.columns.values]

# Shuffle and split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=2)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


def train_predict(clf, X_train, y_train, X_test, y_test):
    # Train the classifier
    clf.fit(X_train, y_train)
    print("the predicted is: ", clf.predict(X_test[1:2]), "and the label is:", str(y_test[1:2]).split("    ")[1])
    y_pred = clf.predict(X_test)
    print("Accuracy : {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))


# Setting the classifier
clf_C = xgb.XGBClassifier(seed=82)

if __name__ == '__main__':
    train_predict(clf_C, X_train, y_train, X_test, y_test)
