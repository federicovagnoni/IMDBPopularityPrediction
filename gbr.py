import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# #############################################################################
# Load data
credits = pd.read_csv("dataset/tmdb_5000_credits.csv")
movies = pd.read_csv("dataset/tmdb_5000_movies.csv")

# Remove all nominal features
movies = movies.drop(
    ["genres", "homepage", "id", "keywords", "original_language", "original_title", "overview", "production_companies",
     "production_countries", "spoken_languages", "status", "tagline", "title", "release_date"], axis=1)

# Get popularity values and remove them from the dataset
y = movies["popularity"]
movies = movies.drop(["popularity"], axis=1)

# Select the random indexes for the test set and
arr = np.arange(movies.shape[0])
index_test = np.random.choice(arr, int(0.25 * movies.shape[0]), replace=False)
index_train = np.setdiff1d(arr, index_test)

# Substitue NaN values with 0
movies = movies.fillna(0)
y = y.fillna(0)

# Use the indexes to form the training and test sets
x_test = movies.loc[index_test]
y_test = y.loc[index_test]
x_train = movies.loc[index_train]
y_train = y.loc[index_train]

xmins = x_train.min()
xmaxs = x_train.max()
ymins = y_train.min()
ymaxs = y_train.max()

x_train -= xmins
x_train /= xmaxs

x_test -= xmins
x_test /= xmaxs

y_train -= ymins
y_train /= ymaxs

y_test -= ymins
y_test /= ymaxs

# #############################################################################
# Fit regression model
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(x_train, y_train)
mse = mean_squared_error(y_test, clf.predict(x_test))
print("MSE: %.4f" % mse)

# #############################################################################
# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(x_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

# #############################################################################
# Plot feature importance

# Importance is calculated for a single decision tree by the amount that each attribute split point improves the
# performance measure, weighted by the number of observations the node is responsible for. The performance measure
# may be the purity (Gini index) used to select the split points or another more specific error function.
# See https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, movies.columns)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

# #############################################################################
# Validate using cross-validation

movies = movies.fillna(0)

movies_min = movies.min()
movies_max = movies.max()
y_min = y.min()
y_max = y.max()

movies -= movies_min
movies /= movies_max

y -= y_min
y /= y_max

params = {'n_estimators': 200, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(x_train, y_train)

scores = cross_val_score(clf, movies, y, cv=5, scoring='neg_mean_squared_error')
print("Cross validation with 5 groups")
print(scores)
