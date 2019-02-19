import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import preprocessing
from statistics import mean, stdev
from sklearn.model_selection import train_test_split

credits = preprocessing.load_tmdb_credits("dataset/tmdb_5000_credits.csv")
movies = preprocessing.load_tmdb_movies("dataset/tmdb_5000_movies.csv")
meta = pd.read_csv("dataset/movie_metadata.csv")
# movies = preprocessing.preProcess(movies, meta, credits)

# # Remove all nominal features
movies = movies.drop(["genres", "homepage", "id", "keywords", "original_language", "original_title", "overview",
                      "production_companies", "production_countries", "spoken_languages", "status", "tagline",
                      "title",
                      "release_date", 'budget'], axis=1)

# Substitue NaN values with 0
movies = movies.fillna(0)
y = movies['popularity']
# Select the random indexes for the test set and
x_train, x_test = train_test_split(movies, test_size=0.2)
y_train = x_train['popularity']
y_test = x_test['popularity']

x_train = x_train.drop(["popularity"], axis=1)
x_test = x_test.drop(["popularity"], axis=1)

# #############################################################################
# Normalize (min-max-scaler)
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
# Define model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma='auto')
svr_lin = SVR(kernel='linear', C=1e3, gamma='auto')
svr_poly_2 = SVR(kernel='poly', C=1e3, degree=2, gamma='auto')
svr_poly_3 = SVR(kernel='poly', C=1e3, degree=3, gamma='auto')

y_rbf = svr_rbf.fit(x_train, y_train).predict(x_test)
y_lin = svr_lin.fit(x_train, y_train).predict(x_test)
y_poly = svr_poly_2.fit(x_train, y_train).predict(x_test)

print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_rbf))

# #############################################################################
# Validate using cross-validation
# See https://scikit-learn.org/stable/modules/model_evaluation.html
# scores = cross_val_score(svr_rbf, movies, y, cv=5, scoring='neg_mean_squared_error')
# print(f"MSE RBF: {mean(scores)}")
# scores = cross_val_score(svr_lin, movies, y, cv=5, scoring='neg_mean_squared_error')
# print(f"MSE Linear: {mean(scores)}")
# cores = cross_val_score(svr_poly_2, movies, y, cv=5, scoring='neg_mean_squared_error')
# print(f"MSE POLYNOMIAL (DEGREE=2): {mean(scores)}")
# scores = cross_val_score(svr_poly_3, movies, y, cv=5, scoring='neg_mean_squared_error')
# print(f"MSE POLYNOMIAL (DEGREE=3): {mean(scores)}")
