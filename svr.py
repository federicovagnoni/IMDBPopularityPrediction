import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR

import preprocessing

credits = preprocessing.load_tmdb_credits("dataset/tmdb_5000_credits.csv")
movies = preprocessing.load_tmdb_movies("dataset/tmdb_5000_movies.csv")
meta = pd.read_csv("dataset/movie_metadata.csv")
movies = preprocessing.preProcess(movies, meta, credits)

# # Remove all nominal features
movies = movies.drop(["genres", "homepage", "id", "keywords", "original_language", "original_title", "overview",
                      "production_companies", "production_countries", "spoken_languages", "status", "tagline",
                      "title",
                      "release_date", 'revenue', 'vote_average', 'vote_count'], axis=1)

# Substitue NaN values with 0
movies = movies.fillna(0)

y = movies['popularity']
movies = movies.drop(['popularity'], axis=1)
# Select the random indexes for the test set and
x_train, x_test, y_train, y_test = train_test_split(
    movies, y, test_size=0.30, random_state=1234)

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

print(y_test.describe())
print(y_train.describe())

# #############################################################################
# Define model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma='auto')
svr_lin = SVR(kernel='linear', C=1e3, gamma='auto')
svr_poly_2 = SVR(kernel='poly', C=1e3, degree=2, gamma='auto')
svr_poly_3 = SVR(kernel='poly', C=1e3, degree=3, gamma='auto')

svr_rbf.fit(x_train, y_train)
y_rbf = svr_rbf.predict(x_test)

svr_lin.fit(x_train, y_train)
y_lin = svr_lin.predict(x_test)

svr_poly_2.fit(x_train, y_train).predict(x_test)
y_poly = svr_poly_2.predict(x_test)

plt.scatter(y_test, y_test, s=30, label="Real popularity")
plt.scatter(y_test, y_poly, s=30, c='r', label="Predicted popularity")
plt.ylabel("popularity")
plt.legend(loc=2)
plt.show()


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


print("Linear")
print("Mean squared error: %.6f"
      % rmse(y_test, y_lin))
print("Mean absolute error: %.6f"
      % mean_absolute_error(y_test, y_lin))

print("\nPolynomial n=2")
print("Mean squared error: %.6f"
      % rmse(y_test, y_poly))
print("Mean absolute error: %.6f"
      % mean_absolute_error(y_test, y_poly))

print("\nRBF")
print("Mean squared error: %.6f"
      % rmse(y_test, y_rbf))
print("Mean absolute error: %.6f"
      % mean_absolute_error(y_test, y_rbf))

#############################################################################
# Validate using cross-validation
# See https://scikit-learn.org/stable/modules/model_evaluation.html
scores = cross_val_score(svr_rbf, movies, y, cv=10, scoring='neg_mean_squared_error')
print(f"MSE RBF: {scores}")
scores = cross_val_score(svr_lin, movies, y, cv=10, scoring='neg_mean_squared_error')
print(f"MSE Linear: {scores}")
cores = cross_val_score(svr_poly_2, movies, y, cv=10, scoring='neg_mean_squared_error')
print(f"MSE POLYNOMIAL (DEGREE=2): {scores}")
scores = cross_val_score(svr_poly_3, movies, y, cv=10, scoring='neg_mean_squared_error')
print(f"MSE POLYNOMIAL (DEGREE=3): {scores}")
