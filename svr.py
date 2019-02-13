import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import preprocessing
from statistics import mean , stdev

credits = preprocessing.load_tmdb_credits("dataset/tmdb_5000_credits.csv")
movies = preprocessing.load_tmdb_movies("dataset/tmdb_5000_movies.csv")
movies = preprocessing.includeProductionCompanies(movies)

# Remove all nominal features
movies = movies.drop(
    ["genres", "homepage", "id", "keywords", "original_language", "original_title", "overview", "production_companies",
     "production_countries", "spoken_languages", "status", "tagline", "title", "release_date"], axis=1)

# Get popularity values and remove them from the dataset
y = movies["popularity"]

movies = movies.drop(["popularity"], axis=1)

# Substitue NaN values with 0 #TODO it can be changed!
movies = movies.fillna(0)

# #############################################################################
# Normalize (min-max-scaler)
movies_min = movies.min()
movies_max = movies.max()
y_min = y.min()
y_max = y.max()

movies -= movies_min
movies /= movies_max

y -= y_min
y /= y_max

# #############################################################################
# Define model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma='auto')
svr_lin = SVR(kernel='linear', C=1e3, gamma='auto')
svr_poly_2 = SVR(kernel='poly', C=1e3, degree=2, gamma='auto')
svr_poly_3 = SVR(kernel='poly', C=1e3, degree=3, gamma='auto')

# y_rbf = svr_rbf.fit(x_train, y_train).predict(x_test)
# y_lin = svr_lin.fit(x_train, y_train).predict(x_test)
# y_poly = svr_poly.fit(x_train, y_train).predict(x_test)

# #############################################################################
# Validate using cross-validation
# See https://scikit-learn.org/stable/modules/model_evaluation.html
scores = cross_val_score(svr_rbf, movies, y, cv=5, scoring='neg_mean_squared_error')
print(f"MSE RBF: {mean(scores)}")
scores = cross_val_score(svr_lin, movies, y, cv=5, scoring='neg_mean_squared_error')
print(f"MSE Linear: {mean(scores)}")
cores = cross_val_score(svr_poly_2, movies, y, cv=5, scoring='neg_mean_squared_error')
print(f"MSE POLYNOMIAL (DEGREE=2): {mean(scores)}")
scores = cross_val_score(svr_poly_3, movies, y, cv=5, scoring='neg_mean_squared_error')
print(f"MSE POLYNOMIAL (DEGREE=3): {mean(scores)}")
