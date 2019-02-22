import numpy as np

np.random.seed(12345)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import preprocessing

# #############################################################################
# Load data
credits = preprocessing.load_tmdb_credits("dataset/tmdb_5000_credits.csv")
movies = preprocessing.load_tmdb_movies("dataset/movies.csv")
#movies = preprocessing.includeProductionCompanies(movies)

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

# Substitute NaN values with 0
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

regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
regr.fit(x_train, y_train)

mse = mean_squared_error(y_test, regr.predict(x_test))
print("MSE: %.4f" % mse)

# ####################cross_val_score#########################################################
# Validate using cross-validation
movies = movies.fillna(0)
regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
regr.fit(x_train, y_train)

scores = cross_val_score(regr, movies, y, cv=5, scoring='neg_mean_squared_error')
print("Cross validation with 5 groups")
print(scores)
