import numpy as np
import pandas as pd

np.random.seed(12345)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import preprocessing

# #############################################################################
# Load data
credits = preprocessing.load_tmdb_credits("dataset/tmdb_5000_credits.csv")
movies = preprocessing.load_tmdb_movies("dataset/tmdb_5000_movies.csv")
meta = pd.read_csv("dataset/movie_metadata.csv")
del credits['title']
movies = pd.concat([movies, credits], axis=1)

y = movies['popularity']
x_train, x_test, y_train, y_test = train_test_split(
    movies, y, test_size=0.30, random_state=1)

x_train, x_test = preprocessing.preProcess(x_train, x_test, meta)

# # Remove all nominal features
x_train = x_train.drop(["genres", "homepage", "id", "keywords", "original_language", "original_title", "overview",
                        "production_companies", "production_countries", "spoken_languages", "status", "tagline",
                        "title",
                        "release_date", 'cast', 'crew'], axis=1)

x_test = x_test.drop(["genres", "homepage", "id", "keywords", "original_language", "original_title", "overview",
                      "production_companies", "production_countries", "spoken_languages", "status", "tagline",
                      "title",
                      "release_date", 'cast', 'crew'], axis=1)

print(x_train.describe())
print(x_test.describe())

y = movies['popularity']
movies = movies.drop(['popularity'], axis=1)

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

best_mse = 1
best_dept = 2
for x in range(2, 20):
    regr = RandomForestRegressor(max_depth=x, random_state=3, n_estimators=100, criterion='mse')
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    print("MSE: %.4f" % mse)

    if mse < best_mse:
        best_mse = mse
        best_dept = x

print(best_dept, best_mse)
# plt.plot(np.arange(len(y_test)), y_test, color='red', label='Real data')
# plt.plot(np.arange(len(y_test)), y_pred, color='blue', label='Predicted data')
# plt.title('Prediction')
# plt.legend()
# plt.show()
# plt.close()
