from statistics import mean, stdev

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVR

import preprocessing

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

credits = preprocessing.load_tmdb_credits("dataset/tmdb_5000_credits.csv")
movies = preprocessing.load_tmdb_movies("dataset/tmdb_5000_movies.csv")
meta = pd.read_csv("dataset/movie_metadata.csv")
del credits['title']
movies = pd.concat([movies, credits], axis=1)
y = movies['popularity']


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


## CROSS  VALIDATION

kf = KFold(n_splits=10)
kf.get_n_splits(movies)

rmse_list = []
for train_index, test_index in kf.split(movies):
    x_train, x_test = movies.loc[train_index], movies.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    x_train, x_test = preprocessing.preProcess(x_train, x_test, meta)

    # # Remove all nominal features
    # x_train = x_train.drop(["genres", "homepage", "id", "keywords", "original_language", "original_title", "overview",
    #                         "production_companies", "production_countries", "spoken_languages", "status", "tagline",
    #                         "title",
    #                         "release_date", 'popularity', 'revenue', 'vote_average', 'vote_count', "cast", "crew",
    #                         "movie_id"], axis=1)
    #
    # x_test = x_test.drop(["genres", "homepage", "id", "keywords", "original_language", "original_title", "overview",
    #                       "production_companies", "production_countries", "spoken_languages", "status", "tagline",
    #                       "title",
    #                       "release_date", 'popularity', 'revenue', 'vote_average', 'vote_count', "cast", "crew",
    #                       "movie_id"], axis=1)
    #
    x_train = x_train.drop(["genres", "homepage", "id", "keywords", "original_language", "original_title", "overview",
                            "production_companies", "production_countries", "spoken_languages", "status", "tagline",
                            "title",
                            "release_date", 'popularity',  "cast", "crew",
                            "movie_id"], axis=1)

    x_test = x_test.drop(["genres", "homepage", "id", "keywords", "original_language", "original_title", "overview",
                          "production_companies", "production_countries", "spoken_languages", "status", "tagline",
                          "title",
                          "release_date", 'popularity', "cast", "crew",
                          "movie_id"], axis=1)

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

    svr_lin = SVR(kernel='linear', C=1, gamma='auto', epsilon=0.001, max_iter=10000)
    svr_lin.fit(x_train, y_train)
    rmse_list.append(rmse(y_test, svr_lin.predict(x_test)))

print(rmse_list)
print("RMSE Mean ", mean(rmse_list))
print("RMSE stddev ", stdev(rmse_list))
