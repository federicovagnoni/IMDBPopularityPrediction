import json
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sklearn

np.random.seed(0)
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

import seaborn as sns

sns.set()


def genresAnalysis(df):
    liste_genres = set()
    for s in df['genres'].str.split('|'):
        liste_genres = set().union(s, liste_genres)
    liste_genres = list(liste_genres)
    liste_genres.remove('')

    df_reduced = df[
        ['title', 'vote_average', 'release_date', 'runtime', 'budget', 'revenue', 'popularity']].reset_index(drop=True)

    for genre in liste_genres:
        df_reduced[genre] = df['genres'].str.contains(genre).apply(lambda x: 1 if x else 0)
    df_reduced[:5]

    mean_per_genre = pd.DataFrame(liste_genres)

    # Mean votes average
    newArray = [] * len(liste_genres)
    for genre in liste_genres:
        newArray.append(df_reduced.groupby(genre, as_index=True)['vote_average'].mean())
    newArray2 = [] * len(liste_genres)
    for i in range(len(liste_genres)):
        newArray2.append(newArray[i][1])

    mean_per_genre['mean_votes_average'] = newArray2

    # Mean budget
    newArray = [] * len(liste_genres)
    for genre in liste_genres:
        newArray.append(df_reduced.groupby(genre, as_index=True)['budget'].mean())
    newArray2 = [] * len(liste_genres)
    for i in range(len(liste_genres)):
        newArray2.append(newArray[i][1])

    mean_per_genre['mean_budget'] = newArray2

    # Mean revenue
    newArray = [] * len(liste_genres)
    for genre in liste_genres:
        newArray.append(df_reduced.groupby(genre, as_index=True)['revenue'].mean())
    newArray2 = [] * len(liste_genres)
    for i in range(len(liste_genres)):
        newArray2.append(newArray[i][1])

    mean_per_genre['mean_revenue'] = newArray2

    # Mean popularity
    newArray = [] * len(liste_genres)
    for genre in liste_genres:
        newArray.append(df_reduced.groupby(genre, as_index=True)['popularity'].mean())
    newArray2 = [] * len(liste_genres)
    for i in range(len(liste_genres)):
        newArray2.append(newArray[i][1])

    mean_per_genre['mean_popularity'] = newArray2

    mean_per_genre['profit'] = mean_per_genre['mean_revenue'] - mean_per_genre['mean_budget']

    print(mean_per_genre.sort_values('mean_popularity', ascending=False))


def includeProductionCompanies(movies_train, movies_test):
    # Learn the company score from the training test
    companiesList = dict()
    for index, row in movies_train.iterrows():
        companies = movies_train.loc[index, 'production_companies']
        for c in companies:
            if c['name'] in companiesList.keys():
                companiesList[c['name']][0] += movies_train.loc[index, 'popularity']
                companiesList[c['name']][1] += 1
            else:
                companiesList[c['name']] = [movies_train.loc[index, 'popularity'], 1]

    # Compute mean
    for e in companiesList.keys():
        companiesList[e] = companiesList[e][0] / companiesList[e][1]

    movies_train = movies_train.copy()
    movies_test = movies_test.copy()

    movies_train.loc[:, 'companies_popularity'] = 0
    movies_test.loc[:, 'companies_popularity'] = 0

    # Apply the company score to the training set
    for index, row in movies_train.iterrows():
        companies = movies_train.loc[index, 'production_companies']
        names = [c['name'] for c in companies]
        for company in names:
            if companiesList[company] > movies_train.loc[index, 'companies_popularity']:
                movies_train.loc[index, 'companies_popularity'] = companiesList[company]

    # Apply the company score to the test set
    for index, row in movies_test.iterrows():
        companies = movies_test.loc[index, 'production_companies']
        names = [c['name'] for c in companies]
        for company in names:
            if company in companiesList:
                if companiesList[company] > movies_test.loc[index, 'companies_popularity']:
                    movies_test.loc[index, 'companies_popularity'] = companiesList[company]

    my_imputer = SimpleImputer()
    X2 = my_imputer.fit_transform(movies_test[['companies_popularity']])
    movies_test['companies_popularity'] = X2

    return movies_train, movies_test


def convertGenres(movies):
    liste_genres = set()
    for s in movies['genres'].str.split('|'):
        liste_genres = set().union(s, liste_genres)
    liste_genres = list(liste_genres)

    for genre in liste_genres:
        if genre != '':
            movies[genre] = movies['genres'].str.contains(genre).apply(lambda x: 1 if x else 0)

    return movies


def convertRuntime(movies):
    my_imputer = SimpleImputer()
    X2 = my_imputer.fit_transform(movies[['runtime']])
    movies['runtime'] = X2

    movies['runtime'] = pd.cut(movies['runtime'], [0, 75, movies['runtime'].describe()['max']], labels=['low', 'high'])

    for length in ["low", "high"]:
        movies[length] = movies['runtime'].str.contains(length).apply(lambda x: 1 if x else 0)

    return movies


def load_tmdb_movies(path):
    df = pd.read_csv(path)
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    df['genres'] = df['genres'].apply(pipe_flatten_names)
    return df


def load_tmdb_credits(path):
    df = pd.read_csv(path)
    json_columns = ['cast', 'crew']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df


def pipe_flatten_names(keywords):
    keys = [x['name'] for x in keywords]
    return '|'.join([x['name'] for x in keywords])


def insertCast(movies, meta):
    meta = meta.drop(['genres', 'budget'], axis=1)

    meta['movie_title'] = meta['movie_title'].apply(lambda x: x.strip())
    movies['title'] = movies['title'].apply(lambda x: x.strip())

    meta = meta[['movie_title', 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes',
                 'director_facebook_likes', 'movie_facebook_likes', 'cast_total_facebook_likes']]
    movies = pd.merge(movies, meta.drop_duplicates(subset=['movie_title']), how='left', left_on=['title'],
                      right_on=['movie_title'])

    movies = movies.drop(['movie_title'], axis=1)

    my_imputer = SimpleImputer()
    X2 = my_imputer.fit_transform(movies[['actor_1_facebook_likes']])
    movies['actor_1_facebook_likes'] = X2

    my_imputer = SimpleImputer()
    X2 = my_imputer.fit_transform(movies[['actor_2_facebook_likes']])
    movies['actor_2_facebook_likes'] = X2

    my_imputer = SimpleImputer()
    X2 = my_imputer.fit_transform(movies[['actor_3_facebook_likes']])
    movies['actor_3_facebook_likes'] = X2

    my_imputer = SimpleImputer()
    X2 = my_imputer.fit_transform(movies[['director_facebook_likes']])
    movies['director_facebook_likes'] = X2

    my_imputer = SimpleImputer()
    X2 = my_imputer.fit_transform(movies[['movie_facebook_likes']])
    movies['movie_facebook_likes'] = X2

    my_imputer = SimpleImputer()
    X2 = my_imputer.fit_transform(movies[['cast_total_facebook_likes']])
    movies['cast_total_facebook_likes'] = X2

    return movies


def castClustering(meta, movies_train, movies_test):
    meta = meta.drop(['genres', 'budget'], axis=1)

    directorDict = dict()
    actorsDict = dict()
    for index, row in meta.iterrows():
        directorDict[row['director_name']] = row['director_facebook_likes']
        actorsDict[row['actor_1_name']] = [row['actor_1_facebook_likes'], 0, 0]
        actorsDict[row['actor_2_name']] = [row['actor_2_facebook_likes'], 0, 0]
        actorsDict[row['actor_2_name']] = [row['actor_3_facebook_likes'], 0, 0]

    keys = actorsDict.keys()
    for index, row in movies_train.iterrows():
        actors = [x['name'] for x in row['cast']]
        for actor in actors:
            if actor in keys:
                actorsDict[actor][1] += row['popularity']
                actorsDict[actor][2] += 1

    df = pd.DataFrame.from_dict(actorsDict, orient='index')
    df.columns = ['facebook_likes', 'popularity', 'movies_number']
    df = df.fillna(0)

    df_min = df.min()
    df_max = df.max()
    df -= df_min
    df /= df_max

    # Convert DataFrame to matrix
    mat = df.values
    # Using sklearn
    km = KMeans(n_clusters=3)
    km.fit(mat)
    # Get cluster assignment labels
    labels = km.labels_
    # Format results as a DataFrame
    results = pd.DataFrame([df.index, labels]).T
    results.columns = ['actor', 'cluster']

    cluster_0 = results.loc[results['cluster'] == 0]['actor'].values
    cluster_1 = results.loc[results['cluster'] == 1]['actor'].values
    cluster_2 = results.loc[results['cluster'] == 2]['actor'].values

    movies_train['actor_0'] = 0
    movies_train['actor_1'] = 0
    movies_train['actor_2'] = 0

    for index, row in movies_train.iterrows():
        actors = [x['name'] for x in row['cast']]
        for actor in actors:
            if actor in cluster_0:
                movies_train.loc[index, 'actor_0'] = 1
                continue
            if actor in cluster_1:
                movies_train.loc[index, 'actor_1'] = 1
                continue
            if actor in cluster_2:
                movies_train.loc[index, 'actor_2'] = 1

    movies_test['actor_0'] = 0
    movies_test['actor_1'] = 0
    movies_test['actor_2'] = 0

    for index, row in movies_test.iterrows():
        actors = [x['name'] for x in row['cast']]
        for actor in actors:
            if actor in cluster_0:
                movies_test.loc[index, 'actor_0'] = 1
                continue
            if actor in cluster_1:
                movies_test.loc[index, 'actor_1'] = 1
                continue
            if actor in cluster_2:
                movies_test.loc[index, 'actor_2'] = 1

    return movies_train, movies_test


def includeProductionCountries(movies_train, movies_test):
    array = []
    for index, row in movies_train.iterrows():
        if len(row['production_countries']) != 0:
            h = row['production_countries'][0]['iso_3166_1']
            movies_train.at[index, 'production_countries'] = h
            if h not in array:
                array.append(h)

    for c in array:
        movies_train[c] = movies_train['production_countries'].str.contains(c).apply(lambda x: 1 if x else 0)
        movies_test[c] = movies_test['production_countries'].str.contains(c).apply(lambda x: 1 if x else 0)

    return movies_train, movies_test


def preProcess(movies_train, movies_test, meta):
    movies_train = movies_train.copy()
    movies_test = movies_test.copy()
    # movies_train = convertGenres(movies_train)
    # movies_test = convertGenres(movies_test)

    # movies_train, movies_test = includeProductionCompanies(movies_train, movies_test)
    # movies_train, movies_test = includeProductionCountries(movies_train, movies_test)
    #
    # movies_train, movies_test = castClustering(meta, movies_train, movies_test)
    # movies_train = insertCast(movies_train, meta)
    # movies_test = insertCast(movies_test, meta)

    my_imputer = SimpleImputer()
    X2 = my_imputer.fit_transform(movies_train[['runtime']])
    movies_train['runtime'] = X2

    my_imputer = SimpleImputer()
    X2 = my_imputer.fit_transform(movies_test[['runtime']])
    movies_test['runtime'] = X2

    return movies_train, movies_test


if __name__ == "__main__":
    # #########
    # Load data
    credits = load_tmdb_credits("dataset/tmdb_5000_credits.csv")
    movies = load_tmdb_movies("dataset/tmdb_5000_movies.csv")
    meta = pd.read_csv("dataset/movie_metadata.csv")
    del credits['title']
    movies = pd.concat([movies, credits], axis=1)

    y = movies['popularity']
    x_train, x_test, y_train, y_test = train_test_split(
        movies, y, test_size=0.30, random_state=1234)

    preProcess(x_train, x_test, meta)
# movies = movies.drop(['popularity'], axis=1)


# prod_co = pd.get_dummies(prod_co, prefix='')
# print(prod_co.head())

# movies = preProcess(movies, meta, credits)

# # ####################
# # Popularity Histogram
# popularity = movies.popularity
# # print(popularity.describe())
# plt.hist(popularity, bins=100)
# plt.hist(movies['companies_popularity'], bins=100)
# plt.legend(["popularity", 'companies_popularity'])
# plt.show()
#
# f, ax = plt.subplots(figsize=(12, 9))
# corrmat = movies.corr()
# print(corrmat)
# cols = corrmat.index
# print(cols.values)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(corrmat, annot=True, cmap='coolwarm')
# plt.show()
