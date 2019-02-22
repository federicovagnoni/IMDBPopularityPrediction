import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from mpl_toolkits.mplot3d import Axes3D

import json
import numpy as np
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

def analyzeActors():
    df = pd.read_csv("./dataset/movies_metadata.csv")


def includeProductionCompanies(movies):
    companiesList = dict()
    for index, row in movies.iterrows():
        companies = row['production_companies']
        # companiesJson = json.loads(companies)
        for c in companies:
            if c['name'] in companiesList.keys():
                companiesList[c['name']][0] += row['popularity']
                companiesList[c['name']][1] += 1
            else:
                companiesList[c['name']] = [row['popularity'], 1]

    # Compute mean
    for e in companiesList.keys():
        companiesList[e] = companiesList[e][0] / companiesList[e][1]

    movies['companies_popularity'] = 0
    for index, row in movies.iterrows():
        companies = row['production_companies']
        # companiesJson = json.loads(companies)
        names = [c['name'] for c in companies]
        for company in names:
            if companiesList[company] > movies.at[index, 'companies_popularity']:
                movies.at[index, 'companies_popularity'] = companiesList[company]

    # print(movies['companies_popularity'].describe())
    # movies.to_csv('./dataset/movies_companies.csv', index=False)
    return movies


def convertGenres(movies):
    #movies['genres'] = movies['genres'].apply(pipe_flatten_names)

    liste_genres = set()
    for s in movies['genres'].str.split('|'):
        liste_genres = set().union(s, liste_genres)
    liste_genres = list(liste_genres)
    liste_genres.remove('')

    for genre in liste_genres:
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


def insertCast(movies):
    meta = pd.read_csv("dataset/movie_metadata.csv")

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

    movies.to_csv("dataset/movies.csv", index=False)
    return movies


def castClustering(meta, movies):
    meta = meta.drop(['genres', 'budget'], axis=1)

    directorDict = dict()
    actorsDict = dict()
    for index, row in meta.iterrows():
        directorDict[row['director_name']] = row['director_facebook_likes']
        actorsDict[row['actor_1_name']] = [row['actor_1_facebook_likes'], 0, 0]
        actorsDict[row['actor_2_name']] = [row['actor_2_facebook_likes'], 0, 0]
        actorsDict[row['actor_2_name']] = [row['actor_3_facebook_likes'], 0, 0]

    keys = actorsDict.keys()
    for index, row in movies.iterrows():
        actors = [x['name'] for x in row['cast']]
        for actor in actors:
            if actor in keys:
                actorsDict[actor][1] += row['popularity']
                actorsDict[actor][2] += 1

    df = pd.DataFrame.from_dict(actorsDict, orient='index')
    df.columns = ['facebook_likes', 'popularity', 'movies_number']
    df = df.fillna(0)

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(df['facebook_likes'].values, df['popularity'].values, df['movies_number'].values)
    # ax.set_xlabel("facebook")
    # ax.set_ylabel("popularity")
    # ax.set_zlabel("movies_number")
    # plt.show()

    df_min = df.min()
    df_max = df.max()
    df -= df_min
    df /= df_max

    # Convert DataFrame to matrix
    mat = df.values
    # Using sklearn
    km = sklearn.cluster.KMeans(n_clusters=3)
    km.fit(mat)
    # Get cluster assignment labels
    labels = km.labels_
    # Format results as a DataFrame
    results = pd.DataFrame([df.index, labels]).T
    results.columns = ['actor', 'cluster']

    cluster_0 = results.loc[results['cluster'] == 0]['actor'].values
    cluster_1 = results.loc[results['cluster'] == 1]['actor'].values
    cluster_2 = results.loc[results['cluster'] == 2]['actor'].values

    movies['actor_0'] = 0
    movies['actor_1'] = 0
    movies['actor_2'] = 0

    for index, row in movies.iterrows():
        actors = [x['name'] for x in row['cast']]
        for actor in actors:
            if actor in cluster_0:
                movies.at[index, 'actor_0'] = 1
                continue
            if actor in cluster_1:
                movies.at[index, 'actor_1'] = 1
                continue
            if actor in cluster_2:
                movies.at[index, 'actor_2'] = 1

    return movies


def includeProductionCountries(movies):
    array = []
    for index, row in movies.iterrows():
        if len(row['production_countries']) != 0:
            h = row['production_countries'][0]['iso_3166_1']
            movies.at[index, 'production_countries'] = h
            if h not in array:
                array.append(h)

    for c in array:
        movies[c] = movies['production_countries'].str.contains(c).apply(lambda x: 1 if x else 0)

    return movies


def preProcess(movies, meta, credits):
    # movies = convertGenres(movies)
    # movies = convertRuntime(movies)
    movies = includeProductionCompanies(movies)
    # movies = includeProductionCountries(movies)
    del credits['title']
    del credits['movie_id']
    movies = pd.concat([movies, credits], axis=1)
    # movies = castClustering(meta, movies)
    del movies['cast']
    del movies['crew']
    # del movies["actor_0"]
    movies = insertCast(movies)
    return movies



def cleaningAndConvertion(movies):
    movies = convertGenres(movies)

    movies = convertRuntime(movies)

    movies = includeProductionCompanies(movies)

    movies = movies.drop(
        ["genres", "homepage", "id", "keywords", "original_language", "original_title", "overview",
         "production_companies", "production_countries", "spoken_languages", "status", "tagline", "title",
         "release_date", "runtime"], axis=1)

    return movies


if __name__ == "__main__":
    # #########
    # Load data
    credits = load_tmdb_credits("dataset/tmdb_5000_credits.csv")
    movies = load_tmdb_movies("dataset/tmdb_5000_movies.csv")
    meta = pd.read_csv("dataset/movie_metadata.csv")

    print(movies['production_countries'].head(100))

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
