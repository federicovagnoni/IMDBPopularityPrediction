import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.preprocessing import Imputer

np.random.seed(0)

import seaborn as sns
sns.set()


def includeProductionCompanies(movies):
    companiesList = dict()
    for index, row in movies.iterrows():
        companies = row['production_companies']
        #companiesJson = json.loads(companies)
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
        #companiesJson = json.loads(companies)
        names = [c['name'] for c in companies]
        for company in names:
            if companiesList[company] > movies.at[index, 'companies_popularity']:
                movies.at[index, 'companies_popularity'] = companiesList[company]

    # print(movies['companies_popularity'].describe())
    # movies.to_csv('./dataset/movies_companies.csv', index=False)
    return movies


def convertGenres(movies):
    movies['genres'] = movies['genres'].apply(pipe_flatten_names)

    liste_genres = set()
    for s in movies['genres'].str.split('|'):
        liste_genres = set().union(s, liste_genres)
    liste_genres = list(liste_genres)
    liste_genres.remove('')

    for genre in liste_genres:
        movies[genre] = movies['genres'].str.contains(genre).apply(lambda x: 1 if x else 0)

    return movies

def convertRuntime(movies):
    my_imputer = Imputer()
    X2 = my_imputer.fit_transform(movies[['runtime']])
    movies['runtime'] = X2

    movies['runtime'] = pd.cut(movies['runtime'], [0, 75, 338], labels=['low', 'high'])

    for length in ["low", "high"]:
        movies[length] = movies['runtime'].str.contains(length).apply(lambda x: 1 if x else 0)

    return movies

def load_tmdb_movies(path):
    df = pd.read_csv(path)
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df

def load_tmdb_credits(path):
    df = pd.read_csv(path)
    json_columns = ['cast', 'crew']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df


def pipe_flatten_names(keywords):
    return '|'.join([x['name'] for x in keywords])

if __name__ == "__main__":
    # #########
    # Load data
    credits = load_tmdb_credits("dataset/tmdb_5000_credits.csv")
    movies = load_tmdb_movies("dataset/tmdb_5000_movies.csv")


    movies = convertGenres(movies)

    movies = convertRuntime(movies)

    movies = includeProductionCompanies(movies)

    movies = movies.drop(
        ["genres", "homepage", "id", "keywords", "original_language", "original_title", "overview",
         "production_companies", "production_countries", "spoken_languages", "status", "tagline", "title", "release_date", "runtime"], axis=1)

    # ####################
    # Popularity Histogram
    popularity = movies.popularity
    # print(popularity.describe())
    plt.hist(popularity, bins=100)
    plt.hist(movies['companies_popularity'], bins=100)
    plt.legend(["popularity", 'companies_popularity'])
    plt.show()


    f, ax = plt.subplots(figsize=(12, 9))
    corrmat = movies.corr()
    print(corrmat)
    cols = corrmat.index
    print(cols.values)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(corrmat, annot=True, cmap='coolwarm')
    plt.show()
