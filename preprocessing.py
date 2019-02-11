import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
np.random.seed(0)

import seaborn as sns
sns.set()


def includeProductionCompanies(movies):
    companiesList = dict()
    for index, row in movies.iterrows():
        companies = row['production_companies']
        companiesJson = json.loads(companies)
        for c in companiesJson:
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
        companiesJson = json.loads(companies)
        names = [c['name'] for c in companiesJson]
        for company in names:
            if companiesList[company] > movies.at[index, 'companies_popularity']:
                movies.at[index, 'companies_popularity'] = companiesList[company]

    # print(movies['companies_popularity'].describe())
    # movies.to_csv('./dataset/movies_companies.csv', index=False)
    return movies


if __name__ == "__main__":
    # #########
    # Load data
    credits = pd.read_csv("dataset/tmdb_5000_credits.csv")
    movies = pd.read_csv("dataset/tmdb_5000_movies.csv")

    # ####################
    # Popularity Histogram
    popularity = movies.popularity
    print(popularity.describe())
    plt.hist(popularity, bins=100)
    plt.show()

    movies = includeProductionCompanies(movies)
    movies = movies.drop(
        ["genres", "homepage", "id", "keywords", "original_language", "original_title", "overview",
         "production_companies", "production_countries", "spoken_languages", "status", "tagline", "title", "release_date"], axis=1)

    f, ax = plt.subplots(figsize=(12, 9))
    corrmat = movies.corr()
    print(corrmat)
    cols = corrmat.index
    print(cols.values)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(corrmat, annot=True, cmap='coolwarm')

    plt.show()
