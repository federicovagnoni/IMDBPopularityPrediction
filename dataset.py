import json
import pandas as pd


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


def safe_access(container, index_values):
    # return a missing value rather than an error upon indexing/key failure
    result = container
    try:
        for idx in index_values:
            result = result[idx]
        return result
    except IndexError or KeyError:
        return pd.np.nan


if __name__ == '__main__':
    movies = load_tmdb_movies("./dataset/tmdb_5000_movies.csv")
    credits = load_tmdb_credits("./dataset/tmdb_5000_credits.csv")

    movies.head(3)
    credits.head(3)

    print(sorted(credits.cast.iloc[0][0].keys()))
    print(sorted(credits.crew.iloc[0][0].keys()))

    print([actor['name'] for actor in credits['cast'].iloc[0][:5]])

    # Add Gender and lead columns
    credits['gender_of_lead'] = credits.cast.apply(lambda x: safe_access(x, [0, 'gender']))
    credits['lead'] = credits.cast.apply(lambda x: safe_access(x, [0, 'name']))
    print(credits.head(3))

    print(credits.gender_of_lead.value_counts())

    # Top 10 by revenue
    df = pd.merge(movies, credits, left_on='id', right_on='movie_id')
    print(df[['original_title', 'revenue', 'lead', 'gender_of_lead']].sort_values(by=['revenue'], ascending=False)[:10])

