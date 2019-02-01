import pandas as pd
import matplotlib.pyplot as plt
import json

credits = pd.read_csv("dataset/tmdb_5000_credits.csv")
movies = pd.read_csv("dataset/tmdb_5000_movies.csv")

popularity = movies.popularity
print(popularity.describe())

plt.hist(popularity, bins=100)
# plt.show()

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

for e in companiesList.keys():
    companiesList[e] = [companiesList[e][0] / companiesList[e][1]]

df = pd.DataFrame.from_dict(companiesList, orient='index')
df = df.sort_values(0, ascending=False)
df = df.head(20).sort_values(0, ascending=True)

companies = df.index.values
print(companies)

for company in companies:
    movies[company] = 0
    for index, row in movies.iterrows():
        companies = row['production_companies']
        companiesJson = json.loads(companies)
        names = [c['name'] for c in companiesJson]
        if company in names:
            movies.at[index, company] = 1

movies.to_csv('./dataset/movies_companies20.csv', index=False)
exit()
