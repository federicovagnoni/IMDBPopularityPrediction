import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

np.random.seed(12345)
import tensorflow as tf

tf.set_random_seed(12345)
import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

credits = preprocessing.load_tmdb_credits("dataset/tmdb_5000_credits.csv")
movies = preprocessing.load_tmdb_movies("dataset/tmdb_5000_movies.csv")
meta = pd.read_csv("dataset/movie_metadata.csv")
del credits['title']
movies = pd.concat([movies, credits], axis=1)


# corrmat = movies.corr()
# hm = sns.heatmap(corrmat, annot=True, cmap='coolwarm')
# plt.show()
# plt.close()

y = movies['popularity']
x_train, x_test, y_train, y_test = train_test_split(
    movies, y, test_size=0.30, random_state=1234)

x_train, x_test = preprocessing.preProcess(x_train, x_test, meta)


# Remove all nominal features
x_train = x_train.drop(
    ["genres", "homepage", "id", "keywords", "original_language", "original_title", "overview", "production_companies",
     "production_countries", "spoken_languages", "status", "tagline", "title", "release_date", "popularity", "cast", "crew", "movie_id",
     'revenue', 'vote_average', 'vote_count'], axis=1)

x_test = x_test.drop(
    ["genres", "homepage", "id", "keywords", "original_language", "original_title", "overview", "production_companies",
     "production_countries", "spoken_languages", "status", "tagline", "title", "release_date", "popularity", "cast", "crew", "movie_id",
     'revenue', 'vote_average', 'vote_count'], axis=1)

print(x_test.columns.values)
print(x_train.columns.values)

# Scale train and test set between 0 and 1 using the max and min values for each attribute the values for each
# attribute are retrieved form the training set and these values will be used on the test set too (e.g. we do not use
#  the max and min value that the attributes of the test set will have, but we will use the ones from the training)
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

best_mse = 100
best_mae = 100
best_model = ""
best_nodes = 1

# Train using 3 neurons in the hidden layer
for num_nodes in range(2, 26):

    MLP = tf.keras.models.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='sigmoid', input_shape=(x_train.shape[1],)),
        # tf.keras.layers.Dense(3, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    MLP.compile(optimizer="adadelta", loss="mse", metrics=['mae'])
    MLP.fit(x_train, y_train, epochs=10)

    mse, mae = MLP.evaluate(x_test, y_test)
    print("Error on the test set for " + str(num_nodes) + " nodes is: ", [mse, mae])

    if mse < best_mse:
        best_nodes = num_nodes
        best_mae = mae
        best_mse = mse
        best_model = MLP

print("The best loss is for " + str(best_nodes) + " nodes with MSE: " + str(best_mse) + ", MAE: " + str(best_mae))

y_pred = best_model.predict(x_test)

# plt.rcParams['legend.numpoints'] = 1
# fig, ax = plt.subplots(figsize=(6, 4))
# for i in range(len(y_pred)):
#    ax.plot([i, i], [y_pred[i], y_test[i]], c="k", linewidth=0.5)
# ax.plot(y_pred, 'o', label='Prediction', color='g')
# ax.plot(y_test, '^', label='Ground Truth', color='r')

plt.plot(np.arange(len(y_test)), y_test, color='red', label='Real data')
plt.plot(np.arange(len(y_test)), y_pred, color='blue', label='Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()
plt.close()

y_pred = np.concatenate(y_pred, axis=0)

plt.plot(np.arange(len(y_test)), y_pred - y_test.values, color='red', label='Real data')
plt.title('Prediction')
plt.legend()
plt.show()
