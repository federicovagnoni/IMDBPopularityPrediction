import numpy as np
import pandas as pd

np.random.seed(12345)
import tensorflow as tf
tf.set_random_seed(12345)
import preprocessing
import matplotlib.pyplot as plt

credits = preprocessing.load_tmdb_credits("dataset/tmdb_5000_credits.csv")
movies = preprocessing.load_tmdb_movies("dataset/tmdb_5000_movies.csv")
movies = preprocessing.includeProductionCompanies(movies)


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

# Substitute NaN values with 0 #TODO it can be changed!
movies = movies.fillna(0)
y = y.fillna(0)

# Use the indexes to form the training and test sets
x_test = movies.loc[index_test]
y_test = y.loc[index_test]
x_train = movies.loc[index_train]
y_train = y.loc[index_train]

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


# Train using 3 neurons in the hidden layer
MLP = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4, activation='sigmoid', input_shape=(movies.shape[1],)),
    tf.keras.layers.Dense(1, activation='linear')
])

MLP.compile(optimizer=tf.keras.optimizers.Adadelta(0.001), loss="mse")
MLP.fit(x_train, y_train, epochs=10)

# Return the loss on the test set
print("Loss on the test set:", MLP.evaluate(x_test, y_test))
