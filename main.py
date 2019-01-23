import pandas as pd
import tensorflow as tf

credits = pd.read_csv("datasets/tmdb_5000_credits.csv")
movies = pd.read_csv("datasets/tmdb_5000_movies.csv")
movies.drop(["budget", "homepage", "id", "release_date"], axis=1)
y = movies["popularity"]



MLP = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, activation='sigmoid', input_shape=(movies.shape[1],)),
    tf.keras.layers.Dense(1, activation='softmax')
])

MLP.compile(optimizer=tf.keras.optimizers.Adadelta(0.5), loss=tf.losses.log_loss)
MLP.fit(movies, y, epochs=100)
