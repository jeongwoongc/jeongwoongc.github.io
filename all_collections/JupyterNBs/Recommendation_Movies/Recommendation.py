#%%
import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")
# check data 
for x in ratings.take(1).as_numpy_iterator():
  pprint.pprint(x)

#%% split ratings/movies into movie titles and user id

ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
})
movies = movies.map(lambda x: x["movie_title"])
#%% shuffle data 

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

#%% unique user ids and movie titles

movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

unique_movie_titles[:10]
# %% Query tower 

embedding_dimension = 32
user_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
    vocabulary=unique_user_ids, mask_token=None),
  # additional embedding to account for unkown tokens
  tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

# %% Candidate tower

movie_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_movie_titles, mask_token=None),
  tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
])
# %% Metrics

metrics = tfrs.metrics.FactorizedTopK(
  candidates=movies.batch(128).map(movie_model)
)

# %% loss 

task = tfrs.tasks.Retrieval(
  metrics=metrics
)

# %% Full model

class MovielensModel(tfrs.Model):
  
  def __init__(self, user_model, movie_model):
    super().__init__()
    self.movie_model: tf.keras.Model = movie_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task
  
  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # User features to user model
    user_embeddings = self.user_model(features["user_id"])
    # movie features to movie model with embeddings
    positive_movie_embeddings = self.movie_model(features["movie_title"])

    # Task method computes loss and metrics
    return self.task(user_embeddings, positive_movie_embeddings)

# %% Fitting

model = MovielensModel(user_model, movie_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
# %% shuffle, batch, cache

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

# %% train model

model.fit(cached_train, epochs=3)

# %% eval

model.evaluate(cached_test, return_dict=True)

# %% make predictions

index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(
  tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
)

_, title = index(tf.constant(["56"]))
print(f"Recommendations for user 56: {title[0, :3]}")

# %%
