import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['MKL_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import random
import time
from joblib import dump

from rectools.models import (PopularModel,
                             LightFMWrapperModel)
from rectools import Columns
from rectools.dataset import Dataset

from pathlib import Path

from lightfm import LightFM


#set seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

DATA_PATH = Path("experiments/data_original")


users = pd.read_csv(DATA_PATH / 'users.csv')
items = pd.read_csv(DATA_PATH / 'items.csv')
interactions = pd.read_csv(DATA_PATH / 'interactions.csv')

Columns.Datetime = 'last_watch_dt'
interactions.drop(interactions[interactions[Columns.Datetime].str.len() != 10].index, inplace=True)
interactions[Columns.Datetime] = pd.to_datetime(interactions[Columns.Datetime], format='%Y-%m-%d')
max_date = interactions[Columns.Datetime].max()
interactions[Columns.Weight] = np.where(interactions['watched_pct'] > 10, 3, 1)

train = interactions.copy()

train.drop(train.query("total_dur < 300").index, inplace=True)

users.fillna('Unknown', inplace=True)
users = users.loc[users[Columns.User].isin(train[Columns.User])].copy()

user_features_frames = []
for feature in ["sex", "age", "income"]:
    feature_frame = users.reindex(columns=[Columns.User, feature])
    feature_frame.columns = ["id", "value"]
    feature_frame["feature"] = feature
    user_features_frames.append(feature_frame)
user_features = pd.concat(user_features_frames)


items = items.loc[items[Columns.Item].isin(train[Columns.Item])].copy()

items["genre"] = items["genres"].str.lower().str.replace(", ", ",", regex=False).str.split(",")
genre_feature = items[["item_id", "genre"]].explode("genre")
genre_feature.columns = ["id", "value"]
genre_feature["feature"] = "genre"

content_feature = items.reindex(columns=[Columns.Item, "content_type"])
content_feature.columns = ["id", "value"]
content_feature["feature"] = "content_type"

item_features = pd.concat((genre_feature, content_feature))

dataset = Dataset.construct(
    interactions_df=train,
    user_features_df=user_features,
    cat_user_features=["sex", "age", "income"],
    item_features_df=item_features,
    cat_item_features=["genre", "content_type"],
)



best_light_fm_params = {
    'loss': 'warp',
    'no_components': 100,
    'learning_rate': 0.0066594231278431434,
    'user_alpha': 0.0005203512950757777,
    'item_alpha': 0.0003116105897809789,
    'random_state': 42,
}

N_EPOCHS = 1
NUM_THREADS = 8
best_model = LightFMWrapperModel(LightFM(**best_light_fm_params),
                                        epochs=N_EPOCHS, num_threads=NUM_THREADS)
best_model.fit(dataset)

#популярные айтемы для холодных пользователей
popular_model = PopularModel()
popular_model.fit(dataset)

dump(best_model,"best_models/main_model.joblib")
dump(popular_model,"best_models/popular_model.joblib")
dump(dataset,"best_models/dataset.joblib")
