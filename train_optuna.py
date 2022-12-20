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

from rectools.metrics import Precision, Recall, MAP, calc_metrics
from rectools.models import (PopularModel, RandomModel, ImplicitALSWrapperModel, 
                             LightFMWrapperModel)
from rectools import Columns
from rectools.dataset import Dataset

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import typing as tp
from tqdm import tqdm

from lightfm import LightFM
from implicit.als import AlternatingLeastSquares

import optuna
import hnswlib

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
train = interactions[interactions[Columns.Datetime] < max_date - 2*pd.Timedelta(days=7)].copy()
valid = interactions[(max_date - 2*pd.Timedelta(days=7) <= interactions[Columns.Datetime]) & \
                     (interactions[Columns.Datetime] <= max_date - pd.Timedelta(days=7))].copy()
test = interactions[interactions[Columns.Datetime] > max_date - pd.Timedelta(days=7)].copy()

print(f"train: {train.shape}")
print(f"valid: {valid.shape}")
print(f"test: {test.shape}")
print(f"{train.shape[0] + valid.shape[0] + test.shape[0] == interactions.shape[0]}")

train.drop(train.query("total_dur < 300").index, inplace=True)

# отфильтруем холодных пользователей из валида
cold_users = set(valid[Columns.User]) - set(train[Columns.User])
valid.drop(valid[valid[Columns.User].isin(cold_users)].index, inplace=True)


# подготовка данных
# user features

users.fillna('Unknown', inplace=True)
users = users.loc[users[Columns.User].isin(train[Columns.User])].copy()

user_features_frames = []
for feature in ["sex", "age", "income"]:
    feature_frame = users.reindex(columns=[Columns.User, feature])
    feature_frame.columns = ["id", "value"]
    feature_frame["feature"] = feature
    user_features_frames.append(feature_frame)
user_features = pd.concat(user_features_frames)


#item features
items = items.loc[items[Columns.Item].isin(train[Columns.Item])].copy()

items["genre"] = items["genres"].str.lower().str.replace(", ", ",", regex=False).str.split(",")
genre_feature = items[["item_id", "genre"]].explode("genre")
genre_feature.columns = ["id", "value"]
genre_feature["feature"] = "genre"
genre_feature.head()

content_feature = items.reindex(columns=[Columns.Item, "content_type"])
content_feature.columns = ["id", "value"]
content_feature["feature"] = "content_type"

item_features = pd.concat((genre_feature, content_feature))

# конструирование датасета
dataset = Dataset.construct(
    interactions_df=train,
    user_features_df=user_features,
    cat_user_features=["sex", "age", "income"],
    item_features_df=item_features,
    cat_item_features=["genre", "content_type"],
)

TEST_USERS = test[Columns.User].unique()
VALID_USERS = valid[Columns.User].unique()


metrics_name = {
    'Precision': Precision,
    'Recall': Recall,
    'MAP': MAP,
}

metrics = {}
for metric_name, metric in metrics_name.items():
    for k in [10]:
        metrics[f'{metric_name}@{k}'] = metric(k=k)
        
        
# Подбор гиперпараметров

K_RECOS = 10
NUM_THREADS = 8
N_EPOCHS = 1 # Lightfm

models = {
    'ALS': AlternatingLeastSquares,
    'LightFM': LightFM
}

# функция для оптимизации в optuna
def opt_metric(trial, dataset, valid_user_ids, top_k, metric, model):
    if model == 'ALS':
        params = {
            'num_threads': NUM_THREADS,
            'random_state': RANDOM_STATE,
            'regularization': trial.suggest_loguniform('regularization', 1e-5, 1.0),
            'factors': trial.suggest_categorical('factors',[32,64,100]),
            'iterations': trial.suggest_categorical('iterations',[20,30,40])           
        }
        rec_model = ImplicitALSWrapperModel(models[model](**params),
                                            fit_features_together = True)
        
    elif model == 'LightFM':
        params = {
            'random_state': RANDOM_STATE,
            'loss': trial.suggest_categorical('loss',['logistic', 'bpr', 'warp']),
            'no_components': trial.suggest_categorical('no_components',
                                                       [32,64,100]),
            'learning_rate': trial.suggest_loguniform('learning_rate',
                                                       1e-5, 0.99),
            'user_alpha': trial.suggest_loguniform('user_alpha',
                                                       1e-5, 0.99),
            'item_alpha': trial.suggest_loguniform('item_alpha',
                                                       1e-5, 0.99)
        }
        rec_model = LightFMWrapperModel(models[model](**params),
                                        epochs=N_EPOCHS, num_threads=NUM_THREADS)
        
    rec_model.fit(dataset)
    recos = rec_model.recommend(
        users=valid_user_ids,
        dataset=dataset,
        k=top_k,
        filter_viewed=True,
    )
    metric_values = calc_metrics(metric, recos, valid)
    map10 = metric_values['MAP@10']
    
    return map10


study = optuna.create_study(direction = 'maximize')

func = lambda trial: opt_metric(trial, dataset = dataset, valid_user_ids = VALID_USERS,
                                top_k = K_RECOS, metric = metrics, model = 'LightFM')

study.optimize(func, n_jobs = -1, n_trials = 1, show_progress_bar = True)


# Сохранение лучшей модели
best_light_fm_params = study.best_params
best_light_fm_params['random_state'] = 42
best_model = LightFMWrapperModel(LightFM(**best_light_fm_params),
                                        epochs=N_EPOCHS, num_threads=NUM_THREADS)

best_model.fit(dataset)

#популярные айтемы для холодных пользователей
popular_model = PopularModel()
popular_model.fit(dataset)

dump(best_model,"new_models/main_model.joblib")
dump(popular_model,"new_models/popular_model.joblib")
dump(dataset,"new_models/dataset.joblib")