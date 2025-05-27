from preprocess import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

def aggregate_monthly_sales(train):
    monthly_sales = (
    train.groupby(['date_block_num', 'shop_id', 'item_id'])
    .agg(item_cnt_month=('item_cnt_day', 'sum'))
    .reset_index()
)
    return monthly_sales

def create_grid(train,monthly_sales):
    #Create shop-month-item grid
    grid = []
    for block_num in range(34):
        cur_shops = train[train['date_block_num'] == block_num]['shop_id'].unique()
        cur_items = train[train['date_block_num'] == block_num]['item_id'].unique()
        grid += list(np.array(np.meshgrid([block_num], cur_shops, cur_items)).T.reshape(-1, 3))

    grid_df = pd.DataFrame(grid, columns=['date_block_num', 'shop_id', 'item_id'])

    # Merge with aggregated monthly sales
    full_data = pd.merge(grid_df, monthly_sales, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    full_data['item_cnt_month'] = full_data['item_cnt_month'].fillna(0)
    return full_data

#Creating Lag Features
def create_lag_feature(df, lags, col):
    for lag in lags:
        temp = df[['date_block_num', 'shop_id', 'item_id', col]].copy()
        temp['date_block_num'] += lag
        temp = temp.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_month'].mean().reset_index()
        temp.rename(columns={col: f'{col}_lag_{lag}'}, inplace=True)
        df = pd.merge(df, temp, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return df

def add_rolling_means(df):
    df['item_cnt_month_rolling_mean_3'] = df[['item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3']].mean(axis=1)
    return df
