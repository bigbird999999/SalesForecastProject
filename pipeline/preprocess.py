import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

def load_data():
    try:
        train = pd.read_csv('data/sales_train.csv')
        items = pd.read_csv('data/items.csv')
        item_categories = pd.read_csv('data/item_categories.csv')
        shops = pd.read_csv('data/shops.csv')
        test = pd.read_csv('data/test.csv')
        print("All CSV files loaded successfully.")
        return train, items, shops, test,item_categories
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Make sure all CSV files are in the same directory.")
        exit()

def preprocess(train):
    train=train[train['item_cnt_day']<1100]
    train=train[train['item_price']<100000]
    median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()
    train.loc[train.item_price<0, 'item_price'] = median
    train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
    train=train.drop_duplicates()
    return train