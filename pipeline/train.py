import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from preprocess import *
from features import *

train, items, shops, test,item_categories=load_data()
train=preprocess(train)
monthly_sales=aggregate_monthly_sales(train)
full_data=create_grid(train,monthly_sales)
full_data = create_lag_feature(full_data, [1, 2, 3], 'item_cnt_month')
full_data=add_rolling_means(full_data)

# Filter training data (exclude the last month for validation)
X = full_data[full_data['date_block_num'] < 33].drop(['item_cnt_month'], axis=1)
y = full_data[full_data['date_block_num'] < 33]['item_cnt_month']

# Validation on month 33
X_val = full_data[full_data['date_block_num'] == 33].drop(['item_cnt_month'], axis=1)
y_val = full_data[full_data['date_block_num'] == 33]['item_cnt_month']

# Train model
model = XGBRegressor(max_depth=8, n_estimators=100, learning_rate=0.1, n_jobs=-1)
model.fit(X, y)

# Predict
y_pred = model.predict(X_val).clip(0, 20)

# Evaluate
rmse = mean_squared_error(y_val, y_pred)
print("Validation RMSE:", rmse)

model.save_model('model.json')
joblib.dump(model, "model.joblib")