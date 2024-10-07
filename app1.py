import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from flask import Flask, render_template, request, redirect, url_for
import joblib
import os
import warnings
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV



data = pd.read_csv('score-mat.csv', sep=',')
le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])
    label_ecoders = le

# joblib.dump(label_ecoders, 'label_ecoders.pkl')

target = 'G3'
features = [col for col in data.columns if col!=target]
X = data[features]
y = data['G3']

scaler = StandardScaler()

linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
mlp_model = MLPRegressor(hidden_layer_sizes=(128,128), max_iter=2000, random_state=42)

def iterative_feature_elimination(X, y, threshold=0):
    prev_column_count = 0  # Biến đếm số lượng cột từ lần lặp trước
    current_column_count = X.shape[1]  # Số lượng cột hiện tại

    while current_column_count != prev_column_count:  # Lặp cho đến khi số lượng cột không thay đổi
        prev_column_count = current_column_count

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ridge_model.fit(X_train_scaled, y_train)

        coefficients = ridge_model.coef_

        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': coefficients})

        important_features = importance_df[importance_df['Importance'] >= threshold]['Feature'].tolist()

        X = X[important_features]

        current_column_count = len(important_features)

        print(f"Số lượng cột sau khi lặp: {current_column_count}")

    return X, coefficients

X_filtered ,coef = iterative_feature_elimination(X, y)
# X_filtered = X

# print(f"Các cột quan trọng còn lại: {important_features}")
print(X_filtered.head())
joblib.dump(X_filtered, "Data_filtered.pkl")
# Vẽ biểu đồ hệ số hồi quy
# coefficients = coef
# features = X_filtered.columns
# indices = np.argsort(coefficients)

# plt.figure(figsize=(10, 6))
# plt.title('Feature Importance (Linear Regression Coefficients)')
# plt.barh(range(len(indices)), coefficients[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Coefficient Value')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#linear
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
joblib.dump(linear_model, 'Linear.pkl')

#ridge
# # Định nghĩa các giá trị alpha cần thử
# alpha_values = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# # Sử dụng GridSearchCV để tìm alpha tốt nhất
# grid_search = GridSearchCV(ridge_model, param_grid=alpha_values, cv=5, scoring='r2')

# # Huấn luyện mô hình với GridSearch
# grid_search.fit(X_train_scaled, y_train)

# # In ra alpha tốt nhất
# best_alpha = grid_search.best_params_['alpha']
# print(f"Alpha tốt nhất: {best_alpha}")

# # Đánh giá hiệu suất trên tập test
# best_ridge_model = grid_search.best_estimator_
# y_pred = best_ridge_model.predict(X_test_scaled)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
joblib.dump(ridge_model, 'Ridge.pkl')

#mlp
# param_grid = {
#     'hidden_layer_sizes': [(64,), (128,), (64, 64), (128, 64), (64, 64, 64), (128,128)],
#     'max_iter': [1500,2000],
    
# }

mlp = MLPRegressor()
# grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, scoring='r2')
# grid_search.fit(X_train_scaled, y_train)

# print(f"Best hidden_layer_sizes: {grid_search.best_params_}")
mlp_model = MLPRegressor(hidden_layer_sizes=(128,128),activation='relu', solver='adam', alpha=0.1, max_iter=1500, early_stopping=True, random_state=42)
mlp_model.fit(X_train_scaled, y_train)
joblib.dump(mlp_model, 'MLP.pkl')



#bagging
bagging_model = BaggingRegressor(estimator=LinearRegression(), n_estimators=10, random_state=42)
bagging_model.fit(X_train_scaled, y_train)
joblib.dump(bagging_model, 'Bagging.pkl')

# y_pred_bagging = bagging_model.predict(X_test_scaled)
# mse_bagging = mean_squared_error(y_test, y_pred_bagging)
# r2_bagging = r2_score(y_test, y_pred_bagging)

#stacking
estimators = [
    ('linear', linear_model),
    ('mlp', mlp_model)
]
stacking_model = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0))
stacking_model.fit(X_train_scaled, y_train)
joblib.dump(stacking_model, 'Stacking.pkl')

# y_pred_stacking = stacking_model.predict(X_test_scaled)
# mse_stacking = mean_squared_error(y_test, y_pred_stacking)
# r2_stacking = r2_score(y_test, y_pred_stacking)

def score_prediction(input_data, model):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]
    return prediction

def nse(observed, predicted):
    return 1 - (np.sum((observed - predicted)**2) / np.sum((observed - np.mean(predicted))**2))

def evaluate_model(model):
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    nses = nse(y_test, y_pred) 
    return mse, r2, mae, nses

def plot_prediction(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=3) 
    plt.xlabel('Giá trị thực tế (y_test)')
    plt.ylabel('Giá trị dự đoán (y_pred)')
    plt.title(f'Biểu đồ dự đoán - {model_name}')
    plt.show()

def plot_evaluate(model, model_name):
    mse, r2, mae, nses = evaluate_model(model)

    metrics = {
        'NSE': nses,
        'MAE': mae,
        'R2': r2,
        'MSE': mse
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Chỉ số đánh giá
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    # Vẽ biểu đồ cột
    ax.barh(metric_names, metric_values, color=['skyblue', 'lightgreen', 'salmon', 'lightcoral'])
    ax.set_title(f'Evaluation Metrics for {model_name}', fontsize=14)
    ax.set_xlabel('Score')
    

    # Điều chỉnh bố cục để các biểu đồ không bị đè lên nhau
    plt.tight_layout()
    # plt.show()

    plot_path = os.path.join('web\static\\', f'{model_name}.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def evaluate_overfitting_with_plot(model, X_train, X_test, y_train, y_test, model_name):
    # Dự đoán trên tập huấn luyện và kiểm tra
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Tính toán các chỉ số đánh giá
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # In kết quả
    print(f"{model_name}")
    print("Hiệu suất trên tập huấn luyện:")
    print(f"Train MSE: {train_mse:.4f}, Train R²: {train_r2:.4f}, Train MAE: {train_mae:.4f}")
    print("Hiệu suất trên tập kiểm tra:")
    print(f"Test MSE: {test_mse:.4f}, Test R²: {test_r2:.4f}, Test MAE: {test_mae:.4f}")

    # So sánh chênh lệch giữa các chỉ số
    labels = ['MSE', 'MAE', 'R²']
    train_scores = [train_mse, train_mae, train_r2]
    test_scores = [test_mse, test_mae, test_r2]

    x = np.arange(len(labels))  # Tạo trục x cho các chỉ số

    width = 0.35  # Độ rộng của các cột

    # Vẽ biểu đồ so sánh MSE, MAE, R² cho tập huấn luyện và kiểm tra
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, train_scores, width, label='Train', color='skyblue')
    ax.bar(x + width/2, test_scores, width, label='Test', color='salmon')

    # Đặt nhãn và tiêu đề cho biểu đồ
    ax.set_xlabel('Chỉ số đánh giá')
    ax.set_title(f'So sánh hiệu suất giữa tập huấn luyện và tập kiểm tra {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.show()
#####Kiểm tra quá khớp#####

##Chạy code
models = {
    linear_model : 'Linear',
    ridge_model : 'Ridge',
    mlp_model: 'MLP',
    stacking_model: 'Stacking',
    bagging_model : 'Bagging'

}

# linear_model = joblib.load('Linear.pkl')
# ridge_model = joblib.load('Ridge.pkl')
# mpl_model = joblib.load('MLP.pkl')
# stacking_model = joblib.load('Stacking.pkl')
# bagging_model = joblib.load('Bagging.pkl')

# for model in models:
#     evaluate_overfitting_with_plot(model, X_train_scaled, X_test_scaled, y_train, y_test, models[model])

# for model in models:
#     plot_prediction(model, X_test_scaled, y_test, models[model])

print("\t\tmse\t\tr2\t\tmae\t\tnse")    
print(f"linear: {evaluate_model(linear_model)}")
print(f"ridge: {evaluate_model(ridge_model)}")
print(f"mlp: {evaluate_model(mlp_model)}")
print(f"stack: {evaluate_model(stacking_model)}")
print(f"bagging: {evaluate_model(bagging_model)}")


# for i in range (len(y_test)):
#     warnings.filterwarnings("ignore", message="X does not have valid feature names")    
#     atual = y_test.iloc[i]
#     input_data = X_test.iloc[i].values.reshape(1, -1)
#     input_data_scaled = scaler.transform(input_data)
    
#     L_predict = linear_model.predict(input_data_scaled)
#     R_predict = ridge_model.predict(input_data_scaled)
#     M_predict = mlp_model.predict(input_data_scaled)
#     S_predict = stacking_model.predict(input_data_scaled)
#     B_predict = bagging_model.predict(input_data_scaled)
#     print(f"{i}\t{atual}\t{L_predict[0]}\t{R_predict[0]}\t{M_predict[0]}\t{S_predict[0]}\t{B_predict[0]}")

plot_prediction(ridge_model, X_test_scaled, y_test, "Ridge")
# plot_prediction(stacking_model, X_test_scaled, y_test, "Stacking")
# plot_prediction(bagging_model, X_test_scaled, y_test, "bagging")

