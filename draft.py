import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from flask import Flask, render_template, request, jsonify
import joblib
import os
import warnings
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

app = Flask(__name__)

data = pd.read_csv('student-mat1.csv', sep=',')
le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])
    label_ecoders = le

joblib.dump(label_ecoders, 'label_ecoders.pkl')


target = 'G3'
features = [col for col in data.columns if col!=target]
X = data[features]
y = data['G3']


# data['school'] = le.fit_transform(data['school'])
# data['gender'] = le.fit_transform(data['gender'])
# data['schoolsup'] = le.fit_transform(data['schoolsup'])
# data['famsup'] = le.fit_transform(data['famsup'])
# data['paid'] = le.fit_transform(data['paid'])
# data['activities'] = le.fit_transform(data['activities'])

# features = ['school', 'gender', 'age', 'medu', 'fedu', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'freetime', 'goout', 'health', 'absences', 'G1', 'G2']
# X = data[features]
# y = data['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
mlp_model = MLPRegressor(hidden_layer_sizes=(64,64), max_iter=1000, random_state=42)


linear_model.fit(X_train_scaled, y_train)
ridge_model.fit(X_train_scaled, y_train)
mlp_model.fit(X_train_scaled, y_train)

def xuly(model):
    coefficients = model.coef_

    
#bagging
bagging_model = BaggingRegressor(estimator=LinearRegression(), n_estimators=10, random_state=42)
bagging_model.fit(X_train_scaled, y_train)

y_pred_bagging = bagging_model.predict(X_test_scaled)
mse_bagging = mean_squared_error(y_test, y_pred_bagging)
r2_bagging = r2_score(y_test, y_pred_bagging)

#stacking
estimators = [
    ('linear', linear_model),
    ('mlp', mlp_model)
]
stacking_model = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0))
stacking_model.fit(X_train_scaled, y_train)

y_pred_stacking = stacking_model.predict(X_test_scaled)
mse_stacking = mean_squared_error(y_test, y_pred_stacking)
r2_stacking = r2_score(y_test, y_pred_stacking)


def plot_prediction(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=3) 
    plt.xlabel('Giá trị thực tế (y_test)')
    plt.ylabel('Giá trị dự đoán (y_pred)')
    plt.title(f'Biểu đồ dự đoán - {model_name}')
    plt.show()

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

# scores = cross_val_score(linear_model, X_train_scaled, y_train)
# print(scores)

coefficients = ridge_model.coef_

# # Vẽ biểu đồ hệ số hồi quy
# features = X.columns
# indices = np.argsort(coefficients)

# plt.figure(figsize=(10, 6))
# plt.title('Feature Importance (Linear Regression Coefficients)')
# plt.barh(range(len(indices)), coefficients[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Coefficient Value')
# plt.show()
####Chạy thử

# plt.plot(mlp_model.loss_curve_)
# plt.title('Training Loss Curve')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.show()

# row_index = 300
# input_data = X.iloc[row_index].values.reshape(1, -1) 
# warnings.filterwarnings("ignore", message="X does not have valid feature names")  
# actual_g3 = y.iloc[row_index]
# print(f"Điểm thực tế (G3): {actual_g3}")
# print(f"Điểm dự đoán (G3): {score_prediction(input_data,stacking_model)}")



# for i in range (len(y_test)):
    # warnings.filterwarnings("ignore", message="X does not have valid feature names")    
#     atual = y_test.iloc[i]
#     input_data = X_test.iloc[i].values.reshape(1, -1)
#     input_data_scaled = scaler.transform(input_data)
#     L_predict = linear_model.predict(input_data_scaled)
#     R_predict = ridge_model.predict(input_data_scaled)
#     S_predict = stacking_model.predict(input_data_scaled)
#     B_predict = bagging_model.predict(input_data_scaled)
#     print(f"{i}\t{atual}\t{L_predict[0]}\t{R_predict[0]}\t{S_predict[0]}\t{B_predict[0]}")

#evalue model
# print("\t\tmse\t\tr2\t\tmae\t\tnse")    
# print(f"linear: {evaluate_model(linear_model)}")
# print(f"ridge: {evaluate_model(ridge_model)}")
# print(f"mlp: {evaluate_model(mlp_model)}")
# print(f"stack: {evaluate_model(stacking_model)}")
# print(f"bagging: {evaluate_model(bagging_model)}")

# y_pred = linear_model.predict(X_test_scaled)
# plot_prediction(stacking_model, X_test_scaled, y_test, "Stacking")
# print(evaluate_model(linear_model))
# print(nse(y_test, y_pred))



# print(f"Bagging - MSE: {mse_bagging}, R2: {r2_bagging}")




# print(f"Stacking - MSE: {mse_stacking}, R2: {r2_stacking}")


# RUN WEB

models = {
    'Linear': linear_model,
    'Ridge': ridge_model,
    'MLP': mlp_model,
    'Stacking': stacking_model,
    'Bagging': bagging_model
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form('model')

    # if model_name == "Linear":
    #         model = linear_model
    # elif model_name == "Ridge":
    #     model = ridge_model
    # elif model_name == "MLP":
    #     model = mlp_model
    # elif model_name == "Bagging":
    #     model = bagging_model
    # else:
    #     model = stacking_model
    model = models[model_name]

    age = float(request.form['age'])
    Medu = float(request.form['Medu'])
    Fedu = float(request.form['Fedu'])
    studytime = float(request.form['studytime'])
    failures = float(request.form['failures'])
    famrel = float(request.form['famrel'])
    freetime = float(request.form['freetime'])
    goout = float(request.form['goout'])
    Dalc = float(request.form['Dalc'])
    Walc = float(request.form['Walc'])
    health = float(request.form['health'])
    absences = float(request.form['absences'])

    # Tạo một numpy array từ dữ liệu nhập
    input_data = np.array([[age, Medu, Fedu, studytime, failures, famrel, freetime, goout, Dalc, Walc, health, absences]])

        
    prediction = score_prediction(input_data,model)
    evaluate = evaluate_model(model)
    plot_path = plot_evaluate(model, model_name)

    return jsonify({'prediction' : prediction, 'evaluate' : evaluate, 'plot_path' : plot_path})    
    

        

if __name__ == '__main__':
    app.run(debug=True)

