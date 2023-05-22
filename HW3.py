import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
from typing import Any, Tuple

def split_data(data, train_frac=0.7, valid_frac=0.1, test_frac=0.2):
    train_data, temp = train_test_split(data, test_size=1-train_frac)
    valid_data, test_data = train_test_split(temp, test_size=test_frac/(valid_frac+test_frac)) 
    return train_data, valid_data, test_data

def MLR(X, y):
    X = np.column_stack((np.ones(X.shape[0]), X)) # add bias term
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w

def BLR(X, y, alpha=1, beta=1):
    X = np.column_stack((np.ones(X.shape[0]), X)) # add bias term
    S_inv = alpha * np.eye(X.shape[1]) + beta * X.T @ X
    S = np.linalg.inv(S_inv)
    m = beta * S @ X.T @ y
    return m, S

def predict(X, w):
    X = np.column_stack((np.ones(X.shape[0]), X)) # add bias term
    return X @ w

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def find_optimal_regression_model(X_train, X_test, y_train, y_test,
):
    models = {
        'Linear Regression': LinearRegression(),
        'Degree-2 Polynomial Regression': make_pipeline(
            PolynomialFeatures(2), LinearRegression()),
        'Degree-3 Polynomial Regression': make_pipeline(
            PolynomialFeatures(3), LinearRegression()),
        'Decision Tree Regression': DecisionTreeRegressor(),
        'Random Forest Regression': RandomForestRegressor(
            n_estimators=100, random_state=42),
        'Gradient Boosting Regression': GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=4, random_state=0, loss='squared_error'),
        'Neural Networks': MLPRegressor(
            hidden_layer_sizes=(100, ), activation='relu', solver='adam', max_iter=1000),
        'SVR': SVR(),
        'Elastic Net': ElasticNet()
    }

    min_mse = np.Infinity
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        if mse < min_mse:
            min_mse = mse
            best_model_name = name
        print(f'Mean Squared Error of {name}: {mse:.6f}')

    return min_mse, best_model_name

def main():
    # Load and preprocess data
    exercise = pd.read_csv('exercise.csv')
    calories = pd.read_csv('calories.csv')
    data = pd.concat([exercise, calories], axis=1)

    # Drop 'User_ID' column
    data = data.drop(columns=['User_ID'])

    # Convert 'Gender' to numeric
    data['Gender'] = data['Gender'].map({'male': 0, 'female': 1})

    train_data, valid_data, test_data = split_data(data)

    # Train models
    X_train = train_data.drop(columns=['Calories']).values
    y_train = train_data['Calories'].values
    X_valid = valid_data.drop(columns=['Calories']).values
    y_valid = valid_data['Calories'].values
    X_test = test_data.drop(columns=['Calories']).values
    y_test = test_data['Calories'].values

    w_mlr = MLR(X_train, y_train)
    w_blr, _ = BLR(X_train, y_train)

    # Predict and evaluate
    y_pred_mlr = predict(X_test, w_mlr)
    y_pred_blr = predict(X_test, w_blr)
    mse_mlr = mse(y_test, y_pred_mlr)
    mse_blr = mse(y_test, y_pred_blr)

    print(f'MSE for MLR: {mse_mlr}')
    print(f'MSE for BLR: {mse_blr}')

    # Find the optimal model
    min_mse, best_model_name = find_optimal_regression_model(X_train, X_test, y_train, y_test)
    print(f'Best model is {best_model_name} with MSE {min_mse:.6f}')

    # Generate predictions for the validation set using the BLR model
    y_preds_blr = predict(X_valid, w_blr)


if __name__ == "__main__":
    main()
