import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(data, train_frac=0.7, valid_frac=0.1, test_frac=0.2):
    train_data, temp = train_test_split(data, test_size=1-train_frac)
    valid_data, test_data = train_test_split(temp, test_size=test_frac/(valid_frac+test_frac)) 
    return train_data, valid_data, test_data

def MLR(X, y):
    X = np.column_stack((np.ones(len(X)), X)) # add bias term
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w

def BLR(X, y, alpha=1, beta=1):
    X = np.column_stack((np.ones(len(X)), X)) # add bias term
    S_inv = alpha * np.eye(X.shape[1]) + beta * X.T @ X
    S = np.linalg.inv(S_inv)
    m = beta * S @ X.T @ y
    return m, S

def predict(X, w):
    X = np.column_stack((np.ones(len(X)), X)) # add bias term
    return X @ w

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)**0.5

def main():
    # Load and preprocess data
    exercise = pd.read_csv('exercise.csv')
    calories = pd.read_csv('calories.csv')
    data = pd.concat([exercise, calories], axis=1)
    train_data, valid_data, test_data = split_data(data)

    # Train models
    w_mlr = MLR(train_data['Duration'].values, train_data['Calories'].values)
    w_blr, _ = BLR(train_data['Duration'].values, train_data['Calories'].values)

    # Predict and evaluate
    y_pred_mlr = predict(test_data['Duration'].values, w_mlr)
    y_pred_blr = predict(test_data['Duration'].values, w_blr)
    mse_mlr = mse(test_data['Calories'].values, y_pred_mlr)
    mse_blr = mse(test_data['Calories'].values, y_pred_blr)

    print(f'MSE for MLR: {mse_mlr}')
    print(f'MSE for BLR: {mse_blr}')

if __name__ == "__main__":
    main()

