import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
    X = np.column_stack((np.ones(len(X)), X)) # add bias term
    return X @ w

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def question_4(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestRegressor

    # Train RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=800, max_depth=40, verbose=2, n_jobs=5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mse(y_test, y_pred_rf)

    print(f'MSE for RandomForestRegressor: {mse_rf}')

def main():
    # Load and preprocess data
    exercise = pd.read_csv('exercise.csv')
    calories = pd.read_csv('calories.csv')
    data = pd.concat([exercise, calories], axis=1)

    # Drop 'User_ID' column
    data = data.drop(columns=['User_ID'])

    # Convert 'Gender' to numeric
    data['Gender'] = data['Gender'].map({'male': -1, 'female': 1})

    train_data, valid_data, test_data = split_data(data)

    # Train models
    X_train = train_data.drop(columns=['Calories']).values
    y_train = train_data['Calories'].values
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

    question_4(X_train, y_train, X_test, y_test)

    # Plot true vs predicted values
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_mlr)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('MLR Model')

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_blr)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('BLR Model')

    plt.tight_layout()
    plt.show()

    

if __name__ == "__main__":
    main()