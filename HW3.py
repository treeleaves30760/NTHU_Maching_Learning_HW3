import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
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

def plot_results(X, y, y_pred, feature_names, model_name):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.ravel()

    for i in range(X.shape[1]):
        axs[i].scatter(X[:, i], y, label='Actual')
        axs[i].scatter(X[:, i], y_pred, label='Predicted')

        # Fit a line using np.polyfit and plot it
        coef = np.polyfit(X[:, i], y_pred, 1)
        poly1d_fn = np.poly1d(coef)
        axs[i].plot(X[:, i], poly1d_fn(X[:, i]), color='red')

        axs[i].set_xlabel(feature_names[i])
        axs[i].set_ylabel('Calories')
        axs[i].set_title(f'{model_name}: Calories vs {feature_names[i]}')
        axs[i].legend()

    # Remove extra subplots
    for i in range(X.shape[1], len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()

def find_optimal_regression_model(X_train, X_test, y_train, y_test):
    models = {
        'Neural Networks': MLPRegressor(hidden_layer_sizes=(400, ), activation='relu', solver='adam', max_iter=1000),
        'SVR': SVR(),
        'Ridge':Ridge(),
        'Elastic Net': ElasticNet(),
        'Linear Regression': LinearRegression(),
        'Degree-2 Polynomial Regression': make_pipeline(PolynomialFeatures(2), LinearRegression()),
        'Degree-3 Polynomial Regression': make_pipeline(PolynomialFeatures(3), LinearRegression()),
        'Degree-4 Polynomial Regression': make_pipeline(PolynomialFeatures(4), LinearRegression()),
        'Decision Tree Regression': DecisionTreeRegressor(),
        'Random Forest Regression': RandomForestRegressor(n_estimators=400, n_jobs=5, random_state=42),
        'Gradient Boosting Regression': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=0, loss='squared_error')
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

    print(f'Mean Squared Error of MLR: {mse_mlr}')
    print(f'Mean Squared Error of BLR: {mse_blr}')

    # Find the optimal model
    min_mse, best_model_name = find_optimal_regression_model(X_train, X_test, y_train, y_test)
    print(f'Best model is {best_model_name} with MSE {min_mse:.6f}')

    # Generate predictions for the validation set using the BLR model
    y_preds_blr = predict(X_valid, w_blr)

    feature_names = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    plot_results(X_test, y_test, y_pred_mlr, feature_names, 'MLR')
    plot_results(X_test, y_test, y_pred_blr, feature_names, 'BLR')

if __name__ == "__main__":
    main()
