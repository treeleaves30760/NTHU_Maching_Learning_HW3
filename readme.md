# Regression Models Comparison

This Python script compares various regression models on a dataset of exercise and calorie burn. The dataset includes features such as gender, age, height, weight, duration of exercise, heart rate, and body temperature.

## Dependencies

The script requires the following Python libraries:

- pandas
- numpy
- sklearn
- matplotlib

## Data

The data is expected to be in two CSV files: `exercise.csv` and `calories.csv`. These files should be in the same directory as the script. The `exercise.csv` file should contain the exercise data, and the `calories.csv` file should contain the corresponding calorie burn data.

The data is expected to have the following columns:

- User_ID
- Gender
- Age
- Height
- Weight
- Duration
- Heart_Rate
- Body_Temp
- Calories

The 'User_ID' column is dropped during preprocessing, and the 'Gender' column is converted to numeric (0 for male, 1 for female).

## Usage

To run the script, simply execute the Python file in your command line:

```bash
python regression_models_comparison.py
```

## Functionality

The script first splits the data into training, validation, and testing sets. It then trains two types of linear regression models: Multiple Linear Regression (MLR) and Bayesian Linear Regression (BLR).

The script evaluates these models using Mean Squared Error (MSE) and plots the actual vs predicted values for each feature.

Next, the script trains a variety of other regression models, including:

- Neural Networks
- Support Vector Regression (SVR)
- Ridge Regression
- Elastic Net Regression
- Polynomial Regression (degrees 2, 3, and 4)
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression

The script then compares the MSE of these models and selects the model with the lowest MSE as the optimal model.

Finally, the script generates predictions for the validation set using the BLR model and plots the results.

## Output

The script outputs the MSE for the MLR and BLR models, the name and MSE of the optimal model, and plots of the actual vs predicted values for each feature for the MLR and BLR models.