import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import plotly.express as px 

def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(y_pred: pd.Series, y_test: pd.Series):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        y_pred: predictions of the price.
        y_test: Testing data for price.
    """
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)

def predict_y(regressor: LinearRegression, X_test: pd.DataFrame):
    """Generates predictions using the trained model.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
    
        
    Returns:
        Predictions for the given test data.
    """
    return regressor.predict(X_test)

def plot_results(y_test: pd.Series, y_pred: pd.Series) -> px.scatter:
    """Plot the results of the model predictions.
    
    Args:
        y_test: pd.Series
        y_pred: pd.Series
    
    Returns:
        fig: plotly.graph_objs.Figure
    """
    
    fig = px.scatter(x=y_test, y=y_pred, labels={"x": "True", "y": "Predicted"})    
    fig.update_layout(title="Model Predictions")
    return fig
