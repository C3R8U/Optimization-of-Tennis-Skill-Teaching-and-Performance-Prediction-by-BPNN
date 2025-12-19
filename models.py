import tensorflow as tf
from tensorflow.keras import models, layers, regularizers, optimizers
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


def build_bpnn_model(input_dim, hidden_neurons=15, l2_reg=0.001, learning_rate=0.01):
    """
    Constructs the 3-layer BPNN as described in the paper.
    - Input Layer: 7 neurons
    - Hidden Layer: 15 neurons, Sigmoid activation, He Initialization
    - Output Layer: 1 neuron, Linear activation
    - Regularization: L2
    """
    model = models.Sequential()

    # Hidden Layer
    model.add(layers.Dense(
        hidden_neurons,
        input_dim=input_dim,
        activation='sigmoid',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='Hidden_Layer'
    ))

    # Output Layer
    model.add(layers.Dense(
        1,
        activation='linear',
        kernel_initializer='glorot_normal',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='Output_Layer'
    ))

    # Optimizer: SGD with Momentum
    optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def get_baselines():
    """Returns a dictionary of configured baseline models."""
    return {
        "LR (Ridge)": Ridge(alpha=1.0),
        "SVR": SVR(kernel='rbf', C=10, gamma=0.1, epsilon=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }