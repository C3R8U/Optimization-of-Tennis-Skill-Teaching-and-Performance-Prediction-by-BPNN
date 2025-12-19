import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def garsons_algorithm(model, feature_names):
    """
    Implements Garson's Algorithm to determine relative feature importance
    based on connection weights in a 3-layer NN.

    Formula: Importance_i = sum(|w_ij * v_j|) / sum(sum(|w_kj * v_j|))
    """
    # Extract weights
    # Layer 0 weights (Input -> Hidden): (n_features, n_hidden)
    w_ih = model.layers[0].get_weights()[0]
    # Layer 1 weights (Hidden -> Output): (n_hidden, 1)
    w_ho = model.layers[1].get_weights()[0]

    n_features = w_ih.shape[0]
    n_hidden = w_ih.shape[1]

    # Calculate contribution of each input neuron via each hidden neuron
    # shape: (n_features, n_hidden)
    contribution = np.abs(w_ih) * np.abs(w_ho.T)

    # Sum contributions for each hidden neuron
    # shape: (1, n_hidden)
    hidden_sum = np.sum(contribution, axis=0)

    # Calculate relative importance for each input feature
    # shape: (n_features,)
    # Input contribution divided by total sum of contributions
    importance = np.sum(contribution / hidden_sum, axis=1)

    # Normalize to 100%
    importance = importance / np.sum(importance) * 100

    # Create DataFrame
    results = pd.DataFrame({
        'Feature': feature_names,
        'Relative_Importance_%': importance
    }).sort_values(by='Relative_Importance_%', ascending=False)

    return results


def run_shap_analysis(model, X_train, X_test, feature_names):
    """
    Runs SHAP analysis using KernelExplainer (model-agnostic for Keras).
    """
    # Using a subset of background data for speed
    background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]

    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(X_test[0:50])  # Analyze first 50 test samples

    return shap_values, explainer