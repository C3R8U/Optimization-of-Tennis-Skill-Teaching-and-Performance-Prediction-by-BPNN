import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf

from data_utils import generate_synthetic_data, preprocess_data
from models import build_bpnn_model, get_baselines
from analysis import garsons_algorithm

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{model_name}] MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
    return mae, rmse, r2


def main():
    print("--- 1. Data Generation & Preprocessing ---")
    raw_df = generate_synthetic_data(n_subjects=120, n_months=6)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names = preprocess_data(raw_df)

    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Validation Samples: {X_val.shape[0]}")
    print(f"Testing Samples: {X_test.shape[0]}")

    print("\n--- 2. Training Baselines ---")
    baselines = get_baselines()
    results = {}

    for name, model in baselines.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = evaluate_model(y_test, y_pred, name)

    print("\n--- 3. Training BPNN (Proposed Model) ---")
    bpnn = build_bpnn_model(input_dim=X_train.shape[1], hidden_neurons=15)

    # Early stopping to prevent overfitting
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = bpnn.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=16,
        callbacks=[callback],
        verbose=0  # Set to 1 to see training logs
    )

    y_pred_bpnn = bpnn.predict(X_test).flatten()
    results['BPNN'] = evaluate_model(y_test, y_pred_bpnn, 'BPNN')

    print("\n--- 4. Interpretability Analysis (Garson's Algorithm) ---")
    garson_results = garsons_algorithm(bpnn, feature_names)
    print(garson_results)

    print("\n--- 5. Saving Results ---")
    # Plotting Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('BPNN Training Process')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    print("Training loss plot saved as 'training_loss.png'")


if __name__ == "__main__":
    main()