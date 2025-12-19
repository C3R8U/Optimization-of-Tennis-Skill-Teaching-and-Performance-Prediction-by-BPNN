import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_synthetic_data(n_subjects=120, n_months=6, random_seed=42):
    """
    Generates synthetic tennis training data to simulate the study's dataset.
    Simulates the correlations between physiological/technical features and performance.
    """
    np.random.seed(random_seed)
    total_samples = n_subjects * n_months

    # Feature generation with realistic ranges
    # 1. Forehand Success Rate (0.4 - 0.9)
    forehand = np.random.uniform(0.4, 0.9, total_samples)

    # 2. Backhand Success Rate (0.3 - 0.8)
    backhand = np.random.uniform(0.3, 0.8, total_samples)

    # 3. Serve Placement Accuracy (distance in meters, lower is better, 0.2 - 1.5)
    serve_acc = np.random.uniform(0.2, 1.5, total_samples)

    # 4. Movement Speed (m/s, 2.0 - 6.0)
    speed = np.random.uniform(2.0, 6.0, total_samples)

    # 5. Training Duration (minutes per week, 300 - 900)
    duration = np.random.uniform(300, 900, total_samples)

    # 6. Resting Heart Rate (RHR, 45 - 80 bpm)
    rhr = np.random.uniform(45, 80, total_samples)

    # 7. HRV (RMSSD in ms, 20 - 100, higher is better)
    hrv = np.random.uniform(20, 100, total_samples)

    # Generate Target Score (0-100) based on non-linear relationships + noise
    # Logic: Higher Acc, Speed, HRV, Duration -> Higher Score. Lower Serve Dist, RHR -> Higher Score.
    score = (
            25 * forehand +
            20 * backhand +
            15 * (2.0 - serve_acc) +  # Invert distance
            10 * (speed / 6.0) +
            15 * (hrv / 100.0) +  # High weight for HRV based on paper findings
            5 * (duration / 900.0) -
            5 * (rhr / 80.0) +
            np.random.normal(0, 5, total_samples)  # Add Gaussian noise
    )

    # Clip scores to 0-100
    score = np.clip(score, 0, 100)

    # Create DataFrame
    data = pd.DataFrame({
        'Forehand_Rate': forehand,
        'Backhand_Rate': backhand,
        'Serve_Accuracy': serve_acc,
        'Movement_Speed': speed,
        'Training_Duration': duration,
        'RHR': rhr,
        'HRV': hrv,
        'Score': score,
        'Subject_ID': np.repeat(range(n_subjects), n_months),
        'Month': np.tile(range(1, n_months + 1), n_subjects)
    })

    return data


def preprocess_data(df):
    """
    Standardizes features and performs chronological splitting.
    Split: Month 1-4 (Train), Month 5 (Val), Month 6 (Test).
    """
    feature_cols = ['Forehand_Rate', 'Backhand_Rate', 'Serve_Accuracy',
                    'Movement_Speed', 'Training_Duration', 'RHR', 'HRV']
    target_col = 'Score'

    # Chronological Split
    train_data = df[df['Month'] <= 4]
    val_data = df[df['Month'] == 5]
    test_data = df[df['Month'] == 6]

    X_train, y_train = train_data[feature_cols].values, train_data[target_col].values
    X_val, y_val = val_data[feature_cols].values, val_data[target_col].values
    X_test, y_test = test_data[feature_cols].values, test_data[target_col].values

    # Standardization (Z-score) - Fit on Train, transform Val/Test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test), feature_cols