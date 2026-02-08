#!/usr/bin/env python3
"""
Dynamic ECG Classification Pipeline - Adjusts to any number of columns
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

print("✓ All imports successful")

# =========================
# CONFIG
# =========================
CSV_PATH = "ecg_data/sample_ecg.csv"

# =========================
# LOAD DATA
# =========================
print(f"Loading: {CSV_PATH}")

try:
    df = pd.read_csv(CSV_PATH)
    print(f"✓ Dataset loaded successfully")
    print(f"Shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Show sample data
    print(f"\nFirst 3 rows of data:")
    print(df.head(3))
    
except FileNotFoundError:
    print(f"❌ File not found: {CSV_PATH}")
    print("Creating sample data for testing...")
    
    # Create sample ECG data with variable columns
    np.random.seed(42)
    n_samples = 1000
    n_columns = np.random.randint(10, 200)  # Random number of columns between 10-200
    
    print(f"Creating sample data with {n_columns} columns...")
    
    # Create synthetic ECG data with variable columns
    column_names = [f'ecg_{i}' for i in range(n_columns)]
    X = np.random.randn(n_samples, n_columns)
    
    # Create labels (70% normal, 30% abnormal)
    y = np.zeros(n_samples)
    y[int(n_samples*0.7):] = 1  # Last 30% are abnormal
    
    # Save for testing
    df = pd.DataFrame(X, columns=column_names)
    df['label'] = y
    os.makedirs('ecg_data', exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"✓ Created sample data at: {CSV_PATH} with {n_columns} features")

# =========================
# DYNAMIC DATA PREPARATION
# =========================
print("\n" + "="*50)
print("DYNAMIC DATA PREPARATION")
print("="*50)

# Find label column if it exists
label_columns = [col for col in df.columns if col.lower() in ['label', 'target', 'class', 'diagnosis', 'outcome']]

if label_columns:
    # Use the first label column found
    label_col = label_columns[0]
    print(f"Found label column: '{label_col}'")
    
    y = df[label_col].values
    
    # Use all other columns as features
    feature_cols = [col for col in df.columns if col != label_col]
    X = df[feature_cols].values
    
    print(f"Using {len(feature_cols)} feature columns")
    
else:
    print("⚠️ No label column found. Creating synthetic labels...")
    
    # Check data characteristics to decide how to handle
    print(f"Data info:")
    print(f"  Number of columns: {len(df.columns)}")
    print(f"  Column dtypes: {df.dtypes.unique()}")
    
    # Strategy 1: If all columns are numeric, use all as features
    if all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
        print("✓ All columns are numeric. Using all columns as features.")
        X = df.values
        
    # Strategy 2: If mixed types, use only numeric columns
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"Using {len(numeric_cols)} numeric columns as features")
        X = df[numeric_cols].values
    
    # Create synthetic labels
    n_samples = len(X)
    y = np.zeros(n_samples)
    
    # Smart label generation based on data patterns
    if n_samples > 0:
        # Calculate some statistics to create meaningful labels
        row_means = np.mean(X, axis=1)
        row_stds = np.std(X, axis=1)
        
        # Create labels based on data patterns
        # Example: Label as abnormal if std > mean std or mean > 2*overall mean
        mean_std_threshold = np.percentile(row_stds, 75)  # 75th percentile
        mean_value_threshold = np.percentile(np.abs(row_means), 75)
        
        for i in range(n_samples):
            if (row_stds[i] > mean_std_threshold or 
                np.abs(row_means[i]) > mean_value_threshold):
                y[i] = 1
        
        # Ensure we have at least some abnormal samples
        if np.sum(y == 1) < n_samples * 0.1:  # Less than 10% abnormal
            n_abnormal = int(n_samples * 0.3)
            abnormal_indices = np.random.choice(n_samples, n_abnormal, replace=False)
            y[abnormal_indices] = 1

print(f"\nFinal data shape: X={X.shape}, y={y.shape}")
print(f"Class distribution: Normal={np.sum(y==0)}, Abnormal={np.sum(y==1)}")

# =========================
# DYNAMIC PREPROCESSING
# =========================
print("\n" + "="*50)
print("DYNAMIC PREPROCESSING")
print("="*50)

# Normalize the data (important for neural networks)
print("Normalizing data...")
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std[X_std == 0] = 1  # Avoid division by zero
X = (X - X_mean) / X_std

# Reshape for model based on input size
n_samples, n_features = X.shape
print(f"Number of features: {n_features}")

# Determine optimal window size for ECG
# ECG signals typically have 187, 256, or 512 time points
if n_features < 50:
    window_size = n_features  # Use all features if small
    print(f"Small dataset ({n_features} features). Using full sequence length.")
elif n_features <= 200:
    window_size = min(187, n_features)  # Standard ECG size or available features
    print(f"Medium dataset. Using window size: {window_size}")
else:
    # For large datasets, we can downsample or use standard size
    window_size = 256 if n_features >= 256 else 187
    print(f"Large dataset ({n_features} features). Using window size: {window_size}")

# If we have more features than window_size, we need to adjust
if n_features > window_size:
    print(f"Downsampling from {n_features} to {window_size} features...")
    
    # Strategy 1: Select evenly spaced features
    indices = np.linspace(0, n_features-1, window_size, dtype=int)
    X = X[:, indices]
    
    print(f"New X shape after downsampling: {X.shape}")

# Ensure X has the right shape for Conv1D (samples, timesteps, features)
if len(X.shape) == 2:
    # Add channel dimension
    X = X[..., np.newaxis]

print(f"Final X shape for model: {X.shape}")

# =========================
# TRAIN/TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# =========================
# DYNAMIC MODEL ARCHITECTURE
# =========================
print("\n" + "="*50)
print("DYNAMIC MODEL ARCHITECTURE")
print("="*50)

# Get input shape
input_shape = X_train.shape[1:]
timesteps = input_shape[0]

print(f"Creating model for input shape: {input_shape}")
print(f"Timesteps: {timesteps}")

# Dynamically adjust model architecture based on input size
if timesteps < 50:
    # Simple model for small sequences
    print("Using simple model for small sequences...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
elif timesteps < 200:
    # Standard ECG model
    print("Using standard ECG model architecture...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # First Conv1D layer
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # Second Conv1D layer
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # Third Conv1D layer
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
else:
    # Advanced model for long sequences
    print("Using advanced model for long sequences...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # Multiple Conv1D layers with increasing filters
        tf.keras.layers.Conv1D(filters=16, kernel_size=7, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # Bidirectional LSTM for sequence learning
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dropout(0.3),
        
        # Dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print("\nModel Summary:")
model.summary()

# =========================
# DYNAMIC TRAINING
# =========================
print("\n" + "="*50)
print("DYNAMIC TRAINING")
print("="*50)

print("Starting training...")

# Adjust batch size based on dataset size
n_train = len(X_train)
if n_train > 10000:
    batch_size = 64
elif n_train > 5000:
    batch_size = 32
else:
    batch_size = 16

print(f"Using batch size: {batch_size}")

# Adjust epochs based on dataset size
if n_train < 1000:
    epochs = 20
else:
    epochs = 15

# Add callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=1
)

# =========================
# EVALUATION
# =========================
print("\n" + "="*50)
print("EVALUATION")
print("="*50)

# Evaluate
print("Evaluating model...")
loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Loss: {loss:.4f}")
print(f"✅ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"✅ Test AUC: {auc:.4f}")

# Predictions
y_pred = model.predict(X_test, verbose=0)
y_pred_class = (y_pred > 0.5).astype(int).flatten()

# Calculate metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print("\nClassification Report:")
print(classification_report(y_test, y_pred_class, target_names=['Normal', 'Abnormal']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_class))

roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC AUC Score: {roc_auc:.4f}")

# =========================
# SAVE MODEL AND METADATA
# =========================
print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)

# Save model
model_filename = f'ecg_model_{timesteps}features.h5'
model.save(model_filename)
print(f"✅ Model saved as '{model_filename}'")

# Save metadata about the model
metadata = {
    'input_shape': input_shape,
    'timesteps': timesteps,
    'n_features_original': n_features,
    'n_samples': n_samples,
    'accuracy': float(accuracy),
    'auc': float(auc),
    'class_distribution': {
        'normal': int(np.sum(y == 0)),
        'abnormal': int(np.sum(y == 1))
    },
    'data_stats': {
        'X_mean': X_mean.tolist() if n_features <= 100 else 'too_large',
        'X_std': X_std.tolist() if n_features <= 100 else 'too_large'
    }
}

import json
with open(f'model_metadata_{timesteps}features.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Model metadata saved")

# =========================
# TEST WITH SAMPLE DATA
# =========================
print("\n" + "="*50)
print("TESTING WITH SAMPLE DATA")
print("="*50)

# Create multiple test samples
n_test_samples = 5
test_samples = []
predictions = []

for i in range(n_test_samples):
    # Create realistic test data
    if i == 0:
        # Normal sample (low variance)
        sample = np.random.randn(1, timesteps, 1) * 0.1
    elif i == 1:
        # Abnormal sample (high variance)
        sample = np.random.randn(1, timesteps, 1) * 0.5
    else:
        # Random sample
        sample = np.random.randn(1, timesteps, 1) * np.random.uniform(0.1, 0.5)
    
    # Normalize like training data
    sample_normalized = sample  # In production, apply same normalization as training
    
    prediction = model.predict(sample_normalized, verbose=0)
    probability = float(prediction[0][0])
    predicted_class = "ABNORMAL" if probability > 0.5 else "NORMAL"
    
    test_samples.append(sample)
    predictions.append((probability, predicted_class))
    
    print(f"\nSample {i+1}:")
    print(f"  Shape: {sample.shape}")
    print(f"  Prediction: {predicted_class}")
    print(f"  Probability: {probability:.4f}")
    print(f"  Confidence: {max(probability, 1-probability)*100:.2f}%")

# =========================
# EXPORT FOR WEB USE
# =========================
print("\n" + "="*50)
print("EXPORTING FOR WEB DASHBOARD")
print("="*50)

# Save sample predictions for testing
np.save('sample_test_data.npy', {
    'X_test_samples': X_test[:5],
    'y_test_samples': y_test[:5],
    'y_pred_samples': y_pred[:5]
})
print("✅ Sample test data saved to 'sample_test_data.npy'")

# Try to export to TensorFlow.js format
try:
    import tensorflowjs as tfjs
    tfjs_dir = f'tfjs_model_{timesteps}features'
    tfjs.converters.save_keras_model(model, tfjs_dir)
    print(f"✅ Model exported in TensorFlow.js format to '{tfjs_dir}/' folder")
    
    # Create a simple loader script for the web
    loader_script = f"""
// ECG Model Loader for {timesteps} features
const ECG_MODEL_CONFIG = {{
    inputShape: [{timesteps}, 1],
    modelPath: '{tfjs_dir}/model.json',
    normalization: {{
        mean: {X_mean.tolist() if n_features <= 100 else '[]'},
        std: {X_std.tolist() if n_features <= 100 else '[]'}
    }}
}};
    """
    
    with open('model_config.js', 'w') as f:
        f.write(loader_script)
    print("✅ Model configuration saved to 'model_config.js'")
    
except ImportError:
    print("⚠️ tensorflowjs not installed. To export for web:")
    print("   pip install tensorflowjs")
    print("   Then run: tensorflowjs_converter --input_format=keras {model_filename} tfjs_model")

print("\n" + "="*50)
print("SCRIPT COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"\nSummary:")
print(f"- Input shape: {input_shape}")
print(f"- Original features: {n_features}")
print(f"- Model features: {timesteps}")
print(f"- Accuracy: {accuracy*100:.2f}%")
print(f"- AUC: {auc:.4f}")
print(f"- Model saved: {model_filename}")