import tensorflow as tf
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from data_loader import load_dataset
from dcgan import make_discriminator_model
from pilae import PILAE

# Paths
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(base_path, "Dataset")
models_dir = os.path.join(base_path, "models")

def get_feature_extractor(discriminator_path):
    """
    Loads the discriminator and creates a feature extractor model.
    """
    print(f"Loading discriminator from {discriminator_path}...")
    discriminator = tf.keras.models.load_model(discriminator_path)
    
    # The 4th Conv block output is at index 11 (Conv(512) -> Leaky -> Drop)
    # Let's verify:
    # 0: Conv64, 1: Leaky, 2: Drop
    # 3: Conv128, 4: Leaky, 5: Drop
    # 6: Conv256, 7: Leaky, 8: Drop
    # 9: Conv512, 10: Leaky, 11: Drop
    # 12: Flatten
    
    # We want the output of layer 11 (Dropout) which is (B, 4, 4, 512)
    # Then we flatten it to (B, 8192)
    
    output = discriminator.layers[11].output
    flat_output = tf.keras.layers.Flatten()(output)
    
    model = tf.keras.Model(inputs=discriminator.inputs, outputs=flat_output)
    return model

def extract_features_and_labels(model, dataset):
    """
    Runs the dataset through the model and collects features and labels.
    """
    print("Extracting features...")
    features_list = []
    labels_list = []
    
    for images, labels in dataset:
        feats = model(images, training=False)
        features_list.append(feats.numpy())
        labels_list.append(labels.numpy())
        
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    return features, labels

def train_classifier():
    # 1. Load Data
    # Note: load_dataset returns a batched dataset.
    # We need to split it into train/test manually or use the loader's logic if we modified it.
    # For simplicity, let's load all and split here, or assume the loader can handle splits (it currently doesn't).
    # Let's just use a simple random split on the extracted features.
    
    ds, count, class_names = load_dataset(dataset_dir, batch_size=128)
    
    # 2. Load Feature Extractor
    # We need a trained discriminator.
    # Check for 'current' model (saved every epoch) or 'final' model
    disc_path_current = os.path.join(models_dir, "discriminator_current.h5")
    disc_path_final = os.path.join(models_dir, "discriminator_final.h5")
    
    if os.path.exists(disc_path_current):
        print(f"Found intermediate model: {disc_path_current}")
        disc_path = disc_path_current
    elif os.path.exists(disc_path_final):
        print(f"Found final model: {disc_path_final}")
        disc_path = disc_path_final
    else:
        print("Error: Trained discriminator not found. Please run train_gan.py first.")
        # For testing purposes, we can create a fresh one
        print("Creating a fresh discriminator for testing (random weights)...")
        discriminator = make_discriminator_model()
        # Same logic for fresh model
        output = discriminator.layers[11].output
        flat_output = tf.keras.layers.Flatten()(output)
        feature_extractor = tf.keras.Model(inputs=discriminator.inputs, outputs=flat_output)
    else:
        feature_extractor = get_feature_extractor(disc_path)

    # 3. Extract Features
    X, y = extract_features_and_labels(feature_extractor, ds)
    print(f"Extracted features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # 4. Split Data (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. One-Hot Encode Labels for PILAE
    enc = OneHotEncoder(sparse_output=False)
    Y_train_oh = enc.fit_transform(y_train.reshape(-1, 1))
    
    # 6. Train PILAE
    pilae = PILAE(input_dim=X_train.shape[1])
    pilae.fit(X_train, Y_train_oh)
    
    # 7. Evaluate
    y_pred_train = pilae.predict(X_train)
    y_pred_test = pilae.predict(X_test)
    
    # Map back to class indices from one-hot if needed, but predict returns indices
    
    print("\nTraining Results:")
    print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    
    print("\nTest Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=class_names))
    
    # 8. Save Model
    pilae.save(os.path.join(models_dir, "pilae_model.npz"))
    print("PILAE model saved.")

if __name__ == "__main__":
    train_classifier()
