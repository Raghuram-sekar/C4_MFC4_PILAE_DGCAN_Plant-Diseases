import numpy as np
import scipy.linalg

class PILAE:
    """
    Pseudoinverse Learning Autoencoder (PILAE) implementation.
    """
    def __init__(self, input_dim, hidden_dim=None):
        """
        Args:
            input_dim: Dimension of input features (e.g., 8192).
            hidden_dim: Dimension of hidden layer. If None, can be set later or unused if single layer.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weights = None

    def fit(self, X, Y):
        """
        Trains the PILAE classifier using the pseudoinverse method.
        W = H^+ * Y
        
        Args:
            X: Input features (N, input_dim)
            Y: One-hot encoded labels (N, num_classes)
        """
        # In the simplest PILAE for classification (as per paper description often implies):
        # We map Input -> Output directly or Input -> Hidden -> Output.
        # The paper mentions "PILAE output classifier includes better-generalized effectiveness when SLFN can be used to train PILAE."
        # And "weights computed analytically using matrix pseudoinverse".
        
        # Let's assume a single layer mapping from Features -> Classes for the classifier part
        # W = pinv(X) * Y
        
        print(f"Training PILAE with input shape {X.shape} and target shape {Y.shape}")
        
        # Add bias term to X
        # X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        
        # Calculate Pseudoinverse of X
        # H^+ = (H^T H)^-1 H^T  or just scipy.linalg.pinv
        # Using pinv2 (SVD based) is more stable
        
        print("Computing pseudoinverse...")
        X_pinv = scipy.linalg.pinv(X)
        
        print("Computing weights...")
        self.weights = np.dot(X_pinv, Y)
        
        print("PILAE training complete.")

    def predict(self, X):
        """
        Predicts class labels for X.
        """
        if self.weights is None:
            raise ValueError("Model not trained yet.")
        
        # X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        logits = np.dot(X, self.weights)
        return np.argmax(logits, axis=1)

    def predict_proba(self, X):
        """
        Predicts class probabilities (softmax).
        """
        if self.weights is None:
            raise ValueError("Model not trained yet.")
            
        logits = np.dot(X, self.weights)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def save(self, filepath):
        np.savez(filepath, weights=self.weights)

    def load(self, filepath):
        data = np.load(filepath)
        self.weights = data['weights']
