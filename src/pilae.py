import numpy as np
import scipy.linalg
from sklearn.utils.extmath import randomized_svd
import gc  # For manual garbage collection

class PILAE:
    """
    Pseudoinverse Learning Autoencoder (PILAE) implementation.
    Based on the paper methodology (Pages 6-7):
    - Uses low-rank approximation via SVD
    - Hidden layer dimensionality reduction with β parameter
    - Weight decay regularization with k parameter
    """
    def __init__(self, input_dim, hidden_dim=None, beta=0.9, k=1e-5):
        """
        Args:
            input_dim: Dimension of input features (e.g., 8192).
            hidden_dim: Dimension of hidden layer. If None, computed as β * input_dim
            beta: Dimensionality reduction factor β ∈ (0,1], default 0.9 (paper uses this)
            k: Regularization parameter for weight decay (paper equation 10)
        """
        self.input_dim = input_dim
        self.beta = beta
        self.k = k
        
        # Hidden neurons: p = β * Dim(x) as per paper equation (4)
        if hidden_dim is None:
            self.hidden_dim = int(beta * input_dim)
        else:
            self.hidden_dim = hidden_dim
            
        self.encoder_weights = None  # We (encoder weight matrix)
        self.decoder_weights = None  # Wd (decoder weight matrix)
        self.weights = None  # Final classification weights

    def fit(self, X, Y):
        """
        Trains the PILAE classifier using the pseudoinverse method with low-rank approximation.
        Based on paper equations (2)-(10):
        1. SVD: X = U Σ V^T
        2. Low-rank approximation: X̂^+ = V̂ Σ'^(-1) U^T (first p rows of V)
        3. Encoder weight: We = X̂^+
        4. Hidden representation: H = X @ We^T
        5. Decoder with regularization: W = (H^T H + kI)^(-1) H^T Y
        
        Args:
            X: Input features (N, input_dim)
            Y: One-hot encoded labels (N, num_classes)
        """
        print(f"Training PILAE with input shape {X.shape} and target shape {Y.shape}")
        print(f"Hidden dimension: {self.hidden_dim} (β={self.beta}, input_dim={self.input_dim})")
        
        # Memory optimization: Convert to float32 (cuts memory usage in half)
        if X.dtype == np.float64:
            print("Converting to float32 for memory efficiency...")
            X = X.astype(np.float32)
            Y = Y.astype(np.float32)
        
        # Determine number of components to compute
        p = min(self.hidden_dim, X.shape[0], X.shape[1])
        print(f"Computing {p} components for low-rank approximation...")
        
        # Step 1 & 2: Use randomized SVD for memory efficiency (computes only first p components)
        # Much faster and uses less memory than full SVD
        # X ≈ U_p Σ_p V_p^T where we only compute first p components
        print("Computing truncated SVD (memory-efficient)...")
        try:
            U_p, Sigma_p, VT_p = randomized_svd(X, n_components=p, random_state=42)
            print(f"✓ Truncated SVD complete: U{U_p.shape}, Σ{Sigma_p.shape}, V^T{VT_p.shape}")
        except MemoryError:
            print("⚠ Memory error with truncated SVD, trying smaller subset...")
            p = p // 2  # Reduce by half
            U_p, Sigma_p, VT_p = randomized_svd(X, n_components=p, random_state=42)
            print(f"✓ Used reduced dimension: {p}")
        
        # Step 2: Low-rank pseudoinverse approximation (paper equation 5)
        # For encoder: we only need the projection from feature space to hidden space
        # W_e = V_p^T @ Σ_p^(-1) (NOT including U_p^T which would give sample dimension)
        print("Computing encoder weights (input_dim → hidden_dim)...")
        Sigma_inv = 1.0 / Sigma_p  # Just the diagonal values
        encoder_weights = VT_p.T * Sigma_inv  # (8192, 7372) element-wise scaling by Sigma_inv
        
        # Free memory immediately
        del U_p, Sigma_p, VT_p, Sigma_inv
        gc.collect()
        
        # Step 3: Encoder weight We (paper page 7)
        self.encoder_weights = encoder_weights.astype(np.float32)
        del encoder_weights
        gc.collect()
        print(f"Encoder weights shape: {self.encoder_weights.shape} (should be {(X.shape[1], p)})")
        
        # Step 4: Compute hidden representation H = X @ We^T
        H = X @ self.encoder_weights  # (N x input_dim) @ (input_dim x p) = (N x p)
        print(f"Hidden representation H shape: {H.shape}")
        
        # Free X memory after computing H
        del X
        gc.collect()
        
        # Step 5: Compute decoder/classification weights with regularization (paper equation 10)
        # W = (H^T H + kI)^(-1) H^T Y
        print(f"Computing classification weights with regularization k={self.k}...")
        
        HTH = H.T @ H  # (p x p)
        reg_term = self.k * np.eye(HTH.shape[0])
        HTH_reg = HTH + reg_term
        
        # W = (H^T H + kI)^(-1) H^T Y
        HTH_inv = scipy.linalg.inv(HTH_reg)
        self.weights = (HTH_inv @ H.T @ Y).astype(np.float32)  # (p x p) @ (p x N) @ (N x classes) = (p x classes)
        
        # Free memory
        del H, HTH, HTH_reg, HTH_inv, Y
        gc.collect()
        
        print(f"Final classification weights shape: {self.weights.shape}")
        print("PILAE training complete.")

    def predict(self, X):
        """
        Predicts class labels for X.
        Uses hidden representation: H = X @ We, then logits = H @ W
        """
        if self.weights is None or self.encoder_weights is None:
            raise ValueError("Model not trained yet.")
        
        # Transform to hidden representation
        H = X @ self.encoder_weights  # (N x input_dim) @ (input_dim x p) = (N x p)
        
        # Compute logits using classification weights
        logits = H @ self.weights  # (N x p) @ (p x classes) = (N x classes)
        
        return np.argmax(logits, axis=1)

    def predict_proba(self, X):
        """
        Predicts class probabilities (softmax).
        """
        if self.weights is None or self.encoder_weights is None:
            raise ValueError("Model not trained yet.")
            
        # Transform to hidden representation
        H = X @ self.encoder_weights
        
        # Compute logits
        logits = H @ self.weights
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def save(self, filepath):
        np.savez(filepath, 
                 encoder_weights=self.encoder_weights,
                 weights=self.weights,
                 beta=self.beta,
                 k=self.k,
                 hidden_dim=self.hidden_dim)

    def load(self, filepath):
        data = np.load(filepath)
        self.encoder_weights = data['encoder_weights']
        self.weights = data['weights']
        self.beta = data['beta']
        self.k = data['k']
        self.hidden_dim = data['hidden_dim']
