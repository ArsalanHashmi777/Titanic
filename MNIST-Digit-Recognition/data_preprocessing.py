# data_preprocessing.py
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def load_and_prep_mnist():
    """
    Fetches MNIST, splits it, and returns scaled data.
    """
    print("Loading MNIST (784 features)...")
    # as_frame=False ensures we get NumPy arrays, which are faster for images
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist["data"], mnist["target"]

    # Normalizing pixels to [0, 1] range (Essential for ML stability)
    X = X / 255.0

    # Professional 60k/10k split
    return train_test_split(X, y, test_size=10000, random_state=42)
    
def train_and_evaluate_xgb(X_train, y_train, X_test, y_test, use_gpu=False):
    """
    Encodes labels, trains an XGBoost model, and returns the model and predictions.
    """
    # 1. Encode labels (MNIST fetch_openml targets are often strings)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # 2. Initialize Model
    # 'gpu_hist' is for Kaggle/NVIDIA GPUs, 'hist' is for CPU
    method = 'gpu_hist' if use_gpu else 'hist'
    
    print(f"Initializing XGBoost with tree_method='{method}'...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        tree_method=method,
        random_state=42,
        n_jobs=-1 # Use all available CPU cores
    )

    # 3. Train
    print("Training XGBoost (this may take a few minutes on CPU)...")
    xgb_model.fit(X_train, y_train_enc)
    
    # 4. Predict
    predictions = xgb_model.predict(X_test)
    
    # Inverse transform back to original labels (strings like '0', '1'...)
    decoded_preds = le.inverse_transform(predictions)
    
    return xgb_model, decoded_preds


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prep_mnist()
    print(f"Data ready. Training shape: {X_train.shape}")