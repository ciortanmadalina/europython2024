import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def get_data(path):
    """
    Load the dataset from a given path.
    This method supports the ADBench files https://github.com/Minqi824/ADBench

    Add here logic to load your own dataset.

    Parameters:
    - path (str): The path to the dataset file.

    Returns:
    - X (ndarray): The input features of the dataset.
    - y (ndarray): The target labels of the dataset.
    """
    try:
        data = np.load(path, allow_pickle=True)
        X, y = data['X'], data['y']
        return X, y
    except FileNotFoundError:
        print("File not found")
        return None, None


def plot_dataset(X, y):
    """
    Plots the dataset and its principal component analysis (PCA) representation.

    Parameters:
    - X (array-like): The input dataset.
    - y (array-like): The target values.

    Returns:
    None
    """
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    plt.figure(figsize=(5, 3))
    plt.plot(y)
    plt.xlabel('Time steps')
    plt.ylabel('Is anomaly')
    plt.title('Ground truth labels of the dataset (0: Normal, 1: Anomaly)')
    plt.show()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(5, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.5, cmap='coolwarm')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Timestep-based PCA of the dataset\n(Color code 0: Normal, 1: Anomaly)')
    plt.colorbar()
    plt.show()


def save_results(path, results):
    """
    Save the results to a given path.

    Parameters:
    - path (str): The path to save the results.
    - results (dict): The results to save.

    Returns:
    None
    """
    # if path contains a folder that does not exist, create it

    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    np.savez(path, **results)
    print(f"Results saved to {path}")

def load_results(path):
    """
    Load the results from a given path.

    Parameters:
    - path (str): The path to load the results.

    Returns:
    - results (dict): The loaded results.
    """
    try:
        results = np.load(path, allow_pickle=True)
        return dict(results)
    except FileNotFoundError:
        print("File not found")
        return None