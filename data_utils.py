import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pysad.utils import Data

def get_data(name = 'data/6_cardio.npz'):
    """
    Load the dataset.

    Parameters:
    name (str): The name of the dataset file to load. Default is 'data/6_cardio.npz'.

    Returns:
    X (ndarray): The input features of the dataset.
    y (ndarray): The target labels of the dataset.
    """
    if name == 'data/6_cardio.npz':
        data = np.load(name, allow_pickle=True)
        X, y = data['X'], data['y']
        
    elif name == 'data/arrhythmia.mat':
        # https://odds.cs.stonybrook.edu/arrhythmia-dataset/
        data = Data("data")
        X, y = data.get_data("arrhythmia.mat")

    ##################################################    
    ########### Insert here your dataset #############
    ##################################################
    else:
        raise FileNotFoundError("Dataset not found")
    
    return X, y
    


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

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(5, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('PCA of the dataset')
    plt.legend()
    plt.show()
