import numpy as np

def generate_simulated_data(n_samples=5000, d=10, theta=np.pi/3, seed=42):
    np.random.seed(seed)
    # Generate X
    X = np.random.normal(0,1, size= (n_samples, d))

    # Generate beta_y
    beta_y = np.random.randn(d)
    beta_y = beta_y / np.linalg.norm(beta_y)

    # Generate a random unit vector orthogonal to beta_y
    v = np.random.randn(d)
    v -= v.dot(beta_y) * beta_y
    v = v / np.linalg.norm(v)

    # Generate beta_z
    beta_z = np.cos(theta) * beta_y + np.sin(theta) * v

    # Generate target variable y and sensitive attribute z
    epsilon_y = np.random.randn(n_samples)
    epsilon_z = np.random.randn(n_samples)

    y = X @ beta_y + epsilon_y
    z = X @ beta_z + epsilon_z

    return X, y, z
