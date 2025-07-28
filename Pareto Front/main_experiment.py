#Requirements
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from Dataset.synthetic_generator import generate_simulated_data
from LEACE.gamma_controlled import gamma_LEACE  # 함수 이름이 다르면 맞게 수정

# 1. Generate data
X,y,z = generate_simulated_data(theta=np.pi/3)

# 2. Experiment config
n_iter = 10
gammas = np.linspace(0, 1, 200)
errors_all = np.zeros((n_iter, len(gammas)))
covariances_all = np.zeros((n_iter, len(gammas)))
errors = []
covariances = []

# 3. Run experiment
for iteration in range(n_iter):
    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, test_size=0.2)
    
    for i,gamma in enumerate(gammas):
        P = gamma_LEACE(X_train, z_train, gamma)
        X_train_proj = X_train @ P
        X_test_proj = X_test @ P

        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1,max_depth=10)
        model.fit(X_train_proj, y_train)
        y_pred = model.predict(X_test_proj)
        mse = mean_squared_error(y_test, y_pred)
        errors_all[iteration, i] = mse

        X_proj_centered = X_test_proj - X_test_proj.mean(axis=0)
        z_centered = z_test - z_test.mean()
        cov_vec = X_proj_centered.T @ z_centered / (len(z) - 1)
        cov_proj = np.linalg.norm(cov_vec)  
        covariances_all[iteration, i] = cov_proj


mean_errors = np.mean(errors_all, axis=0)
std_errors = np.std(errors_all, axis=0)
mean_covs = np.mean(covariances_all, axis=0)
std_covs = np.std(covariances_all, axis=0)
ci_errors = 1.96 * std_errors / np.sqrt(n_iter)
ci_covs = 1.96 * std_covs / np.sqrt(n_iter)

np.save("results/mean_errors.npy", mean_errors)
np.save("results/ci_errors.npy", ci_errors)
np.save("results/mean_covs.npy", mean_covs)
np.save("results/ci_covs.npy", ci_covs)

