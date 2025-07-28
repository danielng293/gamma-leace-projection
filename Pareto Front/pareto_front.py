import numpy as np
import matplotlib.pyplot as plt

mean_errors = np.load("results/mean_errors.npy")
ci_errors = np.load("results/ci_errors.npy")
mean_covs = np.load("results/mean_covs.npy")
ci_covs = np.load("results/ci_covs.npy")

# 1. Trade-off curve
plt.figure(figsize=(8, 6))
plt.plot(gammas, mean_errors, marker='o', linestyle='-', color='blue', label = 'Mean Trade-off')
plt.fill_between(gammas, mean_errors-ci_errors, mean_errors+ci_errors, color = 'blue', alpha = 0.2, label = '95% CI')
plt.xlabel("Covariance Ratio $\\gamma$",fontsize=12)
plt.ylabel("Mean Squared Error",fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("Figure1.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Pareto Front
points = np.array(list(zip(gammas, mean_errors)))

def get_pareto_front(points):
    pareto = []
    for i, (x_i, y_i) in enumerate(points):
        dominated = False
        for j, (x_j, y_j) in enumerate(points):
            if i != j and x_j <= x_i and y_j <= y_i and (x_j < x_i or y_j < y_i):
                dominated = True
                break
        if not dominated:
            pareto.append((x_i, y_i))
    pareto = np.array(pareto)
    return pareto[np.argsort(pareto[:, 0])]  

pareto_points = get_pareto_front(points)

plt.figure(figsize=(8, 6))
plt.scatter(pareto_points[:, 0], pareto_points[:, 1], color='red', label='Pareto Optimal Points')
plt.plot(pareto_points[:, 0], pareto_points[:, 1], color='red',linewidth=2, label='Pareto Front')
plt.scatter(gammas, mean_errors, marker='o', linestyle='-', color='blue',alpha=0.2, label = 'Mean Trade-off')
plt.fill_between(gammas, mean_errors-ci_errors, mean_errors+ci_errors, color = 'blue', alpha = 0.2, label = '95% CI')

plt.xlabel("Covariance Ratio $\\gamma$",fontsize=12)
plt.ylabel("Mean Squared Error", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Figure2.png", dpi=300, bbox_inches='tight')
plt.show()
