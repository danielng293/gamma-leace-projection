# 1. Experiment config
angles = [0, 30, 60, 90]
pareto_all = []
labels = [f"{a}°" for a in angles]

# 2. Experiment Iteration
d = 10
n_samples = 5000
n_iter = 5
gammas = np.linspace(0, 1, 100)
angles = [0, 30, 60, 90]

results = {}

X = np.random.normal(size=(n_samples, d))
for angle in angles:
    errors_all = np.zeros((n_iter, len(gammas)))
    for it in range(n_iter):
        beta_y, beta_z = generate_beta_pair_with_angle(d, angle)
        y = X @ beta_y + np.random.normal(size=n_samples)
        z = X @ beta_z + np.random.normal(size=n_samples)

        X_tr, X_te, y_tr, y_te, z_tr, z_te = train_test_split(
            X, y, z, test_size=0.2
        )

        for i, γ in enumerate(gammas):
            P = alpha_LEACE(X_tr, z_tr, γ)
            Xp_tr = X_tr @ P
            Xp_te = X_te @ P

            model = RandomForestRegressor(
                n_estimators=20, max_depth=10, n_jobs=-1, random_state=42
            )
            model.fit(Xp_tr, y_tr)
            y_pred = model.predict(Xp_te)
            errors_all[it, i] = mean_squared_error(y_te, y_pred)

    mean_err = errors_all.mean(axis=0)
    ci_err = 1.96 * errors_all.std(axis=0) / np.sqrt(n_iter)
    results[angle] = (mean_err, ci_err)

# 3. Visualization
plt.figure(figsize=(8, 6))
colors = {
    0:  '#E41A1C',  # 빨강
    30: '#377EB8',  # 파랑
    60: '#4DAF4A',  # 초록
    90: '#FF7F00',  # 주황
}

for idx, angle in enumerate(angles):
    col = colors[angle]
    mean_err, ci_err = results[angle]

    points = np.column_stack((gammas, mean_err))
    pareto_pts = get_pareto_front(points)  
    pareto_idx = [np.where(np.isclose(gammas, γ_val))[0][0]
                  for γ_val in pareto_pts[:,0]]

    plt.plot(gammas, mean_err, color=col, linewidth=1.5, marker='o', markevery=pareto_idx, markersize=5, label=f"{angle}°")
    plt.fill_between(gammas, mean_err - ci_err, mean_err + ci_err, alpha=0.2, color=col)

plt.xlabel("Covariance Ratio $\\gamma$",fontsize=12)
plt.ylabel("Mean Squared Error",fontsize=12)
plt.legend(title="Angle between $\\beta_y$ and $\\beta_z$",fontsize=11, title_fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure3.png", dpi=300, bbox_inches='tight')
plt.show()
