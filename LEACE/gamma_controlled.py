def gamma_LEACE(X_train, z_train, gamma):

    # Whitening Matrix   
    W_train = est_W(X_train)

    # Whitening the data
    XW_train = X_train @ W_train

    # Calculating v, the whitened covariance between X and z
    XW_train_centered = XW_train - XW_train.mean(axis=0)
    z_centered = z_train - z_train.mean()
    n = len(z_train)
    v = XW_train_centered.T @ z_centered / (n-1)
    v = np.asarray(v)
    v_col = v.reshape(-1, 1) 

    # define the orthogonal projection matrix
    P_proj_v = (v_col @ v_col.T)/np.dot(v, v)

    # using this, define the projection matrix with multiplier
    W_pinv = np.linalg.pinv(W_train)
    I_d = np.eye(W_pinv.shape[0])
    P = I_d - (1-np.sqrt(gamma))*(W_pinv @ P_proj_v @ W_train)

    return P.T
