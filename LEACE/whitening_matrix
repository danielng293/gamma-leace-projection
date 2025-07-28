def est_W(X_train , eps=1e-6):

    # get the covariance matrix
    Cov_X = np.cov(X_train, rowvar=False)

    # get the eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(Cov_X)

    # for numerical stability
    eigenvals[eigenvals < eps] = 0

    # Take reciprocal square root of non-zero eigenvalues
    diag = np.zeros_like(eigenvals)
    nonzero = eigenvals > 0
    diag[nonzero] = 1.0 / np.sqrt(eigenvals[nonzero])

    # Compute whitening matrix
    W = np.dot(eigenvecs, np.dot(np.diag(diag), eigenvecs.T))

    return W
